import argparse
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def load_kitti_flow(flow_path):
    # KITTI flow PNG: 16-bit 3-channel, B=valid mask, G=v, R=u
    # decode: real = (stored - 2^15) / 64
    raw   = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
    u     = (raw[:, :, 2].astype(float) - 2**15) / 64.0
    v     = (raw[:, :, 1].astype(float) - 2**15) / 64.0
    valid = raw[:, :, 0] > 0
    return u, v, valid


def visualize_flow(u, v):
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_metrics(u_pred, v_pred, u_gt, v_gt, valid):
    u_pred[~valid] = 0; v_pred[~valid] = 0
    u_gt  [~valid] = 0; v_gt  [~valid] = 0
    error = np.sqrt((u_pred - u_gt)**2 + (v_pred - v_gt)**2)
    msen  = np.mean(error[valid])
    pepn  = np.mean(error[valid] > 3.0) * 100
    return msen, pepn


def save_flow_img(u, v, path, title=""):
    bgr = visualize_flow(u, v)
    plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_pyflow(img1_path, img2_path):
    import pyflow

    img1 = np.array(Image.open(img1_path).convert("L"), dtype=float) / 255.0
    img2 = np.array(Image.open(img2_path).convert("L"), dtype=float) / 255.0
    img1 = img1[:, :, np.newaxis]
    img2 = img2[:, :, np.newaxis]

    t0 = time.time()
    u, v, _ = pyflow.coarse2fine_flow(
        img1, img2,
        alpha=0.012, ratio=0.75, minWidth=20,
        nOuterFPIterations=7, nInnerFPIterations=1,
        nSORIterations=30, colType=1,
    )
    print(f"  PyFlow elapsed: {time.time() - t0:.2f}s")
    return u, v


def run_flowformer(img1_path, img2_path, checkpoint, ff_dir, target_h, target_w):
    if ff_dir not in sys.path:
        sys.path.insert(0, ff_dir)

    from core.FlowFormer import build_flowformer
    from core.utils.utils import InputPadder
    from configs.kitti import get_cfg

    cfg   = get_cfg()
    model = nn.DataParallel(build_flowformer(cfg))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    def _load(path):
        gray = np.array(Image.open(path).convert("L"))
        rgb  = np.stack([gray, gray, gray], axis=-1)
        t    = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return t[None].cuda()

    img1, img2 = _load(img1_path), _load(img2_path)
    padder = InputPadder(img1.shape)
    img1p, img2p = padder.pad(img1, img2)

    t0 = time.time()
    with torch.no_grad():
        preds = model(img1p, img2p, {})
    print(f"  FlowFormerPlusPlus elapsed: {time.time() - t0:.2f}s")

    flow = preds[-1]
    _, _, h, w = flow.shape
    flow = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
    flow[:, 0] *= target_w / w
    flow[:, 1] *= target_h / h
    flow = flow.cpu().squeeze(0)
    return flow[0].numpy(), flow[1].numpy()


def run_neuflow(img1_path, img2_path, nf_dir, target_h, target_w,
                infer_h=432, infer_w=768):
    if nf_dir not in sys.path:
        sys.path.insert(0, nf_dir)

    from NeuFlow.neuflow import NeuFlow
    from NeuFlow.backbone_v7 import ConvBlock

    def _fuse_conv_bn(conv, bn):
        fused = (
            torch.nn.Conv2d(
                conv.in_channels, conv.out_channels,
                kernel_size=conv.kernel_size, stride=conv.stride,
                padding=conv.padding, dilation=conv.dilation,
                groups=conv.groups, bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )
        w = torch.mm(
            torch.diag(bn.weight / torch.sqrt(bn.eps + bn.running_var)),
            conv.weight.view(conv.out_channels, -1),
        ).view(fused.weight.shape)
        fused.weight.copy_(w)
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
        b_bn   = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
        fused.bias.copy_(
            torch.mm(
                torch.diag(bn.weight / torch.sqrt(bn.eps + bn.running_var)),
                b_conv.view(-1, 1),
            ).view(-1) + b_bn
        )
        return fused

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(device)

    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = _fuse_conv_bn(m.conv1, m.norm1)
            m.conv2 = _fuse_conv_bn(m.conv2, m.norm2)
            delattr(m, "norm1")
            delattr(m, "norm2")
            m.forward = m.forward_fuse

    model.eval().half()
    model.init_bhwd(1, infer_h, infer_w, str(device))

    def _load(path):
        gray = np.array(Image.open(path).convert("L"))
        rgb  = np.stack([gray, gray, gray], axis=-1)
        rgb  = cv2.resize(rgb, (infer_w, infer_h))
        t    = torch.from_numpy(rgb).permute(2, 0, 1).half()
        return t[None].to(device)

    img1, img2 = _load(img1_path), _load(img2_path)

    t0 = time.time()
    with torch.no_grad():
        flow_raw = model(img1, img2)[-1][0]
    print(f"  NeuFlow v2 elapsed: {time.time() - t0:.2f}s")

    flow_np  = flow_raw.permute(1, 2, 0).float().cpu().numpy()
    flow_rsz = cv2.resize(flow_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    flow_rsz[:, :, 0] *= target_w / infer_w
    flow_rsz[:, :, 1] *= target_h / infer_h
    return flow_rsz[:, :, 0], flow_rsz[:, :, 1]


def run_gmflow(img1_path, img2_path, checkpoint, gmflow_dir, target_h, target_w):
    if gmflow_dir not in sys.path:
        sys.path.insert(0, gmflow_dir)

    from gmflow.gmflow import GMFlow

    model = GMFlow(
        num_scales=1,
        upsample_factor=8,
        feature_channels=128,
        attention_type="swin",
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def _load(path):
        gray = np.array(Image.open(path).convert("L"))
        rgb  = np.stack([gray, gray, gray], axis=-1)
        t    = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return t[None].to(device)

    img1, img2 = _load(img1_path), _load(img2_path)

    # GMFlow needs H and W divisible by 64
    padding_factor = 64
    orig_h, orig_w = img1.shape[-2:]
    pad_h = (padding_factor - orig_h % padding_factor) % padding_factor
    pad_w = (padding_factor - orig_w % padding_factor) % padding_factor
    if pad_h > 0 or pad_w > 0:
        img1 = F.pad(img1, (0, pad_w, 0, pad_h))
        img2 = F.pad(img2, (0, pad_w, 0, pad_h))

    t0 = time.time()
    with torch.no_grad():
        out = model(
            img1, img2,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            pred_bidir_flow=False,
        )
    print(f"  GMFlow elapsed: {time.time() - t0:.2f}s")

    flow = out["flow_preds"][-1][:, :, :orig_h, :orig_w]
    _, _, h, w = flow.shape
    if (h, w) != (target_h, target_w):
        flow = F.interpolate(flow, size=(target_h, target_w),
                             mode="bilinear", align_corners=False)
        flow[:, 0] *= target_w / w
        flow[:, 1] *= target_h / h
    flow = flow.cpu().squeeze(0)
    return flow[0].numpy(), flow[1].numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kitti",       default="../KITTI")
    p.add_argument("--seq",         default="000045")
    p.add_argument("--ff_ckpt",     default="FlowFormerPlusPlus/checkpoints/kitti.pth")
    p.add_argument("--gmflow_ckpt", default="GMFlow/checkpoints/gmflow_kitti.pth")
    p.add_argument("--out",         default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    img1_path = os.path.join(args.kitti, "training", "image_0", f"{args.seq}_10.png")
    img2_path = os.path.join(args.kitti, "training", "image_0", f"{args.seq}_11.png")
    flow_path = os.path.join(args.kitti, "training", "flow_noc", f"{args.seq}_10.png")

    u_gt, v_gt, valid = load_kitti_flow(flow_path)
    print(f"GT valid pixels: {np.sum(valid)}")
    save_flow_img(u_gt, v_gt, os.path.join(args.out, "gt_flow.png"), "Ground Truth Flow")

    H, W    = u_gt.shape
    results = {}

    print("\n[PyFlow]")
    u_pf, v_pf = run_pyflow(img1_path, img2_path)
    save_flow_img(u_pf, v_pf, os.path.join(args.out, "pyflow_flow.png"), "PyFlow")
    results["PyFlow"] = compute_metrics(u_pf.copy(), v_pf.copy(), u_gt.copy(), v_gt.copy(), valid)

    print("\n[FlowFormerPlusPlus]")
    ff_dir = os.path.join(os.path.dirname(__file__), "FlowFormerPlusPlus")
    u_ff, v_ff = run_flowformer(img1_path, img2_path, args.ff_ckpt, ff_dir, H, W)
    save_flow_img(u_ff, v_ff, os.path.join(args.out, "flowformer_flow.png"), "FlowFormerPlusPlus")
    results["FlowFormerPlusPlus"] = compute_metrics(u_ff.copy(), v_ff.copy(), u_gt.copy(), v_gt.copy(), valid)

    print("\n[NeuFlow v2]")
    nf_dir = os.path.join(os.path.dirname(__file__), "NeuFlow_v2")
    u_nf, v_nf = run_neuflow(img1_path, img2_path, nf_dir, H, W)
    save_flow_img(u_nf, v_nf, os.path.join(args.out, "neuflow_flow.png"), "NeuFlow v2")
    results["NeuFlow v2"] = compute_metrics(u_nf.copy(), v_nf.copy(), u_gt.copy(), v_gt.copy(), valid)

    print("\n[GMFlow]")
    gmflow_dir = os.path.join(os.path.dirname(__file__), "gmflow")
    u_gm, v_gm = run_gmflow(img1_path, img2_path, args.gmflow_ckpt, gmflow_dir, H, W)
    save_flow_img(u_gm, v_gm, os.path.join(args.out, "gmflow_flow.png"), "GMFlow")
    results["GMFlow"] = compute_metrics(u_gm.copy(), v_gm.copy(), u_gt.copy(), v_gt.copy(), valid)

    print(f"\n{'='*50}")
    print(f"{'Method':<25} {'MSEN':>10} {'PEPN (%)':>12}")
    print("-" * 50)
    for name, (msen, pepn) in results.items():
        print(f"{name:<25} {msen:>10.4f} {pepn:>11.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
