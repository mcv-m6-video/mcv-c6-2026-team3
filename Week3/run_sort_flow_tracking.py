import argparse
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from sort_tracker_flow import SORTTracker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"


try:
    from evaluation.tracking_eval import load_gt_tracks, evaluate_tracking
except ImportError as e:
    print(f"Warning: Could not import evaluation functions locally: {e}")
    load_gt_tracks = None
    evaluate_tracking = None

DEFAULT_VIDEO      = os.path.join(SCRIPT_DIR, "../data/AICity_data/train/S03/c010/vdo.avi")
DEFAULT_XML        = os.path.join(SCRIPT_DIR, "../data/ai_challenge_s03_c010-full_annotation.xml")
DEFAULT_DETS       = os.path.join(SCRIPT_DIR, "detections.txt")
DEFAULT_FF_DIR     = os.path.join(SCRIPT_DIR, "FlowFormerPlusPlus")
DEFAULT_FF_CKPT    = os.path.join(SCRIPT_DIR, "FlowFormerPlusPlus", "checkpoints", "kitti.pth")
DEFAULT_OUT        = os.path.join(SCRIPT_DIR, "results/tracking_sort_flow.mp4")

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid SORT + FlowFormer++ tracker")
    p.add_argument("--video",   default=DEFAULT_VIDEO,   help="Path to input video")
    p.add_argument("--xml",     default=DEFAULT_XML,     help="Path to Ground Truth XML (for metrics)")
    p.add_argument("--dets",    default=DEFAULT_DETS,    help="Path to detections .txt file")
    p.add_argument("--ff_ckpt", default=DEFAULT_FF_CKPT, help="Path to FlowFormer++ checkpoint")
    p.add_argument("--out",     default=DEFAULT_OUT,     help="Output video path")
    p.add_argument("--iou_thr", default=0.1458, type=float, help="IOU threshold for tracker")
    p.add_argument("--max_age", default=5,   type=int,   help="Max frames to keep lost track")
    p.add_argument("--min_hits", default=12, type=int, help="Min hits to confirm track")
    p.add_argument("--scale",   default=0.5, type=float, help="Scale factor for optical flow calculation")
    
    p.add_argument("--alpha",   default=0.5, type=float, help="0.0 = 100% Flow, 1.0 = 100% Kalman, 0.5 = Mix")
    return p.parse_args()

# Load FlowFormer++
if DEFAULT_FF_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_FF_DIR)
from core.FlowFormer import build_flowformer
from core.utils.utils import InputPadder
from configs.kitti import get_cfg

def load_detections(path):
    dets = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            f_id, x, y, w, h, _ = map(float, line.strip().split(','))
            dets[int(f_id)].append([x, y, w, h])
    return dets

def compute_flow_between_frames(model, prev_frame, curr_frame, scale):
    """Computes the optical flow between two frames using FlowFormer."""
    h, w = curr_frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    
    img1_small = cv2.resize(prev_frame, (new_w, new_h))
    img2_small = cv2.resize(curr_frame, (new_w, new_h))

    img1 = torch.from_numpy(cv2.cvtColor(img1_small, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).cuda()
    img2 = torch.from_numpy(cv2.cvtColor(img2_small, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).cuda()
    
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    
    with torch.no_grad():
        preds = model(img1, img2, {})
        flow = padder.unpad(preds[-1])[0].cpu().numpy()
    
    flow_u = cv2.resize(flow[0], (w, h)) * (w / new_w)
    flow_v = cv2.resize(flow[1], (w, h)) * (h / new_h)
    
    return flow_u, flow_v

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ff_dir = os.path.dirname(args.ff_ckpt[:-len("/checkpoints/kitti.pth")]) if args.ff_ckpt.endswith("/checkpoints/kitti.pth") else DEFAULT_FF_DIR
    if ff_dir not in sys.path:
        sys.path.insert(0, ff_dir)

    # Setup Model
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    checkpoint = torch.load(args.ff_ckpt, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Initializing SORT Tracker with alpha={args.alpha} (0: Flow, 1: Kalman, 0.5: Mix)")
    tracker = SORTTracker(iou_threshold=args.iou_thr, max_age=args.max_age, min_hits=args.min_hits)
    
    detections = load_detections(args.dets)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                          int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                          
    out_txt_path = args.out.replace('.mp4', '.txt')
    out_file = open(out_txt_path, 'w')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    prev_frame = None
    f_id = 0
    
    pred_tracks_for_eval = {}

    for _ in tqdm(range(total_frames) if total_frames else iter(int, 1), desc="Tracking (Hybrid SORT)"):
        ret, frame = cap.read()
        if not ret: break
        
        flow_u, flow_v = None, None
        if prev_frame is not None:
            flow_u, flow_v = compute_flow_between_frames(model, prev_frame, frame, scale=args.scale)

        frame_dets = detections.get(f_id, [])
        results = tracker.track(frame_dets, f_id, flow_u=flow_u, flow_v=flow_v, alpha=args.alpha)
        
        if flow_u is not None:
            del flow_u, flow_v
            torch.cuda.empty_cache()

        pred_tracks_for_eval[f_id] = []

        for tid, box in results:
            x1, y1, w_box, h_box = map(int, box)
            x2, y2 = x1 + w_box, y1 + h_box
            
            # Save MOT format
            out_file.write(f"{f_id + 1},{tid},{x1},{y1},{w_box},{h_box},-1,-1,-1,-1\n")
            
            # Save format for trackeval
            pred_tracks_for_eval[f_id].append([tid, [x1, y1, w_box, h_box]])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        out.write(frame)
        prev_frame = frame.copy()
        f_id += 1

    cap.release(); out.release(); out_file.close()
    
    if load_gt_tracks is not None and evaluate_tracking is not None:
        print("\nLoading GT to calculate tracking metrics...")
        try:
            train_frames = int(total_frames * 0.25) if total_frames else 535 
            gt_tracks = load_gt_tracks(args.xml, train_frames=train_frames)
            
            pred_filtered = {k: v for k, v in pred_tracks_for_eval.items() if k >= train_frames}
            
            print(f"Evaluating Tracking from frame {train_frames} to end...")
            metrics = evaluate_tracking(pred_filtered, gt_tracks)
            
            print("\n" + "="*30)
            print(f"SORT + FLOW RESULTS (Alpha={args.alpha})")
            print("="*30)
            print(f"HOTA: {metrics['HOTA']:.4f}")
            print(f"IDF1: {metrics['IDF1']:.4f}")
            print("="*30)
        except Exception as e:
            print(f"Error during metrics evaluation: {e}")
    else:
        print("\nCould not evaluate tracking metrics. Ensure Week2 modules are available.")

if __name__ == "__main__":
    main()