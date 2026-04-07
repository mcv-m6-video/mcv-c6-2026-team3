#!/usr/bin/env python3
"""
visualize_clip.py
-----------------
Picks a random clip from the TEST split, runs inference with your trained
model, and saves an annotated GIF showing:

  • Frame counter  (top-left)   current_frame / total_frames
  • Ground-truth labels         (top panel, green)
  • Model predictions + prob    (top panel, amber/red by confidence)

Usage
-----
    python visualize_clip.py \
        --model <config_name> \
        [--split test]        \   # test | val | train
        [--clip_idx -1]       \   # -1 = random, or pass a fixed index
        [--out clip_vis.gif]  \
        [--fps 8]             \
        [--seed 42]

The script reuses your existing config/*.json + model checkpoint
(checkpoints/checkpoint_best.pt) exactly as main_classification.py does.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from util.io import load_json
from dataset.datasets import get_datasets
from model.model_classification_mod import Model
from model.model_classification_temporal import ModelTemporal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, required=True,
                   help="Config name (matches config/<name>.json)")
    p.add_argument("--split",    type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Which dataset split to sample from")
    p.add_argument("--clip_idx", type=int, default=-1,
                   help="Index of clip to visualize (-1 = random)")
    p.add_argument("--out",      type=str, default="clip_vis.gif",
                   help="Output GIF filename")
    p.add_argument("--fps",      type=int, default=8,
                   help="GIF playback speed (frames per second)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers  (mirrors update_args from main_classification.py)
# ---------------------------------------------------------------------------

def update_args(args, config):
    args.frame_dir       = config["frame_dir"]
    args.save_dir        = config["save_dir"] + "/" + args.model
    args.store_dir       = config["save_dir"] + "/" + "splits"
    args.labels_dir      = config["labels_dir"]
    args.store_mode      = config["store_mode"]
    args.task            = config["task"]
    args.batch_size      = config["batch_size"]
    args.clip_len        = config["clip_len"]
    args.dataset         = config["dataset"]
    args.epoch_num_frames = config["epoch_num_frames"]
    args.feature_arch    = config["feature_arch"]
    args.learning_rate   = config["learning_rate"]
    args.num_classes     = config["num_classes"]
    args.num_epochs      = config["num_epochs"]
    args.warm_up_epochs  = config["warm_up_epochs"]
    args.only_test       = config["only_test"]
    args.device          = config["device"]
    args.num_workers     = config["num_workers"]
    args.use_weighted_bce = config.get("use_weighted_bce", False)
    args.pos_weight_clip  = float(config.get("pos_weight_clip", 20.0))
    args.pos_weight_eps   = float(config.get("pos_weight_eps",  1.0))
    args.temporal_handler = config.get("temporal_handler", None)
    args.freeze_backbone  = config.get("freeze_backbone", False)
    args.unfreeze_num     = config.get("freeze_backbone", 0)
    args.use_focal_loss   = config.get("use_focal_loss", False)
    return args


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

PANEL_H       = 110   # pixels of info panel above the frame
FRAME_W       = 224   # resize frame to this width for the GIF
FRAME_H       = 224   # resize frame to this height
FONT_SIZE     = 13
SMALL_SIZE    = 11

# Palette
BG_COLOR      = (15,  15,  20)   # near-black panel bg
GT_COLOR      = (80,  200, 120)  # green  – ground truth
PRED_HI       = (255, 200,  50)  # amber  – high confidence pred
PRED_LO       = (220,  80,  80)  # red    – low  confidence pred
TEXT_COLOR    = (230, 230, 230)
COUNTER_COLOR = (255, 255, 255)
THRESH        = 0.5               # probability threshold for "positive" pred


def _try_font(size):
    """Load a monospace font if available, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_panel(gt_names, pred_names_probs, frame_idx, total_frames):
    """
    Returns a PIL Image of size (FRAME_W, PANEL_H) with the annotation overlay.

    gt_names          – list[str]  active ground-truth class names
    pred_names_probs  – list[(str, float)]  all classes with prob >= THRESH,
                        sorted descending by probability
    """
    font_sm  = _try_font(SMALL_SIZE)
    font_reg = _try_font(FONT_SIZE)

    panel = Image.new("RGB", (FRAME_W, PANEL_H), BG_COLOR)
    d = ImageDraw.Draw(panel)

    # ── Frame counter ──────────────────────────────────────────────────────
    counter_txt = f"{frame_idx + 1:>3} / {total_frames}"
    d.text((4, 4), counter_txt, font=font_reg, fill=COUNTER_COLOR)

    # ── Divider ────────────────────────────────────────────────────────────
    d.line([(0, 22), (FRAME_W, 22)], fill=(60, 60, 70), width=1)

    # ── Ground truth block ─────────────────────────────────────────────────
    gt_label     = "GT: "
    gt_label_w   = _text_width(d, gt_label, font_sm)
    d.text((4, 26), gt_label, font=font_sm, fill=GT_COLOR)
    if gt_names:
        gt_text = ",  ".join(gt_names)
    else:
        gt_text = "(background)"
    # Draw content starting right after the "GT: " label
    _draw_wrapped(d, gt_text, x=4 + gt_label_w, y=26, font=font_sm,
                  color=GT_COLOR, max_w=FRAME_W - 4 - gt_label_w, line_h=14)

    # ── Predictions block ──────────────────────────────────────────────────
    pred_y = 60
    d.line([(0, pred_y - 2), (FRAME_W, pred_y - 2)], fill=(60, 60, 70), width=1)
    d.text((4, pred_y), "PRED:", font=font_sm, fill=TEXT_COLOR)

    if pred_names_probs:
        # Show up to 3 predictions side-by-side as "[name p%]"
        parts = []
        for name, prob in pred_names_probs[:3]:
            pct   = int(prob * 100)
            color = PRED_HI if prob >= 0.70 else PRED_LO
            parts.append((f"  {name} {pct}%", color))

        x_cur = 4
        y_cur = pred_y + 14
        for txt, color in parts:
            tw = _text_width(d, txt, font_sm)
            if x_cur + tw > FRAME_W - 4:
                x_cur  = 4
                y_cur += 14
            d.text((x_cur, y_cur), txt, font=font_sm, fill=color)
            x_cur += tw + 6
    else:
        d.text((4, pred_y + 14), "  (none above threshold)", font=font_sm,
               fill=(130, 130, 130))

    return panel


def _text_width(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def _draw_wrapped(draw, text, x, y, font, color, max_w, line_h):
    """Draw text, splitting at commas when it overflows max_w."""
    tw = _text_width(draw, text, font)
    if tw <= max_w:
        draw.text((x, y), text, font=font, fill=color)
        return
    # Split into tokens and rebuild lines
    tokens = text.split(",")
    line   = ""
    cy     = y
    for tok in tokens:
        candidate = line + ("," if line else "") + tok
        if _text_width(draw, candidate, font) <= max_w:
            line = candidate
        else:
            if line:
                draw.text((x, cy), line, font=font, fill=color)
                cy += line_h
            line = tok.strip()
    if line:
        draw.text((x, cy), line, font=font, fill=color)


def tensor_frame_to_pil(frame_tensor):
    """
    frame_tensor: float32 tensor, shape (C, H, W), values in [0, 1] or [0, 255].
    Returns a PIL Image resized to (FRAME_W, FRAME_H).
    """
    arr = frame_tensor.cpu().numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")
    img = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = f"config/{args.model}.json"
    config      = load_json(config_path)
    args        = update_args(args, config)

    # ── Load datasets ───────────────────────────────────────────────────────
    classes, train_data, val_data, test_data = get_datasets(args)
    class_names = list(classes.keys())   # 0-indexed list of class names

    split_map = {"train": train_data, "val": val_data, "test": test_data}
    dataset   = split_map[args.split]

    n = len(dataset)
    if n == 0:
        sys.exit(f"[ERROR] Dataset split '{args.split}' is empty.")

    idx = args.clip_idx
    idx = idx % n
    print(f"Visualising clip index {idx} / {n - 1}  (split={args.split})")

    # ── Load sample  ────────────────────────────────────────────────────────
    # dataset[i] returns a dict with at least:
    #   "frame"  – Tensor (clip_len, C, H, W)  uint8 or float
    #   "label"  – Tensor (num_classes,)        multi-hot float
    sample = dataset[idx]
    frames_tensor = sample["frame"]      # (T, C, H, W)
    label_tensor  = sample["label"]      # (num_classes,)

    T = frames_tensor.shape[0]

    # Ground-truth active class names
    gt_indices = (label_tensor > 0.5).nonzero()[0].tolist()
    gt_names   = [class_names[i] for i in gt_indices if i < len(class_names)]

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.save_dir, "checkpoints", "checkpoint_best.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    if args.feature_arch.startswith(("rny002", "rny004", "rny008")):
        model = Model(args=args, pos_weight=None)
    else:
        model = ModelTemporal(args=args, pos_weight=None)

    model.load(torch.load(ckpt_path, map_location="cpu"))
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Run inference on the whole clip ─────────────────────────────────────
    # model.predict expects (T, C, H, W) → adds batch dim internally
    probs = model.predict(frames_tensor)   # numpy (1, num_classes) or (num_classes,)
    probs = probs.squeeze()                # (num_classes,)

    # Build sorted prediction list (all classes, sorted by prob desc)
    pred_names_probs = sorted(
        [(class_names[i], float(probs[i])) for i in range(len(class_names))],
        key=lambda x: x[1],
        reverse=True,
    )
    # Keep only those above threshold
    pred_above_thresh = [(n, p) for n, p in pred_names_probs if p >= THRESH]

    print("\nGround truth :", gt_names if gt_names else ["(background)"])
    print("Predictions  :")
    for name, prob in pred_names_probs[:5]:
        marker = "✓" if prob >= THRESH else " "
        print(f"  [{marker}] {name:<30s} {prob:.3f}")

    # ── Build GIF frames ────────────────────────────────────────────────────
    gif_frames = []
    total_w    = FRAME_W
    total_h    = FRAME_H + PANEL_H

    for t in range(T):
        frame_pil = tensor_frame_to_pil(frames_tensor[t])  # (C,H,W)

        panel = draw_panel(
            gt_names         = gt_names,
            pred_names_probs = pred_above_thresh,
            frame_idx        = t,
            total_frames     = T,
        )

        # Stack panel on top of frame
        combined = Image.new("RGB", (total_w, total_h), BG_COLOR)
        combined.paste(panel,     (0, 0))
        combined.paste(frame_pil, (0, PANEL_H))

        gif_frames.append(combined)

    # ── Save GIF ────────────────────────────────────────────────────────────
    duration_ms = int(1000 / args.fps)
    out_path    = args.out

    gif_frames[0].save(
        out_path,
        save_all   = True,
        append_images = gif_frames[1:],
        duration   = duration_ms,
        loop       = 0,          # loop forever
        optimize   = False,
    )

    print(f"\nSaved GIF  →  {out_path}  ({T} frames @ {args.fps} fps)")
    print(f"Resolution : {total_w} × {total_h} px  (panel {PANEL_H}px + frame {FRAME_H}px)")


if __name__ == "__main__":
    main()