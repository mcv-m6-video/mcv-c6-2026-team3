#!/usr/bin/env python3
"""
Visualize a test-set clip as a GIF with:
  - A running frame counter (white text, black outline)
  - When a spotting is detected (GT or prediction) the GIF freezes for 1 s
    and shows the event name centred at the top in large font:
        GREEN  for ground-truth events
        ORANGE for predicted events
  - Normal frames play at the requested fps without any overlay text.
Usage:
    python visualize_clip.py --model <config_name> --clip_index <int> [--output out.gif] [--fps 10]
"""

import argparse
import torch
import numpy as np
import random
import sys
import os
from PIL import Image, ImageDraw, ImageFont

from util.io import load_json
from dataset.datasets import get_datasets
from model.model_spotting import Model
from model.model_spotting_mod import ModelMod


# ─── args ────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Create a GIF of a test clip with GT / predicted spottings.")
    parser.add_argument('--model',      type=str, required=True,
                        help='Config name (matches config/<model>.json)')
    parser.add_argument('--clip_index', type=int, required=True,
                        help='Index of the clip inside the test set')
    parser.add_argument('--seed',       type=int, default=1)
    parser.add_argument('--output',     type=str, default='clip_visualization.gif')
    parser.add_argument('--fps',        type=int, default=10,
                        help='Frames per second for the output GIF')
    return parser.parse_args()


def update_args(args, config):
    """Mirror the argument-update logic of main_spotting.py."""
    args.frame_dir        = config['frame_dir']
    args.save_dir         = config['save_dir'] + '/' + args.model
    args.store_dir        = config['save_dir'] + '/' + "splits"
    args.labels_dir       = config['labels_dir']
    args.store_mode       = 'load'                       # always load
    args.task             = config['task']
    args.batch_size       = config['batch_size']
    args.clip_len         = config['clip_len']
    args.dataset          = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch     = config['feature_arch']
    args.learning_rate    = config['learning_rate']
    args.num_classes      = config['num_classes']
    args.num_epochs       = config['num_epochs']
    args.warm_up_epochs   = config['warm_up_epochs']
    args.only_test        = config['only_test']
    args.device           = config['device']
    args.num_workers      = config['num_workers']

    # Temporal head
    args.use_temporal_head  = config.get('use_temporal_head', False)
    args.temporal_hidden_dim  = config.get('temporal_hidden_dim', None)
    args.temporal_kernel_size = config.get('temporal_kernel_size', 3)
    args.temporal_dilations   = config.get('temporal_dilations', [1, 2, 4])
    args.temporal_dropout     = config.get('temporal_dropout', 0.2)

    # Misc
    args.clip_aug       = config.get('clip_aug', False)
    args.patience       = config.get('patience', 5)
    args.use_delta      = config.get('use_delta', False)
    args.use_bottleneck = config.get('use_bottleneck', False)
    return args


# ─── helpers ─────────────────────────────────────────────────────────────────

def outlined_text(draw, xy, text, fill, font):
    """Draw text with a 1-px black outline for readability."""
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=fill, font=font)


def detect_onsets(labels, class_names):
    """
    Return a dict  {frame_index: class_name}  for every frame where a
    non-background class *first appears* (i.e. the onset of the event).
    """
    onsets = {}
    prev = 0
    for t, cls_idx in enumerate(labels):
        cls_idx = int(cls_idx)
        if cls_idx != 0 and cls_idx != prev:          # new event starts
            onsets[t] = class_names[cls_idx - 1]
        prev = cls_idx
    return onsets


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Config & datasets
    config = load_json('config/' + args.model + '.json')
    args = update_args(args, config)

    classes, _, _, test_data, _ = get_datasets(args)
    class_names = list(classes.keys())

    # ── Retrieve the requested clip ──────────────────────────────────────
    if args.clip_index < 0 or args.clip_index >= len(test_data):
        sys.exit(f"clip_index {args.clip_index} is out of range "
                 f"(test set has {len(test_data)} items).")

    sample = test_data[args.clip_index]
    frames_tensor = sample['frame']          # (T, C, H, W)  uint8 0-255
    gt_labels     = sample['label']          # (T,)

    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.numpy()
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = np.array(frames_tensor)

    clip_len = frames_np.shape[0]
    print(f"Clip {args.clip_index}: {clip_len} frames, "
          f"shape per frame {frames_np.shape[1:]}")

    # ── Load trained model & predict ─────────────────────────────────────
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint_best.pt')
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    model = ModelMod(args=args) if args.clip_aug else Model(args=args)
    model.load(torch.load(ckpt_path, map_location=args.device))

    pred_probs, _ = model.predict(frames_tensor)   # (1, T, num_classes+1)
    pred_classes  = np.argmax(pred_probs[0], axis=-1)  # (T,)

    # ── Detect event onsets ──────────────────────────────────────────────
    gt_onsets   = detect_onsets(gt_labels,   class_names)
    pred_onsets = detect_onsets(pred_classes, class_names)

    # ── Pick fonts ───────────────────────────────────────────────────────
    try:
        font_counter = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_event   = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font_counter = ImageFont.load_default()
        font_event   = font_counter

    # ── Compose frames with per-frame durations ──────────────────────────
    GREEN      = (0, 255, 0)
    ORANGE     = (255, 165, 0)
    WHITE      = (255, 255, 255)
    normal_ms  = int(1000 / args.fps)
    freeze_ms  = 1000                          # 1-second pause

    img_width  = frames_np.shape[3]            # (T, C, H, W) -> W

    pil_frames = []
    durations  = []

    for t in range(clip_len):
        # (C, H, W) -> (H, W, C)
        frame_rgb = frames_np[t].transpose(1, 2, 0).astype(np.uint8)

        has_gt   = t in gt_onsets
        has_pred = t in pred_onsets

        if has_gt or has_pred:
            # --- build the freeze frame(s) for this timestep ---
            lines = []                         # (text, colour)
            if has_gt:
                lines.append((f"GT: {gt_onsets[t]}", GREEN))
            if has_pred:
                lines.append((f"Pred: {pred_onsets[t]}", ORANGE))

            for text, colour in lines:
                img  = Image.fromarray(frame_rgb.copy())
                draw = ImageDraw.Draw(img)

                # frame counter (top-left)
                outlined_text(draw, (10, 8),
                              f"Frame {t+1}/{clip_len}", WHITE, font_counter)

                # event label centred at top-middle
                bbox = draw.textbbox((0, 0), text, font=font_event)
                tw = bbox[2] - bbox[0]
                x_center = (img_width - tw) // 2
                outlined_text(draw, (x_center, 35), text, colour, font_event)

                pil_frames.append(img)
                durations.append(freeze_ms)

        # --- normal (clean) frame ---
        img  = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)
        outlined_text(draw, (10, 8),
                      f"Frame {t+1}/{clip_len}", WHITE, font_counter)
        pil_frames.append(img)
        durations.append(normal_ms)

    # ── Save GIF ─────────────────────────────────────────────────────────
    pil_frames[0].save(
        args.output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"GIF saved to {args.output}  "
          f"({clip_len} source frames, {len(pil_frames)} GIF frames, "
          f"{args.fps} fps for normal playback)")


if __name__ == '__main__':
    main()