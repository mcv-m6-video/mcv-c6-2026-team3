import argparse
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import csv

from sort_tracker import SORTTracker

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

DEFAULT_VIDEO      = os.path.join(SCRIPT_DIR, "../AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c005/vdo.avi")
DEFAULT_TXT        = os.path.join(SCRIPT_DIR, "../AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c005/gt/gt.txt")
DEFAULT_DETS       = os.path.join(SCRIPT_DIR, "results/S01/c005/detections.txt")
DEFAULT_FF_DIR     = os.path.join(SCRIPT_DIR, "FlowFormerPlusPlus")
DEFAULT_FF_CKPT    = os.path.join(SCRIPT_DIR, "FlowFormerPlusPlus", "checkpoints", "kitti.pth")
DEFAULT_OUT        = os.path.join(SCRIPT_DIR, "results/tracking_sort_flow.mp4")

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid SORT + FlowFormer++ tracker")
    p.add_argument("--video",   default=DEFAULT_VIDEO,   help="Path to input video")
    p.add_argument("--txt",     default=DEFAULT_TXT,     help="Path to Ground Truth XML (for metrics)")
    p.add_argument("--dets",    default=DEFAULT_DETS,    help="Path to detections .txt file")
    p.add_argument("--ff_ckpt", default=DEFAULT_FF_CKPT, help="Path to FlowFormer++ checkpoint")
    p.add_argument("--out",     default=DEFAULT_OUT,     help="Output video path")
    p.add_argument("--iou_thr", default=0.1458, type=float, help="IOU threshold for tracker")
    p.add_argument("--max_age", default=5,   type=int,   help="Max frames to keep lost track")
    p.add_argument("--min_hits", default=12, type=int, help="Min hits to confirm track")
    p.add_argument("--scale",   default=0.5, type=float, help="Scale factor for optical flow calculation")
    p.add_argument("--metrics_csv", default=None,             help="Path to write per-run metrics CSV")

    p.add_argument("--alpha",   default=0.5, type=float, help="0.0 = 100% Flow, 1.0 = 100% Kalman, 0.5 = Mix")
    return p.parse_args()


def load_detections(path):
    dets = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            f_id, x, y, w, h, _ = map(float, line.strip().split(','))
            dets[int(f_id)].append([x, y, w, h])
    return dets


def write_metrics_csv(csv_path, metrics):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["HOTA", "IDF1"])
        writer.writeheader()
        writer.writerow({"HOTA": metrics["HOTA"], "IDF1": metrics["IDF1"]})

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ff_dir = os.path.dirname(args.ff_ckpt[:-len("/checkpoints/kitti.pth")]) if args.ff_ckpt.endswith("/checkpoints/kitti.pth") else DEFAULT_FF_DIR
    if ff_dir not in sys.path:
        sys.path.insert(0, ff_dir)



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

    f_id = 0
    
    pred_tracks_for_eval = {}

    for _ in tqdm(range(total_frames) if total_frames else iter(int, 1), desc="Tracking"):
        ret, frame = cap.read()
        if not ret: break
        
        frame_dets = detections.get(f_id, [])
        results = tracker.track(frame_dets, f_id)
        


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
        f_id += 1

    cap.release(); out.release(); out_file.close()
    
    if load_gt_tracks is not None and evaluate_tracking is not None:
        print("\nLoading GT to calculate tracking metrics...")
        try:
            train_frames = int(total_frames * 0.25) if total_frames else 535 
            gt_tracks = load_gt_tracks(args.txt, train_frames=train_frames)
            
            pred_filtered = {k: v for k, v in pred_tracks_for_eval.items() if k >= train_frames}
            
            print(f"Evaluating Tracking from frame {train_frames} to end...")
            metrics = evaluate_tracking(pred_filtered, gt_tracks)
            
            print("\n" + "="*30)
            print(f"SORT + FLOW RESULTS (Alpha={args.alpha})")
            print("="*30)
            print(f"HOTA: {metrics['HOTA']:.4f}")
            print(f"IDF1: {metrics['IDF1']:.4f}")
            print("="*30)
            
            if args.metrics_csv:
                write_metrics_csv(args.metrics_csv, metrics)
                print(f"Metrics saved to {args.metrics_csv}")
                
        except Exception as e:
            print(f"Error during metrics evaluation: {e}")
    else:
        print("\nCould not evaluate tracking metrics. Ensure Week2 modules are available.")

if __name__ == "__main__":
    main()