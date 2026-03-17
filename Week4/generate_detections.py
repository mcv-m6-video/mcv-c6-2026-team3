"""
Generate YOLO detections filtered by ROI for MTSC pipeline.
Output: <frame_id>,-1,<x_tl>,<y_tl>,<width>,<height>,<conf>,-1,-1,-1
"""

import argparse
import gc
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

SEQUENCES = ["S01", "S03", "S04"]


def is_detection_in_roi(x1, y1, x2, y2, roi_mask):
    """Check if detection center is within ROI (white pixels = valid region)."""
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    if 0 <= center_x < roi_mask.shape[1] and 0 <= center_y < roi_mask.shape[0]:
        return roi_mask[center_y, center_x] > 128
    return False


def process_camera(model: YOLO, video_path: Path, roi_path: Path, output_path: Path, conf_thr: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping.")
        return

    # Load ROI mask
    if not roi_path.exists():
        print(f"  [WARN] ROI mask {roi_path} not found, processing entire frame.")
        roi_mask = None
    else:
        roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(output_path, "w") as f_out:
        for frame_id in tqdm(range(1, total_frames + 1), desc=f"    {video_path.parent.name}", unit="frame", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame,
                conf=conf_thr,
                iou=0.5,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                xyxy = results[0].boxes.xyxy.cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()

                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    if roi_mask is None or is_detection_in_roi(x1, y1, x2, y2, roi_mask):
                        w = x2 - x1
                        h = y2 - y1
                        f_out.write(f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")

            del results

    cap.release()


def main(data_root: str, weights: str, conf_thr: float) -> None:
    root = Path(data_root)

    print(f"Loading model: {weights}")
    model = YOLO(weights)

    for seq in SEQUENCES:
        seq_path = root / seq
        if not seq_path.exists():
            print(f"[WARN] Sequence {seq} not found, skipping.")
            continue

        cam_dirs = sorted([p for p in seq_path.iterdir() if p.is_dir()])
        print(f"\n[{seq}] {len(cam_dirs)} camera(s): {[c.name for c in cam_dirs]}")

        for cam_dir in cam_dirs:
            video_path = cam_dir / "vdo.avi"
            roi_path = cam_dir / "roi.jpg"
            output_path = Path("outputs") / seq / cam_dir.name / "det_yolo.txt"

            if not video_path.exists():
                print(f"  [WARN] {video_path} not found, skipping.")
                continue

            print(f"  {seq}/{cam_dir.name} → {output_path}")
            process_camera(model, video_path, roi_path, output_path, conf_thr)

            torch.cuda.empty_cache()
            gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/path/to/AI_CITY_CHALLENGE_2022_TRAIN/train", help="Path to train/ folder")
    parser.add_argument("--weights", default="best_yolo.pt", help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    main(args.data_root, args.weights, args.conf)
