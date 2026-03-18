"""
Extract ground truth crops from videos for Siamese network training.
Saves cropped images organized by track_id.
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import os

SEQUENCES = ["S01", "S04"]


def load_gt_annotations(gt_file):
    """Load ground truth annotations from gt.txt file."""
    annotations = {}

    if not gt_file.exists():
        print(f"  [WARN] GT file {gt_file} not found")
        return annotations

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue

                if frame_id not in annotations:
                    annotations[frame_id] = []

                annotations[frame_id].append({
                    'track_id': track_id,
                    'bbox': (int(x), int(y), int(x + w), int(y + h))
                })

    return annotations


def extract_crops_from_video(video_path, annotations, output_dir, seq, cam):
    """Extract crops from video based on ground truth annotations."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops_count = 0

    for frame_id in tqdm(range(1, total_frames + 1),
                        desc=f"    {cam}",
                        unit="frame",
                        leave=False):

        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in annotations:
            for ann in annotations[frame_id]:
                track_id = ann['track_id']
                x1, y1, x2, y2 = ann['bbox']

                # Track folder name includes sequence prefix to avoid ID collision
                # (e.g., ID 5 in S01 is a different vehicle than ID 5 in S04)
                track_name = f"{seq}_ID_{track_id:04d}"
                track_dir = output_dir / track_name
                track_dir.mkdir(parents=True, exist_ok=True)

                # Extract crop with padding
                h, w = frame.shape[:2]
                pad = 5
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)

                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                # Skip very small crops
                if crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue

                # Include camera name in filename to avoid overwriting crops
                # when the same vehicle appears in multiple cameras at the same frame
                crop_filename = track_dir / f"{cam}_frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(crop_filename), crop)
                crops_count += 1

    cap.release()
    print(f"    Extracted {crops_count} crops from {cam}")


def main():
    parser = argparse.ArgumentParser(description="Extract GT crops for Siamese training")
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to dataset train/ folder"
    )
    parser.add_argument(
        "--output_dir",
        default="data/gt_crops",
        help="Output directory for crops (default: data/gt_crops)"
    )
    parser.add_argument(
        "--seqs",
        nargs='+',
        default=SEQUENCES,
        help="Sequences to process (default: S01 S04)"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    print(f"Dataset: {data_root}")
    print(f"Output: {output_dir.absolute()}")

    total_crops = 0

    for seq in args.seqs:
        seq_path = data_root / seq
        if not seq_path.exists():
            print(f"[WARN] Sequence {seq} not found")
            continue

        cam_dirs = sorted([p for p in seq_path.iterdir() if p.is_dir()])
        print(f"\n[{seq}] Processing {len(cam_dirs)} cameras: {[c.name for c in cam_dirs]}")

        for cam_dir in cam_dirs:
            video_path = cam_dir / "vdo.avi"
            gt_file = cam_dir / "gt" / "gt.txt"

            if not video_path.exists():
                print(f"  [WARN] Video {video_path} not found")
                continue

            if not gt_file.exists():
                print(f"  [WARN] GT file {gt_file} not found")
                continue

            print(f"  Processing {seq}/{cam_dir.name}")

            # Load annotations
            annotations = load_gt_annotations(gt_file)
            if not annotations:
                print(f"    No valid annotations found")
                continue

            print(f"    Found annotations for {len(annotations)} frames")

            # Extract crops
            extract_crops_from_video(video_path, annotations, output_dir, seq, cam_dir.name)

    print(f"\nDone! Crops saved to: {output_dir.absolute()}")
    print("Structure: gt_crops/<SEQ>_ID_<global_id>/<cam>_frame_<N>.jpg")
    print("Compatible with PyTorch ImageFolder!")


if __name__ == "__main__":
    main()