"""
Pipeline orchestrator: iterates S01/S03/S04 → c00X subfolders.

For each sequence/camera:
  1. If detections.txt is missing → run YOLO to generate it.
  2. If flow tracking results already exist   → skip flow tracker.
  3. If sort-flow tracking results already exist → skip sort-flow tracker.
  4. Run whichever trackers still need to be run and append metrics to CSV.
"""

import argparse
import os
import sys
import csv
import subprocess
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEQUENCES  = ["S01", "S03", "S04"]
BASE_DATA  = os.path.join(SCRIPT_DIR, "../AI_CITY_CHALLENGE_2022_TRAIN/train")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FF_CKPT    = os.path.join(SCRIPT_DIR, "FlowFormerPlusPlus", "checkpoints", "kitti.pth")
CSV_OUTPUT = os.path.join(RESULTS_DIR, "tracking_metrics.csv")

CSV_FIELDNAMES = [
    "sequence", "camera", "tracker",
    "HOTA", "IDF1",
    "alpha",        # only relevant for sort-flow
    "iou_thr", "max_age", "min_hits",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def ensure_csv_header(csv_path):
    """Create CSV with header row if it doesn't exist yet."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def append_csv_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def find_cameras(sequence_path):
    """Return sorted list of camera sub-folder names (e.g. c001, c010)."""
    if not os.path.isdir(sequence_path):
        return []
    return sorted(
        d for d in os.listdir(sequence_path)
        if os.path.isdir(os.path.join(sequence_path, d)) and d.startswith("c")
    )


def detections_path(seq, cam):
    return os.path.join(RESULTS_DIR, seq, cam, "detections.txt")


def tracking_results_exist(seq, cam, tracker):
    """Return True if the tracker's output .txt already exists and is non-empty."""
    name = "tracking_flow" if tracker == "flow" else "tracking_sort_flow"
    path = os.path.join(RESULTS_DIR, seq, cam, f"{name}.txt")
    return os.path.isfile(path) and os.path.getsize(path) > 0


def run_yolo(seq, cam):
    """
    Run run_yolo.py for the given sequence/camera to produce detections.txt.
    Returns True on success.
    """
    data_dir    = os.path.join(BASE_DATA, seq, cam)
    results_dir = os.path.join(RESULTS_DIR, seq, cam)
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_yolo.py"),
        "--data",    data_dir,
        "--results", results_dir,
    ]
    rc = run_script(cmd, f"YOLO DETECTION  {seq}/{cam}")
    if rc != 0:
        print(f"  [WARN] YOLO exited with code {rc} for {seq}/{cam}")
        return False
    if not os.path.isfile(detections_path(seq, cam)):
        print(f"  [WARN] YOLO finished but detections.txt was not created for {seq}/{cam}")
        return False
    return True


def run_script(cmd, label):
    """Run a subprocess command, streaming stdout/stderr. Returns exit code."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True)
    return result.returncode

# ── per-tracker runners ───────────────────────────────────────────────────────

def run_flow_tracker(seq, cam, dets_path, args):
    """
    Run run_flow_tracking_script.py for one sequence/camera.
    Returns a metrics dict or None on failure.
    """
    video   = os.path.join(BASE_DATA, seq, cam, "vdo.avi")
    gt_txt  = os.path.join(BASE_DATA, seq, cam, "gt", "gt.txt")
    out_dir = os.path.join(RESULTS_DIR, seq, cam)
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, "tracking_flow.mp4")
    metrics_csv = os.path.join(out_dir, "flow_metrics.csv")   # per-run metrics sink

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_flow_tracking_script.py"),
        "--video",   video,
        "--txt",     gt_txt,
        "--dets",    dets_path,
        "--ff_ckpt", args.ff_ckpt,
        "--out",     out_mp4,
        "--iou_thr", str(args.flow_iou_thr),
        "--max_age", str(args.flow_max_age),
        "--scale",   str(args.scale),
        "--metrics_csv", metrics_csv,   # new arg consumed by patched script
    ]
    rc = run_script(cmd, f"FLOW TRACKER  {seq}/{cam}")
    if rc != 0:
        print(f"  [WARN] Flow tracker exited with code {rc} for {seq}/{cam}")
        return None

    return _read_single_row_csv(metrics_csv)


def run_sort_flow_tracker(seq, cam, dets_path, args):
    """
    Run run_sort_flow_tracking_script.py for one sequence/camera.
    Returns a metrics dict or None on failure.
    """
    video   = os.path.join(BASE_DATA, seq, cam, "vdo.avi")
    gt_txt  = os.path.join(BASE_DATA, seq, cam, "gt", "gt.txt")
    out_dir = os.path.join(RESULTS_DIR, seq, cam)
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, "tracking_sort_flow.mp4")
    metrics_csv = os.path.join(out_dir, "sort_flow_metrics.csv")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_sort_flow_tracking_script.py"),
        "--video",    video,
        "--txt",      gt_txt,
        "--dets",     dets_path,
        "--ff_ckpt",  args.ff_ckpt,
        "--out",      out_mp4,
        "--iou_thr",  str(args.sort_iou_thr),
        "--max_age",  str(args.sort_max_age),
        "--min_hits", str(args.sort_min_hits),
        "--scale",    str(args.scale),
        "--alpha",    str(args.alpha),
        "--metrics_csv", metrics_csv,   # new arg consumed by patched script
    ]
    rc = run_script(cmd, f"SORT+FLOW TRACKER  {seq}/{cam}  alpha={args.alpha}")
    if rc != 0:
        print(f"  [WARN] SORT+Flow tracker exited with code {rc} for {seq}/{cam}")
        return None

    return _read_single_row_csv(metrics_csv)


def _read_single_row_csv(path):
    """Read the first data row from a single-row CSV written by a tracker script."""
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return None

# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch tracking pipeline over S01/S03/S04")
    p.add_argument("--ff_ckpt",       default=FF_CKPT)
    p.add_argument("--scale",         default=0.5,    type=float)
    # flow tracker
    p.add_argument("--flow_iou_thr",  default=0.3,    type=float)
    p.add_argument("--flow_max_age",  default=5,      type=int)
    # sort+flow tracker
    p.add_argument("--sort_iou_thr",  default=0.1458, type=float)
    p.add_argument("--sort_max_age",  default=5,      type=int)
    p.add_argument("--sort_min_hits", default=12,     type=int)
    p.add_argument("--alpha",         default=0.5,    type=float)
    # control
    p.add_argument("--skip_flow",      action="store_true", help="Skip the flow-only tracker")
    p.add_argument("--skip_sort_flow", action="store_true", help="Skip the SORT+flow tracker")
    p.add_argument("--csv_out",        default=CSV_OUTPUT,  help="Path for aggregate CSV results")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_csv_header(args.csv_out)

    total_ran     = 0
    total_skipped = 0
    total_yolo    = 0

    for seq in SEQUENCES:
        seq_path = os.path.join(BASE_DATA, seq)
        cameras  = find_cameras(seq_path)

        if not cameras:
            print(f"[INFO] No camera folders found under {seq_path}, skipping.")
            continue

        for cam in cameras:
            dets = detections_path(seq, cam)

            # ── Step 1: run YOLO if detections are missing ─────────────────
            if not os.path.isfile(dets):
                print(f"\n[YOLO] {seq}/{cam} — detections.txt missing, running YOLO...")
                ok = run_yolo(seq, cam)
                if not ok:
                    print(f"  [SKIP] Could not produce detections for {seq}/{cam}, skipping trackers.")
                    total_skipped += 1
                    continue
                total_yolo += 1
            else:
                print(f"\n[OK]   {seq}/{cam} — detections.txt found, skipping YOLO.")

            # ── Step 2: Flow tracker ────────────────────────────────────────
            if not args.skip_flow:
                if tracking_results_exist(seq, cam, "flow"):
                    print(f"  [SKIP] flow results already exist for {seq}/{cam}, skipping.")
                else:
                    flow_metrics = run_flow_tracker(seq, cam, dets, args)
                    if flow_metrics:
                        append_csv_row(args.csv_out, {
                            "sequence": seq,
                            "camera":   cam,
                            "tracker":  "flow",
                            "HOTA":     flow_metrics.get("HOTA", ""),
                            "IDF1":     flow_metrics.get("IDF1", ""),
                            "alpha":    "",
                            "iou_thr":  args.flow_iou_thr,
                            "max_age":  args.flow_max_age,
                            "min_hits": "",
                        })
                        print(f"  [flow]      HOTA={flow_metrics.get('HOTA')}  IDF1={flow_metrics.get('IDF1')}")

            # ── Step 3: SORT + Flow tracker ─────────────────────────────────
            if not args.skip_sort_flow:
                if tracking_results_exist(seq, cam, "sort_flow"):
                    print(f"  [SKIP] sort_flow results already exist for {seq}/{cam}, skipping.")
                else:
                    sort_metrics = run_sort_flow_tracker(seq, cam, dets, args)
                    if sort_metrics:
                        append_csv_row(args.csv_out, {
                            "sequence": seq,
                            "camera":   cam,
                            "tracker":  "sort_flow",
                            "HOTA":     sort_metrics.get("HOTA", ""),
                            "IDF1":     sort_metrics.get("IDF1", ""),
                            "alpha":    args.alpha,
                            "iou_thr":  args.sort_iou_thr,
                            "max_age":  args.sort_max_age,
                            "min_hits": args.sort_min_hits,
                        })
                        print(f"  [sort_flow] HOTA={sort_metrics.get('HOTA')}  IDF1={sort_metrics.get('IDF1')}")

            total_ran += 1

    print(f"\n{'='*60}")
    print(f"Done.  Processed: {total_ran}  |  YOLO runs: {total_yolo}  |  Skipped (error): {total_skipped}")
    print(f"Aggregate results → {args.csv_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()