"""
compare_detectors.py
--------------------
Runs the full MTMC tracking + evaluation pipeline for each detector
(default / finetuned / large) across all available sequences and prints
a comparison table.

Usage
-----
    python compare_detectors.py \
        --data_root ../AI_CITY_CHALLENGE_2022_TRAIN/train \
        --detections_root results \
        --out_dir results/comparison

Optional arguments mirror main.py tracker / ReID hyper-parameters so you
can compare detectors under a fixed tracking configuration.
"""

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Make Week4 modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as mtmc

DETECTORS  = ["default", "finetuned", "large"]
SEQUENCES  = ["S01", "S03", "S04"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare detector quality via MTMC tracking metrics")

    p.add_argument("--data_root", required=True,
                   help="Root of the dataset, e.g. .../AI_CITY_CHALLENGE_2022_TRAIN/train")
    p.add_argument("--detections_root", default="results",
                   help="Root that contains default/ finetuned/ large/ subdirs (default: results)")
    p.add_argument("--detectors", nargs="+", default=DETECTORS,
                   help=f"Detectors to compare (default: {DETECTORS})")
    p.add_argument("--sequences", nargs="+", default=SEQUENCES,
                   help=f"Sequences to evaluate (default: {SEQUENCES})")
    p.add_argument("--out_dir", default="results/comparison",
                   help="Output directory for per-run tracking results and summary CSV")

    # ---- Tracker / ReID params (forwarded to main.run_mtmc) ----
    p.add_argument("--tracker",    default="sort", choices=["sort", "bytetrack"])
    p.add_argument("--iou_thr",    default=0.1458, type=float)
    p.add_argument("--max_age",    default=5,       type=int)
    p.add_argument("--min_hits",   default=12,      type=int)
    p.add_argument("--match_thr",  default=0.30,    type=float)
    p.add_argument("--lookback",   default=3.0,     type=float)
    p.add_argument("--geo_weight", default=0.3,     type=float)
    p.add_argument("--max_speed",  default=30.0,    type=float)
    p.add_argument("--geo_sigma",  default=50.0,    type=float)
    p.add_argument("--velocity_window", default=5,  type=int)
    p.add_argument("--scorer",     default="histogram", choices=["histogram", "siamese"])
    p.add_argument("--siamese_ckpt", default=None)
    p.add_argument("--device",     default="cuda")

    # ByteTrack params
    p.add_argument("--bt_track_thresh", default=0.25, type=float)
    p.add_argument("--bt_track_buffer", default=30,   type=int)
    p.add_argument("--bt_match_thresh", default=0.8,  type=float)

    p.add_argument("--conf_thr", default=0.0, type=float,
                   help="Discard detections below this confidence")
    p.add_argument("--use_roi", action="store_true",
                   help="Filter detections outside the camera ROI mask")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_run_args(args, detector: str, sequence: str, out_dir: str) -> argparse.Namespace:
    """Build a fake args Namespace for main.run_mtmc."""
    run = copy.deepcopy(args)

    data_root        = Path(args.data_root)
    detections_root  = Path(args.detections_root)

    run.scenario_dir  = str(data_root / sequence)
    run.dets_dir      = str(detections_root / detector / sequence)
    run.out_dir       = str(Path(out_dir) / detector / sequence)
    run.eval          = True
    run.write_video   = False
    run.montage       = False
    run.metrics_csv   = None
    run.gs_csv        = None
    run.grid_search   = False

    # Grid search attrs required by run_mtmc even when not grid-searching
    for attr in ["gs_match_thr", "gs_lookback", "gs_geo_weight", "gs_max_speed",
                 "gs_geo_sigma", "gs_velocity_window", "gs_iou_thr",
                 "gs_max_age", "gs_min_hits", "gs_conf_thr"]:
        setattr(run, attr, None)

    return run


def _print_table(rows: list, detectors: list, sequences: list):
    """Pretty-print a comparison table to stdout."""
    metric_pairs = [("HOTA", "IDF1")]

    # Header
    col_w = 10
    det_w = 12
    print(f"\n{'='*70}")
    print(f"  Detector comparison  (HOTA / IDF1)")
    print(f"{'='*70}")

    header = f"  {'Detector':<{det_w}}  {'Sequence':<8}"
    header += f"  {'HOTA':>{col_w}}  {'IDF1':>{col_w}}"
    print(header)
    print(f"  {'-'*60}")

    for row in rows:
        line = f"  {row['detector']:<{det_w}}  {row['sequence']:<8}"
        hota = row.get("HOTA")
        idf1 = row.get("IDF1")
        line += f"  {hota:>{col_w}.4f}" if hota is not None else f"  {'N/A':>{col_w}}"
        line += f"  {idf1:>{col_w}.4f}" if idf1 is not None else f"  {'N/A':>{col_w}}"
        print(line)

    # Per-detector mean across sequences
    print(f"  {'-'*60}")
    for det in detectors:
        det_rows = [r for r in rows if r["detector"] == det and r.get("HOTA") is not None]
        if not det_rows:
            continue
        mean_hota = np.mean([r["HOTA"] for r in det_rows])
        mean_idf1 = np.mean([r["IDF1"] for r in det_rows])
        line = f"  {det:<{det_w}}  {'MEAN':<8}"
        line += f"  {mean_hota:>{col_w}.4f}  {mean_idf1:>{col_w}.4f}"
        print(line)

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "detector_comparison.csv")
    rows = []

    total = len(args.detectors) * len(args.sequences)
    done  = 0

    for detector in args.detectors:
        for sequence in args.sequences:
            done += 1
            dets_path = Path(args.detections_root) / detector / sequence
            if not dets_path.exists():
                print(f"  [{done}/{total}] {detector}/{sequence}  SKIP (no detections at {dets_path})")
                continue

            print(f"\n  [{done}/{total}] Running  detector={detector}  sequence={sequence}")

            run_args = _base_run_args(args, detector, sequence, out_dir)

            result = mtmc.run_mtmc(run_args)

            row = {"detector": detector, "sequence": sequence}
            if result:
                row["HOTA"] = result["HOTA"]
                row["IDF1"]  = result["IDF1"]
                print(f"         HOTA={result['HOTA']:.4f}  IDF1={result['IDF1']:.4f}")
            else:
                row["HOTA"] = None
                row["IDF1"]  = None
                print(f"         No GT available – metrics skipped")

            rows.append(row)

    # Print and save results
    _print_table(rows, args.detectors, args.sequences)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "sequence", "HOTA", "IDF1"])
        writer.writeheader()
        writer.writerows(rows)
        # Append per-detector means
        for det in args.detectors:
            det_rows = [r for r in rows if r["detector"] == det and r.get("HOTA") is not None]
            if det_rows:
                writer.writerow({
                    "detector": det,
                    "sequence": "MEAN",
                    "HOTA": np.mean([r["HOTA"] for r in det_rows]),
                    "IDF1": np.mean([r["IDF1"] for r in det_rows]),
                })

    print(f"  Results saved → {csv_path}")


if __name__ == "__main__":
    main()
