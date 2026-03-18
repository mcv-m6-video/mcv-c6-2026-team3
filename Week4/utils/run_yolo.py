import os
import sys
import time
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors import YOLODetector
from pipelines import DetectionPipeline

SCRIPT_DIR = Path(__file__).parent

MODEL_CONFIGS = {
    "default":   {"model_name": "yolo26n.pt",   "finetune": False},
    "finetuned": {"model_name": "best_yolo.pt", "finetune": True},
    "large":     {"model_name": "yolo11x.pt",   "finetune": False},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run YOLO detection on all cameras in a scenario."
    )
    p.add_argument(
        "--scenario_dir", required=True,
        help="Path to scenario directory, e.g. .../train/S01",
    )
    p.add_argument(
        "--results_dir", default=None,
        help="Root output directory. Defaults to <script_dir>/results/<scenario>",
    )
    p.add_argument(
        "--model", default="default", choices=list(MODEL_CONFIGS.keys()),
        help=(
            "Model to use: "
            "'default' (yolo26n.pt), "
            "'finetuned' (best_yolo.pt, fine-tuned on traffic data), "
            "'large' (yolo11x.pt, highest accuracy)"
        ),
    )
    p.add_argument(
        "--cameras", nargs="+", default=None,
        help="Subset of cameras to process, e.g. --cameras c001 c002. "
             "Defaults to all cameras found in scenario_dir.",
    )
    return p.parse_args()


def discover_cameras(scenario_dir: str):
    return sorted(
        d for d in os.listdir(scenario_dir)
        if d.startswith("c") and os.path.isdir(os.path.join(scenario_dir, d))
    )


def main():
    args = parse_args()

    scenario_dir  = Path(args.scenario_dir)
    scenario_name = scenario_dir.name
    results_root  = Path(args.results_dir) if args.results_dir \
                    else SCRIPT_DIR / "results" / scenario_name

    cameras = args.cameras or discover_cameras(str(scenario_dir))
    if not cameras:
        print(f"No camera subdirectories found in {scenario_dir}")
        return

    cfg = MODEL_CONFIGS[args.model]
    print(f"\nModel   : {args.model}  ({cfg['model_name']})")
    print(f"Scenario: {scenario_name}")
    print(f"Cameras : {cameras}")
    print(f"Output  : {results_root}\n")

    total_start = time.time()

    for cam in cameras:
        vdo_path = scenario_dir / cam / "vdo.avi"
        if not vdo_path.exists():
            print(f"  [{cam}] WARNING: {vdo_path} not found – skipped")
            continue

        out_dir = results_root / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "detections.txt"

        print(f"  [{cam}] Processing {vdo_path} ...")
        t0 = time.time()

        detector = YOLODetector(model_name=cfg["model_name"], finetune=cfg["finetune"])
        pipeline = DetectionPipeline(detector)
        pipeline(vdo_path, out_dir)

        print(f"  [{cam}] Done in {time.time() - t0:.1f}s  →  {out_file}\n")

    print(f"All cameras done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
