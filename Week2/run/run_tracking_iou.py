import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers import IOUTracker
from utils import set_args
from pipelines import TrackingPipeline
from config import build_config


def main():
    args = set_args()
    
    # Configuration parameters
    IOU_THRESHOLD = 0.1869655175330422 #Best parameter found with optuna
    TRAIN_PERCENTAGE = 0.25
    DETECTIONS_FILE = Path(f"{args.results}/yolo_finetuned_run/detections.txt")
    
    config = build_config(args, "tracking_from_detections_0.2_finetuned")
    
    print(f"Initializing IOU Tracker (threshold={IOU_THRESHOLD})")
    tracker = IOUTracker(iou_threshold=IOU_THRESHOLD)
    
    pipeline = TrackingPipeline(tracker, detector=None)
    
    print(f"Train/Test split: {TRAIN_PERCENTAGE:.0%} train, {1-TRAIN_PERCENTAGE:.0%} test")
    
    start_time = time.time()
    metrics = pipeline(config.input_path, config.output_path, config.xml_path,
                      train_percentage=TRAIN_PERCENTAGE, 
                      detections_file=DETECTIONS_FILE,
                      save=True)
    end_time = time.time()
    
    print("TRACKING RESULTS")
    print(f"HOTA: {metrics['HOTA']:.4f}")
    print(f"IDF1: {metrics['IDF1']:.4f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
