import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers import SORTTracker
from utils import set_args
from pipelines import TrackingPipeline
from config import build_config


def main():
    args = set_args()
    
    # SORT Configuration parameters
    MAX_AGE = 1  # Maximum number of frames to keep alive a track without associated detections
    MIN_HITS = 3  # Minimum number of associated detections before track is confirmed
    IOU_THRESHOLD = 0.3  # Minimum IoU for matching
    TRAIN_PERCENTAGE = 0.25
    DETECTIONS_FILE = Path(f"{args.results}/yolo_finetuned_run/detections.txt")
    
    config = build_config(args, "tracking_sort_finetuned")
    
    tracker = SORTTracker(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
    
    pipeline = TrackingPipeline(tracker, detector=None)
    
    print(f"Train/Test split: {TRAIN_PERCENTAGE:.0%} train, {1-TRAIN_PERCENTAGE:.0%} test")
    
    start_time = time.time()
    metrics = pipeline(config.input_path, config.output_path, config.xml_path,
                      train_percentage=TRAIN_PERCENTAGE, 
                      detections_file=DETECTIONS_FILE,
                      save=True)
    end_time = time.time()
    
    print("SORT TRACKING RESULTS")
    print(f"HOTA: {metrics['HOTA']:.4f}")
    print(f"IDF1: {metrics['IDF1']:.4f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
