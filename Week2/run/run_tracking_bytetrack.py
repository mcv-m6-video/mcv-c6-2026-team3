import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers import ByteTrackTracker
from utils import set_args
from pipelines import TrackingPipeline
from config import build_config


def main():
    """
    Run ByteTrack tracking on pre-computed detections.
    
    ByteTrack is a state-of-the-art multi-object tracker that uses:
    - Two-stage association (high/low confidence detections)
    - Kalman filtering for motion prediction
    - Advanced track lifecycle management
    
    Configuration Tips:
    - track_thresh (0.2-0.5): Lower values track more objects but may include false positives
    - track_buffer (10-60): Higher values help maintain tracks during long occlusions
    - match_thresh (0.7-0.9): Higher values reduce ID switches but may lose tracks
    - frame_rate: Set to actual video FPS (AICity dataset is 10 FPS)
    """
    args = set_args()
    
    # ByteTrack Configuration parameters
    TRACK_THRESH = 0.6859066160175378
    TRACK_BUFFER = 60    # Number of frames to keep lost tracks before deletion
    MATCH_THRESH = 0.6519245296076033

    FRAME_RATE = 10      # Frame rate of the video (AICity is 10 FPS)
    TRAIN_PERCENTAGE = 0.25
    DETECTIONS_FILE = Path(f"{args.results}/yolo_finetuned_run/detections.txt")
    
    config = build_config(args, "tracking_bytetrack_finetuned")
    
    print(f"Initializing ByteTrack Tracker")
    
    tracker = ByteTrackTracker(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        frame_rate=FRAME_RATE
    )
    
    pipeline = TrackingPipeline(tracker, detector=None)
    
    print(f"Train/Test split: {TRAIN_PERCENTAGE:.0%} train, {1-TRAIN_PERCENTAGE:.0%} test")
    
    start_time = time.time()
    metrics = pipeline(config.input_path, config.output_path, config.xml_path,
                      train_percentage=TRAIN_PERCENTAGE, 
                      detections_file=DETECTIONS_FILE,
                      save=True)
    end_time = time.time()
    
    print("\n" + "="*80)
    print("BYTETRACK TRACKING RESULTS")
    print("="*80)
    print(f"HOTA: {metrics['HOTA']:.4f}")
    print(f"IDF1: {metrics['IDF1']:.4f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()
