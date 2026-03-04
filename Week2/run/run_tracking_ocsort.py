import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers import OCSORTTracker
from utils import set_args
from pipelines import TrackingPipeline
from config import build_config


def main():
    """
    Run OC-SORT (Observation-Centric SORT) tracking on pre-computed detections.
    
    OC-SORT is an improved version of SORT that addresses occlusion challenges through:
    - Observation-Centric Momentum (OCM): Uses historical observations for better motion prediction
    - Observation-Centric Re-association (OCR): Re-associates tracks using past observations
    - Support for multiple association metrics (IoU, GIoU, DIoU, CIoU)
    
    Configuration Tips:
    - det_thresh (0.1-0.3): Detection confidence threshold for filtering
    - max_age (10-50): Higher values maintain tracks longer during occlusions
    - min_hits (1-5): Lower values initialize tracks faster
    - iou_threshold (0.2-0.4): Matching threshold for association
    - delta_t (1-5): Time steps for observation-centric momentum (3 is typical)
    - asso_func: Association metric ("iou", "giou", "ciou", "diou")
    - inertia (0.1-0.5): Weight for observation-centric momentum
    - use_byte: Enable ByteTrack-style second matching with low-confidence detections
    """
    args = set_args()
    
    # OC-SORT Configuration parameters
    DET_THRESH = 0.2069259351554724
    MAX_AGE = 13
    MIN_HITS = 5
    IOU_THRESHOLD = 0.3164438451238905
    DELTA_T = 2
    ASSO_FUNC = "iou"
    INERTIA = 0.22885026775270556
    USE_BYTE = False
    
    TRAIN_PERCENTAGE = 0.25
    DETECTIONS_FILE = Path(f"{args.results}/yolo_finetuned_run/detections.txt")
    
    config = build_config(args, "tracking_ocsort_finetuned")
    
    print(f"Initializing OC-SORT Tracker")
    print(f"  - Association function: {ASSO_FUNC}")
    print(f"  - Delta T: {DELTA_T}")
    print(f"  - Inertia: {INERTIA}")
    print(f"  - Use ByteTrack matching: {USE_BYTE}")
    
    tracker = OCSORTTracker(
        det_thresh=DET_THRESH,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        iou_threshold=IOU_THRESHOLD,
        delta_t=DELTA_T,
        asso_func=ASSO_FUNC,
        inertia=INERTIA,
        use_byte=USE_BYTE
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
    print("OC-SORT TRACKING RESULTS")
    print("="*80)
    print(f"HOTA: {metrics['HOTA']:.4f}")
    print(f"IDF1: {metrics['IDF1']:.4f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()
