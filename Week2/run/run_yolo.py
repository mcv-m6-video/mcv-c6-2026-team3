import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors import YOLODetector
from utils import set_args
from pipelines import DetectionPipeline
from config import build_config

def main():
    args = set_args()
    
    # Configuration parameters
    YOLO_MODEL = "yolo26m.pt"  # Use n for nano, s for small, m for medium, l for large
    TRAIN_PERCENTAGE = 0.25  # First 25% of frames are train (skipped), rest is test
    
    config = build_config(args, "yolo_run")

    print(f"Loading YOLO model: {YOLO_MODEL}")
    detector = YOLODetector(model_name=YOLO_MODEL)
    
    pipeline = DetectionPipeline(detector)
    
    print("Running YOLO Detection Pipeline...")
    print(f"Train/Test split: {TRAIN_PERCENTAGE:.0%} train, {1-TRAIN_PERCENTAGE:.0%} test")
    print("-" * 80)
    
    start_time = time.time()
    metrics = pipeline(config.input_path, config.output_path, config.xml_path, 
                      train_percentage=TRAIN_PERCENTAGE, save=True)
    end_time = time.time()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"mAP@0.5  : {metrics['mAP50']:.4f}")
    print(f"mAP@0.75 : {metrics['mAP75']:.4f}")
    print(f"mAP      : {metrics['mAP']:.4f}")
    print(f"mIoU     : {metrics['mIoU']:.4f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
