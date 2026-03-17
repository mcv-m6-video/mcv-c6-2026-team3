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
    YOLO_MODEL = "yolo26n.pt"
    
    config = build_config(args, "yolo_finetuned_run")

    detector = YOLODetector(model_name=YOLO_MODEL, finetune=False)
    
    pipeline = DetectionPipeline(detector)
    
    print("Running YOLO Detection Pipeline...")
    
    start_time = time.time()
    pipeline(config.input_path, config.output_path)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
