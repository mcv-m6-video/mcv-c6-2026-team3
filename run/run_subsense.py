import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SubsenseModel
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def main():
    args = set_args()
    
    # Configuration parameters
    MIN_CC_PIXELS = 300
    BG_PERCENTAGE = 0.25
    
    config = build_config(args, "subsense_run")

    model = SubsenseModel()
    
    detector = CCDetector(min_pixels=MIN_CC_PIXELS)
    # SuBSENSE uses no preprocessing
    preprocess_fn = None
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)
    
    print("Running SuBSENSE...")
    start_time = time.time()
    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
    end_time = time.time()

    print(f"Obtained mAP@0.5 = {mAP}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
