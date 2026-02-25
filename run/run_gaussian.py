import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GaussianModelShadow
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def main():
    args = set_args()
    
    # Configuration parameters
    K = 2.087023
    MIN_CC_PIXELS = 700
    BG_PERCENTAGE = 0.25
    OPEN_MORPH = 5
    CLOSE_MORPH = 50
    
    config = build_config(args, "gaussian_shadow_run")

    model = GaussianModelShadow(K=K)
    
    detector = CCDetector(min_pixels=MIN_CC_PIXELS)
    preprocess_fn = preprocess.generate_morph_func(OPEN_MORPH, CLOSE_MORPH)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)
    
    print("Running Gaussian Model (Shadow)...")
    start_time = time.time()
    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
    end_time = time.time()

    print(f"Obtained mAP@0.5 = {mAP}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
