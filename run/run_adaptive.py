import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptative_gaussian_model import AdaptiveGaussianModel
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def main():
    args = set_args()
    
    # Configuration parameters
    K = 2.633764
    P = 0.005414
    
    MIN_CC_PIXELS = 547
    BG_PERCENTAGE = 0.25
    
    OPEN_MORPH = 4
    CLOSE_MORPH_X = 33
    CLOSE_MORPH_Y = 44
    CLOSE_MORPH = (CLOSE_MORPH_X, CLOSE_MORPH_Y)
    
    config = build_config(args, "adaptive_gaussian_run")

    model = AdaptiveGaussianModel(K=K, p=P)
    
    detector = CCDetector(min_pixels=MIN_CC_PIXELS)
    preprocess_fn = preprocess.generate_morph_func_adap(OPEN_MORPH, CLOSE_MORPH)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)
    
    print("Running Adaptive Gaussian Model...")
    start_time = time.time()
    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
    end_time = time.time()

    print(f"Obtained mAP@0.5 = {mAP}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
