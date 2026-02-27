import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MOG2Model
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def main():
    args = set_args()
    
    # Configuration parameters
    HISTORY = 466
    VAR_THRESHOLD = 61.76900846523375
    LEARNING_RATE = 0.005094593594879748
    DETECT_SHADOWS = True # Assuming default or specific requirement not mentioned, keeping generic or True
    
    MIN_CC_PIXELS = 431
    BG_PERCENTAGE = 0.25
    OPEN_MORPH = 3
    CLOSE_MORPH = 43
    
    config = build_config(args, "mog2_run")

    model = MOG2Model(
        history=HISTORY,
        varThreshold=VAR_THRESHOLD,
        detectShadows=DETECT_SHADOWS,
        learningRate=LEARNING_RATE
    )
    
    detector = CCDetector(min_pixels=MIN_CC_PIXELS)
    preprocess_fn = preprocess.generate_morph_func(OPEN_MORPH, CLOSE_MORPH)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)
    
    print("Running MOG2...")
    start_time = time.time()
    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
    end_time = time.time()

    print(f"Obtained mAP@0.5 = {mAP}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
