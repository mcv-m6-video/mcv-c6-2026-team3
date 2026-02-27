import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MOGModel
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def main():
    args = set_args()
    
    # Configuration parameters
    HISTORY = 999
    NMIXTURES = 7
    BACKGROUND_RATIO = 0.5584106065962365
    NOISE_SIGMA = 19.97907990458267
    LEARNING_RATE = 0.008268579072404952
    
    MIN_CC_PIXELS = 252
    BG_PERCENTAGE = 0.25
    OPEN_MORPH = 4
    CLOSE_MORPH = 30
    
    config = build_config(args, "mog_run")

    model = MOGModel(
        history=HISTORY,
        nmixtures=NMIXTURES,
        backgroundRatio=BACKGROUND_RATIO,
        noiseSigma=NOISE_SIGMA,
        learningRate=LEARNING_RATE
    )
    
    detector = CCDetector(min_pixels=MIN_CC_PIXELS)
    preprocess_fn = preprocess.generate_morph_func(OPEN_MORPH, CLOSE_MORPH)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)
    
    print("Running MOG...")
    start_time = time.time()
    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
    end_time = time.time()
    
    print(f"Obtained mAP@0.5 = {mAP}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
