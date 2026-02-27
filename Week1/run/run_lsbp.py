import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LSBPModel
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def main():
    args = set_args()
    
    # Configuration parameters
    N_SAMPLES = 18
    LSBP_RADIUS = 15
    T_LOWER = 1.5380357641477462
    T_UPPER = 26.87192947642451
    T_INC = 1.637602322668896
    T_DEC = 0.032027178922089304
    R_SCALE = 14.89802851356109
    LSBP_THRESHOLD = 13
    MIN_COUNT = 1
    LEARNING_RATE = 0.0005944188489098032

    MIN_CC_PIXELS = 410
    BG_PERCENTAGE = 0.25
    OPEN_MORPH = 1
    CLOSE_MORPH = 18
    
    config = build_config(args, "lsbp_run")

    model = LSBPModel(
        nSamples=N_SAMPLES,
        LSBPRadius=LSBP_RADIUS,
        Tlower=T_LOWER,
        Tupper=T_UPPER,
        Tinc=T_INC,
        Tdec=T_DEC,
        Rscale=R_SCALE,
        LSBPthreshold=LSBP_THRESHOLD,
        minCount=MIN_COUNT,
        learningRate=LEARNING_RATE
    )
    
    detector = CCDetector(min_pixels=MIN_CC_PIXELS)
    preprocess_fn = preprocess.generate_morph_func(OPEN_MORPH, CLOSE_MORPH)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)
    
    print("Running LSBP...")
    start_time = time.time()
    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
    end_time = time.time()

    print(f"Obtained mAP@0.5 = {mAP}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
