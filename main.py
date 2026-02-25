from models import *
from detectors import *
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import time

args = set_args()
K = 2.09
MIN_CC_PIXELS = 700
BG_PERCENTAGE = 0.25
config = build_config(args, f"gaussian_optimal_alpha")

model = GaussianModelShadow(K=K)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

preprocess_fn = preprocess.generate_morph_func(5, 50)

pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn, save_steps=True)

mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
print(f"Obtained mAP@0.5 = {mAP}")