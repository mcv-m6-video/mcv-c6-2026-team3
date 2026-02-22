from models import GaussianModel
from detectors import CCDetector
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import time

args = set_args()
K = 6
MIN_CC_PIXELS = 480
BG_PERCENTAGE = 0.25
config = build_config(args, f"gaussian_k_{K}_cc_{MIN_CC_PIXELS}")

model = GaussianModel(K, use_median = False)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

preprocess_fn = preprocess.generate_morph_func(5, 60)

pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

start = time.time()
mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
print(f"Obtained mAP@0.5 = {mAP}")