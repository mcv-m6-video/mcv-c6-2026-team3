from models import *
from detectors import *
from models.adaptative_gaussian_model import AdaptiveGaussianModel
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import time

args = set_args()
K = 2.67
MIN_CC_PIXELS = 500
BG_PERCENTAGE = 0.25
config = build_config(args, f"simple_gaussian_test")

model = AdaptiveGaussianModel(K=K, p=0.005)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

preprocess_fn = preprocess.generate_morph_func_adap(4, (20, 45))

pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
print(f"Obtained mAP@0.5 = {mAP}")