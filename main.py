from models import *
from detectors import *
from models.adaptative_gaussian_model import AdaptiveGaussianModel
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import time

args = set_args()
K = 4
MIN_CC_PIXELS = 325
BG_PERCENTAGE = 0.25
config = build_config(args, f"simple_gaussian_test")

model = AdaptiveGaussianModel(K=2.5, p=0.01)
detector = CCDetector(min_pixels=0)

preprocess_fn = preprocess.generate_morph_func(5, 50)

pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
print(f"Obtained mAP@0.5 = {mAP}")