from models import *
from detectors import *
from models.subsense_lobster_model import SubsenseModel, LobsterModel
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import time

args = set_args()
K = 2.67
MIN_CC_PIXELS = 500
BG_PERCENTAGE = 0.25
config = build_config(args, f"subsense_test")

model = SubsenseModel()
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

preprocess_fn = preprocess.generate_morph_func_adap(4, (20, 45))

pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
print(f"Obtained mAP@0.5 = {mAP}")