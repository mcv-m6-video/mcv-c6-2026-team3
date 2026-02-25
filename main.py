from models import *
from detectors import *
from models.subsense_lobster_model import SubsenseModel, LobsterModel
from models.bgscnn_model import BGsCNNModel
from utils import *
from pipelines import DetectionPipepline
from pipelines.dl_detection_pipeline import DLDetectionPipeline
from config import build_config
import preprocess
import time

args = set_args()
K = 2.67
MIN_CC_PIXELS = 500
BG_PERCENTAGE = 0.25
config = build_config(args, f"bgscnn")

model    = BGsCNNModel(epochs=10, learning_rate=1e-4)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)
pipeline = DLDetectionPipeline(model, detector, preprocess_fn=None)

mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=True)
print(f"Obtained mAP@0.5 = {mAP}")