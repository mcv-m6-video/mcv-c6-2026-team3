from models import GaussianModel
from detectors import CCDetector
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess

args = set_args()
K = args.k
MIN_CC_PIXELS = args.min
BG_PERCENTAGE = 0.25
config = build_config(args, f"gaussian_k_{K}_cc_{MIN_CC_PIXELS}")

model = GaussianModel(K)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess.preprocess_morph)
pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE)