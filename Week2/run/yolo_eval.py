from sklearn.model_selection import KFold
from detectors import YOLODetector
from pipelines import EvaluationPipeline, NoDetectronPipeline
from torch.utils.data import Subset
from datasets import AICityDataset
from config import *
from utils import *
from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import time
from tqdm import tqdm
import gc

args = set_args()
    
# Configuration parameters
YOLO_MODEL = "yolo26n.pt"  # Use n for nano, s for small, m for medium, l for large
TRAIN_PERCENTAGE = 0.25
FOLDS = 4

config = build_config(args, "yolo_base")

dataset = AICityDataset(
    config.input_path, 
    config.xml_path,
    evaluation=True,
    car_class=0
)

dataset_size = len(dataset)
split = int(dataset_size * 0.25)

train_idx = list(range(0, split))
val_idx = list(range(split, dataset_size))

val_dataset = Subset(dataset, val_idx)

detector = YOLODetector(model_name=YOLO_MODEL)

pipeline = NoDetectronPipeline(detector)

start_time = time.time()

metrics = pipeline(
    dataset, 
    output=config.output_path,
    subset=val_dataset,
    save=True
)

end_time = time.time()

print("\nRESULTS")
print(f"mAP@0.5  : {metrics['map_50']:.4f}")
print(f"mAP@0.75 : {metrics['map_75']:.4f}")
print(f"mAP      : {metrics['map']:.4f}")
print(f"mIoU     : {metrics['miou']:.4f}")
print(f"Execution time: {end_time - start_time:.2f} seconds")
    