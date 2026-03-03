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

args = set_args()
    
# Configuration parameters
YOLO_MODEL = "best.pt"  # Use n for nano, s for small, m for medium, l for large
TRAIN_PERCENTAGE = 0.25
FOLDS = 4

config = build_config(args, "yolo_finetune_25_prob")

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

with open(f'{config.yolo_path}/train.txt', 'w') as f:
    for idx in train_idx:
        f.write(f'{os.path.abspath(f"{config.yolo_path}/images/frame_{idx}.jpg")}\n')

with open(f'{config.yolo_path}/val.txt', 'w') as f:
    for idx in val_idx:
        f.write(f'{os.path.abspath(f"{config.yolo_path}/images/frame_{idx}.jpg")}\n')

yaml_content = {
    'path': os.path.abspath(f'{config.yolo_path}'),   # must point to parent of images/ and labels/
    'train': os.path.abspath(f'{config.yolo_path}/train.txt'),
    'val': os.path.abspath(f'{config.yolo_path}/val.txt'),
    'nc': 1,
    'names': ['car']
}

with open(f'{config.yolo_path}/dataset.yaml', 'w') as f:
    yaml.dump(yaml_content, f)

# models_dir = Path("./models")
# model_path = models_dir / YOLO_MODEL

# model = YOLO(str(model_path))
# model.train(data=f'{config.yolo_path}/dataset.yaml', epochs=20, imgsz=640, batch=16, project="models", name="Finetune0.25", freeze=21)

val_dataset = Subset(dataset, val_idx)

detector = YOLODetector(model_name=YOLO_MODEL, finetune=True)

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