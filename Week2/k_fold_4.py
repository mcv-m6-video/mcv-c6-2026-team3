from sklearn.model_selection import KFold
from detectors import YOLODetector
from pipelines import EvaluationPipeline
from torch.utils.data import Subset
from datasets import AICityDataset
from config import *
from utils import *
from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import time
from collections import deque
import numpy as np
import gc

args = set_args()
    
# Configuration parameters
YOLO_MODEL = "yolo26n.pt"  # Use n for nano, s for small, m for medium, l for large
TRAIN_PERCENTAGE = 0.25
FOLDS = 4

config = build_config(args, "yolo_fold_random")

dataset = AICityDataset(
    config.input_path, 
    config.xml_path,
    evaluation=True,
    car_class=0
    )

k = 4
kfold = KFold(n_splits=k, shuffle=True, random_state=0)
indices = list(range(len(dataset)))

mAP50 = []
mAP75 = []
mAP = []
mIoU = []

for fold, (val_idx, train_idx) in enumerate(kfold.split(indices)):
    print(f"\n--- Fold {fold+1}/{k} ---")

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

    models_dir = Path("./models")
    model_path = models_dir / YOLO_MODEL

    model = YOLO(str(model_path))
    model.train(data=f'{config.yolo_path}/dataset.yaml', epochs=20, imgsz=640, batch=16, project="models", name=f"fold_{fold}_train_random", freeze=21)

    val_dataset = Subset(dataset, val_idx)

    detector = YOLODetector(model_name=YOLO_MODEL, model=model, finetune=True)

    pipeline = EvaluationPipeline(detector)

    os.makedirs(config.output_path / f"fold_{fold}", exist_ok=True)

    metrics = pipeline(dataset, config.output_path / f"fold_{fold}", val_dataset, save=True)
    
    #Let's see if erasing them will make memory not get overloaded after some time
    del metrics
    del pipeline
    del detector
    del val_dataset
    del model
    gc.collect()
    
    
    
for i in range(k):
    
    with open(f'{config.output_path}/fold_{i}/metrics.txt', 'r') as f:
        last_lines = list(deque(f, maxlen=4))
        
        ap50 = float(last_lines[0].strip().split(':')[1].strip())
        ap75 = float(last_lines[1].strip().split(':')[1].strip())
        ap   = float(last_lines[2].strip().split(':')[1].strip())
        iou  = float(last_lines[3].strip().split(':')[1].strip())
        
        mAP50.append(ap50)
        mAP75.append(ap75)
        mAP.append(ap)
        mIoU.append(iou)
    
with open(config.output_path / "final_results.txt", "w") as f:
    f.write(f"mAP@50 = {np.mean(mAP50)} ± {np.std(mAP50)}\n")
    f.write(f"mAP@75 = {np.mean(mAP75)} ± {np.std(mAP75)}\n")
    f.write(f"mAP = {np.mean(mAP)} ± {np.std(mAP)}\n")
    f.write(f"mIoU = {np.mean(mIoU)} ± {np.std(mIoU)}")
    
    
print(f"Results can be found on {config.output_path / 'final_results.txt'}")