from torch.utils.data import Subset
from sklearn.model_selection import KFold
from datasets import AICityDataset
from config import *
from utils import *
from ultralytics import YOLO
from pathlib import Path
from torch.utils.data import DataLoader
import os
import cv2
from tqdm import tqdm

args = set_args()

config = build_config(args, "yolo_run", create=False)

dataset = AICityDataset(
    config.input_path, 
    config.xml_path,
    evaluation=False,
    car_class=0
    )

TARGET_DIR = "../data/yolo"

os.makedirs(f'{TARGET_DIR}/images', exist_ok=True)
os.makedirs(f'{TARGET_DIR}/labels', exist_ok=True)

for idx in tqdm(range(len(dataset)), desc="Creating YOLO dataset: ", unit=" images"):
    
    frame, annotations = dataset[idx]
    
    cv2.imwrite(f'{TARGET_DIR}/images/frame_{idx}.jpg', frame)
    
    with open(f'{TARGET_DIR}/labels/frame_{idx}.txt', 'w') as f:
        
        for bbox, cls in zip(annotations['bboxes'], annotations['classes']):
            
            x, y, w, h = bbox
            f.write(f'{int(cls)} {x:.4f} {y:.4f} {w:.4f} {h:.4f}\n')