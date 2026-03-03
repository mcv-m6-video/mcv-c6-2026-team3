import time
import os
from pathlib import Path
from trackers import IOUTracker
from utils import set_args
from pipelines import TrackingPipeline
from config import build_config
import optuna
import numpy as np
import gc

args = set_args()
config = build_config(args, "study_iou", create=False)
TRAIN_PERCENTAGE = 0.25
DETECTIONS_FILE = 'results/detections.txt'

def objective(trial : optuna.Trial):
    
    iou_threshold = trial.suggest_float('iou_threshold', 0.0, 1.0)
    
    tracker = IOUTracker(iou_threshold=iou_threshold)
    
    pipeline = TrackingPipeline(tracker, detector=None)
    
    metrics = pipeline(config.input_path, config.output_path, config.xml_path,
                      train_percentage=TRAIN_PERCENTAGE,
                      detections_file=DETECTIONS_FILE,
                      save=False)
    
    hota = metrics['HOTA']
    idf1 = metrics['IDF1']
    
    loss = np.sqrt((1 - idf1)**2 + (1 - hota)**2)
    
    del pipeline
    del tracker
    gc.collect()
    
    return loss



os.makedirs('results/studies', exist_ok=True)
    
storage_name = "sqlite:///results/studies/optimization_iou.db"
study_name = "all_params_study"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='minimize',
    load_if_exists=True
)

print(f"Initializing optimization for: {study_name}...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best mAP: {study.best_value}")
print(f"Best Hyperparameters: {study.best_params}")
