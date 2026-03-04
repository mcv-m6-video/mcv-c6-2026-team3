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
config = build_config(args, "study_sort", create=False)
TRAIN_PERCENTAGE = 0.25
DETECTIONS_FILE = Path(f"{args.results}/yolo_finetuned_run/detections.txt")

def objective(trial : optuna.Trial):
    
    iou_threshold = trial.suggest_float('iou_threshold', 0.0, 1.0)
    max_age = trial.suggest_int('max_age', 0, 30)
    
    tracker = IOUTracker(iou_threshold=iou_threshold, max_age=max_age)
    
    pipeline = TrackingPipeline(tracker, detector=None)
    
    metrics = pipeline(config.input_path, config.output_path, config.xml_path,
                      train_percentage=TRAIN_PERCENTAGE,
                      detections_file=DETECTIONS_FILE,
                      save=False)
    
    hota = metrics['HOTA']
    idf1 = metrics['IDF1']
    
    del pipeline
    del tracker
    gc.collect()
    
    return hota, idf1



os.makedirs('results/studies', exist_ok=True)
    
storage_name = "sqlite:///results/studies/optimization.db"
study_name = "iou_params_study"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    directions=['maximize', 'maximize'],
    load_if_exists=True
)

print(f"Initializing optimization for: {study_name}...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Number of Pareto optimal trials: {len(study.best_trials)}")
print(f"\nBest trials (Pareto front):")
for i, trial in enumerate(study.best_trials[:5]):
    print(f"  Trial {i+1}: HOTA={trial.values[0]:.4f}, IDF1={trial.values[1]:.4f}")
    print(f"    Params: {trial.params}")
