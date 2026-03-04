import time
import os
from pathlib import Path
from trackers import IOUTracker
from utils import set_args
from pipelines import TrackingPipeline
from config import build_config
import optuna
from optuna.samplers import GridSampler
import numpy as np
import gc

args = set_args()
config = build_config(args, "study_sort", create=False)
TRAIN_PERCENTAGE = 0.25
DETECTIONS_FILE = Path(f"{args.results}/yolo_finetuned_run/detections.txt")

def objective(trial : optuna.Trial):
    
    iou_threshold = trial.suggest_float('iou_threshold', 0.0, 0.75)
    max_age = trial.suggest_int('max_age', 1, 15)
    
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

# Define grid search space
# iou_threshold: 0 to 0.75 with step 0.05 -> 16 values
# max_age: 1 to 15 -> 15 values
# Total combinations: 16 * 15 = 240
search_space = {
    'iou_threshold': [i * 0.05 for i in range(16)],  # [0.0, 0.05, 0.10, ..., 0.75]
    'max_age': list(range(1, 16))  # [1, 2, 3, ..., 15]
}

sampler = GridSampler(search_space)
    
storage_name = "sqlite:///results/studies/optimization.db"
study_name = "iou_params_study_grid_search"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    directions=['maximize', 'maximize'],
    sampler=sampler,
    load_if_exists=True
)

print(f"Initializing Grid Search for: {study_name}...")
print(f"Grid size: {len(search_space['iou_threshold'])} iou_threshold values × {len(search_space['max_age'])} max_age values = {len(search_space['iou_threshold']) * len(search_space['max_age'])} total combinations")
study.optimize(objective, n_trials=240, show_progress_bar=True)

print(f"Number of Pareto optimal trials: {len(study.best_trials)}")
print(f"\nBest trials (Pareto front):")
for i, trial in enumerate(study.best_trials[:5]):
    print(f"  Trial {i+1}: HOTA={trial.values[0]:.4f}, IDF1={trial.values[1]:.4f}")
    print(f"    Params: {trial.params}")
