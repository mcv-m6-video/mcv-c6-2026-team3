import optuna
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GaussianModel
from detectors import CCDetector
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def objective(trial):
    # Suggest hyperparameters
    k = trial.suggest_float('k', 2.0, 11.0)
    cc_pixels = trial.suggest_int('min_cc_pixels', 300, 700)
    open_morph = trial.suggest_int("open_morph", 1, 10)
    close_morph = trial.suggest_int("close_morph", 30, 80)

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, f"gaussian_k_{k}_cc_{cc_pixels}", create=False)

    model = GaussianModel(k)
    detector = CCDetector(min_pixels=cc_pixels)
    
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)

    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    return mAP


os.makedirs('results/studies', exist_ok=True)
    
storage_name = "sqlite:///results/studies/optuna_optimization_simple.db"
study_name = "all_params_study"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)

print(f"Initializing optimization for: {study_name}...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best mAP: {study.best_value}")
print(f"Best Hyperparameters: {study.best_params}")