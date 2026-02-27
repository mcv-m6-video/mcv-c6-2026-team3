import optuna
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GaussianModel, GaussianModelShadow
from models.adaptative_gaussian_model import AdaptiveGaussianModel
from detectors import CCDetector
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def objective(trial):
    # Suggest hyperparameters
    k = trial.suggest_float('k', 1.0, 11.0)
    cc_pixels = 700
    open_morph = 5
    close_morph = 50

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, f"gaussian_k_{k}_cc_{cc_pixels}", create=False)

    model = GaussianModelShadow(k)
    detector = CCDetector(min_pixels=cc_pixels)
    
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)

    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    return mAP


os.makedirs('results/studies', exist_ok=True)
    
storage_name = "sqlite:///results/studies/optimization_k.db"
study_name = "all_params_study"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)

print(f"Initializing optimization for: {study_name}...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best mAP: {study.best_value}")
print(f"Best Hyperparameters: {study.best_params}")

k_best = study.best_params["k"]

def objective_2(trial):
    k = k_best
    p = trial.suggest_float('p', 0.001, 0.1, log=True) 
    cc_pixels = 547
    open_morph = 4
    close_morph_x = 33
    close_morph_y =  44

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, "temp_study", create=False)
    close_morph = (close_morph_x, close_morph_y)

    model = AdaptiveGaussianModel(K=k, p=p)
    detector = CCDetector(min_pixels=cc_pixels)
    preprocess_fn = preprocess.generate_morph_func_adap(open_morph, close_morph)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    
    return mAP

storage_name = "sqlite:///results/studies/p_optimization.db"
study_name = "all_params_study"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)

print(f"Initializing optimization for: {study_name}...")
study.optimize(objective_2, n_trials=50, show_progress_bar=True)