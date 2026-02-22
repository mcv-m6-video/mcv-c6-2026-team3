import optuna
from models import GaussianModel
from detectors import CCDetector
from utils import *
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import os
import pickle

def objective(trial):
    # Suggest hyperparameters
    k = trial.suggest_float('k', 2.0, 11.0)
    cc_pixels = trial.suggest_int('min_cc_pixels', 50, 500)
    open_morph = trial.suggest_int("open_morph", 5, 50)
    close_morph = trial.suggest_int("close_morph", 10, 60)

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, f"gaussian_k_{k}_cc_{cc_pixels}", create=False)

    model = GaussianModel(k)
    detector = CCDetector(min_pixels=cc_pixels)
    
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)

    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    return mAP


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70, show_progress_bar=True)


print("Best Accuracy: ", study.best_value)
print("Best Hyperparameters: ", study.best_params)

with open('results/best_params.pkl', 'wb') as f:
    pickle.dump(study.best_params, f)