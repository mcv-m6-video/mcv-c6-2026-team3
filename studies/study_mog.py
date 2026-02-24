import optuna
import os
from models import MOGModel
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def objective(trial):
    history = trial.suggest_int('history', 100, 1000)
    nmixtures = trial.suggest_int('nmixtures', 3, 7)
    backgroundRatio = trial.suggest_float('backgroundRatio', 0.5, 0.9)
    noiseSigma = trial.suggest_float('noiseSigma', 0, 20)
    learningRate = trial.suggest_float('learningRate', 0.0, 0.01)
    
    cc_pixels = trial.suggest_int('min_cc_pixels', 100, 500)
    open_morph = trial.suggest_int("open_morph", 1, 10)
    close_morph = trial.suggest_int("close_morph", 5, 50)

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, "temp_study", create=False)

    model = MOGModel(
        history=history,
        nmixtures=nmixtures,
        backgroundRatio=backgroundRatio,
        noiseSigma=noiseSigma,
        learningRate=learningRate
    )
    detector = CCDetector(min_pixels=cc_pixels)
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    
    return mAP

if __name__ == "__main__":
    os.makedirs('results/studies', exist_ok=True)
    
    storage_name = "sqlite:///results/studies/optuna_optimization.db"
    study_name = "mog_study"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )
    
    print(f"Initializing optimization for: {study_name}...")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print(f"Best mAP: {study.best_value}")
    print(f"Best Hyperparameters: {study.best_params}")
