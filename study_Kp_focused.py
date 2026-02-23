import optuna
import os
from models.adaptative_gaussian_model import AdaptiveGaussianModel
from detectors import CCDetector
from utils import set_args
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def objective(trial):
    k = trial.suggest_float('k', 1.0, 10.0)
    p = trial.suggest_float('p', 0.01, 0.99)

    cc_pixels = 400 
    open_morph = 10
    close_morph = 30

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, "temp_study_kp", create=False)

    model = AdaptiveGaussianModel(K=k, p=p)
    detector = CCDetector(min_pixels=cc_pixels)
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)
    
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    
    return mAP

if __name__ == "__main__":
    os.makedirs('results/studies', exist_ok=True)
    
    storage_name = "sqlite:///results/studies/optuna_optimization.db"
    study_name = "Kp_study"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )
    
    print(f"Initializing optimization for: {study_name}...")
    study.optimize(objective, n_trials=300, show_progress_bar=True)

    print(f"Best mAP: {study.best_value}")
    print(f"Best Hyperparameters: {study.best_params}")