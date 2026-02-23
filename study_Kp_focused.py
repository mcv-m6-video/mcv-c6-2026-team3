import optuna
import pickle
import os
import matplotlib.pyplot as plt
from models.adaptative_gaussian_model import AdaptiveGaussianModel
from detectors import CCDetector
from utils import set_args, ConvergenceEarlyStopping
from pipelines import DetectionPipepline
from config import build_config
import preprocess
import numpy as np

def objective(trial):
    k = trial.suggest_float('k', 1.0, 10.0)
    p = trial.suggest_float('p', 0.01, 0.99)

    cc_pixels = 480
    open_morph = 10
    close_morph = 30

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, "temp_study", create=False)

    model = AdaptiveGaussianModel(k, p)
    detector = CCDetector(min_pixels=cc_pixels)
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    return mAP

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', study_name="Kp_study")
    early_stop = ConvergenceEarlyStopping(patience=100, tolerance=0.001) #High patience to allow for the more complex search space
    
    study.optimize(objective, n_trials=300, callbacks=[early_stop], show_progress_bar=True)

    with open('results/studies/study_Kp.pkl', 'wb') as f:
        pickle.dump(study, f)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ks = [t.params['k'] for t in study.trials if t.state.name == 'COMPLETE']
    ps = [t.params['p'] for t in study.trials if t.state.name == 'COMPLETE']
    f1s = [t.value for t in study.trials if t.state.name == 'COMPLETE']
    
    ax.plot_trisurf(ks, ps, f1s, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Alpha (K)')
    ax.set_ylabel('Rho (p)')
    ax.set_zlabel('mAP Score')
    plt.title("Surface mAP Score vs Alpha and Rho")
    plt.savefig('results/studies/plot_Kp_surface.png')