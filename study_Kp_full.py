import optuna
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.adaptative_gaussian_model import AdaptiveGaussianModel
from detectors import CCDetector
from utils import set_args, ConvergenceEarlyStopping
from pipelines import DetectionPipepline
from config import build_config
import preprocess

def objective(trial):
    k = trial.suggest_float('k', 1.0, 10.0)
    p = trial.suggest_float('p', 0.01, 0.99, log=True)
    cc_pixels = trial.suggest_int('min_cc_pixels', 50, 500)
    open_morph = trial.suggest_int("open_morph", 5, 50)
    close_morph = trial.suggest_int("close_morph", 10, 60)

    args = set_args()
    BG_PERCENTAGE = 0.25
    config = build_config(args, f"adapt_k_{k}_p_{p}_cc_{cc_pixels}", create=False)

    model = AdaptiveGaussianModel(k, p)
    detector = CCDetector(min_pixels=cc_pixels)
    preprocess_fn = preprocess.generate_morph_func(open_morph, close_morph)
    pipeline = DetectionPipepline(model, detector, preprocess_fn=preprocess_fn)

    mAP = pipeline(config.input_path, config.output_path, config.xml_path, BG_PERCENTAGE, save=False)
    
    return mAP

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', study_name="all_params_study")
    early_stop = ConvergenceEarlyStopping(patience=150, tolerance=0.001) #High patience to allow for the more complex search space
    
    study.optimize(objective, n_trials=750, callbacks=[early_stop], show_progress_bar=True)

    os.makedirs('results/studies', exist_ok=True)
    with open('results/studies/study_all_params_Kp.pkl', 'wb') as f:
        pickle.dump(study, f)

    print("Best mAP: ", study.best_value)
    print("Best Hyperparameters: ", study.best_params)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ks = [t.params['k'] for t in study.trials if t.state.name == 'COMPLETE']
    ps = [t.params['p'] for t in study.trials if t.state.name == 'COMPLETE']
    f1s = [t.value for t in study.trials if t.state.name == 'COMPLETE']
    
    scatter = ax.scatter(ks, ps, f1s, c=f1s, cmap='viridis')
    ax.set_xlabel('Alpha (K)')
    ax.set_ylabel('Rho (p)')
    ax.set_zlabel('mAP Score')
    plt.colorbar(scatter, label='mAP Score')
    plt.title("mAP Score vs Alpha and Rho (All Params)")
    plt.savefig('results/studies/plot_all_params_3d.png')