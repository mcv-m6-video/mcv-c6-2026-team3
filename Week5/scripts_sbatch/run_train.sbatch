#!/bin/bash
#SBATCH --job-name=wbse_baseline_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mlow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

# La carpeta logs debe existir también antes de sbatch
mkdir -p logs

# Inicializar conda correctamente en shell no interactiva
source /ghome/group03/miniconda3/etc/profile.d/conda.sh
conda activate soccer

echo "Conda env: ${CONDA_DEFAULT_ENV:-N/A}"
echo "Python: $(which python)"
python --version

cd /ghome/group03/shinto/c6_soccer/CVMasterActionRecognitionSpotting-main

echo "Lanzando entrenamiento..."
python3 main_classification.py --model baseline_weighted_bce

echo "========================================"
echo "End time: $(date)"
echo "========================================"