#!/bin/bash
#SBATCH --job-name=qd_classical_tasks
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_baselines_%j.log
#SBATCH --error=logs/slurm_baselines_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python train_subtask_baseline.py --task combat --max-seconds 7200
python train_subtask_baseline.py --task navigation --max-seconds 7200
