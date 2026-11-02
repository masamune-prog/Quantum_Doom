#!/bin/bash
#SBATCH --job-name=qd_dodge_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_dodge_smoke_%j.log
#SBATCH --error=logs/slurm_dodge_smoke_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python train_subtask_baseline.py --task dodge --total-steps 1024 \
  --num-envs 4 --steps 32 --eval-episodes 2 --final-episodes 2 \
  --output subtask_baseline_smoke
