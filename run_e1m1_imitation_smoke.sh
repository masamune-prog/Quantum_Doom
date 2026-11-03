#!/bin/bash
#SBATCH --job-name=qd_imit_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_e1m1_imit_smoke_%j.log
#SBATCH --error=logs/slurm_e1m1_imit_smoke_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
python train_e1m1_imitation.py --num-envs 4 --iterations 1 --initial-steps 3000 --iter-steps 2000 \
  --initial-epochs 1 --epochs 1 --capacity 20000 --eval-episodes 4 --final-episodes 4 \
  --max-episode 300 --output pixel_e1m1_runs/imitation_smoke
