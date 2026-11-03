#!/bin/bash
#SBATCH --job-name=qd_e1m1_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_pixel_e1m1_smoke_%j.log
#SBATCH --error=logs/slurm_pixel_e1m1_smoke_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# CPU only, so it can run next to a GPU job. Builds the distance field, checks the
# env, then runs a few PPO updates end to end with the classical head.
python e1m1_distance.py
python diagnose_pixel_e1m1.py
python train_pixel_nav.py --task e1m1 --model classical --seed 0 --decisions 8192 \
  --num-envs 4 --steps 128 --eval-every 8 --eval-episodes 4 --final-episodes 4 \
  --log-every 2 --random-baseline --output pixel_e1m1_runs/smoke
