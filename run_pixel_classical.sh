#!/bin/bash
#SBATCH --job-name=qd_pixel_classical
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_pixel_classical_%j.log
#SBATCH --error=logs/slurm_pixel_classical_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# Validate the CNN encoder and auxiliary losses with the classical head before
# spending ~5 GPU hours per quantum run. Seeds run sequentially (one-job limit).
for seed in 0 1 2; do
  extra=()
  [ "$seed" = 0 ] && extra=(--random-baseline)
  python train_pixel_nav.py --model classical --seed "$seed" --decisions 3000000 \
    --eval-every 250 --eval-episodes 32 --final-episodes 100 --max-seconds 6000 \
    "${extra[@]}" --output "pixel_nav_runs/classical_seed$seed"
done
