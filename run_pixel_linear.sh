#!/bin/bash
#SBATCH --job-name=qd_pixel_linear
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_pixel_linear_%j.log
#SBATCH --error=logs/slurm_pixel_linear_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# Control for the circuit head: same encoder, 52-parameter affine readout.
# Uses the classical learning rate; seeds run sequentially.
for seed in 0 1 2; do
  python train_pixel_nav.py --model linear --seed "$seed" --decisions 3000000 \
    --eval-every 250 --eval-episodes 32 --final-episodes 100 --max-seconds 6000 \
    --output "pixel_nav_runs/linear_seed$seed"
done
