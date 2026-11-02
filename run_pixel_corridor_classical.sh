#!/bin/bash
#SBATCH --job-name=qd_corr_classical
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_pixel_corridor_classical_%j.log
#SBATCH --error=logs/slurm_pixel_corridor_classical_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# Deadly Corridor (skill 3), screen-only: validate the encoder with the classical head
# before the circuit and linear heads. Seeds run sequentially.
for seed in ${SEEDS:-0 1 2}; do
  extra=()
  [ "$seed" = 0 ] && extra=(--random-baseline)
  python train_pixel_nav.py --task corridor --model classical --seed "$seed" --decisions 3000000 \
    --eval-every 250 --eval-episodes 32 --final-episodes 100 --max-seconds 6000 \
    "${extra[@]}" --output "pixel_corridor_runs/classical_seed$seed"
done
