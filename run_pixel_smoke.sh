#!/bin/bash
#SBATCH --job-name=qd_pixel_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm_pixel_smoke_%j.log
#SBATCH --error=logs/slurm_pixel_smoke_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# 10 updates of 8x128 decisions per head, to check correctness and measure throughput.
for model in quantum classical; do
  python train_pixel_nav.py --model "$model" --decisions 10240 --log-every 1 \
    --eval-every 10 --eval-episodes 8 --final-episodes 8 --random-baseline \
    --output "pixel_nav_smoke/$model"
done
