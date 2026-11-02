#!/bin/bash
#SBATCH --job-name=qd_human_nav_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_human_nav_smoke_%j.log
#SBATCH --error=logs/slurm_human_nav_smoke_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
for memory in off on; do
  python train_human_nav.py --memory "$memory" --updates 1 --num-envs 2 \
    --steps 8 --epochs 1 --minibatch 16 --eval-every 1 \
    --eval-episodes 1 --final-episodes 1 --max-seconds 600 \
    --output "human_nav_smoke/$memory"
done
