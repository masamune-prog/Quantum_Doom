#!/bin/bash
#SBATCH --job-name=qd_human_nav
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_human_nav_%j.log
#SBATCH --error=logs/slurm_human_nav_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
for memory in off on; do
  python train_human_nav.py --memory "$memory" --updates 1000 \
    --num-envs 8 --steps 32 --epochs 2 --minibatch 64 \
    --eval-every 100 --eval-episodes 5 --final-episodes 20 \
    --max-seconds 9200 --output "human_nav_runs/$memory"
done
