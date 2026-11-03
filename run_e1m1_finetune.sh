#!/bin/bash
#SBATCH --job-name=qd_e1m1_ft
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_e1m1_finetune_%x_%j.log
#SBATCH --error=logs/slurm_e1m1_finetune_%x_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# PPO fine-tuning of an E1M1 imitation student with the environment reward only (no teacher).
# The starting policy is evaluated first and kept as best.pt, so a worse result cannot replace it.
MODEL=${MODEL:-classical}
SEED=${SEED:-0}
INIT=${INIT:-pixel_e1m1_runs/imitation_${MODEL}_v2_seed0/best.pt}
DECISIONS=${DECISIONS:-3000000}
TAG=${TAG:-}
EXTRA=${EXTRA:-}   # later flags override the defaults below
python train_pixel_nav.py --task e1m1 --model "$MODEL" --seed "$SEED" --decisions "$DECISIONS" \
  --init-checkpoint "$INIT" --value-warmup 100 --lr 5e-5 --head-lr 1e-4 --entropy-coef 0.005 \
  --curriculum uniform --real-fraction 0.6 --eval-every 50 --eval-episodes 64 \
  --final-episodes 100 --max-seconds 19500 $EXTRA \
  --output "pixel_e1m1_runs/finetune_${MODEL}${TAG}_seed${SEED}"
