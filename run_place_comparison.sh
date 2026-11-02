#!/bin/bash
#SBATCH --job-name=qd_place_compare
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_place_compare_%j.log
#SBATCH --error=logs/slurm_place_compare_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
python train_human_nav.py --task navigation --memory on --memory-kind place \
  --model classical_mlp --updates 1000 --num-envs 8 --steps 32 \
  --epochs 2 --minibatch 64 --lr 0.002 --eval-every 250 \
  --eval-episodes 20 --final-episodes 20 --max-seconds 3600 \
  --output human_place_comparison/classical_mlp
python train_human_nav.py --task navigation --memory on --memory-kind place \
  --model quantum_bridged --updates 1000 --num-envs 8 --steps 32 \
  --epochs 2 --minibatch 64 --lr 0.003 --eval-every 250 \
  --eval-episodes 20 --final-episodes 20 --max-seconds 12000 \
  --output human_place_comparison/quantum_bridged
