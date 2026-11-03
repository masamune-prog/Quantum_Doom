#!/bin/bash
#SBATCH --job-name=qd_ft_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_e1m1_ft_smoke_%j.log
#SBATCH --error=logs/slurm_e1m1_ft_smoke_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
python train_pixel_nav.py --task e1m1 --model classical --seed 0 --decisions 12288 \
  --num-envs 4 --steps 128 --eval-every 6 --eval-episodes 4 --final-episodes 4 --log-every 2 \
  --init-checkpoint pixel_e1m1_runs/imitation_classical_v2_seed0/best.pt \
  --value-warmup 3 --lr 5e-5 --curriculum uniform --real-fraction 0.6 \
  --output pixel_e1m1_runs/finetune_smoke
