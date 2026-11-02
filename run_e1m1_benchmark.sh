#!/bin/bash
#SBATCH --job-name=qd_e1m1_bench
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_e1m1_bench_%j.log
#SBATCH --error=logs/slurm_e1m1_bench_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python train_e1m1.py --skill 3 --num-envs 8 --steps 32 --updates 4 \
  --starts 19 --eval-episodes 2 --max-seconds 1200 --output e1m1_benchmark
