#!/bin/bash
#SBATCH --job-name=qd_extend_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_extend_smoke_%j.log
#SBATCH --error=logs/slurm_extend_smoke_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for task in combat navigation dodge; do
  python train_subtask_baseline.py --task "$task" --total-steps 263168 \
    --num-envs 8 --steps 128 --eval-episodes 2 --final-episodes 2 \
    --resume-from subtask_baselines --output subtask_baselines_extended_smoke
done
