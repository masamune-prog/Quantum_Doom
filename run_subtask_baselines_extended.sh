#!/bin/bash
#SBATCH --job-name=qd_extend_baselines
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_baselines_extended_%j.log
#SBATCH --error=logs/slurm_baselines_extended_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for task in combat navigation dodge; do
  python train_subtask_baseline.py --task "$task" --total-steps 524288 \
    --max-seconds 6000 --resume-from subtask_baselines \
    --output subtask_baselines_extended
done
