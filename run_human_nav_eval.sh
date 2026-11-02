#!/bin/bash
#SBATCH --job-name=qd_human_nav_eval
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_human_nav_eval_%j.log
#SBATCH --error=logs/slurm_human_nav_eval_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
python evaluate_human_nav.py --root human_nav_runs --episodes 20 --seed 20000
