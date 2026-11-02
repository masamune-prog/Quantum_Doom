#!/bin/bash
#SBATCH --job-name=qd_e1m1_memory
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_e1m1_memory_%j.log
#SBATCH --error=logs/slurm_e1m1_memory_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
python diagnose_e1m1_unstick.py
