#!/bin/bash
#SBATCH --job-name=qd_human_nav_diag
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_human_nav_diag_%j.log
#SBATCH --error=logs/slurm_human_nav_diag_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export OMP_NUM_THREADS=1
python diagnose_human_nav.py
