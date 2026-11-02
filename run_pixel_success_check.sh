#!/bin/bash
#SBATCH --job-name=qd_pixel_succ
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_pixel_succ_%j.log
#SBATCH --error=logs/slurm_pixel_succ_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
python diagnose_pixel_success.py
