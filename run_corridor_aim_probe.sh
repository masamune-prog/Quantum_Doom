#!/bin/bash
#SBATCH --job-name=qd_aim_probe
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_aim_probe_%j.log
#SBATCH --error=logs/slurm_aim_probe_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
python probe_corridor_aim.py "$@"
