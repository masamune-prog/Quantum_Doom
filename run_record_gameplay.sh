#!/bin/bash
#SBATCH --job-name=qd_videos
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_videos_%j.log
#SBATCH --error=logs/slurm_videos_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# CPU only: greedy play of every saved best checkpoint on held-out seeds from 20000; saves the first successful episode.
python record_gameplay.py "$@"
