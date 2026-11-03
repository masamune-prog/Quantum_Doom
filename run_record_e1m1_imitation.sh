#!/bin/bash
#SBATCH --job-name=qd_imit_video
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_e1m1_imit_video_%j.log
#SBATCH --error=logs/slurm_e1m1_imit_video_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
# CPU only: first successful greedy E1M1 episode of an imitation student (screen-only policy).
python record_gameplay.py "${CKPT:-pixel_e1m1_runs/imitation_classical_v2_seed0/best.pt}" --out videos_e1m1_imitation
