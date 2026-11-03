#!/bin/bash
#SBATCH --job-name=qd_e1m1_video
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_e1m1_video_%j.log
#SBATCH --error=logs/slurm_e1m1_video_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# CPU only. Greedy play of an E1M1 checkpoint from the real start and from curriculum
# waypoints, 500 decisions each, failures included.
CKPT=${CKPT:-pixel_e1m1_runs/classical_warm_seed0/best.pt}
python record_gameplay.py "$CKPT" --out videos_e1m1 --any-outcome --max-steps 500 --first-seed 20000 --episodes 1 --tag _real "$@"
for wp in 12 8; do
  python record_gameplay.py "$CKPT" --out videos_e1m1 --any-outcome --max-steps 500 --first-seed 1 --episodes 1 --start-wp $wp --tag _wp$wp "$@"
done
