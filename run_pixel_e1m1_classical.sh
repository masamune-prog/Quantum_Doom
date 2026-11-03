#!/bin/bash
#SBATCH --job-name=qd_e1m1_cls
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_pixel_e1m1_classical_%x_%j.log
#SBATCH --error=logs/slurm_pixel_e1m1_classical_%x_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# E1M1, classical head, reverse-start curriculum, evaluation from the real start.
# VARIANT=warm starts from the corridor classical seed-0 encoder; VARIANT=scratch does not.
# The GPU job is `sbatch --gres=gpu:1`; without --gres this runs on CPU.
VARIANT=${VARIANT:-warm}
SEED=${SEED:-0}
CURRICULUM=${CURRICULUM:-reverse}   # reverse | uniform | none
TAG=${TAG:-}                         # suffix for the output directory
extra=()
if [ "$VARIANT" = warm ]; then
  extra=(--init-encoder pixel_corridor_runs/classical_seed0/best.pt)
fi
python train_pixel_nav.py --task e1m1 --model classical --seed "$SEED" --decisions 10000000 \
  --eval-every 250 --eval-episodes 32 --final-episodes 100 --random-baseline \
  --max-seconds 19500 --curriculum "$CURRICULUM" "${extra[@]}" \
  --output "pixel_e1m1_runs/classical_${VARIANT}_${CURRICULUM}${TAG}_seed${SEED}"
