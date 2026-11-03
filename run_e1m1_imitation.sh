#!/bin/bash
#SBATCH --job-name=qd_e1m1_imit
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_e1m1_imitation_%x_%j.log
#SBATCH --error=logs/slurm_e1m1_imitation_%x_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# DAgger imitation of the scripted E1M1 teacher (private state) from screens only.
# MODEL=classical|linear|quantum. Warm start from the corridor classical encoder.
MODEL=${MODEL:-classical}
SEED=${SEED:-0}
TAG=${TAG:-}
WEIGHT_POWER=${WEIGHT_POWER:-0.75}
EXTRA=${EXTRA:-}   # e.g. EXTRA="--head-lr 0.003 --initial-epochs 3 --epochs 2" for the circuit
python train_e1m1_imitation.py --model "$MODEL" --seed "$SEED" \
  --init-encoder pixel_corridor_runs/classical_seed0/best.pt \
  --weight-power "$WEIGHT_POWER" $EXTRA \
  --output "pixel_e1m1_runs/imitation_${MODEL}${TAG}_seed${SEED}"
