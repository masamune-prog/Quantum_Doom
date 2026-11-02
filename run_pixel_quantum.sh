#!/bin/bash
#SBATCH --job-name=qd_pixel_quantum
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_pixel_quantum_%j.log
#SBATCH --error=logs/slurm_pixel_quantum_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
# Same encoder, bottleneck, auxiliary losses and PPO settings as run_pixel_classical.sh;
# only the head differs. One seed per job: ~165 decisions/s fills the 6 h limit.
SEED=${SEED:-0}
python train_pixel_nav.py --model quantum --seed "$SEED" --decisions 3000000 \
  --head-lr 0.003 --eval-every 250 --eval-episodes 32 --final-episodes 100 \
  --max-seconds 19500 --output "pixel_nav_runs/quantum_seed$SEED"
