#!/bin/bash
#SBATCH --job-name=qd_e1m1_quantum
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_e1m1_train_%j.log
#SBATCH --error=logs/slurm_e1m1_train_%j.err

set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib_qdoom"
mkdir -p "$MPLCONFIGDIR"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
python -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable"; print(torch.cuda.get_device_name(0))'
python -c 'import torch; torch.set_default_device("cuda"); from e1m1_quantum_model import QuantumE1M1Policy; m=QuantumE1M1Policy(); logits,value=m(torch.zeros(2,12)); (logits.square().mean()+value.square().mean()).backward(); assert m.theta.grad is not None; print("Quantum circuit GPU forward/backward OK")'

# Establish that the map route and observation can support an exit before training.
python validate_e1m1_route.py --skills 3 --episodes 5
python -c 'import json; r=json.load(open("e1m1_route_check.json"))["3"]; assert r["wins"] > 0, "Scripted route failed: inspect route check before training"'

python train_e1m1.py --skill 3 --num-envs 8 --steps 32 --updates 2800 \
  --adaptive --starts 21 20 19 16 12 6 0 --eval-episodes 20 \
  --max-seconds 16000 --output e1m1_adaptive_run --resume
