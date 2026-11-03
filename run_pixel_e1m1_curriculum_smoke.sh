#!/bin/bash
#SBATCH --job-name=qd_e1m1_cur_smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_pixel_e1m1_cur_smoke_%j.log
#SBATCH --error=logs/slurm_pixel_e1m1_cur_smoke_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# CPU only. Warm start from the corridor encoder, curriculum with a low bar so a stage
# advance is likely, and a resume from the saved checkpoint.
python diagnose_pixel_e1m1.py 2>&1 | grep -E "warp|Error|Trace"
python - <<'PY'
import numpy as np
from train_pixel_nav import Curriculum
rng = np.random.default_rng(0)
c = Curriculum()
draws = [c.sample(rng) for _ in range(5000)]
print("stage 0 sample:", {k: draws.count(k) for k in sorted(set(draws))})
for _ in range(20): c.record(21, True)
print("advanced:", c.maybe_advance(), "stage", c.stage)
draws = [c.sample(rng) for _ in range(5000)]
print("stage 1 sample:", {k: draws.count(k) for k in sorted(set(draws))})
PY
python train_pixel_nav.py --task e1m1 --model classical --seed 0 --decisions 12288 \
  --num-envs 4 --steps 128 --eval-every 6 --eval-episodes 4 --final-episodes 4 \
  --log-every 2 --init-encoder pixel_corridor_runs/classical_seed0/best.pt \
  --curriculum uniform --output pixel_e1m1_runs/smoke_cur
python train_pixel_nav.py --task e1m1 --model classical --seed 0 --decisions 16384 \
  --num-envs 4 --steps 128 --eval-every 8 --eval-episodes 4 --final-episodes 4 \
  --log-every 2 --resume --curriculum uniform --output pixel_e1m1_runs/smoke_cur
