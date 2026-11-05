#!/usr/bin/env bash
# Local: PYTHON=/path/to/python bash run_frozen_perception.sh pilot
# Full GPU run: DEVICE=cuda bash run_frozen_perception.sh full --output /path/to/new/run
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-full}
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
python_bin=${PYTHON:-python3}
case "$mode" in
  pilot)
    exec "$python_bin" train_frozen_perception.py \
      --output frozen_perception_runs/e1m1_pilot --device "${DEVICE:-cpu}" \
      --threads 2 --num-envs 4 --seeds 0 \
      --perception-steps 4096 --validation-steps 512 --perception-epochs 10 \
      --initial-steps 2048 --iter-steps 256 --iterations 1 --head-epochs 6 \
      --capacity 4096 --max-episode 256 --batch 64 \
      --eval-episodes 2 --final-episodes 4 "$@"
    ;;
  full)
    exec "$python_bin" train_frozen_perception.py --device "${DEVICE:-cuda}" "$@"
    ;;
  *) echo "Usage: $0 [pilot|full] [train_frozen_perception.py options]" >&2; exit 2 ;;
esac
