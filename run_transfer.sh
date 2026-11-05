#!/usr/bin/env bash
# GPU allocation is external. Set CUDA_VISIBLE_DEVICES and PYTHON for this machine.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python3}" train_transfer.py "$@"
