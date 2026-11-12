#!/usr/bin/env bash
cd /mnt/hdd2/zhengrong/Quantum_Doom
O=frozen_perception_runs/e1m1_gpu_full_20261103/offline_probe/wide
export OMP_NUM_THREADS=2
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u probe_frozen_heads.py --heads wide20x4 --seeds $s --eval-batch 256 \
    --output $O/wide20x4_seed$s.json > $O/wide20x4_seed$s.log 2>&1
done
