#!/usr/bin/env bash
# Qubit-scaling sweep on cached frozen features (probe_frozen_heads.py, wide_circuit.py).
cd /mnt/hdd2/zhengrong/Quantum_Doom
O=frozen_perception_runs/e1m1_gpu_full_20261103/offline_probe/wide
P=".venv/bin/python -u probe_frozen_heads.py"
export OMP_NUM_THREADS=2
job() { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $P --heads $name --output $O/$name"$TAG".json "$@" > $O/$name"$TAG".log 2>&1; }
( job 0 wide12x4; job 0 wide14x4; job 0 wide16x4; job 0 wide18x4 ) &
( job 1 wide16x23 ) &
( job 2 wide12x31; TAG=_seed1 job 2 wide20x4 --seeds 1 ) &
( TAG=_seed0 job 3 wide20x4 --seeds 0; TAG=_seed2 job 3 wide20x4 --seeds 2 ) &
wait
