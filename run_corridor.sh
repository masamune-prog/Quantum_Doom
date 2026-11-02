#!/bin/bash
#SBATCH --job-name=quantum_doom_corridor
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_corridor_%j.log
#SBATCH --error=logs/slurm_corridor_%j.err

echo "=== Quantum Doom - Deadly Corridor Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-16}"
echo "Start: $(date)"

cd /tc1home/UG/zgoh033/quantum_doom

# Use the uv venv with cu126 PyTorch (supports V100 CC 7.0)
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 1: Quick sanity check of the environment
echo ""
echo "=== Step 1: Sanity Check ==="
python -c "
from doom_env import QuantumDoomEnv
import os
cfg = os.path.join(os.path.dirname(os.path.abspath('doom_env.py')), 'deadly_corridor_quantum.cfg')
env = QuantumDoomEnv(n_qubits=8, frame_skip=4, use_composite_actions=True, custom_cfg_path=cfg)
obs, _ = env.reset(seed=42)
print(f'Corridor env OK! Actions: {env.n_actions}, Obs: {obs.round(2)}')
for i in range(5):
    obs, r, t, _, _ = env.step(env.action_space.sample())
    print(f'  Step {i}: reward={r:.2f}, terminated={t}')
env.close()
print('Sanity check passed!')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Sanity check failed!"
    exit 1
fi

# Step 2: Train the agent
# 524288 frames with 16 parallel envs on 16 CPU cores
NUM_CPUS=${SLURM_CPUS_PER_TASK:-16}
echo ""
echo "=== Step 2: Training (524288 frames, $NUM_CPUS parallel envs) ==="
python train_corridor.py \
    --seed 42 \
    --total-steps 524288 \
    --num-envs "$NUM_CPUS" \
    --n-steps 128

# Step 3: Record gameplay video with the best checkpoint
echo ""
echo "=== Step 3: Recording Gameplay Video ==="
CKPT="checkpoint_corridor_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="checkpoint_corridor_seed42.pt"
fi

python record_agent.py \
    --checkpoint "$CKPT" \
    --output quantum_corridor_gameplay.mp4 \
    --corridor \
    --custom-cfg deadly_corridor_quantum.cfg \
    --episodes 5 \
    --max-steps 2000 \
    --fps 15 \
    --greedy

echo ""
echo "=== Done! ==="
echo "End: $(date)"
ls -lh checkpoint_corridor*.pt quantum_corridor_gameplay.mp4 corridor_training_curve.png 2>/dev/null
