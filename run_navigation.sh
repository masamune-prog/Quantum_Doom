#!/bin/bash
#SBATCH --job-name=quantum_doom_nav
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_nav_%j.log
#SBATCH --error=logs/slurm_nav_%j.err

echo "=== Quantum Doom - Multi-Room Navigation (10 Qubits, 7 Actions) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-16}"
echo "Start: $(date)"

cd /tc1home/UG/zgoh033/quantum_doom
set -eo pipefail

# Use the uv venv with cu126 PyTorch
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 1: Sanity Check
echo ""
echo "=== Step 1: Sanity Check (10 Qubits, 7 Actions) ==="
python -c "
from doom_nav_env import QuantumNavigationEnv
env = QuantumNavigationEnv(n_qubits=10)
obs, _ = env.reset(seed=42)
print(f'Nav env OK! Actions: {env.n_actions}, Obs ({len(obs)}q): {obs.round(2)}')
for i in range(5):
    obs, r, t, _, _ = env.step(env.action_space.sample())
    print(f'  Step {i}: reward={r:.2f}, term={t}, stuck_warn={obs[8]:.2f}, nearby={obs[9]:.2f}')
env.close()
print('Sanity check passed!')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Sanity check failed!"
    exit 1
fi

# Step 2: Training (262,144 frames across 16 parallel envs)
NUM_CPUS=${SLURM_CPUS_PER_TASK:-16}
echo ""
echo "=== Step 2: Training (262,144 frames, $NUM_CPUS parallel envs) ==="
python train_navigation.py \
    --seed 42 \
    --total-steps 262144 \
    --num-envs "$NUM_CPUS" \
    --n-steps 128 \
    --n-qubits 10 \
    --n-actions 7

# Step 3: Record gameplay video with the best checkpoint
echo ""
echo "=== Step 3: Recording Navigation Gameplay Video ==="
CKPT="checkpoint_nav_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="checkpoint_nav_seed42.pt"
fi

python record_nav_agent.py \
    --checkpoint "$CKPT" \
    --output quantum_navigation_gameplay.mp4 \
    --episodes 3 \
    --fps 15 \
    --greedy \
    --n-qubits 10 \
    --n-actions 7

echo ""
echo "=== Done! ==="
echo "End: $(date)"
ls -lh checkpoint_nav*.pt quantum_navigation_gameplay.mp4 nav_training_curve.png 2>/dev/null
