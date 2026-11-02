#!/bin/bash
#SBATCH --job-name=qd_e1m1_sanity
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_e1m1_sanity_%j.log
#SBATCH --error=logs/slurm_e1m1_sanity_%j.err

echo "=== Quantum Doom - E1M1 Environment Sanity Check ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

cd /tc1home/UG/zgoh033/quantum_doom
set -eo pipefail
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python - <<'EOF'
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from e1m1_env import QuantumE1M1Env

env = QuantumE1M1Env(skill=3, render_rgb=True, cheats=True)
obs, _ = env.reset(seed=0)
pos, angle = env._player()
print(f"[1] Loaded E1M1. Start pos {pos}, angle {np.rad2deg(angle):.0f} deg (WAD says [1056 -3616], 90)")
print(f"    Obs (12q): {obs.round(2)}")

# Random rollout: check stepping, label names, termination flags
rng = np.random.default_rng(0)
seen_labels, cue_counts = set(), {"door": 0, "exit": 0}
total, steps = 0.0, 0
for steps in range(1, 401):
    obs, r, term, trunc, info = env.step(int(rng.integers(7)))
    total += r
    state = env.game.get_state()
    if state is not None:
        seen_labels.update(l.object_name for l in state.labels)
    cue_counts["door"] += int(np.isclose(obs[11], np.pi / 2))
    cue_counts["exit"] += int(np.isclose(obs[11], np.pi))
    if term or trunc:
        break
print(f"[2] Random rollout: {steps} steps, return {total:.2f}, last info {info}, term={term}, trunc={trunc}")
print(f"    Label names seen: {sorted(seen_labels)}")
print(f"    Interaction cue counts: {cue_counts}")

# Exit test: warp next to the exit switch (west wall, x=2912), face west, press USE
obs, _ = env.reset(seed=1)
obs = env.warp(2960, -4768)
pos, angle = env._player()
print(f"[3] Warped to {pos}, angle {np.rad2deg(angle):.0f} deg")
for _ in range(200):
    if abs(((np.rad2deg(env._player()[1]) - 180 + 180) % 360) - 180) < 4:
        break
    env.game.make_action(env.ACTIONS[4], 1)
pos, angle = env._player()
obs = env._extract_features(env.game.get_state())
print(f"    Facing {np.rad2deg(angle):.0f} deg, interaction cue = {obs[11]:.2f} (expect {np.pi:.2f})")

frame = env.game.get_state().screen_buffer.copy()
view_obs = obs.copy()
result = None
for _ in range(10):
    obs, r, term, trunc, info = env.step(6)  # USE
    if term or trunc:
        result = (r, info)
        break
print(f"    After USE: {'EXIT REACHED' if result and result[1]['exit_reached'] else 'no exit'} -> {result}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(frame)
axes[0].set_title("Agent view at exit switch")
colors = ["#ef4444" if v > 0 else "#3b82f6" for v in view_obs[:8]] + ["#7c3aed"] * 4
axes[1].bar(range(12), view_obs, color=colors)
axes[1].set_ylim(-np.pi, np.pi)
axes[1].set_title("12-qubit input angles")
plt.tight_layout()
plt.savefig("e1m1_sanity.png", dpi=120)
env.close()
print("Saved e1m1_sanity.png. Sanity check finished.")
EOF

echo "=== Done! End: $(date) ==="
