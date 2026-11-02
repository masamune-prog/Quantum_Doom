"""Compute-node check of auxiliary target ranges under random actions."""
import numpy as np
from pixel_nav_env import PixelNavigationEnv, N_SECTORS

env = PixelNavigationEnv()
rng = np.random.default_rng(0)
targets = []
try:
    for seed in range(5):
        obs, info = env.reset(seed=500 + seed)
        targets.append(info["aux"])
        for _ in range(300):
            obs, _, term, trunc, info = env.step(int(rng.integers(3)))
            if term or trunc:
                break
            targets.append(info["aux"])
    state = env.game.get_state()
    print("raw depth min/median/max", state.depth_buffer.min(), np.median(state.depth_buffer),
          state.depth_buffer.max())
finally:
    env.close()
t = np.array(targets)
print("steps", len(t), "obs", obs.shape, obs.dtype)
print("depth target mean", t[:, :N_SECTORS].mean().round(4), "std", t[:, :N_SECTORS].std().round(4),
      "min", t[:, :N_SECTORS].min().round(4), "max", t[:, :N_SECTORS].max().round(4))
print("goal visible: fraction of steps any sector", (t[:, N_SECTORS:].max(1) > 0).mean().round(4),
      "per-sector rate", t[:, N_SECTORS:].mean(0).round(4).tolist())
