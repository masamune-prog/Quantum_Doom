"""Compute-node check of the screen-only Deadly Corridor env and its targets."""
import numpy as np
from pixel_corridor_env import PixelCorridorEnv
from pixel_nav_env import N_SECTORS

env = PixelCorridorEnv()
rng = np.random.default_rng(0)
try:
    obs, info = env.reset(seed=1)
    print("objects", sorted({o.name for o in env.game.get_state().objects}), flush=True)
    print("goal", env.goal_pos, "start", env.start_pos, "obs", obs.shape, obs.dtype, flush=True)
    for policy in ("forward", "random"):
        targets, outcomes = [], []
        for seed in range(5):
            obs, info = env.reset(seed=100 + seed)
            total = 0.0
            for step in range(1, 526):
                action = 0 if policy == "forward" else int(rng.integers(6))
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                if term or trunc:
                    break
                targets.append(info["aux"])
            outcomes.append((info["success"], info["episode"], step, round(total, 1)))
        for o in outcomes:
            print(policy, o, flush=True)
        t = np.array(targets)
        print(policy, "depth mean/std/min/max", t[:, :N_SECTORS].mean().round(3),
              t[:, :N_SECTORS].std().round(3), t[:, :N_SECTORS].min().round(3),
              t[:, :N_SECTORS].max().round(3), "enemy sector rate", t[:, N_SECTORS:].mean().round(3),
              "steps with any enemy", (t[:, N_SECTORS:].max(1) > 0).mean().round(3), flush=True)
finally:
    env.close()
