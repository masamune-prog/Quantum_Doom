"""Compute-node validation of RGB keyframes without engine-pose input."""
import numpy as np
from human_nav_env import HumanNavigationEnv

for enabled in (False, True):
    env = HumanNavigationEnv(memory=enabled, memory_kind="place")
    try:
        obs, _ = env.reset(seed=42)
        goal = env.goal_pos.copy()
        env.goal_pos = goal + np.array([999., -777.], dtype=np.float32)
        assert np.array_equal(obs, env._observe(env.last_rgb))
        env.goal_pos = goal
        rng = np.random.default_rng(42)
        for _ in range(150):
            action = int(rng.integers(3))
            obs, reward, term, trunc, _ = env.step(action)
            assert obs.shape == (12,) and np.all(np.isfinite(obs)) and np.isfinite(reward)
            if term or trunc:
                break
        print('memory',enabled,'keyframes',len(env.travel_map.keyframes),
              'graph_edges',len(env.travel_map.edges),
              'observed_memory',obs[-4:].round(3).tolist(),flush=True)
        if enabled:
            assert len(env.travel_map.keyframes) > 1
        else:
            assert np.all(obs[-4:] == 0)
    finally:
        env.close()
