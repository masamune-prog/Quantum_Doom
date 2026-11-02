"""Compute-node smoke check for private progress rewards and policy observations."""
import numpy as np

from human_nav_env import HumanNavigationEnv
from human_e1m1_env import HumanE1M1Env
from reward_shaping import progress_reward, kill_reward

for name, cls in (("navigation", HumanNavigationEnv), ("e1m1", HumanE1M1Env)):
    env = cls(memory=True)
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (12,) and np.all(np.isfinite(obs))
        original_goal = env.goal_pos.copy()
        env.goal_pos = original_goal + np.array([777.0, -333.0], dtype=np.float32)
        assert np.array_equal(obs, env._observe(env.last_rgb)), "goal leaked into observation"
        env.goal_pos = original_goal
        print(name, "reset observation independent of private goal", flush=True)
        for action in (0, 0, 0, 0, 4 if name == "e1m1" else 1):
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == (12,) and np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            print(name, "action", action, "reward", round(reward, 4),
                  "memory", np.round(obs[-4:], 2).tolist(), flush=True)
            if term or trunc:
                break
    finally:
        env.close()
assert progress_reward((0, 0), (10, 0), (100, 0), distance_scale=200) > 0
assert progress_reward((0, 0), (-10, 0), (100, 0), distance_scale=200) < 0
assert kill_reward(2, 3) == 2.0 and kill_reward(2, 2) == 0.0
print("reward unit checks passed", flush=True)
