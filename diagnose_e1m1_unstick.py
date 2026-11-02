"""Test whether E1M1 route following needs the position-history recovery rule."""

from e1m1_env import QuantumE1M1Env
from validate_e1m1_route import ScriptedFollower


def main():
    env = QuantumE1M1Env(skill=3)
    try:
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            policy = ScriptedFollower(seed=seed)
            for step in range(1, 1601):
                obs, _, term, trunc, info = env.step(policy.act(obs, stuck=False))
                if term or trunc:
                    break
            print(f"seed={seed} steps={step} waypoint={info['waypoint_idx']} "
                  f"exit={info['exit_reached']} dead={info['dead']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
