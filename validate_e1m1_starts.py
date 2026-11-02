"""Check that each E1M1 curriculum warp is usable in the live game."""

import argparse
import numpy as np

from e1m1_env import QuantumE1M1Env
from validate_e1m1_route import ScriptedFollower


def run(env, start_wp, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    if start_wp:
        target = env.waypoints[start_wp]
        obs = env.warp(*target)
        pos, _ = env._player()
        error = float(np.linalg.norm(pos - target))
        if error > 96 or env.wp_idx != start_wp:
            raise RuntimeError(
                f"start {start_wp}: warp landed at {pos}, target {target}, "
                f"error {error:.1f}, waypoint index {env.wp_idx}"
            )
    else:
        error = 0.0

    policy = ScriptedFollower(seed=seed)
    positions = []
    info = None
    for step in range(1, max_steps + 1):
        pos, _ = env._player()
        positions.append(pos)
        stuck = len(positions) > 10 and np.linalg.norm(positions[-1] - positions[-10]) < 8.0
        obs, _, term, trunc, info = env.step(policy.act(obs, stuck))
        if term or trunc:
            break
    return step, error, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", nargs="+", type=int, default=[19, 16, 12, 6, 0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1600)
    args = parser.parse_args()

    env = QuantumE1M1Env(skill=3, cheats=True)
    try:
        for start_wp in args.starts:
            for seed in range(args.seeds):
                steps, error, info = run(env, start_wp, seed, args.max_steps)
                print(f"start={start_wp:2d} seed={seed} warp_error={error:.1f} "
                      f"steps={steps} waypoint={info['waypoint_idx']}/{len(env.waypoints)} "
                      f"exit={info['exit_reached']} dead={info['dead']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
