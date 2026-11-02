"""
Validate the planned E1M1 route with a hand-written policy that only sees the
same 12-qubit observation the quantum agent gets.

If this scripted follower can finish the level, the observation and the waypoints
carry enough information to complete E1M1; any remaining gap is down to learning.
"""

import argparse
import json
import numpy as np

from e1m1_env import QuantumE1M1Env
from e1m1_map import DoomMap

FORWARD, ATTACK, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, USE = range(7)


class ScriptedFollower:
    def __init__(self, turn_tolerance: float = 0.2, seed: int = 0):
        self.turn_tolerance = turn_tolerance
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.unstick = 0

    def act(self, obs, stuck: bool) -> int:
        self.t += 1
        radar, bearing, cue = obs[:8], obs[8], obs[11]

        if self.unstick > 0:
            self.unstick -= 1
            return self.unstick_action
        if stuck:
            self.unstick = 6
            self.unstick_action = int(self.rng.choice([MOVE_LEFT, MOVE_RIGHT]))
            return self.unstick_action

        monsters = np.where(radar > 0)[0]
        if len(monsters):
            if np.any((monsters == 3) | (monsters == 4)):
                return ATTACK
            return TURN_LEFT if monsters.mean() < 3.5 else TURN_RIGHT

        if np.isclose(cue, np.pi) or (np.isclose(cue, np.pi / 2) and self.t % 2 == 0):
            return USE
        if abs(bearing) > self.turn_tolerance:
            return TURN_LEFT if bearing > 0 else TURN_RIGHT
        return FORWARD


def run_episode(env, seed, max_steps=1600):
    obs, _ = env.reset(seed=seed)
    policy = ScriptedFollower(seed=seed)
    traj, positions = [], []
    info = {"exit_reached": False, "dead": False, "waypoint_idx": 0, "health": 100.0}
    total, steps = 0.0, 0
    for steps in range(1, max_steps + 1):
        pos, _ = env._player()
        positions.append(pos)
        traj.append(pos.tolist())
        stuck = len(positions) > 10 and np.linalg.norm(positions[-1] - positions[-10]) < 8.0
        obs, r, term, trunc, info = env.step(policy.act(obs, stuck))
        total += r
        if term or trunc:
            break
    return {"seed": seed, "steps": steps, "return": round(total, 1), **info, "traj": traj,
            "truncated": bool(trunc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    summary = {}
    for skill in args.skills:
        env = QuantumE1M1Env(skill=skill)
        n_wp = len(env.waypoints)
        results = [run_episode(env, seed) for seed in range(args.episodes)]
        env.close()

        for r in results:
            outcome = "EXIT" if r["exit_reached"] else ("DIED" if r["dead"] else "TIMEOUT/STUCK")
            print(f"skill {skill} seed {r['seed']}: {outcome:13s} steps {r['steps']:4d} "
                  f"return {r['return']:7.1f} waypoints {r['waypoint_idx']}/{n_wp} "
                  f"health {r['health']:.0f} kills {r['kills']:.0f}")
        wins = sum(r["exit_reached"] for r in results)
        print(f"skill {skill}: completed without dying {wins}/{len(results)}\n")
        summary[skill] = {"wins": wins, "episodes": len(results),
                          "results": [{k: v for k, v in r.items() if k != "traj"} for r in results]}

        DoomMap().plot(f"e1m1_route_check_skill{skill}.png", skill=skill,
                       waypoints=env.waypoints.tolist(),
                       trajectories={f"seed {r['seed']}": r["traj"] for r in results})

    with open("e1m1_route_check.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Saved e1m1_route_check.json and e1m1_route_check_skill*.png")


if __name__ == "__main__":
    main()
