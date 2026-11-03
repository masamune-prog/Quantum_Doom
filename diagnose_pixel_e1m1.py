"""Smoke checks for the screen-only E1M1 env: distance field, warp starts, doors, exit.

The scripted follower below reads private pose and waypoints. It is a test fixture
for the environment's rewards and USE handling, not a policy.
"""

import time

import numpy as np

from e1m1_distance import DistanceField
from e1m1_env import _wrap_angle
from pixel_e1m1_env import AUX_SIZE_E1M1, PixelE1M1Env

DOOR_WAYPOINTS = (5, 17, 20)


def check_field(env):
    field = env.field
    costs = [field(wp) for wp in env.waypoints]
    print(f"[field] start distance {field.start_distance:.1f}; goal cost {field(env.goal_pos):.1f}")
    print("[field] cost at waypoints 0..21:", " ".join(f"{c:.0f}" for c in costs))
    rises = [i for i in range(len(costs) - 1) if costs[i + 1] > costs[i] + 1e-6]
    print(f"[field] waypoints where cost rises along the route: {rises or 'none'}")
    assert None not in costs


def check_warps(env):
    env.reset(seed=0)
    pos, angle = env._pose()
    print(f"[warp] real start pos {pos.round(0)} angle {np.rad2deg(angle):.0f} "
          f"cost {env.route_cost:.1f}")
    for k in (21, 16, 12, 6, 0):
        obs, info = env.reset(seed=k, options={"start_wp": k})
        pos, angle = env._pose()
        wp = env.waypoints[k]
        ahead = env.waypoints[k + 1] if k + 1 < len(env.waypoints) else env.exit_mid
        error = np.rad2deg(_wrap_angle(np.arctan2(*(ahead - pos)[::-1]) - angle))
        print(f"[warp] wp {k:2d}: off by {np.linalg.norm(pos - wp):6.1f} units, heading error "
              f"{error:+5.0f} deg, cost {env.route_cost:7.1f}, health {env.health:.0f}, "
              f"obs {obs.shape} aux {info['aux'].shape}")
        assert obs.shape == (12, 60, 80) and info["aux"].shape == (AUX_SIZE_E1M1,)
        sin, cos, progress = info["aux"][-3:]
        print(f"[warp]        route target: direction {np.rad2deg(np.arctan2(sin, cos)):+5.0f} deg "
              f"(waypoint heading error above), progress {progress:.2f}")


def follow(env, seed, start_wp=None, max_steps=1575):
    """Scripted route follower: face the next waypoint, walk, shoot, USE doors once and the switch."""
    options = {"start_wp": start_wp} if start_wp is not None else None
    env.reset(seed=seed, options=options)
    exit_mid = env.exit_mid
    index = 0 if start_wp is None else start_wp
    total, info, last_use = 0.0, {}, -100
    for step in range(1, max_steps + 1):
        pos, angle = env._pose()
        while index < len(env.waypoints) and np.linalg.norm(env.waypoints[index] - pos) < 64:
            index += 1
        target = env.waypoints[index] if index < len(env.waypoints) else exit_mid
        error = _wrap_angle(np.arctan2(target[1] - pos[1], target[0] - pos[0]) - angle)
        aux = env._aux(env.game.get_state())
        if aux[8 + 3] or aux[8 + 4]:
            action = 1                      # monster near the centre of the screen: ATTACK
        elif env._interact_ahead(pos, angle) and step - last_use > 25:
            # USE on a door that is already opening closes it again, so press once, then walk.
            action, last_use = 6, step
        elif abs(error) > np.deg2rad(12):
            action = 4 if error > 0 else 5
        else:
            action = 0
        _, reward, term, trunc, info = env.step(action)
        total += reward
        if term or trunc:
            break
    episode = info.get("episode", {})
    outcome = "EXIT" if info["success"] else ("DIED" if episode.get("dead") else "TIMEOUT")
    start = "real start" if start_wp is None else f"wp {start_wp}"
    print(f"[follow] skill-1 follower from {start}, seed {seed}: {outcome} in {step} steps, "
          f"return {total:.1f}, reached wp {index}/{len(env.waypoints)}, "
          f"kills {episode.get('kills')}, health {episode.get('health')}, "
          f"progress {episode.get('progress', 0):.2f}")
    return outcome


def baseline(env, mode, seeds):
    rng = np.random.default_rng(0)
    start, decisions, rows = time.monotonic(), 0, []
    for seed in seeds:
        env.reset(seed=seed)
        total = 0.0
        for step in range(1, 2000):
            _, reward, term, trunc, info = env.step(0 if mode == "forward"
                                                    else int(rng.integers(7)))
            total += reward
            if term or trunc:
                break
        decisions += step
        episode = info["episode"]
        rows.append((step, total, episode["progress"], episode["dead"]))
    rate = decisions / (time.monotonic() - start)
    print(f"[{mode}] steps/return/progress/dead per episode: "
          + "; ".join(f"{s}/{r:.1f}/{p:.2f}/{int(d)}" for s, r, p, d in rows)
          + f"  ({rate:.0f} decisions/s)")


def main():
    env = PixelE1M1Env(skill=1)
    check_field(env)
    check_warps(env)
    for seed in (0, 1, 2):
        follow(env, seed)
    for start_wp in (4, 15, 19):           # just before each door
        follow(env, 0, start_wp)
    env.close()
    env = PixelE1M1Env(skill=3)
    baseline(env, "random", range(20000, 20003))
    baseline(env, "forward", range(20000, 20003))
    env.close()


if __name__ == "__main__":
    main()
