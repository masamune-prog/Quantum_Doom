"""How well does the scripted teacher play E1M1? (It reads private state; see e1m1_teacher.py.)"""

import sys
from collections import Counter

import numpy as np

from pixel_e1m1_env import PixelE1M1Env

skill, episodes = int(sys.argv[1]), int(sys.argv[2])
start_wp = int(sys.argv[3]) if len(sys.argv) > 3 else None
env = PixelE1M1Env(skill=skill, teacher=True)
rows, actions = [], Counter()
for seed in range(20000, 20000 + episodes):
    _, info = env.reset(seed=seed, options={"start_wp": start_wp} if start_wp is not None else None)
    for step in range(1, 1576):
        action = info["teacher"]
        actions[env.ACTION_NAMES[action]] += 1
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    ep = info["episode"]
    rows.append((bool(info["success"]), ep["dead"], step, ep["kills"], ep["health"], ep["progress"]))
exits = sum(r[0] for r in rows)
print(f"skill {skill} start {start_wp}: exits {exits}/{episodes}, died {sum(r[1] for r in rows)}, "
      f"timeouts {sum((not r[0]) and (not r[1]) for r in rows)}, mean steps {np.mean([r[2] for r in rows]):.0f}, "
      f"kills {np.mean([r[3] for r in rows]):.2f}, health {np.mean([r[4] for r in rows]):.0f}, "
      f"progress {np.mean([r[5] for r in rows]):.2f}")
print("actions", dict(actions))
