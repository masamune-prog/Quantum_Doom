"""Do the two small-task specialists play E1M1 with no retraining?

  navigation specialist (My Way Home, 3 actions)  -> E1M1 actions 0, 4, 5
  combat specialist (Deadly Corridor, 6 actions)  -> E1M1 actions 0-5
Modes: nav only, combat only, and an ORACLE gate that uses the combat specialist when a monster
is in the central four screen sectors and the navigation specialist otherwise. The gate reads
the environment's private monster-visibility target, so this is an upper bound on composing the
two skills without learning the switch. E1M1, skill 3, real start, seeds 20000-20099.

usage: python evaluate_specialists_on_e1m1.py OUT.json head:nav_ckpt:combat_ckpt [...]
"""

import json
import sys

import numpy as np
import torch

from pixel_nav_env import EnvPool
from pixel_nav_models import PixelNavPolicy
from train_pixel_nav import FINAL_SEED

NAV_TO_E1M1 = [0, 4, 5]


def run(pool, nav, combat, mode, sampled, seeds, device):
    pending, obs, aux, active, rows = list(seeds), {}, {}, {}, []

    def begin(i):
        seed = pending.pop(0)
        obs[i], info = pool.reset([i], [seed])[0]
        aux[i] = info["aux"]
        active[i] = {"seed": seed, "steps": 0, "gated": 0}

    def pick(logits):
        if sampled:
            return torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(-1).tolist()
        return logits.argmax(-1).tolist()

    for i in range(min(pool.n, len(pending))):
        begin(i)
    while active:
        ids = sorted(active)
        x = torch.as_tensor(np.stack([obs[i] for i in ids]), device=device)
        with torch.no_grad():
            nav_action = [NAV_TO_E1M1[a] for a in pick(nav(x)[0])]
            combat_action = pick(combat(x)[0])
        actions = []
        for k, i in enumerate(ids):
            fighting = bool(aux[i][8 + 2:8 + 6].max() > 0.5)
            active[i]["gated"] += int(fighting)
            actions.append(nav_action[k] if mode == "nav" else combat_action[k] if mode == "combat"
                           else combat_action[k] if fighting else nav_action[k])
        for i, (o, _, term, trunc, info) in zip(ids, pool.step(ids, actions)):
            active[i]["steps"] += 1
            obs[i], aux[i] = o, info["aux"]
            if term or trunc:
                rows.append({"success": bool(info["success"]), "steps": active[i]["steps"],
                             "fighting": active[i]["gated"] / active[i]["steps"],
                             **info["episode"]})
                del active[i]
                if pending:
                    begin(i)
    return {"exits": sum(r["success"] for r in rows), "episodes": len(rows),
            **{f"mean_{k}": round(float(np.mean([r[k] for r in rows])), 3)
               for k in ("kills", "dead", "progress", "steps", "fighting")}}


def load(head, path, n_actions, device):
    model = PixelNavPolicy(head=head, n_actions=n_actions).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval()


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    pool = EnvPool(8, task="e1m1")
    seeds = range(FINAL_SEED, FINAL_SEED + 100)
    results = {}
    try:
        for spec in specs:
            head, nav_path, combat_path = spec.split(":")
            nav, combat = load(head, nav_path, 3, device), load(head, combat_path, 6, device)
            for mode in ("nav", "combat", "gated"):
                for sampled in (False, True):
                    key = f"{head} {mode} {'sampled' if sampled else 'greedy'}"
                    results[key] = run(pool, nav, combat, mode, sampled, seeds, device)
                    print(key, results[key], flush=True)
    finally:
        pool.close()
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
