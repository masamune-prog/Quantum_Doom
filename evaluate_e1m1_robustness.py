"""Does an E1M1 student follow what it sees, or recall a memorised route?

Perturbs the student's input or its start state and measures exits, with no retraining:
  none            unmodified screens, real start (baseline)
  hud_masked      bottom rows (the HUD: health, ammo) blanked, so the HUD cannot act as a progress clock
  static_frame    all 4 stacked frames replaced by the newest one (no motion cues)
  bright_0.6/1.4  brightness scaled
  noise           gaussian noise, std 25 grey levels
  offroute        same screens, but the episode starts at a random reachable position with a random
                  heading (not one of the 22 waypoints used in training)
The scripted teacher is run on the same off-route starts as a reference (it follows the distance
field, so it should not care where it starts). Episodes are cut at 800 decisions.

usage: python evaluate_e1m1_robustness.py OUT.json N_EPISODES name=head:path [...]
"""

import json
import sys

import numpy as np
import torch

from e1m1_distance import DistanceField
from e1m1_map import DoomMap
from pixel_nav_env import EnvPool
from pixel_nav_models import PixelNavPolicy

CAP, FIRST_SEED = 800, 40000
CONDITIONS = ("none", "hud_masked", "static_frame", "bright_0.6", "bright_1.4", "noise", "offroute")


def perturb(x, name):
    if name in ("none", "offroute"):
        return x
    f = x.float()
    if name == "hud_masked":
        f[:, :, 50:, :] = 0.0
    elif name == "static_frame":
        f = f[:, 9:12].repeat(1, 4, 1, 1)
    elif name.startswith("bright"):
        f = f * float(name.split("_")[1])
    elif name == "noise":
        f = f + 25.0 * torch.randn_like(f)
    return f.clamp(0, 255).to(torch.uint8)


class OffRouteStarts:
    """Random reachable positions at least 56 units from any door or the exit switch."""

    def __init__(self):
        field = DistanceField()
        doom_map = DoomMap()
        segments = [(l["p1"], l["p2"]) for l in doom_map.door_lines() + doom_map.exit_lines()]
        cells = np.argwhere(np.isfinite(field.dist) & (field.dist > 10) & (field.dist < 310))
        centres = field.origin + (cells + 0.5) * 16.0
        keep = []
        for point in centres:
            near = min(np.linalg.norm(point - np.clip(
                a + np.clip(np.dot(point - a, b - a) / max(np.dot(b - a, b - a), 1e-6), 0, 1)
                * (b - a), None, None)) for a, b in segments)
            keep.append(near > 56.0)
        self.centres = centres[np.array(keep)]

    def __call__(self, seed):
        rng = np.random.default_rng(seed)
        x, y = self.centres[int(rng.integers(len(self.centres)))]
        return {"start_xy": [float(x), float(y)], "start_heading": float(rng.uniform(-np.pi, np.pi))}


def run(pool, model, condition, mode, episodes, starts, device):
    seeds = list(range(FIRST_SEED, FIRST_SEED + episodes))
    pending, obs, info, active, rows = list(seeds), {}, {}, {}, []

    def begin(i):
        seed = pending.pop(0)
        options = starts(seed) if condition == "offroute" else None
        obs[i], info[i] = pool.reset([i], [seed], options)[0]
        active[i] = {"steps": 0}

    for i in range(min(pool.n, len(pending))):
        begin(i)
    while active:
        ids = sorted(active)
        if mode == "teacher":
            actions = [info[i]["teacher"] for i in ids]
        else:
            x = perturb(torch.as_tensor(np.stack([obs[i] for i in ids]), device=device), condition)
            with torch.no_grad():
                logits, _ = model(x)
            actions = (torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(-1).tolist()
                       if mode == "sampled" else logits.argmax(-1).tolist())
        for i, (o, _, term, trunc, inf) in zip(ids, pool.step(ids, actions)):
            active[i]["steps"] += 1
            obs[i], info[i] = o, inf
            if term or trunc or active[i]["steps"] >= CAP:
                done = term or trunc
                rows.append({"success": bool(done and inf["success"]),
                             "dead": bool(done and inf["episode"]["dead"]),
                             "progress": inf["episode"]["progress"] if done else float("nan")})
                del active[i]
                if pending:
                    begin(i)
    return {"exits": sum(r["success"] for r in rows), "episodes": len(rows),
            "died": sum(r["dead"] for r in rows)}


def main():
    out, episodes, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    pool = EnvPool(8, task="e1m1", teacher=True)
    starts = OffRouteStarts()
    print(f"{len(starts.centres)} candidate off-route start cells", flush=True)
    results = {}
    try:
        for condition in ("none", "offroute"):
            results[f"teacher {condition}"] = run(pool, None, condition, "teacher", episodes,
                                                  starts, device)
            print(f"teacher {condition}: {results[f'teacher {condition}']}", flush=True)
        for spec in specs:
            name, rest = spec.split("=")
            head, path = rest.split(":")
            model = PixelNavPolicy(head=head, n_actions=7, aux_pos_weight=5.0, aux_size=20,
                                   aux_route=3).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
            model.eval()
            for condition in CONDITIONS:
                for mode in ("greedy", "sampled"):
                    key = f"{name} {condition} {mode}"
                    results[key] = run(pool, model, condition, mode, episodes, starts, device)
                    print(key, results[key], flush=True)
    finally:
        pool.close()
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
