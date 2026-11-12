"""Does a history of frame-only features replace the route targets? (offline probe)

The original perception was supervised on 20 targets, three of which (route direction
sin/cos and route progress, targets 17-19) come from the map and the agent's pose, not
from the screen. Those three nearly spell out the teacher's action, so a linear readout of
the 12 frozen angles is already a good policy.

This probe trains two encoders on the SAME collected data with the same seed:
  frame : depth + monster/door visibility only (targets 0-16, all computable from the frame)
  all   : the original 20 targets
then gives linear and MLP heads the last k steps of 12 features (k = 1, 4, 16, 32),
stacked within an episode. "+pose" sources append the agent's (x, y, sin, cos of angle) at
the current step, so the head itself must map position to route (option 3). "Oracle" inputs (the true targets 0-16 / 0-19 plus cooldown)
separate "the information is not there" from "the encoder did not learn it".

Head data is episode-disjoint from perception data; held-out head data is disjoint from both.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from frozen_perception_models import COOLDOWN, N_ACTIONS, GeometryPerception
from pixel_nav_env import EnvPool
from train_frozen_perception import Replay, collect, encode_replay

FRAME_KEYS = ("depth", "visibility_door")


def pretrain(perception, training, validation, keys, args, device):
    optimizer = torch.optim.Adam(perception.parameters(), lr=args.perception_lr)
    rng = np.random.default_rng(args.perception_seed)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.perception_epochs):
        perception.train()
        for idx in np.array_split(rng.permutation(training.size),
                                  max(1, training.size // args.batch)):
            images, _, geometry, cooldown = training.tensors(idx, device)
            losses = perception.geometry_loss(images, cooldown, geometry)
            loss = sum(losses[k] for k in keys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        perception.eval()
        totals = dict.fromkeys(("depth", "visibility_door", "direction", "progress"), 0.0)
        with torch.no_grad():
            for start in range(0, validation.size, args.batch):
                idx = np.arange(start, min(start + args.batch, validation.size))
                images, _, geometry, cooldown = validation.tensors(idx, device)
                for k, v in perception.geometry_loss(images, cooldown, geometry).items():
                    totals[k] += float(v) * len(idx) / validation.size
        score = sum(totals[k] for k in keys)
        if score < best:
            best, best_state = score, copy.deepcopy(perception.state_dict())
        history.append({"epoch": epoch + 1, **totals})
        print(json.dumps({"stage": "perception", "keys": keys, **history[-1]}), flush=True)
    perception.load_state_dict(best_state)
    return perception.freeze(), history


def stack(features, starts, k, stride):
    """[N, k*d]: steps t-(k-1)*stride .. t, clamped to the episode's first step."""
    t = torch.arange(len(features), device=features.device)
    rows = [torch.maximum(t - j * stride, starts) for j in range(k - 1, -1, -1)]
    return torch.cat([features[r] for r in rows], dim=1)


def make_head(kind, d):
    if kind == "linear":
        return nn.Linear(d, N_ACTIONS)
    hidden = 256 if kind == "mlp256" else 64
    return nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, N_ACTIONS))


@torch.no_grad()
def scores(head, x, y):
    logits = torch.cat([head(c) for c in x.split(8192)])
    correct = logits.argmax(-1) == y
    per = [float(correct[y == a].float().mean()) for a in range(N_ACTIONS) if (y == a).any()]
    return {"ce": float(F.cross_entropy(logits, y)), "accuracy": float(correct.float().mean()),
            "balanced_accuracy": float(np.mean(per))}


def fit(kind, x, y, vx, vy, seed, args):
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x, vx = (x - mean) / std, (vx - mean) / std
    head = make_head(kind, x.shape[1]).to(x.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)
    weights = torch.bincount(y, minlength=N_ACTIONS).float().clamp_min(1).rsqrt()
    weights /= weights.mean()
    gen = torch.Generator(device=x.device).manual_seed(seed)
    curve = []
    for _ in range(args.head_epochs):
        for idx in torch.randperm(len(y), device=x.device, generator=gen).split(args.batch):
            loss = F.cross_entropy(head(x[idx]), y[idx], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        curve.append(scores(head, vx, vy)["balanced_accuracy"])
    return {"train": scores(head, x, y), "heldout": scores(head, vx, vy), "curve": curve,
            "parameters": sum(p.numel() for p in head.parameters())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="frozen_perception_runs/feature_stack_probe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--skill", type=int, default=3)
    p.add_argument("--perception-seed", type=int, default=42)
    p.add_argument("--perception-steps", type=int, default=100_000)
    p.add_argument("--validation-steps", type=int, default=10_000)
    p.add_argument("--head-steps", type=int, default=100_000)
    p.add_argument("--heldout-steps", type=int, default=20_000)
    p.add_argument("--perception-epochs", type=int, default=20)
    p.add_argument("--perception-lr", type=float, default=3e-4)
    p.add_argument("--random-probability", type=float, default=0.25)
    p.add_argument("--max-episode", type=int, default=1575)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--head-epochs", type=int, default=15)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sources", nargs="+",
                   default=["frame", "all", "oracle_frame", "oracle_all", "pose",
                            "frame+pose", "oracle_frame+pose"])
    p.add_argument("--heads", nargs="+", default=["linear", "mlp", "mlp256"])
    args = p.parse_args()
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    started = time.monotonic()

    data = output / "data.pt"
    if data.exists():
        cache = torch.load(data, weights_only=False)
        sets = cache["sets"]
    else:
        pool = EnvPool(args.num_envs, task="e1m1", teacher=True, skill=args.skill)
        sets = {}
        try:
            for name, budget, base in (("perception", args.perception_steps, 1_000_000),
                                       ("validation", args.validation_steps, 2_000_000),
                                       ("head", args.head_steps, 3_000_000),
                                       ("heldout", args.heldout_steps, 5_000_000)):
                replay = Replay(budget)
                stats = collect(pool, replay, budget,
                                np.random.default_rng(base + args.perception_seed), device,
                                base, args.max_episode,
                                random_probability=args.random_probability)
                print(json.dumps({"stage": "collection", "set": name, **stats,
                                  "seconds": time.monotonic() - started}), flush=True)
                sets[name] = replay
        finally:
            pool.close()
        torch.save({"sets": sets}, data, pickle_protocol=4)

    features = {}
    for encoder, keys in (("frame", FRAME_KEYS),
                          ("all", ("depth", "visibility_door", "direction", "progress"))):
        torch.manual_seed(args.perception_seed)
        perception, history = pretrain(GeometryPerception().to(device), sets["perception"],
                                       sets["validation"], keys, args, device)
        torch.save({"model": perception.state_dict(), "history": history},
                   output / f"perception_{encoder}.pt")
        for split in ("head", "heldout"):
            angles, labels = encode_replay(perception, sets[split], args.batch, device)
            features[(encoder, split)] = angles
            features[("labels", split)] = labels
    for split in ("head", "heldout"):
        replay = sets[split]
        aux = torch.as_tensor(replay.aux[:replay.size], device=device)
        cool = torch.as_tensor(2 * replay.cooldown[:replay.size] / COOLDOWN - 1,
                               device=device).unsqueeze(1)
        features[("oracle_frame", split)] = torch.cat((aux[:, :17], cool), 1)
        features[("oracle_all", split)] = torch.cat((aux[:, :20], cool), 1)
        features[("starts", split)] = torch.as_tensor(replay.start[:replay.size], device=device)
        pose = torch.as_tensor(replay.pose[:replay.size], device=device)
        if not pose.abs().sum():
            raise RuntimeError("Replay has no pose; re-collect with a pose-reporting env")
        features[("pose", split)] = torch.cat(
            (pose[:, :2], pose[:, 2:].sin(), pose[:, 2:].cos()), 1)
    torch.save({k[0] + "/" + k[1]: v.cpu() for k, v in features.items()},
               output / "features.pt")

    y, vy = features[("labels", "head")], features[("labels", "heldout")]
    results = {"args": vars(args), "label_counts": torch.bincount(y).tolist(), "runs": []}
    def inputs(source, split, k):
        base, _, pose = source.partition("+")
        if base == "pose":
            return features[("pose", split)]
        x = stack(features[(base, split)], features[("starts", split)], k, args.stride)
        return torch.cat((x, features[("pose", split)]), 1) if pose else x

    for source in args.sources:
        for k in ([1] if source == "pose" else args.ks):
            x, vx = inputs(source, "head", k), inputs(source, "heldout", k)
            for kind in args.heads:
                for seed in args.seeds:
                    row = {"source": source, "k": k, "head": kind, "seed": seed,
                           **fit(kind, x, y, vx, vy, seed, args)}
                    results["runs"].append(row)
                    print(json.dumps({key: v for key, v in row.items() if key != "curve"}),
                          flush=True)
                    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    results["seconds"] = time.monotonic() - started
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
