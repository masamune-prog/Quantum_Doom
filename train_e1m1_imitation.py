"""Imitate the scripted E1M1 teacher from screen frames (DAgger), then evaluate the student.

The student sees only stacked 80x60 screens. The teacher (e1m1_teacher.py) reads private
state and only supplies the action label for each state; it is not an input anywhere.
Iteration 0 trains on teacher-driven demonstrations. Later iterations let the student (or a
beta mix of student and teacher) drive, label every visited state with the teacher's action,
add it to the dataset, and retrain on everything collected so far.

Training also uses the auxiliary targets (depth, monster visibility, door ahead, route
direction and progress). Evaluation starts at the real start, like the RL runs.
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from pixel_nav_env import EnvPool
from pixel_nav_models import PixelNavPolicy
from train_pixel_nav import (AUX_ROUTE, AUX_SIZES, FINAL_SEED, SELECTION_SEED, UniformStarts,
                             evaluate, summary)

N_ACTIONS = 7
AUX_SIZE = AUX_SIZES["e1m1"]
ACTION_NAMES = ("FORWARD", "ATTACK", "STRAFE_LEFT", "STRAFE_RIGHT", "TURN_LEFT", "TURN_RIGHT",
                "USE")


class Dataset:
    """Newest frames of every visited state; a state's 4-frame stack is rebuilt on demand."""

    def __init__(self, capacity):
        self.capacity, self.size = capacity, 0
        self.frames = np.zeros((capacity, 3, 60, 80), np.uint8)
        self.actions = np.zeros(capacity, np.int64)
        self.aux = np.zeros((capacity, AUX_SIZE), np.float32)
        self.start = np.zeros(capacity, np.int64)  # index of the first frame of the episode

    def add_episode(self, frames, actions, aux):
        n = min(len(actions), self.capacity - self.size)
        if n <= 0:
            return 0
        s = self.size
        self.frames[s:s + n] = np.asarray(frames[:n])
        self.actions[s:s + n] = actions[:n]
        self.aux[s:s + n] = np.asarray(aux[:n])
        self.start[s:s + n] = s
        self.size += n
        return n

    def batch(self, idx):
        # obs at step t is frames t-3..t, with the first frame repeated at episode start.
        stack = np.stack([np.maximum(idx - k, self.start[idx]) for k in (3, 2, 1, 0)], axis=1)
        return (self.frames[stack].reshape(len(idx), 12, 60, 80), self.actions[idx],
                self.aux[idx])


def collect(pool, dataset, n_steps, model, beta, rng, starts, device, seed_base, max_episode):
    """Drive the envs for n_steps decisions; label every visited state with the teacher's action.

    With model=None the teacher drives. Otherwise each decision follows the teacher with
    probability beta and a sample from the student otherwise.
    """
    counter = [0]
    buffers, steps, obs, info = {}, {}, {}, {}

    def begin(i):
        counter[0] += 1
        wp = starts.sample(rng)
        obs[i], info[i] = pool.reset([i], [seed_base + counter[0]], {"start_wp": wp})[0]
        buffers[i], steps[i] = {"frames": [], "actions": [], "aux": []}, 0

    for i in range(pool.n):
        begin(i)
    collected, agree, seen, labels = 0, 0, 0, Counter()
    while collected < n_steps:
        ids = list(range(pool.n))
        student = greedy = None
        if model is not None:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(np.stack([obs[i] for i in ids]), device=device))
            student = torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(-1).tolist()
            greedy = logits.argmax(-1).tolist()
        chosen = []
        for i in ids:
            teacher = int(info[i]["teacher"])
            buffers[i]["frames"].append(obs[i][9:12].copy())
            buffers[i]["actions"].append(teacher)
            buffers[i]["aux"].append(info[i]["aux"])
            labels[teacher] += 1
            if model is not None:
                agree += int(greedy[i] == teacher)
                seen += 1
            chosen.append(teacher if model is None or rng.random() < beta else student[i])
        for i, (o, _, term, trunc, inf) in zip(ids, pool.step(ids, chosen)):
            steps[i] += 1
            if term or trunc or steps[i] >= max_episode:
                b = buffers[i]
                collected += dataset.add_episode(b["frames"], b["actions"], b["aux"])
                begin(i)
            else:
                obs[i], info[i] = o, inf
    return {"collected": collected, "agreement": (agree / seen) if seen else None,
            "labels": {ACTION_NAMES[k]: v for k, v in sorted(labels.items())}}


def fit(model, optimizer, dataset, epochs, batch_size, device, aux_coef, rng, weight_power=0.5):
    counts = np.bincount(dataset.actions[:dataset.size], minlength=N_ACTIONS) + 1.0
    weights = counts.sum() / counts
    weights = torch.as_tensor((weights / weights.mean()) ** weight_power, dtype=torch.float32,
                              device=device)
    last = {}
    for _ in range(epochs):
        order = rng.permutation(dataset.size)
        totals, correct, n = Counter(), 0, 0
        for s in range(0, dataset.size - batch_size + 1, batch_size):
            x, action, aux = dataset.batch(order[s:s + batch_size])
            x = torch.as_tensor(x, device=device)
            action = torch.as_tensor(action, device=device)
            features, z = model.encode(x)
            logits, _ = model.head(z)
            ce = F.cross_entropy(logits, action, weight=weights)
            depth, goal, route = model.aux_loss(features, torch.as_tensor(aux, device=device))
            loss = ce + aux_coef * (depth + goal + route)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            totals.update({"ce": ce.item(), "route": route.item(), "n": 1})
            correct += int((logits.argmax(-1) == action).sum())
            n += len(action)
        count = totals.pop("n")
        last = {k: round(v / count, 4) for k, v in totals.items()}
        last["accuracy"] = round(correct / max(n, 1), 4)
    return last


def teacher_eval(pool, seeds):
    """The teacher's own results on the same seeds, for reference."""
    pending, active, rows = list(seeds), {}, []
    info = {}

    def begin(i):
        seed = pending.pop(0)
        _, info[i] = pool.reset([i], [seed], None)[0]
        active[i] = {"seed": seed, "steps": 0}

    for i in range(min(pool.n, len(pending))):
        begin(i)
    while active:
        ids = sorted(active)
        for i, (_, _, term, trunc, inf) in zip(ids, pool.step(ids, [info[i]["teacher"]
                                                                    for i in ids])):
            active[i]["steps"] += 1
            info[i] = inf
            if term or trunc:
                rows.append({"seed": active[i]["seed"], "success": bool(inf["success"]),
                             "steps": active[i]["steps"], **inf["episode"]})
                del active[i]
                if pending:
                    begin(i)
    hard = [r for r in rows if r["seed"] != 20017]
    return {"mode": "teacher", "successes": sum(r["success"] for r in rows),
            "episodes": len(rows), "successes_without_easy_seed": sum(r["success"] for r in hard),
            "episodes_without_easy_seed": len(hard),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            **{f"mean_{k}": float(np.mean([r[k] for r in rows]))
               for k in ("kills", "health", "dead", "progress")}}


def main(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    os.makedirs(args.output, exist_ok=True)
    pool = EnvPool(args.num_envs, task="e1m1", teacher=True, skill=args.skill)
    model = PixelNavPolicy(head=args.model, n_actions=N_ACTIONS, aux_pos_weight=5.0,
                           aux_size=AUX_SIZE, aux_route=AUX_ROUTE["e1m1"]).to(device)
    if args.init_encoder:
        source = torch.load(args.init_encoder, map_location=device, weights_only=False)["model"]
        model.load_state_dict({k: v for k, v in source.items() if k.startswith("encoder.")},
                              strict=False)
        print(f"loaded encoder from {args.init_encoder}", flush=True)
    head_ids = {id(p) for p in model.head.parameters()}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
        {"params": list(model.head.parameters()), "lr": args.head_lr or args.lr}], eps=1e-5)
    dataset, starts = Dataset(args.capacity), UniformStarts(real_fraction=args.real_fraction)
    betas = [float(b) for b in args.betas.split(",")]
    history, best_score, begin = [], -1.0, time.monotonic()
    print(f"imitation model={args.model} skill={args.skill} envs={args.num_envs} device={device} "
          f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    try:
        for iteration in range(args.iterations + 1):
            beta = betas[min(iteration, len(betas) - 1)]
            n_steps = args.initial_steps if iteration == 0 else args.iter_steps
            stats = collect(pool, dataset, n_steps, None if iteration == 0 else model, beta, rng,
                            starts, device, 5_000_000 * (args.seed + 1) + 100_000 * iteration,
                            args.max_episode)
            losses = fit(model, optimizer, dataset,
                         args.initial_epochs if iteration == 0 else args.epochs, args.batch,
                         device, args.aux_coef, rng, args.weight_power)
            result = evaluate(model, pool, device,
                              range(SELECTION_SEED, SELECTION_SEED + args.eval_episodes),
                              "greedy", N_ACTIONS)
            score = result["successes"] / result["episodes"] + 0.01 * result["mean_progress"]
            row = {"iteration": iteration, "beta": beta, "dataset": dataset.size, **stats,
                   **losses, "selection": summary(result), "score": round(score, 4),
                   "seconds": round(time.monotonic() - begin, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "update": iteration, "score": score},
                       os.path.join(args.output, "latest.pt"))
            if score >= best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "update": iteration, "score": score},
                           os.path.join(args.output, "best.pt"))
            if time.monotonic() - begin > args.max_seconds:
                print("time budget reached", flush=True)
                break
        best = torch.load(os.path.join(args.output, "best.pt"), map_location=device,
                          weights_only=False)
        model.load_state_dict(best["model"])
        seeds = range(FINAL_SEED, FINAL_SEED + args.final_episodes)
        results = {"model_type": args.model, "skill": args.skill, "best_iteration": best["update"],
                   "history": history, "dataset": dataset.size,
                   "greedy": evaluate(model, pool, device, seeds, "greedy", N_ACTIONS),
                   "sampled": evaluate(model, pool, device, seeds, "sampled", N_ACTIONS,
                                       rng_seed=args.seed),
                   "teacher": teacher_eval(pool, seeds)}
    finally:
        pool.close()
    with open(os.path.join(args.output, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    for key in ("greedy", "sampled", "teacher"):
        print(f"FINAL imitation {args.model} best_iteration={best['update']} "
              f"{summary(results[key])}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="classical",
                   choices=("classical", "linear", "quantum", "quantum_noent", "quantum_frozen"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skill", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--initial-steps", type=int, default=200_000)
    p.add_argument("--iter-steps", type=int, default=60_000)
    p.add_argument("--initial-epochs", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--betas", default="1.0,0.3,0.0", help="teacher mix per iteration; last repeats")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-lr", type=float, default=None, help="head learning rate (circuit: 3e-3)")
    p.add_argument("--aux-coef", type=float, default=0.5)
    p.add_argument("--weight-power", type=float, default=0.5,
                   help="class weights = (1/frequency)^power, so rare actions like USE count more")
    p.add_argument("--capacity", type=int, default=750_000)
    p.add_argument("--max-episode", type=int, default=600)
    p.add_argument("--real-fraction", type=float, default=0.25)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--final-episodes", type=int, default=100)
    p.add_argument("--max-seconds", type=int, default=10**9,
                   help="stop iterating after this long, then run the final evaluation")
    p.add_argument("--init-encoder", default=None)
    p.add_argument("--output", required=True)
    main(p.parse_args())
