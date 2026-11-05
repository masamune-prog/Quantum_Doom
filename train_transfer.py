"""Multi-task semantic perception + goal-conditioned DAgger, with E1M2 held out.

Compare E1M1-only versus multi-map perception with equal pretraining state budgets.
All policies receive the same balanced multi-task action labels and DAgger union.
GRU and MLP policies compare memory while keeping perception frozen. Test maps are
never used to train perception, collect action labels, or select checkpoints.
"""

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from frozen_perception_models import fingerprint, next_cooldown
from pixel_nav_models import PixelNavPolicy
from train_pixel_nav import UniformStarts
from transfer_env import ACTION_MAPS, TRAIN_MAPS, TEST_MAPS, TransferPool, context
from transfer_models import SemanticPerception, GoalActor, TransferPolicy


class Replay:
    def __init__(self, capacity):
        self.capacity, self.size = capacity, 0
        self.frames = np.zeros((capacity, 3, 60, 80), np.uint8)
        self.labels = np.zeros(capacity, np.int64)
        self.targets = np.zeros((capacity, 27), np.float32)
        self.contexts = np.zeros((capacity, 18), np.float32)
        self.starts = np.zeros(capacity, np.int64)
        self.episodes = []

    def append(self, buffer):
        count = len(buffer["labels"])
        if not count:
            return
        begin, end = self.size, self.size + count
        if end > self.capacity:
            raise ValueError("Replay capacity exceeded")
        for key in ("frames", "labels", "targets", "contexts"):
            getattr(self, key)[begin:end] = buffer[key]
        self.starts[begin:end] = begin
        self.episodes.append((begin, end))
        self.size = end

    def batch(self, ids, device):
        indices = np.stack([np.maximum(ids - k, self.starts[ids]) for k in (3, 2, 1, 0)], 1)
        images = self.frames[indices].reshape(len(ids), 12, 60, 80)
        return tuple(torch.as_tensor(array, device=device) for array in
                     (images, self.labels[ids], self.targets[ids], self.contexts[ids]))


class Teachers:
    def __init__(self, device, navigation, corridor):
        self.device, self.models = device, {}
        for task, path, actions in (("navigation", navigation, 3), ("corridor", corridor, 6)):
            model = PixelNavPolicy(head="classical", n_actions=actions).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
            self.models[task] = model.eval().requires_grad_(False)

    @torch.no_grad()
    def actions(self, task, images, infos):
        if task == "e1m1":
            return [int(info["teacher"]) for info in infos]
        logits, _ = self.models[task](torch.as_tensor(np.stack(images), device=self.device))
        return [ACTION_MAPS[task][a] for a in logits.argmax(-1).tolist()]


class PerceptionSubset:
    def __init__(self, source, size, seed):
        self.source, self.size = source, size
        self.ids = np.random.default_rng(seed).permutation(source.size)[:size]

    def batch(self, ids, device):
        return self.source.batch(self.ids[ids], device)


def collect(task, replay, budget, teachers, args, seed_base, policy=None, beta=1.0,
            random_probability=0.0):
    if task not in TRAIN_MAPS:
        raise ValueError("Test map requested for training collection")
    pool = TransferPool(args.num_envs, task, teacher=task == "e1m1")
    rng, starts = np.random.default_rng(seed_base), UniformStarts()
    obs, info, memory, buffers, previous, cooldown, lengths = {}, {}, {}, {}, {}, {}, {}
    serial, total, wins, ended = 0, 0, 0, 0

    def begin(i):
        nonlocal serial
        serial += 1
        options = {"start_wp": starts.sample(rng)} if task == "e1m1" else None
        obs[i], info[i] = pool.reset([i], [seed_base + serial], options)[0]
        memory[i], previous[i], cooldown[i], lengths[i] = None, 7, 0, 0
        buffers[i] = {k: [] for k in ("frames", "labels", "targets", "contexts")}

    try:
        for i in range(pool.n):
            begin(i)
        while total < budget:
            ids = list(range(min(pool.n, budget - total)))
            labels = teachers.actions(task, [obs[i] for i in ids], [info[i] for i in ids])
            ctx = np.stack([context(task, previous[i], cooldown[i]) for i in ids])
            student = None
            if policy is not None:
                states = state_batch(memory, ids, policy.actor, args.device)
                with torch.no_grad():
                    logits, new_states = policy.step(
                        torch.as_tensor(np.stack([obs[i] for i in ids]), device=args.device),
                        torch.as_tensor(ctx, device=args.device), states)
                student = logits.argmax(-1).tolist()
                for j, i in enumerate(ids):
                    memory[i] = None if new_states is None else new_states[:, j:j+1].detach()
            actions = []
            for j, i in enumerate(ids):
                b = buffers[i]
                b["frames"].append(obs[i][9:12].copy())
                b["labels"].append(labels[j])
                b["targets"].append(info[i]["semantic"])
                b["contexts"].append(ctx[j])
                if rng.random() < random_probability:
                    action = int(rng.choice(ACTION_MAPS[task]))
                else:
                    action = labels[j] if policy is None or rng.random() < beta else student[j]
                actions.append(action)
                previous[i], cooldown[i] = action, next_cooldown(cooldown[i], action)
            for i, (image, _, term, trunc, inf) in zip(ids, pool.step(ids, actions)):
                total += 1
                lengths[i] += 1
                if term or trunc or lengths[i] >= args.max_episode:
                    ended += 1
                    wins += int(inf.get("success", False))
                    replay.append(buffers[i])
                    begin(i)
                else:
                    obs[i], info[i] = image, inf
        for i in range(pool.n):
            replay.append(buffers[i])
        return {"task": task, "states": total, "episodes_ended_or_capped": ended,
                "successes": wins}
    finally:
        pool.close()


def state_batch(states, ids, actor, device):
    if actor.kind != "gru":
        return None
    return torch.cat([states[i] if states[i] is not None else
                      torch.zeros(1, 1, actor.core.hidden_size, device=device) for i in ids], 1)


@torch.no_grad()
def perception_metrics(model, replay, args):
    sums = Counter()
    for start in range(0, replay.size, args.batch):
        ids = np.arange(start, min(start + args.batch, replay.size))
        images, _, targets, ctx = replay.batch(ids, args.device)
        for key, value in model.losses(images, ctx[:, :2], targets).items():
            sums[key] += float(value) * len(ids)
    return {k: v / replay.size for k, v in sums.items()}


def fit_perception(training, validation, condition, args, output):
    torch.manual_seed(args.perception_seed)
    model = SemanticPerception().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.perception_lr)
    rng = np.random.default_rng(args.perception_seed)
    tasks = ("e1m1",) if condition == "single" else TRAIN_MAPS
    samples = [(task, index) for task in tasks for index in range(training[task].size)]
    best, saved, history = float("inf"), None, []
    for epoch in range(args.perception_epochs):
        order = rng.permutation(len(samples))
        for start in range(0, len(order), args.batch):
            chosen = [samples[i] for i in order[start:start + args.batch]]
            images, targets, goals = [], [], []
            for task in tasks:
                ids = np.array([i for t, i in chosen if t == task], dtype=np.int64)
                if not len(ids):
                    continue
                x, _, y, ctx = training[task].batch(ids, args.device)
                images.append(x); targets.append(y); goals.append(ctx[:, :2])
            loss = sum(model.losses(torch.cat(images), torch.cat(goals), torch.cat(targets)).values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = {task: perception_metrics(model, validation[task], args) for task in TRAIN_MAPS}
        score = float(np.mean([sum(metrics[task].values()) for task in tasks]))
        if score < best:
            best, saved = score, copy.deepcopy(model.state_dict())
        row = {"stage": "perception", "condition": condition, "epoch": epoch + 1,
               "validation": metrics, "selection_score": score}
        history.append(row)
        print(json.dumps(row), flush=True)
    model.load_state_dict(saved)
    model.freeze()
    torch.save({"model": saved, "fingerprint": fingerprint(model), "history": history,
                "condition": condition, "training_states": len(samples)},
               output / f"perception_{condition}.pt")
    return model


@torch.no_grad()
def cache(model, datasets, args):
    episodes = []
    for task, replay in datasets.items():
        features = []
        for start in range(0, replay.size, args.batch):
            ids = np.arange(start, min(start + args.batch, replay.size))
            images, _, _, ctx = replay.batch(ids, args.device)
            semantic = model.features(images, ctx[:, :2])
            features.append(torch.cat((semantic, ctx), -1))
        inputs = torch.cat(features)
        labels = torch.as_tensor(replay.labels[:replay.size], device=args.device)
        for begin, end in replay.episodes:
            episodes.append((task, inputs[begin:end], labels[begin:end]))
    return episodes


def fit_actor(actor, optimizer, episodes, args, seed):
    rng = np.random.default_rng(seed)
    labels = torch.cat([y for _, _, y in episodes])
    counts = torch.bincount(labels, minlength=7).float().clamp_min(1)
    weights = counts.rsqrt(); weights /= weights.mean()
    for _ in range(args.actor_epochs):
        order = rng.permutation(len(episodes))
        for begin in range(0, len(order), args.sequence_batch):
            chosen = [episodes[i] for i in order[begin:begin + args.sequence_batch]]
            inputs = pad_sequence([x for _, x, _ in chosen], batch_first=True)
            targets = pad_sequence([y for _, _, y in chosen], batch_first=True, padding_value=-100)
            logits, _ = actor(inputs)
            loss = F.cross_entropy(logits.reshape(-1, 7), targets.flatten(), weight=weights,
                                   ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def action_metrics(actor, episodes):
    totals = {}
    for task in TRAIN_MAPS:
        correct = torch.zeros(7, device=next(actor.parameters()).device)
        count = correct.clone()
        loss, n = 0.0, 0
        for t, x, y in episodes:
            if t != task:
                continue
            logits, _ = actor(x.unsqueeze(0))
            logits = logits[0]
            loss += float(F.cross_entropy(logits, y, reduction="sum"))
            n += len(y)
            count += torch.bincount(y, minlength=7)
            correct += torch.bincount(y[logits.argmax(-1) == y], minlength=7)
        recalls = (correct / count.clamp_min(1)).tolist()
        totals[task] = {"ce": loss / n, "accuracy": float(correct.sum() / count.sum()),
                        "recall": recalls, "class_counts": count.tolist()}
    return totals


@torch.no_grad()
def evaluate(task, policy, teachers, args, base, episodes, teacher=False, probes=None):
    pool = TransferPool(args.num_envs, task, teacher=teacher and task == "e1m1")
    pending, active, rows, memory = list(range(base, base + episodes)), {}, [], {}
    probe_totals = {name: Counter() for name in (probes or {})}
    probe_states, decisions = 0, 0

    def begin(i):
        seed = pending.pop(0)
        image, info = pool.reset([i], [seed])[0]
        active[i] = {"image": image, "info": info, "seed": seed, "previous": 7,
                     "cooldown": 0, "steps": 0}
        memory[i] = None

    try:
        for i in range(min(pool.n, len(pending))):
            begin(i)
        while active:
            ids = sorted(active)
            if probes and decisions % args.semantic_probe_every == 0:
                probe_images = torch.as_tensor(np.stack([active[i]["image"] for i in ids]),
                                               device=args.device)
                probe_goals = torch.as_tensor(np.stack([context(task)[:2] for _ in ids]),
                                              device=args.device)
                probe_targets = torch.as_tensor(np.stack([active[i]["info"]["semantic"]
                                                            for i in ids]), device=args.device)
                for name, model in probes.items():
                    for key, loss in model.losses(probe_images, probe_goals, probe_targets).items():
                        probe_totals[name][key] += float(loss) * len(ids)
                probe_states += len(ids)
            decisions += 1
            if teacher:
                actions = teachers.actions(task, [active[i]["image"] for i in ids],
                                           [active[i]["info"] for i in ids])
            else:
                images = torch.as_tensor(np.stack([active[i]["image"] for i in ids]),
                                         device=args.device)
                ctx = torch.as_tensor(np.stack([context(task, active[i]["previous"],
                                                        active[i]["cooldown"]) for i in ids]),
                                      device=args.device)
                logits, states = policy.step(images, ctx, state_batch(memory, ids, policy.actor,
                                                                       args.device))
                actions = logits.argmax(-1).tolist()
                for j, i in enumerate(ids):
                    memory[i] = None if states is None else states[:, j:j+1]
            for i, action, (image, _, term, trunc, info) in zip(ids, actions, pool.step(ids, actions)):
                row = active[i]
                row["steps"] += 1
                capped = row["steps"] >= args.eval_max_steps and not (term or trunc)
                if term or trunc or capped:
                    rows.append({"seed": row["seed"], "steps": row["steps"],
                                 "success": bool(info.get("success", False)),
                                 "evaluation_capped": capped, **info.get("episode", {})})
                    del active[i]
                    del memory[i]
                    if pending:
                        begin(i)
                else:
                    row.update(image=image, info=info, previous=action,
                               cooldown=next_cooldown(row["cooldown"], action))
        rows.sort(key=lambda row: row["seed"])
        return {"task": task, "episodes": len(rows),
                "successes": sum(row["success"] for row in rows),
                "capped_episodes": sum(row["evaluation_capped"] for row in rows), "rows": rows,
                "perception_probe_states": probe_states,
                "paired_perception_losses": {name: {key: value / max(1, probe_states)
                    for key, value in total.items()} for name, total in probe_totals.items()}}
    finally:
        pool.close()


def run(args):
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    if (output / "config.json").exists() or (output / "results.json").exists() or list(output.glob("perception_*.pt")):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    (output / "versions.json").write_text(json.dumps({name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pennylane", "vizdoom", "gymnasium")}, indent=2))
    teachers = Teachers(args.device, args.navigation_teacher, args.corridor_teacher)
    started = time.monotonic()
    # Equal total pretraining samples: single has N E1M1 states; multi has N/3 per map.
    geometry, validation = {}, {}
    for index, task in enumerate(TRAIN_MAPS):
        count = args.perception_steps if task == "e1m1" else args.perception_steps // 3
        geometry[task], validation[task] = Replay(count), Replay(args.validation_steps)
        for replay, budget, base in ((geometry[task], count, 1_000_000),
                                     (validation[task], args.validation_steps, 2_000_000)):
            stats = collect(task, replay, budget, teachers, args, base + index * 100_000,
                            random_probability=.2)
            print(json.dumps({"stage": "geometry_collection", **stats}), flush=True)
    perceptions = {}
    for condition in args.conditions:
        if condition == "single":
            training = {"e1m1": geometry["e1m1"]}
        else:
            training = dict(geometry)
            training["e1m1"] = PerceptionSubset(geometry["e1m1"], args.perception_steps // 3,
                                                 args.perception_seed)
        perceptions[condition] = fit_perception(training, validation, condition, args, output)
    del geometry, training
    hashes = {name: fingerprint(model) for name, model in perceptions.items()}
    validation_cache = {name: cache(model, validation, args) for name, model in perceptions.items()}
    del validation
    capacity = args.initial_steps + args.iterations * args.iter_steps * len(args.conditions) * len(args.models)
    datasets, initial = {}, {}
    for index, task in enumerate(TRAIN_MAPS):
        datasets[task] = Replay(capacity)
        print(json.dumps({"stage": "initial_demonstrations", **collect(
            task, datasets[task], args.initial_steps, teachers, args, 3_000_000 + index * 100_000)}),
              flush=True)
        initial[task] = (datasets[task].size, len(datasets[task].episodes))
    results = {"protocol": "multitask_semantic_transfer_v1", "args": vars(args),
               "train_maps": list(TRAIN_MAPS), "test_maps": list(TEST_MAPS),
               "perception_hashes": hashes, "runs": []}
    for seed in args.seeds:
        for task, replay in datasets.items():
            replay.size = initial[task][0]
            replay.episodes = replay.episodes[:initial[task][1]]
        policies, optimizers, best_states, best_scores = {}, {}, {}, {}
        for condition, model in perceptions.items():
            for kind in args.models:
                name = f"{condition}_{kind}"
                torch.manual_seed(seed)
                hidden = args.gru_hidden if kind == "gru" else args.mlp_hidden
                policies[name] = TransferPolicy(model, GoalActor(kind, hidden).to(args.device))
                optimizers[name] = torch.optim.Adam(policies[name].actor.parameters(), lr=args.lr)
                best_scores[name] = -1
        history = []
        for iteration in range(args.iterations + 1):
            if iteration:
                for name, policy in policies.items():
                    for index, task in enumerate(TRAIN_MAPS):
                        stats = collect(task, datasets[task], args.iter_steps, teachers, args,
                                        10_000_000 + seed * 1_000_000 + iteration * 100_000 + index * 10_000,
                                        policy=policy, beta=args.beta)
                        print(json.dumps({"stage": "dagger_collection", "seed": seed,
                                          "iteration": iteration, "policy": name, **stats}), flush=True)
            caches = {condition: cache(model, datasets, args) for condition, model in perceptions.items()}
            row = {"seed": seed, "iteration": iteration, "policies": {}}
            row["shared_data"] = {task: {"states": replay.size,
                "label_counts": np.bincount(replay.labels[:replay.size], minlength=7).tolist(),
                "label_context_sha256": hashlib.sha256(
                    replay.labels[:replay.size].tobytes() +
                    replay.contexts[:replay.size].tobytes()).hexdigest()}
                for task, replay in datasets.items()}
            for name, policy in policies.items():
                condition = name.split("_")[0]
                fit_actor(policy.actor, optimizers[name], caches[condition], args,
                          seed * 100 + iteration)
                training = action_metrics(policy.actor, caches[condition])
                diagnostic = action_metrics(policy.actor, validation_cache[condition])
                selection = {task: evaluate(task, policy, teachers, args, 10_000,
                                             args.eval_episodes) for task in TRAIN_MAPS}
                score = float(np.mean([r["successes"] / r["episodes"] for r in selection.values()]))
                score += .001 * float(np.mean([diagnostic[task]["accuracy"] for task in TRAIN_MAPS]))
                if score > best_scores[name]:
                    best_scores[name] = score
                    best_states[name] = copy.deepcopy(policy.actor.state_dict())
                    torch.save({"actor": best_states[name], "kind": policy.actor.kind,
                                "hidden": args.gru_hidden if policy.actor.kind == "gru" else args.mlp_hidden,
                                "condition": condition, "iteration": iteration,
                                "perception_hash": hashes[condition]},
                               output / f"{name}_seed{seed}_best.pt")
                assert fingerprint(perceptions[condition]) == hashes[condition]
                stats = {"stage": "policy_fit", "seed": seed, "iteration": iteration,
                         "policy": name, "training": training, "validation": diagnostic,
                         "actor_parameters": sum(p.numel() for p in policy.actor.parameters()),
                         "selection": selection, "score": score}
                row["policies"][name] = stats
                print(json.dumps(stats), flush=True)
            history.append(row)
            (output / f"history_seed{seed}.json").write_text(json.dumps(history, indent=2))
        final = {"seed": seed, "history": history, "policies": {}}
        for name, policy in policies.items():
            policy.actor.load_state_dict(best_states[name])
            scores = {task: evaluate(task, policy, teachers, args, 20_000,
                                     args.final_episodes, probes=perceptions)
                      for task in TRAIN_MAPS + TEST_MAPS}
            final["policies"][name] = scores
            print(json.dumps({"stage": "final", "seed": seed, "policy": name,
                              "results": scores}), flush=True)
        results["runs"].append(final)
        results["perception_unchanged"] = all(fingerprint(model) == hashes[name]
                                                for name, model in perceptions.items())
        results["seconds"] = time.monotonic() - started
        (output / "results.json").write_text(json.dumps(results, indent=2))
    results["teacher_reference"] = {task: evaluate(task, None, teachers, args, 20_000,
                                                   args.final_episodes, teacher=True, probes=perceptions)
                                    for task in TRAIN_MAPS}
    results["seconds"] = time.monotonic() - started
    (output / "results.json").write_text(json.dumps(results, indent=2))
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--navigation-teacher", default="pixel_nav_runs/classical_seed0/best.pt")
    p.add_argument("--corridor-teacher", default="pixel_corridor_runs/classical_seed0/best.pt")
    p.add_argument("--conditions", nargs="+", choices=("single", "multi"), default=["single", "multi"])
    p.add_argument("--models", nargs="+", choices=("gru", "mlp"), default=["gru", "mlp"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--perception-seed", type=int, default=42)
    p.add_argument("--perception-steps", type=int, default=120_000)
    p.add_argument("--validation-steps", type=int, default=5_000, help="per training map")
    p.add_argument("--perception-epochs", type=int, default=20)
    p.add_argument("--perception-lr", type=float, default=3e-4)
    p.add_argument("--initial-steps", type=int, default=50_000, help="per training map")
    p.add_argument("--iter-steps", type=int, default=10_000, help="per policy, per map, per round")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--actor-epochs", type=int, default=6)
    p.add_argument("--sequence-batch", type=int, default=8)
    p.add_argument("--gru-hidden", type=int, default=64)
    p.add_argument("--mlp-hidden", type=int, default=411, help="matches GRU actor parameter budget")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=.3)
    p.add_argument("--max-episode", type=int, default=1575)
    p.add_argument("--eval-max-steps", type=int, default=1575)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--final-episodes", type=int, default=100)
    p.add_argument("--semantic-probe-every", type=int, default=8)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.perception_steps % 3 or args.perception_steps < 3:
        raise ValueError("perception-steps must be a positive multiple of three")
    if not 0 <= args.beta <= 1 or args.iterations < 0:
        raise ValueError("Invalid beta/iterations")
    for key in ("threads", "num_envs", "validation_steps", "perception_epochs", "initial_steps",
                "iter_steps", "actor_epochs", "sequence_batch", "max_episode", "eval_max_steps",
                "batch", "eval_episodes", "final_episodes", "semantic_probe_every",
                "gru_hidden", "mlp_hidden"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("conditions", "models", "seeds"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            raise ValueError(f"Duplicate {key}")
    run(args)
