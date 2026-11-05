"""Frozen-perception E1M1 benchmark with a shared, aggregated DAgger dataset.

Stages: collect geometry-only data; select perception on separate episodes; freeze;
collect teacher demonstrations; fit all heads on identical labels and angles;
aggregate equal-budget rollouts from EVERY head before the next fitting round.
Selection and final environment seeds are separate from all training data.
"""

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from frozen_perception_models import (HEADS, N_ACTIONS, FrozenPolicy,
                                      GeometryPerception, fingerprint, make_head,
                                      next_cooldown)
from pixel_nav_env import EnvPool
from train_e1m1_imitation import Dataset, teacher_eval
from train_pixel_nav import UniformStarts


class Replay(Dataset):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cooldown = np.zeros(capacity, np.float32)

    def append(self, episode):
        count = len(episode["actions"])
        if self.size + count > self.capacity:
            raise RuntimeError("Replay capacity exhausted; increase --capacity")
        begin = self.size
        self.add_episode(episode["frames"], episode["actions"], episode["aux"])
        self.cooldown[begin:self.size] = episode["cooldown"]

    def tensors(self, indices, device):
        images, actions, geometry = self.batch(indices)
        return (torch.as_tensor(images, device=device),
                torch.as_tensor(actions, device=device),
                torch.as_tensor(geometry, device=device),
                torch.as_tensor(self.cooldown[indices], device=device))


def collect(pool, replay, budget, rng, device, seed_base, max_episode, policy=None,
            teacher_probability=1.0, random_probability=0.0):
    """Exactly budget states, including partial episodes; no discarded slow episodes.

    Separate calls always start fresh episodes, making training/validation episode-disjoint.
    Cooldown follows chosen actions, never counterfactual teacher labels.
    """
    starts = UniformStarts()
    observations, infos, buffers, lengths, cooldowns = {}, {}, {}, {}, {}
    serial, collected, returns = 0, 0, []
    labels = Counter()

    def begin(i):
        nonlocal serial
        serial += 1
        observations[i], infos[i] = pool.reset(
            [i], [seed_base + serial], {"start_wp": starts.sample(rng)})[0]
        buffers[i] = {key: [] for key in ("frames", "actions", "aux", "cooldown")}
        lengths[i], cooldowns[i] = 0, 0

    for i in range(pool.n):
        begin(i)
    while collected < budget:
        ids = list(range(min(pool.n, budget - collected)))
        student = None
        if policy is not None:
            with torch.no_grad():
                logits, _ = policy(
                    torch.as_tensor(np.stack([observations[i] for i in ids]), device=device),
                    torch.as_tensor([cooldowns[i] for i in ids], device=device))
                student = logits.argmax(-1).tolist()
        chosen = []
        for j, i in enumerate(ids):
            label = int(infos[i]["teacher"])
            labels[label] += 1
            buffers[i]["frames"].append(observations[i][9:12].copy())
            buffers[i]["actions"].append(label)
            buffers[i]["aux"].append(infos[i]["aux"])
            buffers[i]["cooldown"].append(cooldowns[i])
            if rng.random() < random_probability:
                action = int(rng.integers(N_ACTIONS))
            elif policy is None or rng.random() < teacher_probability:
                action = label
            else:
                action = student[j]
            chosen.append(action)
            cooldowns[i] = next_cooldown(cooldowns[i], action)
        for i, (image, _, term, trunc, info) in zip(ids, pool.step(ids, chosen)):
            collected += 1
            lengths[i] += 1
            if term or trunc or lengths[i] >= max_episode:
                replay.append(buffers[i])
                returns.append(bool(info.get("success", False)))
                begin(i)
            else:
                observations[i], infos[i] = image, info
    for i in range(pool.n):
        if buffers[i]["actions"]:
            replay.append(buffers[i])
    return {"states": collected, "completed_or_capped_episodes": len(returns),
            "successes": sum(returns), "labels": dict(sorted(labels.items()))}


def geometry_metrics(perception, replay, batch, device):
    totals = Counter()
    with torch.no_grad():
        for offset in range(0, replay.size, batch):
            idx = np.arange(offset, min(offset + batch, replay.size))
            images, _, geometry, cooldown = replay.tensors(idx, device)
            for key, value in perception.geometry_loss(images, cooldown, geometry).items():
                totals[key] += float(value) * len(idx)
    return {key: value / replay.size for key, value in totals.items()}


def pretrain(perception, training, validation, args, device, output):
    optimizer = torch.optim.Adam(perception.parameters(), lr=args.perception_lr)
    rng = np.random.default_rng(args.perception_seed)
    best_score, best_state, history = float("inf"), None, []
    for epoch in range(args.perception_epochs):
        perception.train()
        for idx in np.array_split(rng.permutation(training.size),
                                 max(1, int(np.ceil(training.size / args.batch)))):
            images, _, geometry, cooldown = training.tensors(idx, device)
            # Action labels and rewards are deliberately unused here.
            loss = sum(perception.geometry_loss(images, cooldown, geometry).values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        perception.eval()
        metrics = geometry_metrics(perception, validation, args.batch, device)
        score = sum(metrics.values())
        if score < best_score:
            best_score, best_state = score, copy.deepcopy(perception.state_dict())
        row = {"stage": "perception", "epoch": epoch + 1, "validation": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
    perception.load_state_dict(best_state)
    perception.freeze()
    metrics = geometry_metrics(perception, validation, args.batch, device)
    torch.save({"model": perception.state_dict(), "fingerprint": fingerprint(perception),
                "validation": metrics, "history": history, "args": vars(args)},
               output / "perception.pt")
    return metrics


def constant_geometry_metrics(training, validation):
    """Predict training means on validation episodes; same losses as perception."""
    mean = np.clip(training.aux[:training.size].mean(0), 1e-6, 1 - 1e-6)
    # Sine/cosine are signed regression targets, so do not clip these two coordinates.
    mean[17:19] = training.aux[:training.size, 17:19].mean(0)
    targets = validation.aux[:validation.size]
    positive = targets[:, 8:17]
    probability = 5 * mean[8:17] / (1 + 4 * mean[8:17])
    return {"depth": float(np.mean((targets[:, :8] - mean[:8]) ** 2)),
            "visibility_door": float(np.mean(-5 * positive * np.log(probability)
                                               - (1 - positive) * np.log1p(-probability))),
            "direction": float(np.mean((targets[:, 17:19] - mean[17:19]) ** 2)),
            "progress": float(np.mean((targets[:, 19] - mean[19]) ** 2))}


@torch.no_grad()
def encode_replay(perception, replay, batch, device):
    angles = []
    for start in range(0, replay.size, batch):
        idx = np.arange(start, min(start + batch, replay.size))
        images, _, _, cooldown = replay.tensors(idx, device)
        angles.append(perception(images, cooldown))
    return torch.cat(angles), torch.as_tensor(replay.actions[:replay.size], device=device)


def fit_heads(policies, optimizers, angles, labels, epochs, batch, seed):
    # Every head gets the same samples, minibatch order, CE objective and epoch count.
    rng = np.random.default_rng(seed)
    counts = torch.bincount(labels, minlength=N_ACTIONS).float().clamp_min(1)
    weights = counts.rsqrt()
    weights /= weights.mean()
    metrics = {}
    orders = [rng.permutation(len(labels)) for _ in range(epochs)]
    for name, policy in policies.items():
        total, correct, seen = 0.0, 0, 0
        for order in orders:
            for start in range(0, len(order), batch):
                idx = torch.as_tensor(order[start:start + batch], device=angles.device)
                logits, _ = policy.head(angles[idx])
                loss = F.cross_entropy(logits, labels[idx], weight=weights)
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.head.parameters(), 0.5)
                optimizers[name].step()
                total += float(loss.detach()) * len(idx)
                correct += int((logits.argmax(-1) == labels[idx]).sum())
                seen += len(idx)
        metrics[name] = {"training_ce": total / seen, "training_accuracy": correct / seen}
        print(json.dumps({"stage": "head_training_complete", "head": name,
                          **metrics[name]}), flush=True)
    return metrics


@torch.no_grad()
def heldout_metrics(policies, angles, labels, batch):
    result = {}
    for name, policy in policies.items():
        prediction = torch.cat([policy.head(x)[0] for x in angles.split(batch)])
        correct = prediction.argmax(-1) == labels
        per_action = {str(action): float(correct[labels == action].float().mean())
                      for action in range(N_ACTIONS) if (labels == action).any()}
        result[name] = {"ce": float(F.cross_entropy(prediction, labels)),
                        "accuracy": float(correct.float().mean()),
                        "balanced_accuracy": float(np.mean(list(per_action.values()))),
                        "per_action_accuracy": per_action}
    return result


@torch.no_grad()
def evaluate(policy, pool, device, seeds, mode="greedy", rng_seed=0):
    """Real-start episodes; pixels and executed-action cooldown are the only inputs."""
    pending, active, rows = list(seeds), {}, []

    def begin(i):
        seed = pending.pop(0)
        image, _ = pool.reset([i], [seed])[0]
        active[i] = {"image": image, "seed": seed, "cooldown": 0, "steps": 0,
                     "rng": np.random.default_rng(rng_seed + seed)}

    for i in range(min(pool.n, len(pending))):
        begin(i)
    while active:
        ids = sorted(active)
        logits, _ = policy(
            torch.as_tensor(np.stack([active[i]["image"] for i in ids]), device=device),
            torch.as_tensor([active[i]["cooldown"] for i in ids], device=device))
        probabilities = logits.softmax(-1).cpu().numpy()
        actions = [int(logits[j].argmax()) if mode == "greedy" else
                   int(active[i]["rng"].choice(N_ACTIONS, p=probabilities[j] /
                                               probabilities[j].sum()))
                   for j, i in enumerate(ids)]
        for i, action, (image, _, term, trunc, info) in zip(ids, actions,
                                                          pool.step(ids, actions)):
            episode = active[i]
            episode["steps"] += 1
            episode["image"] = image
            episode["cooldown"] = next_cooldown(episode["cooldown"], action)
            if term or trunc:
                rows.append({"seed": episode["seed"], "steps": episode["steps"],
                             "success": bool(info["success"]), **info["episode"]})
                del active[i]
                if pending:
                    begin(i)
    rows.sort(key=lambda row: row["seed"])
    return {"successes": sum(row["success"] for row in rows), "episodes": len(rows),
            "mean_progress": float(np.mean([row["progress"] for row in rows])),
            "rows": rows}


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    # default.qubit builds its initial state on the default torch device.
    torch.set_default_device(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists() or (output / "perception.pt").exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pennylane", "vizdoom", "gymnasium")}
    versions["python"] = platform.python_version()
    (output / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    started = time.monotonic()
    pool = EnvPool(args.num_envs, task="e1m1", teacher=True, skill=args.skill)
    try:
        torch.manual_seed(args.perception_seed)
        perception = GeometryPerception().to(device)
        training, validation = Replay(args.perception_steps), Replay(args.validation_steps)
        for dataset, budget, seed_base in ((training, args.perception_steps, 1_000_000),
                                           (validation, args.validation_steps, 2_000_000)):
            stats = collect(pool, dataset, budget,
                            np.random.default_rng(seed_base + args.perception_seed), device,
                            seed_base, args.max_episode, random_probability=0.25)
            print(json.dumps({"stage": "geometry_collection", **stats}), flush=True)
        geometry = pretrain(perception, training, validation, args, device, output)
        # Compare against constant predictions to reveal a poor/collapsed representation.
        geometry["constant_prediction"] = constant_geometry_metrics(training, validation)
        geometry["target_std"] = validation.aux[:validation.size].std(0).tolist()
        del training
        frozen_hash = fingerprint(perception)
        validation_angles, validation_labels = encode_replay(perception, validation,
                                                              args.batch, device)
        torch.save({"angles": validation_angles.cpu(), "labels": validation_labels.cpu(),
                    "perception_fingerprint": frozen_hash}, output / "validation_features.pt")
        del validation
        replay = Replay(args.capacity)
        initial = collect(pool, replay, args.initial_steps, np.random.default_rng(3_000_000),
                          device, 3_000_000, args.max_episode)
        print(json.dumps({"stage": "demonstrations", **initial}), flush=True)
        # All runs start from exactly the same teacher dataset; per-seed DAgger data is reset.
        initial_size = replay.size
        results = {"protocol": "frozen_geometry_shared_dagger_v1", "args": vars(args),
                   "versions": versions,
                   "perception_fingerprint": frozen_hash, "geometry_validation": geometry,
                   "initial_collection": initial, "runs": []}
        for seed in args.seeds:
            replay.size = initial_size
            policies, optimizers = {}, {}
            for name in args.heads:
                torch.manual_seed(seed)
                policy = FrozenPolicy(perception, make_head(name).to(device))
                policies[name] = policy
                lr = args.quantum_lr if name.startswith(("quantum", "product")) else args.lr
                optimizers[name] = torch.optim.Adam(policy.head.parameters(), lr=lr, eps=1e-5)
            best_states, best_scores, history = {}, {name: -1.0 for name in policies}, []
            best_iterations = {}
            for iteration in range(args.iterations + 1):
                if iteration:
                    for name, policy in policies.items():
                        stats = collect(
                            pool, replay, args.iter_steps,
                            np.random.default_rng(4_000_000 + seed * 100_000 + iteration),
                            device, 4_000_000 + seed * 100_000 + iteration * 10_000,
                            args.max_episode, policy=policy,
                            teacher_probability=args.beta)
                        print(json.dumps({"stage": "dagger_collection", "seed": seed,
                                          "iteration": iteration, "head": name, **stats}),
                              flush=True)
                angles, labels = encode_replay(perception, replay, args.batch, device)
                torch.save({"angles": angles.cpu(), "labels": labels.cpu(),
                            "perception_fingerprint": frozen_hash},
                           output / f"features_seed{seed}_iteration{iteration}.pt")
                training_metrics = fit_heads(policies, optimizers, angles, labels,
                                              args.head_epochs, args.batch,
                                              seed * 1000 + iteration)
                diagnostic = heldout_metrics(policies, validation_angles, validation_labels,
                                              args.batch)
                row = {"seed": seed, "iteration": iteration, "dataset_size": replay.size,
                       "training": training_metrics, "heldout_actions": diagnostic,
                       "selection": {}}
                for name, policy in policies.items():
                    selection = evaluate(policy, pool, device,
                                         range(10_000, 10_000 + args.eval_episodes))
                    score = selection["successes"] / selection["episodes"]
                    score += 0.01 * selection["mean_progress"]
                    row["selection"][name] = selection
                    if score > best_scores[name]:
                        best_scores[name] = score
                        best_iterations[name] = iteration
                        best_states[name] = copy.deepcopy(policy.head.state_dict())
                        torch.save({"head": best_states[name], "name": name, "seed": seed,
                                    "iteration": iteration,
                                    "perception_fingerprint": frozen_hash},
                                   output / f"{name}_seed{seed}_best.pt")
                    if fingerprint(perception) != frozen_hash:
                        raise AssertionError("Frozen perception changed during head training")
                    torch.save({"head": policy.head.state_dict(), "seed": seed,
                                "iteration": iteration, "name": name,
                                "perception_fingerprint": frozen_hash},
                               output / f"{name}_seed{seed}_latest.pt")
                    print(json.dumps({"stage": "selection", "seed": seed,
                                      "iteration": iteration, "head": name,
                                      "successes": selection["successes"],
                                      "episodes": selection["episodes"], "score": score}),
                          flush=True)
                history.append(row)
                print(json.dumps({"stage": "head_fit", **row}), flush=True)
                (output / f"history_seed{seed}.json").write_text(json.dumps(history, indent=2))
            run_result = {"seed": seed, "history": history, "heads": {}}
            for name, policy in policies.items():
                policy.head.load_state_dict(best_states[name])
                torch.save({"head": best_states[name], "name": name, "seed": seed,
                            "iteration": best_iterations[name],
                            "perception_fingerprint": frozen_hash},
                           output / f"{name}_seed{seed}_best.pt")
                final = {"parameters": sum(p.numel() for p in policy.head.parameters()),
                         "best_iteration": best_iterations[name],
                         "selection_score": best_scores[name],
                         "heldout_actions": heldout_metrics(
                             {name: policy}, validation_angles, validation_labels,
                             args.batch)[name]}
                for mode in ("greedy", "sampled"):
                    final[mode] = evaluate(policy, pool, device,
                                           range(20_000, 20_000 + args.final_episodes),
                                           mode=mode, rng_seed=seed)
                run_result["heads"][name] = final
                print(json.dumps({"stage": "final", "seed": seed, "head": name,
                                  **final}), flush=True)
            results["runs"].append(run_result)
            if "teacher" not in results:
                results["teacher"] = teacher_eval(
                    pool, range(20_000, 20_000 + args.final_episodes))
            results["perception_unchanged"] = fingerprint(perception) == frozen_hash
            results["seconds"] = time.monotonic() - started
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return results
    finally:
        pool.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="frozen_perception_runs/e1m1_full")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--skill", type=int, default=3)
    p.add_argument("--perception-seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--heads", choices=HEADS, nargs="+", default=list(HEADS))
    p.add_argument("--perception-steps", type=int, default=100_000)
    p.add_argument("--validation-steps", type=int, default=10_000)
    p.add_argument("--perception-epochs", type=int, default=20)
    p.add_argument("--perception-lr", type=float, default=3e-4)
    p.add_argument("--initial-steps", type=int, default=50_000)
    p.add_argument("--iter-steps", type=int, default=10_000, help="per head, per DAgger round")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--head-epochs", type=int, default=6)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--capacity", type=int, default=250_000)
    p.add_argument("--max-episode", type=int, default=1575)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quantum-lr", type=float, default=3e-3)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--final-episodes", type=int, default=100)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    for key in ("perception_steps", "validation_steps", "perception_epochs", "initial_steps",
                "iter_steps", "head_epochs", "num_envs", "batch", "eval_episodes",
                "final_episodes", "max_episode", "threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.iterations < 0 or not 0 <= args.beta <= 1:
        raise ValueError("iterations must be nonnegative and beta must be in [0,1]")
    if len(set(args.heads)) != len(args.heads) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate heads/seeds would bias the comparison")
    required = args.initial_steps + args.iterations * args.iter_steps * len(args.heads)
    if args.capacity < required:
        raise ValueError(f"--capacity must be at least {required}")
    run(args)
