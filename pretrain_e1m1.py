"""Imitate successful E1M1 routes with the circuit-only action policy."""

import argparse
from collections import Counter
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from e1m1_env import QuantumE1M1Env
from e1m1_quantum_model import QuantumE1M1Policy
from train_e1m1 import evaluate
from validate_e1m1_route import ScriptedFollower


def collect(env, seeds, max_steps):
    observations, actions = [], []
    wins = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy = ScriptedFollower(seed=seed)
        positions = []
        episode = []
        for _ in range(max_steps):
            pos, _ = env._player()
            positions.append(pos)
            stuck = len(positions) > 10 and np.linalg.norm(positions[-1] - positions[-10]) < 8.0
            recovering = stuck or policy.unstick > 0
            action = policy.act(obs, stuck)
            if not recovering:
                episode.append((obs.copy(), action))
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        if info["exit_reached"]:
            wins += 1
            observations.extend(x for x, _ in episode)
            actions.extend(a for _, a in episode)
    if not observations:
        raise RuntimeError("No successful demonstrations were collected")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64), wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output", default="e1m1_imitation")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    env = QuantumE1M1Env(skill=3)
    try:
        obs, actions, wins = collect(env, range(args.episodes), 1600)
    finally:
        env.close()
    counts = Counter(actions.tolist())
    print(f"demonstrations={wins}/{args.episodes} samples={len(actions)} "
          f"action_counts={dict(sorted(counts.items()))}", flush=True)

    x = torch.as_tensor(obs, device=device)
    y = torch.as_tensor(actions, device=device)
    count = torch.bincount(y, minlength=7).float()
    weights = count.clamp(min=1).rsqrt()
    weights /= weights.mean()
    model = QuantumE1M1Policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.output, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for ids in torch.randperm(len(y), device=device).split(args.batch_size):
            logits, _ = model(x[ids])
            loss = F.cross_entropy(logits, y[ids], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(ids)
        mean_loss = total_loss / len(y)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "loss": mean_loss, "args": vars(args)},
                       os.path.join(args.output, "best.pt"))
        print(f"epoch={epoch} weighted_ce={mean_loss:.4f}", flush=True)

    saved = torch.load(os.path.join(args.output, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    with torch.no_grad():
        predicted = torch.cat([model(b)[0].argmax(-1) for b in x.split(args.batch_size)])
    accuracy = float((predicted == y).float().mean())
    print(f"training_action_accuracy={accuracy:.3f}", flush=True)

    eval_env = QuantumE1M1Env(skill=3)
    try:
        results = evaluate(model, eval_env, device, episodes=args.eval_episodes)
    finally:
        eval_env.close()
    summary = {"demonstration_wins": wins, "samples": len(y),
               "training_action_accuracy": accuracy,
               "eval_wins": sum(r["exit_reached"] for r in results),
               "evaluation": results}
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"full-start greedy wins={summary['eval_wins']}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
