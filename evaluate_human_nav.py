"""Held-out sampled and random policy evaluation for screen-only navigation."""
import argparse
import json
import os

import numpy as np
import torch

from e1m1_quantum_model import QuantumE1M1Policy
from human_nav_env import HumanNavigationEnv


def run(model, device, memory, policy, episodes, seed):
    env = HumanNavigationEnv(memory=memory)
    rows = []
    try:
        for ep in range(episodes):
            torch.manual_seed(seed + ep)
            rng = np.random.default_rng(seed + ep)
            obs, _ = env.reset(seed=seed + ep)
            info = {"success": False}
            counts = [0, 0, 0]
            for step in range(1, 526):
                if policy == "random":
                    action = int(rng.integers(3))
                else:
                    with torch.no_grad():
                        logits, _ = model(torch.as_tensor(obs[None], dtype=torch.float32, device=device))
                        action = (int(logits.argmax(-1).item()) if policy == "greedy" else
                                  int(torch.distributions.Categorical(logits=logits).sample().item()))
                counts[action] += 1
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            rows.append({"seed": seed + ep, "success": bool(info["success"]),
                         "steps": step, "actions": counts,
                         "visited_cells": len(env.travel_map.visits)})
            if (ep + 1) % 5 == 0:
                print(f"memory={memory} policy={policy} episodes={ep+1} "
                      f"wins={sum(r['success'] for r in rows)}", flush=True)
    finally:
        env.close()
    return {"successes": sum(r["success"] for r in rows), "episodes": len(rows),
            "mean_visited_cells": float(np.mean([r["visited_cells"] for r in rows])),
            "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="human_nav_runs")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    results = {}
    for memory in ("off", "on"):
        model = QuantumE1M1Policy(n_qubits=12, n_layers=2, n_actions=3).to(device)
        with open(os.path.join(args.root, memory, "results.json")) as f:
            results[memory] = {"best_greedy": json.load(f)["evaluation"]}
        for checkpoint_name in ("best", "latest"):
            checkpoint = torch.load(os.path.join(args.root, memory, checkpoint_name + ".pt"),
                                    map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            for policy in (("sampled",) if checkpoint_name == "best" else
                           ("greedy", "sampled")):
                key = checkpoint_name + "_" + policy
                results[memory][key] = run(model, device, memory == "on", policy,
                                           args.episodes, args.seed)
                print(f"FINAL memory={memory} {key} "
                      f"{results[memory][key]['successes']}/{args.episodes}", flush=True)
    results["random"] = run(None, device, False, "random", args.episodes, args.seed)
    print(f"FINAL random {results['random']['successes']}/{args.episodes}", flush=True)
    with open(os.path.join(args.root, "sampled_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
