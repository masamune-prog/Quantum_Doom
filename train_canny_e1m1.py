"""Fully quantum E1M1 policy from hand-made screen features (no CNN), trained by DAgger.

The policy input is only the current screen's 8 sector features (brightness + Canny edge
density in [-1, 1], green-pixel override, times pi; canny_e1m1_env.py), padded to 12.
There is no learned encoder, frame stack, memory or cooldown: the circuit is the whole
policy. Labels come from the scripted E1M1 teacher (private state, labels only).

Protocol follows train_frozen_perception.py: shared initial demonstrations, each DAgger
round adds equal-budget rollouts from EVERY head to the dataset seen by all heads,
identical minibatch orders, selection on real-start seeds 10000+, final test on 20000+.
Classical heads on the same features are controls for whether the features suffice.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from frozen_perception_models import N_ACTIONS, make_head
from pixel_nav_env import EnvPool
from train_e1m1_imitation import teacher_eval
from train_frozen_perception import evaluate, fit_heads, heldout_metrics
from train_pixel_nav import UniformStarts
from wide_circuit import WideCircuitPolicy

HEADS = ("quantum2", "quantum4", "linear", "classical")
QUANTUM = ("quantum",)


def build_head(name):
    # Same circuit as QuantumE1M1Policy (12 qubits, RY(x*s) RZ RX + CNOT ring, scale 1.0),
    # simulated in torch (test_wide_circuit.py checks it against PennyLane).
    if name == "quantum2":
        return WideCircuitPolicy(12, 2, n_actions=N_ACTIONS, input_scale_init=1.0)
    if name == "quantum4":
        return WideCircuitPolicy(12, 4, n_actions=N_ACTIONS, input_scale_init=1.0)
    return make_head(name)


class FeaturePolicy(nn.Module):
    """evaluate() passes (observations, cooldown); the cooldown is ignored."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features, cooldown=None):
        return self.head(features.float())


class Data:
    def __init__(self, capacity):
        self.x = np.zeros((capacity, 12), np.float32)
        self.y = np.zeros(capacity, np.int64)
        self.size = 0

    def add(self, x, y):
        if self.size + len(y) > len(self.y):
            raise RuntimeError("capacity exhausted; increase --capacity")
        self.x[self.size:self.size + len(y)] = x
        self.y[self.size:self.size + len(y)] = y
        self.size += len(y)

    def tensors(self, device):
        return (torch.as_tensor(self.x[:self.size], device=device),
                torch.as_tensor(self.y[:self.size], device=device))


def collect(pool, data, budget, rng, device, seed_base, max_episode, policy=None,
            teacher_probability=1.0, random_probability=0.0):
    """Exactly budget teacher-labelled states; separate calls use disjoint episodes."""
    starts = UniformStarts()
    obs, infos, lengths = {}, {}, {}
    serial, collected, outcomes, labels = 0, 0, [], Counter()
    xs, ys = [], []

    def begin(i):
        nonlocal serial
        serial += 1
        obs[i], infos[i] = pool.reset([i], [seed_base + serial],
                                      {"start_wp": starts.sample(rng)})[0]
        lengths[i] = 0

    for i in range(pool.n):
        begin(i)
    while collected < budget:
        ids = list(range(min(pool.n, budget - collected)))
        student = None
        if policy is not None:
            with torch.no_grad():
                student = policy(torch.as_tensor(np.stack([obs[i] for i in ids]),
                                                 device=device))[0].argmax(-1).tolist()
        chosen = []
        for j, i in enumerate(ids):
            label = int(infos[i]["teacher"])
            labels[label] += 1
            xs.append(obs[i].copy())
            ys.append(label)
            if rng.random() < random_probability:
                chosen.append(int(rng.integers(N_ACTIONS)))
            elif policy is None or rng.random() < teacher_probability:
                chosen.append(label)
            else:
                chosen.append(student[j])
        for i, (o, _, term, trunc, info) in zip(ids, pool.step(ids, chosen)):
            collected += 1
            lengths[i] += 1
            if term or trunc or lengths[i] >= max_episode:
                outcomes.append(bool(info.get("success", False)))
                begin(i)
            else:
                obs[i], infos[i] = o, info
    data.add(np.stack(xs), np.asarray(ys))
    return {"states": collected, "completed_or_capped_episodes": len(outcomes),
            "successes": sum(outcomes), "labels": dict(sorted(labels.items()))}


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    started = time.monotonic()
    pool = EnvPool(args.num_envs, task="e1m1_canny", teacher=True, skill=args.skill)
    try:
        validation = Data(args.validation_steps)
        stats = collect(pool, validation, args.validation_steps,
                        np.random.default_rng(2_000_000), device, 2_000_000,
                        args.max_episode, random_probability=0.25)
        print(json.dumps({"stage": "validation_collection", **stats}), flush=True)
        val_x, val_y = validation.tensors(device)
        data = Data(args.capacity)
        initial = collect(pool, data, args.initial_steps, np.random.default_rng(3_000_000),
                          device, 3_000_000, args.max_episode)
        print(json.dumps({"stage": "demonstrations", **initial}), flush=True)
        initial_size = data.size
        results = {"protocol": "canny_features_shared_dagger_v1", "args": vars(args),
                   "initial_collection": initial, "runs": []}
        for seed in args.seeds:
            data.size = initial_size
            policies, optimizers = {}, {}
            for name in args.heads:
                torch.manual_seed(seed)
                policies[name] = FeaturePolicy(build_head(name).to(device))
                lr = args.quantum_lr if name.startswith(QUANTUM) else args.lr
                optimizers[name] = torch.optim.Adam(policies[name].head.parameters(),
                                                    lr=lr, eps=1e-5)
            best_states, best_scores, best_iterations, history = {}, {}, {}, []
            for iteration in range(args.iterations + 1):
                if iteration:
                    for name, policy in policies.items():
                        stats = collect(
                            pool, data, args.iter_steps,
                            np.random.default_rng(4_000_000 + seed * 100_000 + iteration),
                            device, 4_000_000 + seed * 100_000 + iteration * 10_000,
                            args.max_episode, policy=policy, teacher_probability=args.beta)
                        print(json.dumps({"stage": "dagger_collection", "seed": seed,
                                          "iteration": iteration, "head": name, **stats}),
                              flush=True)
                x, y = data.tensors(device)
                training = fit_heads(policies, optimizers, x, y, args.head_epochs,
                                     args.batch, seed * 1000 + iteration)
                row = {"seed": seed, "iteration": iteration, "dataset_size": data.size,
                       "training": training,
                       "heldout_actions": heldout_metrics(policies, val_x, val_y, 4096),
                       "selection": {}}
                for name, policy in policies.items():
                    selection = evaluate(policy, pool, device,
                                         range(10_000, 10_000 + args.eval_episodes))
                    score = (selection["successes"] / selection["episodes"]
                             + 0.01 * selection["mean_progress"])
                    row["selection"][name] = selection
                    if score > best_scores.get(name, -1.0):
                        best_scores[name], best_iterations[name] = score, iteration
                        best_states[name] = copy.deepcopy(policy.head.state_dict())
                        torch.save({"head": best_states[name], "name": name, "seed": seed,
                                    "iteration": iteration},
                                   output / f"{name}_seed{seed}_best.pt")
                    print(json.dumps({"stage": "selection", "seed": seed,
                                      "iteration": iteration, "head": name,
                                      "successes": selection["successes"],
                                      "episodes": selection["episodes"],
                                      "mean_progress": selection["mean_progress"],
                                      "heldout_balanced":
                                          row["heldout_actions"][name]["balanced_accuracy"]}),
                          flush=True)
                history.append(row)
                (output / f"history_seed{seed}.json").write_text(json.dumps(history, indent=2))
            run_result = {"seed": seed, "heads": {}}
            for name, policy in policies.items():
                policy.head.load_state_dict(best_states[name])
                final = {"parameters": sum(p.numel() for p in policy.head.parameters()),
                         "best_iteration": best_iterations[name],
                         "selection_score": best_scores[name],
                         "heldout_actions": heldout_metrics({name: policy}, val_x, val_y,
                                                            4096)[name]}
                for mode in ("greedy", "sampled"):
                    final[mode] = evaluate(policy, pool, device,
                                           range(20_000, 20_000 + args.final_episodes),
                                           mode=mode, rng_seed=seed)
                run_result["heads"][name] = final
                print(json.dumps({"stage": "final", "seed": seed, "head": name,
                                  "parameters": final["parameters"],
                                  "best_iteration": final["best_iteration"],
                                  **{m: {k: final[m][k] for k in
                                         ("successes", "episodes", "mean_progress")}
                                     for m in ("greedy", "sampled")}}), flush=True)
            results["runs"].append(run_result)
            if "teacher" not in results:
                results["teacher"] = teacher_eval(
                    pool, range(20_000, 20_000 + args.final_episodes))
                print(json.dumps({"stage": "teacher", **{k: v for k, v in
                                  results["teacher"].items() if k != "rows"}}), flush=True)
            results["seconds"] = time.monotonic() - started
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return results
    finally:
        pool.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--skill", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--heads", choices=HEADS, nargs="+", default=list(HEADS))
    p.add_argument("--validation-steps", type=int, default=10_000)
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
    run(parser().parse_args())
