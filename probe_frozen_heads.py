"""Offline head probe on cached frozen-perception features (no game, no CNN).

Every head sees the same 12 frozen angles and the same teacher labels. Training data is
one seed's final aggregated DAgger feature file; held-out data is the episode-disjoint
validation_features.pt from the same run. Reports held-out CE / accuracy / balanced
accuracy per epoch so under-training is visible.

Extra circuit variants beyond train_frozen_perception.py:
  *_frozen : random circuit, theta and input_scale fixed; only the two output gains train.
  *_half   : input_scale initialised at 0.5, so angles stay in [-pi/2, pi/2]. At +-pi the
             RY encoding of x and -x differ only by a global phase, so saturated features
             lose their sign; halving removes that aliasing.
  wide{N}x{L}: same gate set on N qubits and L layers (wire w re-uploads input w mod 12),
             input_scale initialised at 0.5, simulated by wide_circuit.py.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from frozen_perception_models import N_ACTIONS, make_head
from wide_circuit import WideCircuitPolicy

HEADS = ("linear", "small_mlp", "classical", "quantum2", "quantum2_frozen", "quantum2_half",
         "quantum2_half_frozen", "product2_half", "quantum4_half")


def head_name(name):
    if name in HEADS or (name.startswith("wide") and "x" in name):
        return name
    raise argparse.ArgumentTypeError(f"unknown head {name}")


def build(name):
    if name.startswith("wide"):
        n_qubits, n_layers = map(int, name[4:].split("x"))
        return WideCircuitPolicy(n_qubits, n_layers, n_actions=N_ACTIONS)
    base = name.replace("_half", "").replace("_frozen", "")
    head = make_head(base)
    if "_half" in name:
        with torch.no_grad():
            head.input_scale.fill_(0.5)
    if "_frozen" in name:
        head.theta.requires_grad_(False)
        head.input_scale.requires_grad_(False)
    return head


@torch.no_grad()
def evaluate(head, angles, labels, batch):
    logits = torch.cat([head(x)[0] for x in angles.split(batch)])
    correct = logits.argmax(-1) == labels
    per_action = [float(correct[labels == a].float().mean())
                  for a in range(N_ACTIONS) if (labels == a).any()]
    return {"ce": float(F.cross_entropy(logits, labels)),
            "accuracy": float(correct.float().mean()),
            "balanced_accuracy": float(np.mean(per_action))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="frozen_perception_runs/e1m1_gpu_full_20261103")
    p.add_argument("--iteration", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--heads", nargs="+", type=head_name, default=list(HEADS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quantum-lr", type=float, default=3e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_default_device(device)
    run = Path(args.run)
    held = torch.load(run / "validation_features.pt")
    val_x, val_y = held["angles"].to(device), held["labels"].to(device)
    output = Path(args.output or run / "offline_probe.json")
    results = {"args": vars(args), "runs": []}
    for seed in args.seeds:
        data = torch.load(run / f"features_seed{seed}_iteration{args.iteration}.pt")
        assert data["perception_fingerprint"] == held["perception_fingerprint"]
        x, y = data["angles"].to(device), data["labels"].to(device)
        weights = torch.bincount(y, minlength=N_ACTIONS).float().clamp_min(1).rsqrt()
        weights /= weights.mean()
        rng = np.random.default_rng(seed)
        orders = [torch.as_tensor(rng.permutation(len(y)), device=device)
                  for _ in range(args.epochs)]
        for name in args.heads:
            torch.manual_seed(seed)
            head = build(name).to(device)
            params = [q for q in head.parameters() if q.requires_grad]
            lr = args.quantum_lr if name.startswith(("quantum", "product", "wide")) else args.lr
            optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
            curve, started = [evaluate(head, val_x, val_y, args.eval_batch)], time.monotonic()
            for order in orders:
                for idx in order.split(args.batch):
                    loss = F.cross_entropy(head(x[idx])[0], y[idx], weight=weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 0.5)
                    optimizer.step()
                curve.append(evaluate(head, val_x, val_y, args.eval_batch))
            row = {"seed": seed, "head": name, "trainable": sum(q.numel() for q in params),
                   "train": evaluate(head, x, y, args.eval_batch), "heldout_curve": curve,
                   "best_heldout_balanced": max(c["balanced_accuracy"] for c in curve),
                   "final_heldout": curve[-1], "seconds": time.monotonic() - started}
            results["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "heldout_curve"}),
                  flush=True)
            output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
