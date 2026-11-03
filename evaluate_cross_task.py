"""Run E1M1-trained policies on My Way Home and Deadly Corridor with no retraining.

The observation is identical (4 stacked 80x60 screens). The action sets differ, so the
policy's 7 E1M1 logits are restricted to the actions the other task has:
  My Way Home:    FORWARD, TURN_LEFT, TURN_RIGHT  <- E1M1 actions 0, 4, 5
  Deadly Corridor: FORWARD, ATTACK, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT <- 0..5
Greedy and sampled play are scored on the held-out seeds 20000-20099, next to random actions
(and always-FORWARD in the corridor) and the dedicated models' results from the README.

usage: python evaluate_cross_task.py OUT.json name=head:path[:rich] ...
"""

import json
import sys

import torch
from torch import nn

from pixel_nav_env import EnvPool
from pixel_nav_models import PixelNavPolicy
from train_pixel_nav import FINAL_SEED, evaluate, summary

TASKS = {"navigation": [0, 4, 5], "corridor": [0, 1, 2, 3, 4, 5]}


class Restricted(nn.Module):
    """Keeps only the logits of the actions the target task has."""

    def __init__(self, model, allowed):
        super().__init__()
        self.model, self.allowed = model, allowed

    def forward(self, obs):
        logits, value = self.model(obs)
        return logits[..., self.allowed], value


def load(spec, device):
    name, rest = spec.split("=")
    head, path = rest.split(":")[:2]
    rich = rest.endswith(":rich")
    model = PixelNavPolicy(head=head, n_actions=7, aux_pos_weight=5.0, aux_size=20, aux_route=3,
                           rich_critic=rich).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return name, model.eval()


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    models = [load(spec, device) for spec in specs]
    seeds = range(FINAL_SEED, FINAL_SEED + 100)
    results = {}
    for task, allowed in TASKS.items():
        pool = EnvPool(8, task=task)
        try:
            results[task] = {}
            baselines = ("random", "forward") if task == "corridor" else ("random",)
            for mode in baselines:
                result = evaluate(Restricted(models[0][1], allowed),
                                  pool, device, seeds, mode, len(allowed))
                result.pop("rows")
                results[task][mode] = result
                print(f"[{task}] {mode}: {summary(result)}", flush=True)
            for name, model in models:
                wrapped = Restricted(model, allowed)
                for mode in ("greedy", "sampled"):
                    result = evaluate(wrapped, pool, device, seeds, mode, len(allowed))
                    result.pop("rows")
                    results[task][f"{name} {mode}"] = result
                    print(f"[{task}] {name} {summary(result)}", flush=True)
        finally:
            pool.close()
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
