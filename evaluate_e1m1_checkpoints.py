"""Evaluate E1M1 checkpoints on the same many held-out real-start seeds (greedy and sampled).

usage: python evaluate_e1m1_checkpoints.py OUT.json FIRST_SEED N_SEEDS name=path[:rich] ...
(`:rich` marks checkpoints trained with --rich-critic.)
"""

import json
import sys

import numpy as np
import torch

from pixel_nav_env import EnvPool
from pixel_nav_models import PixelNavPolicy
from train_pixel_nav import evaluate, summary

def main():
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    pool = EnvPool(8, task="e1m1")
    seeds = range(first, first + count)
    results = {}
    try:
        for spec in sys.argv[4:]:
            name, path = spec.split("=")
            rich = path.endswith(":rich")
            path = path.removesuffix(":rich")
            model = PixelNavPolicy(head="classical", n_actions=7, aux_pos_weight=5.0, aux_size=20,
                                   aux_route=3, rich_critic=rich).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
            model.eval()
            results[name] = {}
            for mode in ("greedy", "sampled"):
                result = evaluate(model, pool, device, seeds, mode, 7, rng_seed=0)
                rows = result.pop("rows")
                results[name][mode] = result
                print(f"{name} {summary(result)}", flush=True)
    finally:
        pool.close()
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
