"""Audit a saved perception checkpoint and the teacher on the benchmark's test seeds.

Geometry recollection uses the original seeds/actions and compares against constant
training-set predictions. Useful for auditing pilot runs or an interrupted head run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frozen_perception_models import GeometryPerception, fingerprint
from pixel_nav_env import EnvPool
from train_e1m1_imitation import teacher_eval
from train_frozen_perception import (Replay, collect, constant_geometry_metrics,
                                     geometry_metrics)


def audit(run, device, recollect_geometry):
    args = argparse.Namespace(**json.loads((run / "config.json").read_text()))
    torch.set_num_threads(args.threads)
    torch.set_default_device(device)
    checkpoint = torch.load(run / "perception.pt", map_location=device, weights_only=False)
    perception = GeometryPerception().to(device)
    perception.load_state_dict(checkpoint["model"])
    perception.freeze()
    if fingerprint(perception) != checkpoint["fingerprint"]:
        raise AssertionError("Perception checkpoint fingerprint mismatch")
    pool = EnvPool(args.num_envs, task="e1m1", teacher=True, skill=args.skill)
    try:
        report = {"perception_fingerprint_verified": True,
                  "teacher": teacher_eval(pool, range(20_000, 20_000 + args.final_episodes))}
        if recollect_geometry:
            training, validation = Replay(args.perception_steps), Replay(args.validation_steps)
            for dataset, budget, seed_base in ((training, args.perception_steps, 1_000_000),
                                               (validation, args.validation_steps, 2_000_000)):
                collect(pool, dataset, budget,
                        np.random.default_rng(seed_base + args.perception_seed),
                        device, seed_base, args.max_episode, random_probability=0.25)
            report["geometry_validation"] = geometry_metrics(perception, validation,
                                                              args.batch, device)
            report["constant_geometry_validation"] = constant_geometry_metrics(training,
                                                                                validation)
            saved = checkpoint["validation"]
            report["validation_reproduced"] = all(
                np.isclose(value, saved[key], atol=1e-6)
                for key, value in report["geometry_validation"].items())
        return report
    finally:
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--recollect-geometry", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.run, torch.device(args.device), args.recollect_geometry)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
