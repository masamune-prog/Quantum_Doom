"""Summarize seen-task learning and held-out-map transfer separately."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(results):
    args, runs = results["args"], results["runs"]
    names = list(runs[0]["policies"])
    complete = len(runs) == len(args["seeds"])
    report = {"completed_policy_seeds": [run["seed"] for run in runs],
              "all_requested_seeds_complete": complete, "args": args,
              "train_maps": results["train_maps"], "held_out_map_variants": results["test_maps"],
              "perception_unchanged": results["perception_unchanged"],
              "seconds": results["seconds"], "policies": {}, "multi_minus_single": {}}
    report["scope"] = ("full" if complete and len(runs) >= 3 and
                       args["final_episodes"] >= 100 and args["eval_max_steps"] >= 1575 else "pilot")
    for name in names:
        report["policies"][name] = {}
        for task in results["train_maps"] + results["test_maps"]:
            outcomes = [run["policies"][name][task] for run in runs]
            rates = [r["successes"] / r["episodes"] for r in outcomes]
            report["policies"][name][task] = {
                "successes_by_seed": [r["successes"] for r in outcomes],
                "episodes_by_seed": [r["episodes"] for r in outcomes],
                "capped_episodes_by_seed": [r["capped_episodes"] for r in outcomes],
                "mean_success_rate": float(np.mean(rates)),
                "seed_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else None}
            total_probes = sum(r.get("perception_probe_states", 0) for r in outcomes)
            if total_probes:
                paired = {}
                for condition in args["conditions"]:
                    sums = {}
                    for outcome in outcomes:
                        for key, loss in outcome.get("paired_perception_losses", {}).get(condition, {}).items():
                            sums[key] = sums.get(key, 0) + loss * outcome["perception_probe_states"]
                    paired[condition] = {key: value / total_probes for key, value in sums.items()}
                report["policies"][name][task]["paired_perception_losses"] = paired
                report["policies"][name][task]["perception_probe_states"] = total_probes
    for kind in args["models"]:
        left, right = f"multi_{kind}", f"single_{kind}"
        if left not in names or right not in names:
            continue
        report["multi_minus_single"][kind] = {}
        for task in results["train_maps"] + results["test_maps"]:
            deltas = []
            for run in runs:
                a = {r["seed"]: int(r["success"]) for r in run["policies"][left][task]["rows"]}
                b = {r["seed"]: int(r["success"]) for r in run["policies"][right][task]["rows"]}
                if set(a) != set(b):
                    raise ValueError("Evaluation seeds do not match")
                deltas.append([a[s] - b[s] for s in sorted(a)])
            report["multi_minus_single"][kind][task] = float(np.mean(deltas))
    if "teacher_reference" in results:
        report["teacher_reference"] = {task: {k: r[k] for k in
            ("successes", "episodes", "capped_episodes")} for task, r in
            results["teacher_reference"].items()}
    report["interpretation"] = (
        "E1M2 is one held-out layout evaluated with and without granted keys. "
        "The keys variant controls for a mechanic absent from the training tasks. "
        "Pilot or capped evaluations validate the pipeline and do not establish transfer. "
        "Results are conditional on one perception initialization and the recorded teachers.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(json.loads(args.results.read_text())), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
