"""Summarize completed frozen-perception runs without pooling away policy seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(results):
    runs = results["runs"]
    names = list(runs[0]["heads"])
    report = {"policy_seeds": [run["seed"] for run in runs],
              "perception_unchanged": results["perception_unchanged"],
              "seconds": results["seconds"], "heads": {}, "paired_differences": {}}
    if "args" in results:
        report["budget"] = {key: results["args"][key] for key in (
            "perception_steps", "validation_steps", "perception_epochs", "initial_steps",
            "iter_steps", "iterations", "head_epochs", "eval_episodes", "final_episodes")}
    if "geometry_validation" in results:
        report["geometry_validation"] = results["geometry_validation"]
    report["scope"] = ("full" if len(runs) >= 3 and all(
        run["heads"][names[0]]["greedy"]["episodes"] >= 100 for run in runs) else "pilot")
    if "teacher" in results:
        report["teacher"] = results["teacher"]
    report["interpretation"] = (
        "Descriptive comparison under the recorded data budget and learning rates; "
        "bootstrap intervals do not account for perception-seed variation."
        if report["scope"] == "full" else
        "Small-budget pilot; success rates and intervals do not establish a head advantage.")
    for name in names:
        entry = {"parameters": runs[0]["heads"][name]["parameters"]}
        for mode in ("greedy", "sampled"):
            outcomes = [run["heads"][name][mode] for run in runs]
            rates = [outcome["successes"] / outcome["episodes"] for outcome in outcomes]
            entry[mode] = {"successes_by_seed": [x["successes"] for x in outcomes],
                           "episodes_by_seed": [x["episodes"] for x in outcomes],
                           "mean_success_rate": float(np.mean(rates)),
                           "seed_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                           "mean_progress": float(np.mean([x["mean_progress"] for x in outcomes]))}
        entry["heldout_action_accuracy"] = float(np.mean([
            run["heads"][name]["heldout_actions"]["accuracy"] for run in runs]))
        entry["heldout_balanced_accuracy"] = float(np.mean([
            run["heads"][name]["heldout_actions"]["balanced_accuracy"] for run in runs]))
        report["heads"][name] = entry
    # Match episodes by their actual seed, never by row position. Bootstrap policy and
    # environment seeds, retaining the pairing between heads within each draw.
    for quantum in (name for name in names if name.startswith("quantum")):
        for baseline in (name for name in names if name != quantum):
            comparisons = {}
            for mode in ("greedy", "sampled"):
                differences = []
                for run in runs:
                    left = {r["seed"]: int(r["success"]) for r in
                            run["heads"][quantum][mode]["rows"]}
                    right = {r["seed"]: int(r["success"]) for r in
                             run["heads"][baseline][mode]["rows"]}
                    if set(left) != set(right):
                        raise ValueError("Cannot pair different evaluation seeds")
                    differences.append([left[s] - right[s] for s in sorted(left)])
                delta = np.asarray(differences)
                comparison = {"mean_success_rate_difference": float(delta.mean()),
                              "paired_bootstrap_95_percent": None}
                if delta.shape[0] < 3 or delta.shape[1] < 20:
                    comparison["interval_note"] = "Too few policy or episode seeds for an interval"
                    comparisons[mode] = comparison
                    continue
                if np.all(delta == delta.flat[0]):
                    comparison["interval_note"] = "No paired variation; empirical bootstrap is degenerate"
                    comparisons[mode] = comparison
                    continue
                rng = np.random.default_rng(12345)
                draws = []
                for _ in range(2000):
                    policy_ids = rng.integers(delta.shape[0], size=delta.shape[0])
                    episode_ids = rng.integers(delta.shape[1], size=delta.shape[1])
                    draws.append(delta[np.ix_(policy_ids, episode_ids)].mean())
                comparison["paired_bootstrap_95_percent"] = np.quantile(
                    draws, [0.025, 0.975]).tolist()
                comparisons[mode] = comparison
            report["paired_differences"][f"{quantum}_minus_{baseline}"] = comparisons
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(json.loads(args.results.read_text()))
    audit_path = args.results.parent / "audit.json"
    if audit_path.exists():
        report["audit"] = json.loads(audit_path.read_text())
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
