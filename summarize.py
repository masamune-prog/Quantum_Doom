"""Print the report's result tables (Tables 2-5 and the Section 4.4 numbers) from run outputs.

usage: python summarize.py [ROOT]        ROOT = runs (default) or pretrained

Reads the results.json / train.log files written by scripts/*.sh. Missing runs are shown as "-",
so this can be run at any point while replicating.
"""

import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"


def load(path):
    path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def first_reaching(run, threshold):
    """Decisions at the first selection evaluation scoring >= threshold of 32 (from train.log)."""
    path = os.path.join(ROOT, run, "train.log")
    if not os.path.exists(path):
        return "-"
    with open(path) as handle:
        for line in handle:
            m = re.match(r"eval update=\d+ decisions=(\d+) selection greedy (\d+)/32", line)
            if m and int(m.group(2)) >= threshold:
                return f"{int(m.group(1)) / 1e6:.2f}M"
    return "not reached"


def hms(seconds):
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def g(result, mode, key="successes"):
    return "-" if result is None or mode not in result else result[mode][key]


def table(title, header, rows):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


# Table 2 ----------------------------------------------------------------------------------
rows = []
for task, root, threshold in (("My Way Home", "pixel_nav_runs", 28),
                              ("Deadly Corridor", "pixel_corridor_runs", 32)):
    for head in ("classical", "linear", "quantum"):
        for seed in (0, 1, 2):
            run = f"{root}/{head}_seed{seed}"
            r = load(f"{run}/results.json")
            if r is None:
                continue
            rows.append((task, head, seed, r["head_parameters"], g(r, "greedy"), g(r, "sampled"),
                         first_reaching(run, threshold), f"{r['training_decisions'] / 1e6:.2f}M",
                         hms(r["training_seconds"])))
    base = load(f"{root}/classical_seed0/results.json")
    if base and "random" in base:
        rows.append((task, "random actions", "", "", g(base, "random"), "", "", "", ""))
    if base and "forward" in base:
        rows.append((task, "always FORWARD", "", "", g(base, "forward"), "", "", "", ""))
table("Table 2: feasibility pilots (PPO), exits out of 100 held-out seeds",
      ("Scenario", "Head", "Seed", "Head params", "Greedy", "Sampled", "Decisions to threshold",
       "Decisions trained", "Train time"), rows)
print("\nThreshold: selection score >= 28/32 (maze) or 32/32 (corridor). Train time excludes "
      "the final evaluation; for the corridor circuit it is the --resume job only (the first "
      "job used another 19,500 s).")

# Table 3 ----------------------------------------------------------------------------------
rows = []
for name, run in (("1 reverse curriculum", "classical_warm_reverse_seed0"),
                  ("2 uniform starts", "classical_warm_uniform_noroute_seed0"),
                  ("3 uniform + route aux", "classical_warm_uniform_route_seed0"),
                  ("4 fine-tune (rich critic)", "finetune_classical_richcritic_seed0")):
    r = load(f"pixel_e1m1_runs/{run}/results.json")
    if r is None:
        rows.append((name, "-", "-", "-", "-"))
        continue
    rows.append((name, g(r, "greedy"), g(r, "sampled"),
                 f"{r['greedy']['mean_progress']:.2f} / {r['sampled']['mean_progress']:.2f}",
                 f"{r['training_decisions'] / 1e6:.1f}M"))
r = load("pixel_e1m1_runs/classical_warm_reverse_seed0/results.json")
if r and "random" in r:
    rows.append(("random actions", g(r, "random"), "",
                 f"{r['random']['mean_progress']:.2f}", ""))
    rows.append(("always FORWARD", g(r, "forward"), "",
                 f"{r['forward']['mean_progress']:.2f}", ""))
table("Table 3: PPO on E1M1 (classical head, real start, 100 held-out seeds)",
      ("Attempt", "Exits greedy", "Exits sampled", "Progress greedy / sampled", "Decisions"), rows)
c = load("pixel_e1m1_runs/compare_imitation_vs_finetune.json")
if c:
    print("\nAttempt 4 on 300 fresh seeds (30000-30299): "
          + ", ".join(f"{k} greedy {c[k]['greedy']['successes']} sampled {c[k]['sampled']['successes']}"
                      for k in ("imitation", "finetuned")))

# Table 4 ----------------------------------------------------------------------------------
rows, teacher = [], None
ppo_runs = [load(f"pixel_e1m1_runs/{run}/results.json") for run in
            ("classical_warm_reverse_seed0", "classical_warm_uniform_route_seed0")]
ppo = [r for r in ppo_runs if r]
for attempt, r in zip((1, 3), ppo_runs):
    if r:
        rows.append((f"PPO from scratch, attempt {attempt}", r["head_parameters"], g(r, "greedy"),
                     g(r, "sampled"), f"{r['greedy']['mean_dead']:.0%}", ""))
for name, run, params in (("BC only, 200K teacher states", "imitation_classical_bc200k_seed0", 1096),
                          ("BC only, 680K teacher states", "imitation_classical_bc680k_seed0", 1096),
                          ("DAgger, classical MLP head", "imitation_classical_v2_seed0", 1096),
                          ("DAgger, linear head", "imitation_linear_v2_seed0", 104),
                          ("DAgger, 12-qubit circuit head", "imitation_quantum_v2_seed0", 74)):
    r = load(f"pixel_e1m1_runs/{run}/results.json")
    if r is None:
        rows.append((name, params, "-", "-", "-", "-"))
        continue
    last = r["history"][-1]
    sel = [int(re.search(r"greedy (\d+)/", h["selection"]).group(1)) for h in r["history"]]
    seconds = r["history"][-1]["seconds"]
    rows.append((name, params, g(r, "greedy"), g(r, "sampled"), f"{r['greedy']['mean_dead']:.0%}",
                 f"acc {last['accuracy']:.0%}, ce {last['ce']:.2f}, selection {min(sel)}-{max(sel)}/32, "
                 f"{hms(seconds)}"))
    teacher = teacher or r.get("teacher")
if teacher:
    rows.append(("Scripted teacher (private state)", "-", teacher["successes"], teacher["successes"],
                 f"{teacher['mean_dead']:.0%}", ""))
if ppo and "random" in ppo[0]:
    rows.append(("Random actions", "-", g(ppo[0], "random"), "-",
                 f"{ppo[0]['random']['mean_dead']:.0%}", ""))
table("Table 4: E1M1, skill 3, real start, 100 held-out seeds",
      ("Method", "Head params", "Exits greedy", "Exits sampled", "Died greedy",
       "Final-round fit, selection range, train time (Sec. 4.4)"), rows)

# Table 5 ----------------------------------------------------------------------------------
rob = load("pixel_e1m1_runs/robustness_results.json")
if rob:
    conditions = (("Unmodified screens, real start", "none"), ("HUD masked", "hud_masked"),
                  ("Newest frame only (x4)", "static_frame"), ("Brightness x0.6", "bright_0.6"),
                  ("Brightness x1.4", "bright_1.4"), ("Gaussian noise std 25", "noise"),
                  ("Random off-route start", "offroute"))
    rows = []
    for label, key in conditions:
        cells = [rob.get(f"{h} {key} greedy", {}).get("exits", "") for h in ("classical", "linear", "circuit")]
        rows.append((label, *cells, rob.get(f"teacher {key}", {}).get("exits", "")))
    table("Table 5: memorisation tests, exits out of 100 (greedy), episodes cut at 800 decisions",
          ("Condition", "Classical", "Linear", "Circuit", "Teacher"), rows)

# Transfer (Section 4.4) ---------------------------------------------------------------------
cross = load("pixel_e1m1_runs/cross_task_results.json")
if cross:
    rows = []
    for task, results in cross.items():
        for name, r in results.items():
            died = f"{r['mean_dead']:.0%}" if "mean_dead" in r else ""
            rows.append((task, name, r["successes"], died))
    table("E1M1 students zero-shot on the pilot scenarios (100 seeds)",
          ("Scenario", "Policy", "Exits", "Died"), rows)
spec = load("pixel_e1m1_runs/specialists_on_e1m1.json")
if spec:
    rows = [(name, r.get("successes", r.get("exits")), f"{r.get('mean_progress', float('nan')):.2f}")
            for name, r in spec.items() if isinstance(r, dict)]
    table("Pilot-trained models zero-shot on E1M1 (100 seeds)", ("Policy", "Exits", "Progress"), rows)
elif os.path.exists(os.path.join(ROOT, "pixel_e1m1_runs/specialists_on_e1m1.log")):
    print(f"\nPilot models on E1M1: see {ROOT}/pixel_e1m1_runs/specialists_on_e1m1.log")
