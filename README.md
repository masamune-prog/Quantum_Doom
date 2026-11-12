# Training Quantum Neural Networks to play Doom: replication package

This repository has the code for the SC4055 report. The repository root reproduces the main
experiment (Sections 1–4): a CNN encoder followed by a 12-qubit data-re-uploading circuit
(74 parameters), compared with a classical MLP head and a linear readout on ViZDoom.
[`privileged_navigation/`](privileged_navigation/) holds the complementary experiment of
Section 5 (see [below](#section-5-hybrid-qnn-with-privileged-information)).

It covers:

| Report item | What it shows | Script |
|---|---|---|
| Table 2 | Feasibility pilots: PPO on My Way Home and Deadly Corridor, 3 heads | `01_*.sh`, `02_*.sh` |
| Table 3 | PPO on E1M1 does not learn the level (attempts 1–4) | `03_e1m1_ppo.sh`, `06_*.sh` |
| §3.2 | Scripted teacher exit rate (84/100 at skill 3, 29/30 at skill 1) | `04_e1m1_teacher.sh` |
| Table 4, §4.4 | Behaviour cloning and DAgger students on E1M1, 3 heads | `05_e1m1_imitation.sh` |
| Table 5 | Memorisation and perturbation tests of the E1M1 students | `07_e1m1_memorisation.sh` |
| §4.4 | Zero-shot transfer between E1M1 and the pilot scenarios | `08_transfer.sh` |
| Figures 1–2 | RL process diagram and circuit drawing; gameplay videos | `09_figures_and_videos.sh` |

All circuits are simulated exactly (PennyLane `default.qubit`, statevector, no shots, no noise).
Nothing runs on quantum hardware.

---

### Section 5: hybrid QNN with privileged information

The complementary experiment (report Section 5, Tables 6–7) was developed by a teammate in
**[atMiaobo/quantum-playing-doom](https://github.com/atMiaobo/quantum-playing-doom)**. It uses a
six-qubit circuit with a classical residual pathway, privileged game-state features and a
geometry-aware route planner. It is included in [`privileged_navigation/`](privileged_navigation/)
with its full commit history (imported with `git subtree` from commit `ea93380`). It is a
separate Python package with its own dependencies (a custom PyTorch statevector simulator, no
PennyLane) and a container/Nix launcher. See
[`privileged_navigation/README.md`](privileged_navigation/README.md) for setup and commands, and
its `docs/` for the Deadly Corridor, full-game and MAP01 navigation results.

To pull later updates from the teammate's repository:

```bash
git subtree pull --prefix=privileged_navigation https://github.com/atMiaobo/quantum-playing-doom.git main
```

The upstream repository currently documents the Deadly Corridor and Freedoom MAP01 results. The
E1M1–E1M3 numbers in Tables 6–7 are not yet documented there.

---

## 1. Setup

Requirements: Linux, Python 3.12, and an NVIDIA GPU for the circuit runs. The report used one
V100 with 8 ViZDoom processes on CPU cores. Classical heads also train on CPU, but much more slowly.

```bash
git clone https://github.com/masamune-prog/Quantum_Doom.git && cd Quantum_Doom
python3.12 -m venv .venv          # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install pennylane==0.45.1 vizdoom==1.3.2 gymnasium==1.4.0 opencv-python matplotlib imageio imageio-ffmpeg
# exact versions of every package used for the report: requirements.txt
```

ViZDoom ships with the My Way Home and Deadly Corridor scenarios. E1M1 is read from the
shareware `wads/doom1.wad` (Doom v1.9, MD5 `f0cefca49926d00903cf57551d901abe`), which is
included.

**Check the install (a few minutes):**

```bash
bash scripts/00_smoke_test.sh
```

This checks the E1M1 map, route, distance field and teacher. It then runs a few PPO updates for
every task/head pair and one tiny DAgger round for the classical and circuit heads. The scores
it prints are meaningless. It only checks that every code path runs.

All scripts accept these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python` | interpreter to use |
| `OUT` | `runs` | where outputs go (`$OUT/pixel_nav_runs/...`, etc.) |
| `CKPT` | auto | checkpoints for evaluation: `runs` (yours) or `pretrained` (the report's). Auto uses yours when present, otherwise `pretrained`. |
| `CUDA_VISIBLE_DEVICES` | – | choose the GPU, e.g. `CUDA_VISIBLE_DEVICES=3` |

---

## 2. Fast path: check the reported numbers without training

`pretrained/` holds the best checkpoint, `results.json` and training log of every run behind
Tables 2–5 (about 50 MB). To print all the report's tables from them:

```bash
python summarize.py pretrained
```

To re-run the evaluations on those checkpoints yourself (no training; GPU optional; roughly
10 min to 2 h each):

```bash
CKPT=pretrained bash scripts/07_e1m1_memorisation.sh       # Table 5
CKPT=pretrained bash scripts/08_transfer.sh                # §4.4 transfer
CKPT=pretrained bash scripts/06_e1m1_finetune_300_seeds.sh # Table 3, attempt 4 (300 seeds)
python summarize.py runs
```

Evaluation is deterministic given a checkpoint and seed list, up to GPU floating-point noise, so
these should match the report closely.

---

## 3. Full replication from scratch

Run the steps in this order. **The E1M1 runs warm-start their CNN encoder from the Deadly
Corridor classical seed-0 run**, so do step 2 first, or set
`INIT_ENCODER=pretrained/pixel_corridor_runs/classical_seed0/best.pt`.
Times are for one V100.

```bash
# 1. Table 2, My Way Home (3M decisions per run; classical/linear ~31 min, circuit 5 h 25 min budget)
HEAD=classical bash scripts/01_pilot_my_way_home.sh       # seeds 0 1 2; seed 0 also runs random actions
HEAD=linear    bash scripts/01_pilot_my_way_home.sh
HEAD=quantum SEEDS=0 bash scripts/01_pilot_my_way_home.sh # one seed per call is convenient: ~5.4 h each
HEAD=quantum SEEDS=1 bash scripts/01_pilot_my_way_home.sh
HEAD=quantum SEEDS=2 bash scripts/01_pilot_my_way_home.sh

# 2. Table 2, Deadly Corridor (skill 3)
HEAD=classical bash scripts/02_pilot_deadly_corridor.sh   # seed 0 also runs random / always-forward
HEAD=linear    bash scripts/02_pilot_deadly_corridor.sh
HEAD=quantum   bash scripts/02_pilot_deadly_corridor.sh   # seed 0: 19,500 s, then --resume for 19,500 s (~11 h)

# 3. §3.2 teacher (CPU only)
bash scripts/04_e1m1_teacher.sh

# 4. Table 4, E1M1 imitation (each run also evaluates the teacher on the same 100 seeds)
VARIANT=classical bash scripts/05_e1m1_imitation.sh       # DAgger, ~25 min
VARIANT=linear    bash scripts/05_e1m1_imitation.sh       # DAgger, ~25 min
VARIANT=quantum   bash scripts/05_e1m1_imitation.sh       # DAgger, ~1 h 53 min
VARIANT=bc200k    bash scripts/05_e1m1_imitation.sh       # behaviour cloning only
VARIANT=bc680k    bash scripts/05_e1m1_imitation.sh

# 5. Table 3, PPO on E1M1 (classical head; ~2.2 h per 10M-decision run)
ATTEMPT=1 bash scripts/03_e1m1_ppo.sh                     # reverse curriculum, 10M decisions
ATTEMPT=2 bash scripts/03_e1m1_ppo.sh                     # uniform starts (cancelled at ~3M in the report)
ATTEMPT=3 bash scripts/03_e1m1_ppo.sh                     # + route auxiliary targets, 10M decisions
for a in 4naive 4kl1 4kl01 4kill8 4rich; do ATTEMPT=$a bash scripts/03_e1m1_ppo.sh; done  # fine-tune the DAgger student
bash scripts/06_e1m1_finetune_300_seeds.sh                # imitation vs fine-tuned (4rich), 300 fresh seeds

# 6. Table 5 and §4.4
bash scripts/07_e1m1_memorisation.sh
bash scripts/08_transfer.sh

# 7. Figures and videos
bash scripts/09_figures_and_videos.sh

# All tables from your runs
python summarize.py runs
```

Each run writes `best.pt` (the checkpoint chosen on the selection seeds), `latest.pt`,
`results.json` (final 100-seed evaluation, with per-episode rows) and `train.log` into its
directory.

### What to expect

Results come from a single training seed per configuration, and GPU nondeterminism changes the
training trajectory. Expect your numbers to vary roughly as much as the report's seeds vary
among themselves, not to match exactly:

* **Pilots:** all heads reach ≥ 96/100 greedy exits. The circuit needs more decisions and about
  10× the wall time.
* **E1M1 PPO:** 0 exits in every attempt, with route progress around 0.15–0.25.
* **E1M1 imitation:** DAgger students reach about 70–77 exits; BC alone reaches about 48–58 greedy.
  Selection scores fluctuate by several episodes between DAgger rounds, so with one seed the
  heads cannot be ranked.
* **Table 5:** HUD masking gives 0 exits for every student, and random off-route starts give
  about 25–30 exits against the teacher's 92.

---

## 4. Protocol (shared by all experiments)

| Item | Setting |
|---|---|
| Observation | 160×120 RGB with HUD, downsampled to 80×60, last 4 frames stacked: (12, 60, 80) uint8 |
| Decision | one action held for 4 game tics |
| Encoder | Conv 8×8/4 (32), Conv 4×4/2 (64), Conv 3×3/1 (64), Linear 1536→256; bottleneck z = π·tanh(Linear 256→12) |
| Heads | classical MLP actor 12→32→A + critic 12→32→1 (964/1,063/1,096 params); linear 12→A+1 (52/91/104); 12-qubit circuit, 2 re-uploading layers (74) |
| Auxiliary losses (training only) | per-sector depth (MSE), per-sector monster/goal visibility (BCE), door/switch ahead; on E1M1 also route direction and progress |
| PPO | 8 envs × 128 steps, 4 epochs, minibatch 256, γ 0.99, λ 0.95, clip 0.1, entropy 0.01, aux 0.5, lr 2.5e-4, circuit lr 3e-3 |
| DAgger | 200K teacher states, then 8 rounds × 60K student-driven states (teacher mix 0.3, then 0); weighted CE with w_a ∝ f_a^-0.75; episodes start at the real start (25%) or a non-door waypoint; cut at 600 decisions |
| Seeds | training seeds ≥ 1,000,000; checkpoint selection 10000–10031 (32 episodes); final test 20000–20099 (100 episodes), greedy and sampled; Table 5 uses 40000–40099; attempt 4 re-test uses 30000–30299 |
| E1M1 | skill 3, real start for every evaluation, 1,575 decisions per episode, 7 actions (forward, attack, strafe L/R, turn L/R, use) |

Privileged game state (position, depth and label buffers, kill counter, walking-distance field)
is used only for rewards, auxiliary targets and the teacher's labels. It is never a policy input.

---

## 5. Files

| File | Purpose |
|---|---|
| `train_pixel_nav.py` | PPO training, checkpoint selection, final evaluation, resume (all three tasks); E1M1 curricula, fine-tuning options |
| `train_e1m1_imitation.py` | Behaviour cloning / DAgger of the scripted teacher, then evaluation |
| `pixel_nav_models.py` | `PixelNavPolicy`: CNN encoder, bottleneck, auxiliary head, head selection |
| `e1m1_quantum_model.py` | `QuantumE1M1Policy`, the 12-qubit circuit head (PennyLane) |
| `human_nav_models.py` | `ClassicalNavPolicy`, the classical MLP head |
| `pixel_nav_env.py`, `my_way_home_quantum.cfg` | My Way Home env and the subprocess `EnvPool` |
| `pixel_corridor_env.py`, `pixel_corridor.cfg` | Deadly Corridor env (skill 3) |
| `pixel_e1m1_env.py`, `pixel_e1m1.cfg` | E1M1 env: rewards (Eq. 10), auxiliary targets, warp starts |
| `human_nav_env.py`, `reward_shaping.py`, `ppo_common.py` | Shared env base class, potential-based progress reward, GAE |
| `e1m1_map.py`, `e1m1_planner.py`, `e1m1_waypoints.json` | WAD parser and Dijkstra route planner (22 waypoints, ~4,972 units) |
| `e1m1_distance.py`, `e1m1_distance.npz` | Walking-cost field to the exit, used by the progress reward |
| `e1m1_env.py`, `e1m1_quantum.cfg` | Door-waypoint list and angle helper used by the E1M1 env |
| `e1m1_teacher.py`, `test_e1m1_teacher.py` | Scripted teacher (reads private state) and its evaluation |
| `diagnose_pixel_e1m1.py` | E1M1 checks: distance field, warps, scripted follower, baselines |
| `evaluate_e1m1_robustness.py` | Table 5 perturbation tests |
| `evaluate_e1m1_checkpoints.py` | Many-seed comparison of E1M1 checkpoints |
| `evaluate_cross_task.py`, `evaluate_specialists_on_e1m1.py` | Zero-shot transfer in both directions |
| `record_gameplay.py`, `draw_circuit.py`, `draw_rl_diagram.py` | Videos and figures |
| `summarize.py` | Prints Tables 2–5 from `runs/` or `pretrained/` |
| `scripts/` | One script per experiment (above) |
| `pretrained/` | Checkpoints, results and logs behind the report's numbers |
| `privileged_navigation/` | Section 5 experiment, from [atMiaobo/quantum-playing-doom](https://github.com/atMiaobo/quantum-playing-doom) |

`e1m1_waypoints.json` and `e1m1_distance.npz` are included. To regenerate them, run
`python e1m1_planner.py && python e1m1_distance.py`.

---

## 6. Notes and known differences

* **Time budgets depend on hardware.** The circuit runs are capped at 19,500 s
  (`--max-seconds`, from the cluster's 6-hour job limit). On a V100 the My Way Home circuit ran
  out of time just short of 3M decisions. A faster GPU trains further within the same budget.
  Set `MAX_SECONDS` to change the cap.
* **PPO attempts 1 and 2** ran before the route-direction/progress auxiliary targets were added.
  `--no-route-aux` reproduces that model (aux head 256→17, 496,389 parameters).
* **Attempt 4 flags** were reconstructed from the run notes and logs: naive (encoder lr 5e-5);
  frozen encoder with KL anchor 1.0 and head lr 5e-5; KL anchor 0.1 with head lr 2e-4 and entropy
  0.01; the same with kill reward 8; and the same with a critic on the 256 encoder features
  (`4rich`, the run evaluated on 300 seeds).
* **Circuit epochs under DAgger:** the circuit used 3 epochs in round 0 and 2 per round after,
  against 6 and 3 for the classical heads, to fit its time budget. This makes the comparison
  slightly unfair to the circuit.
* **Corridor classical seed 0** first scored 32/32 on the selection seeds at 2.30M decisions
  (`pretrained/pixel_corridor_runs/classical_seed0/train.log`). Table 2 lists 2.56M.
* The pilot-models-on-E1M1 job hit its time limit before the circuit's oracle-gated mode
  finished. The ten settings that completed are in
  `pretrained/pixel_e1m1_runs/specialists_on_e1m1.log`.
