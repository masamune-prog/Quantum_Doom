# Quantum Doom

Can a variational quantum circuit act as the policy of a Doom-playing agent? This project
trains PPO agents in ViZDoom whose action and value outputs come from a **simulated
12-qubit circuit**, and compares them with matched classical heads. The long-term target is
completing the original E1M1 (Hangar) without dying.

The aim is feasibility, not quantum advantage. All circuits are simulated with PennyLane;
nothing runs on quantum hardware.

**Current approach.** The policy sees only what a human player sees: the rendered screen,
including the HUD. A classical CNN compresses the last four frames to 12 numbers, and a small
head turns those into action probabilities and a value estimate. The head is either a
12-qubit circuit (74 parameters), a classical MLP (~1k parameters) or a single linear layer
(52–91 parameters). Privileged game information such as position, depth, object labels and
kills is used only for the training reward and auxiliary losses, never as policy input.

## Results at a glance

Each table reports episodes that reached the goal, on 100 spawn seeds never seen in
training (20000–20099). "Greedy" takes the most likely action and "sampled" draws from the
policy. A *decision* is one policy action, held for 4 game tics. One PPO update uses
8 environments × 128 decisions = 1,024 decisions.

**My Way Home** (maze, 3 actions; find the vest):

| Head (same CNN encoder) | Head params | Greedy, seeds 0/1/2 | Sampled | Decisions to ≥ 28/32 selection | Wall time |
|---|---:|---|---|---|---:|
| Classical MLP | 964 | 99 / 100 / 100 | 100 ×3 | 1.28M / 0.77M / 1.28M | ~31 min |
| Linear readout | 52 | 99 / 98 / 96 | 100 ×3 | 0.77M ×3 | ~31 min |
| **12-qubit circuit** | **74** | **96 / 99 / 97** | **100 ×3** | 1.79M / 2.56M / 1.79M | 5h 25m |
| Random actions | — | 10 | — | — | — |

**Deadly Corridor**, skill 3 (6 actions; fight 6 enemies and reach the vest):

| Run | Head params | First 32/32 selection | Greedy | Sampled | Kills (of 6) | Died | HP left | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical MLP seed 0 | 1,063 | 2.56M | 99 | 98 | 5.99 | 1% | 90 | 32 min |
| Classical MLP seed 1 | 1,063 | 1.79M | 99 | 93 | 5.91 | 0% | 83 | 31 min |
| Classical MLP seed 2 | 1,063 | 2.82M | 99 | 94 | 5.78 | 1% | 85 | 31 min |
| Linear readout seed 0 | 91 | not reached (30/32 at 3.0M) | 98 | 95 | 4.98 | 2% | 61 | 31 min |
| Linear readout seed 1 | 91 | 2.82M | 100 | 97 | 6.00 | 0% | 98 | 30 min |
| **12-qubit circuit seed 0** | **74** | **4.35M** | **98** | **99** | **5.99** | **0%** | **96** | ~11 h |
| Always FORWARD | — | — | 27 | — | 0.49 | 73% | — | — |
| Random actions | — | — | 0 | — | 0.91 | 100% | — | — |

In this table, Kills, Died and HP left are for greedy play. Linear seed 2 is still training
(job 71605).

**What this shows so far**

- **Both tasks:** a circuit head trained end to end with a CNN encoder solves both tasks
  from the screen alone, and matches the classical heads in final success.
- **Learning speed:** the circuit needed more decisions to learn. On My Way Home it took
  2–3× as many as the linear readout. On the corridor it took 1.5–2.4× as many as the
  classical seeds. The linear readout was slow there too: seed 1 first scored 32/32 at
  2.82M, and seed 0 had not reached it by 3M. With one circuit seed, the gap cannot yet be
  separated from seed-to-seed variation.
- **Wall-clock cost:** simulating the circuit makes training about 9–10× slower per
  decision (165 vs 1,450 decisions/s).
- **No evidence the circuit is needed:** a linear readout does as well as the circuit on
  My Way Home (52 parameters) and on the corridor so far (91 parameters). The CNN does the
  perception, and any small head suffices.

Gameplay videos of one successful held-out episode per checkpoint are in `videos/`; see
[Gameplay videos](#gameplay-videos).

## Architecture

```
 screen 160×120 RGB, HUD on
        │  downsample to 80×60 (INTER_AREA), stack last 4 frames
        ▼
 obs (12, 60, 80) uint8 ──► CNN encoder ─────────────────────────► 256 features
                            Conv 8×8/4, 32 ─ Conv 4×4/2, 64 ─            │
                            Conv 3×3/1, 64 ─ Linear 1536→256             │
                                                    ┌────────────────────┼─────────────────┐
                                                    ▼                                      ▼
                                        bottleneck Linear 256→12            aux head Linear 256→16
                                        z = π·tanh(·) ∈ [-π, π]¹²           (training loss only)
                                                    │
                                    one of three heads, same input z
              ┌─────────────────────────────────────┼─────────────────────────────┐
              ▼                                     ▼                             ▼
     classical MLP                           linear readout              12-qubit circuit
     actor 12→32→A, critic 12→32→1           12 → A+1                    2 re-uploading layers
              └─────────────────────────────────────┼─────────────────────────────┘
                                                    ▼
                                    A action logits + value estimate → PPO
```

### CNN encoder and bottleneck

The encoder and bottleneck have ~491k parameters (`pixel_nav_models.py`, `PixelNavPolicy`).
The encoder is a Nature-DQN-style CNN. Its 256 features are squashed to 12 values in
[-π, π], so they can be used directly as rotation angles. Every head sees exactly these
12 values.

### Heads

| Head | Structure | Parameters (3 / 6 actions) |
|---|---|---:|
| `classical` (`ClassicalNavPolicy`) | Separate actor 12→32→A and critic 12→32→1, tanh hidden layers | 964 / 1,063 |
| `linear` (`LinearNavHead`) | One affine map 12 → A + 1 | 52 / 91 |
| `quantum` (`QuantumE1M1Policy`) | 12 qubits, 2 layers. Each layer applies RY(zᵢ·sᵢ), RZ(θ), RX(θ) to every qubit, then a ring of CNOTs. ⟨Z⟩ on qubits 0…A-1 gives the action logits and ⟨Z⟩ on qubit A gives the value, each times a learned gain. | 74 / 74 |

The circuit's 74 parameters are 48 rotation angles, 24 input scales, a logit gain and a value
gain. It is simulated with PennyLane `default.qubit` with torch backprop on the GPU.

### Auxiliary losses (training only)

The auxiliary head predicts targets for 8 vertical sectors of the screen. The targets are
computed from privileged buffers and never enter the policy:

- **Depth:** mean of the depth buffer in the middle third of each sector, log-scaled to
  [0, 1]. MSE on a sigmoid.
- **Visibility:** whether the goal (My Way Home) or a monster (Deadly Corridor) appears in each
  sector, from the labels buffer. BCE with `pos_weight` 20 (goal pixels are rare) or 5
  (monsters).

These losses, weighted 0.5, give the encoder a useful learning signal before the task
reward is found.

### Training and evaluation protocol

`train_pixel_nav.py` contains the training and evaluation code:

- **Environments:** one ViZDoom process per environment, 8 in total (`EnvPool`). The trainer
  chooses every reset seed, and training seeds never overlap the selection or held-out
  seeds.
- **PPO settings:** 128 steps × 8 environments per update, 4 epochs, minibatch 256,
  lr 2.5e-4 (circuit parameters 3e-3), clip 0.1, entropy 0.01, GAE λ 0.95.
- **Checkpoint selection:** every 250 updates, greedy evaluation on 32 selection seeds
  (10000–10031). The score is success rate + 0.01 × mean progress, and ties go to the later
  checkpoint.
- **Final evaluation:** the best checkpoint plays 100 held-out seeds (20000–20099), greedy and
  sampled, next to random and always-forward baselines. Results are also reported without
  seed 20017, which random play solves.
- **Budget:** circuit runs stop at 19,500 s. `--resume` continues from `latest.pt` with fresh,
  disjoint training seeds.

### Tasks and rewards

The environment wrappers compute each reward term once. Position comes from the engine and
is used only for the progress term.

| | My Way Home (`pixel_nav_env.py`) | Deadly Corridor (`pixel_corridor_env.py`, skill 3) |
|---|---|---|
| Actions | FORWARD, TURN_LEFT, TURN_RIGHT | FORWARD, ATTACK, STRAFE_LEFT/RIGHT, TURN_LEFT/RIGHT |
| Step cost | −0.01 | −0.01 |
| Goal | +100 for reaching the vest | +100 for reaching the vest |
| Progress | Potential-based distance to vest, scale 1024 | Potential-based distance to vest, weight 30, scale 1500 |
| Combat | — | +2 per kill, −0.05 per HP lost, −30 for death |
| Success | Episode ends early with total game reward > 0.5 | Episode ends early while alive (the vest script exits the level) |

## Experiments with the screen-only pipeline

### My Way Home

1. **Classical head first (jobs 71010, 71026).** These runs checked that the learned encoder
   and auxiliary losses solve the task before any circuit training. They did: all three
   classical seeds reached ≥ 99/100 greedy. The earlier fixed RGB-sector features (see
   [Earlier experiments](#earlier-experiments)) had never beaten random.
2. **Success-detection bug.** Job 71025 found that the vest's +1 game reward can land one tic
   before ViZDoom reports the episode as finished. The final step's reward was then about
   −0.0001, so the old test `raw_reward > 0` scored real pickups as failures and withheld the
   +100 bonus during training. Both screen-only environments now test
   `get_total_reward() > 0.5`. Rechecking earlier screen-only evaluations found only three
   uncounted pickups. Pre-fix runs are kept in `pixel_nav_runs_success_bug/`.
3. **Circuit and linear controls (jobs 71027, 71093, 71094, 71132).** Each head was trained on
   three seeds with identical settings; the results are in the table above. Per-run
   greedy episode lengths were 54–64 steps (classical), 57–72 (linear) and 65–73 (circuit),
   against 496 for random play. The circuit runs hit the time budget just short of
   3M decisions.

Outputs: `pixel_nav_runs/{classical,linear,quantum}_seed{0,1,2}/`.

### Deadly Corridor

1. **Baselines (smoke job 71181).** Running forward every step reaches the vest about a
   quarter of the time (27/100) and dies otherwise. Random play never reaches it. A learned
   policy has to beat both.
2. **First reward attempt failed (job 71183).** With a death penalty of −50 and progress
   weight 5, the classical agent learned to camp at the start: about 2.8 kills, no deaths,
   3–12% progress and 0/32 successes. The whole corridor was worth about 5 reward, while
   advancing risked −50. With the current weights (−30, 30), clearing enemies and then
   advancing pays, and a blind charge still scores below camping. Logs are in
   `logs/slurm_pixel_corridor_classical_71183.log` and outputs in `pixel_corridor_runs_camping/`.
3. **Classical head (jobs 71188, 71368).** All three seeds reached 99/100 greedy. Their
   learning speed varied widely: seed 1 first hit 32/32 on the selection seeds at 1.79M
   decisions, seed 2 only at 2.82M.
4. **Circuit head (job 71203, then resumed as job 71255).** The circuit stayed at 0/32 through
   2.56M decisions, jumped to 19/32 at 2.82M and hit the time budget there. That snapshot
   reached the vest 53/100 times greedy (4.5 kills, 31% died) and 19/100 sampled. It used
   ATTACK on 62% of steps. The resumed run scored 32/32 from 4.35M decisions onward. Its best
   checkpoint (update 4750, 4.86M decisions) reached 98/100 greedy and 99/100 sampled. The
   2.8M snapshot is kept as `best_2.8M.pt` / `results_2.8M.json`.
5. **Linear-readout control (job 71605).** Seed 0 reached 98/100 greedy, but it was still
   improving at the 3M-decision cap (30/32 selection) and averaged only 4.98 kills with
   61 HP left. Seed 1 reached 100/100 with 6.00 kills. Selection scores stayed at 0–1/32
   through 2.05M decisions (seed 1) and 2.56M (seed 0), close to where the circuit broke
   through at 2.82M.

Outputs: `pixel_corridor_runs/{classical,linear}_seed{0,1,2}/`, `pixel_corridor_runs/quantum_seed0/`.


#### Why the early circuit could not kill: aiming probe

At 2.82M decisions the circuit fired 21.4 shots per kill on its own greedy episodes;
classical seed 0 fired 2.3. `probe_corridor_aim.py` (jobs 71367 and later) scores checkpoints
on the same 5,946 frames. The frames were collected by classical seed 0, sampling with 25%
random actions. The labels buffer bins each frame by the horizontal offset of the largest
visible monster: centred (within ±6% of screen width), left, right or none. The bin is
never a policy input.

| Checkpoint | Turn toward enemy* | P(ATTACK) centred | P(ATTACK) off-centre | P(ATTACK) no enemy | Shots per kill† |
|---|---:|---:|---:|---:|---:|
| Classical seed 0, update 2250 | +0.58 | 0.43 | 0.12 | 0.14 | 2.3 |
| Circuit, update 2750 (2.8M) | +0.31 | 0.35 | 0.24 | 0.19 | 21.4 |
| Circuit, update 3500 | +0.58 | 0.31 | 0.17 | 0.14 | — |
| Circuit, update 4000 | +0.48 | 0.23 | 0.19 | 0.14 | — |
| Circuit, update 4750 (best) | +0.60 | 0.23 | 0.10 | 0.12 | 1.8 |
| Circuit, update 5250 | +0.65 | 0.24 | 0.10 | 0.05 | — |

\*Mean of P(turn toward) − P(turn away) over frames with an enemy to the left or right.
†On the checkpoint's own greedy episodes on held-out seeds.

How the circuit's aiming developed:

- **At 2.8M decisions** it turned toward enemies half as strongly as the classical policy and
  fired almost regardless of enemy position.
- **By update 3500** its turning had caught up. The remaining gap was fire timing.
- **By update 4750** it fired less overall and more selectively, at 1.8 shots per kill on its
  own play.
- **Remaining difference:** on these shared frames it separates centred from off-centre
  enemies less sharply than the classical policy (2.4× vs 3.6×).
- **Its own style:** at update 4750 it puts 0.39 probability on STRAFE_RIGHT when an enemy
  is to the right, which the classical policy does not do.

The early fire-timing gap was a stage of training, not a fixed limit of the circuit. These
results come from one seed and one frame distribution, and the left bin has only 195 frames.
Results are in `aim_probe/results.json` and `aim_probe/results_final.json`.

### Running and next steps

- **Running:** job 71605, corridor linear-readout seed 2.
- **Next:** more circuit seeds on the corridor, to tell circuit slowness apart from seed
  variance. Each needs two six-hour jobs: train, then `--resume`.

## Gameplay videos

`record_gameplay.py` plays each saved best checkpoint greedily on held-out seeds from 20000
upward, exactly as in evaluation, and saves the first successful episode. Each video shows
every game tic of the native 160×120 screen, upscaled 4×, at 35 fps. A caption strip shows
the run, seed, step, action and kill count. Job 71610 recorded one video per checkpoint into
`videos/<task>_<run>.mp4`, and every checkpoint succeeded on seed 20000. Per-episode
statistics are in `videos/summary.json`.

```bash
sbatch run_record_gameplay.sh                                      # every run with results.json
sbatch run_record_gameplay.sh pixel_corridor_runs/quantum_seed0/best.pt   # one checkpoint
```

## Running

The environment is a `uv` virtualenv in `.venv` (Python 3.12):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126   # V100 = CUDA CC 7.0
uv pip install pennylane vizdoom gymnasium matplotlib opencv-python imageio imageio-ffmpeg
```

Tested versions: torch 2.14.1+cu126, PennyLane 0.45.1, ViZDoom 1.3.2, Gymnasium 1.4.0.

Anything that starts ViZDoom or torch runs through Slurm on partition `UGGPU-TC1`, never on
the login node. Logs go to `logs/`. Each user may have 2 submitted jobs and 1 GPU at a time;
CPU-only jobs (videos, probes) can run alongside a GPU job.

| Script | What it runs |
|---|---|
| `run_pixel_{classical,linear}.sh` | My Way Home, 3 seeds sequentially, 3M decisions each |
| `run_pixel_quantum.sh` | My Way Home circuit head, one seed (`SEED=n`), 19,500 s budget |
| `run_pixel_corridor_{classical,linear}.sh` | Deadly Corridor, seeds `${SEEDS:-0 1 2}` |
| `run_pixel_corridor_quantum.sh`, `..._quantum_resume.sh` | Corridor circuit head; resume toward 6M decisions |
| `run_corridor_aim_probe.sh` | Aiming probe (CPU) |
| `run_record_gameplay.sh` | Gameplay videos (CPU) |
| `run_pixel_smoke.sh`, `run_pixel_corridor_smoke.sh` | Short smoke tests |

Directly, for example:

```bash
python train_pixel_nav.py --task corridor --model quantum --seed 0 --head-lr 0.003 \
    --max-seconds 19500 --output pixel_corridor_runs/quantum_seed0 [--resume]
```

## Repository layout

**Screen-only pipeline (current)**

| File | Purpose |
|---|---|
| `pixel_nav_env.py` | My Way Home screen-only env, auxiliary targets, subprocess `EnvPool` |
| `pixel_corridor_env.py`, `pixel_corridor.cfg` | Deadly Corridor screen-only env (skill 3, HUD and damage flashes on) |
| `pixel_nav_models.py` | CNN encoder, bottleneck, auxiliary head, and the three heads |
| `e1m1_quantum_model.py` | `QuantumE1M1Policy`, the 12-qubit circuit head |
| `human_nav_models.py` | `ClassicalNavPolicy`, the classical MLP head |
| `reward_shaping.py` | Potential-based progress reward and kill reward |
| `train_pixel_nav.py` | PPO training, checkpoint selection, final evaluation, resume |
| `probe_corridor_aim.py` | Aiming probe on shared frames |
| `record_gameplay.py` | Gameplay videos of successful held-out episodes |
| `diagnose_pixel_*.py` | Checks for auxiliary targets, success detection and the corridor env |

**Earlier experiments**

| File | Purpose |
|---|---|
| `models.py` | Hand-crafted-feature circuit actor (`DataReuploadingPQC`), classical baseline |
| `doom_env.py`, `doom_nav_env.py` | 8-, 10- and 12-qubit feature envs for `basic.cfg`, Deadly Corridor and My Way Home |
| `train_ablation.py`, `train_corridor.py`, `train_navigation.py` | PPO for those envs |
| `record_agent.py`, `record_nav_agent.py` | Videos with a feature HUD for those envs |
| `e1m1_map.py`, `e1m1_planner.py`, `e1m1_env.py`, `train_e1m1.py` | E1M1 map reader, route planner, waypoint env and trainer |
| `validate_e1m1_route.py`, `validate_e1m1_starts.py` | Scripted route follower checks |
| `subtask_envs.py`, `train_subtask_baseline.py` | Classical subtask baselines |
| `human_nav_env.py`, `human_e1m1_env.py`, `train_human_nav.py` | RGB-sector features with travel and place memory |
| `wads/doom1.wad` | Shareware Doom v1.9 (E1M1); MD5 `f0cefca49926d00903cf57551d901abe` |

## Earlier experiments

These experiments led to the current design. Most used one seed and 20 evaluation episodes,
so they are preliminary.

### Hand-crafted features into a circuit

The first agents used no CNN. Each frame was reduced to one angle per qubit from the depth
and labels buffers: 8 field-of-view sectors (monster visible, or wall distance) plus
task-specific qubits. The actor was `DataReuploadingPQC` (RX(xᵢ·λ), RY(θ), RZ(θ), a CNOT
ring, 4 layers, ⟨Z⟩ on every qubit, linear head), with a classical MLP critic.

| Experiment | Result |
|---|---|
| Ablation, `basic.cfg`, 3 actions, 32K frames | Classical MLP (171 params) scored 10.1 mean step reward, the entangled circuit (123) 8.4 and the unentangled circuit −1.7. Without CNOTs each ⟨Zᵢ⟩ depends on one input only. Learning rates were not matched (0.01 vs 0.002). |
| Deadly Corridor, 8 qubits, 6 actions, 524K frames | Entropy collapsed and the agent pressed FORWARD on 95–100% of steps (about 4/10 vests). x-progress was rewarded twice, outweighing death. |
| My Way Home, 10 qubits, 7 actions, 262K frames (job 69349) | 14/20 greedy and 14/20 sampled, against 2/20 random. Qubit 9 used the true distance to the goal, which is privileged information. `doom_nav_env.py` still uses the old success test, so this may be slightly undercounted. |

### E1M1 with a planned route

The E1M1 work built the following pieces:

- **Map reader (`e1m1_map.py`):** parses the WAD. At skill 3 the map has 4 Zombiemen and
  2 Imps, 8 door lines and 4 damaging sectors.
- **Route planner (`e1m1_planner.py`):** Dijkstra on a 16-unit grid with clearance, step and
  nukage costs. It produces a 22-waypoint route of ~4,972 units.
- **Waypoint env (`e1m1_env.py`):** 12 qubits (8 sectors, waypoint bearing and proximity,
  health, door ahead) and 7 actions, with a reverse-start curriculum.

What happened:

- **Scripted follower (job 69819):** the route is followable. It exited 4/5 times at skill 1
  and 3/5 at skill 3.
- **Circuit-only policy:** the PPO runs did not finish the level. Job 69826 (fixed curriculum)
  scored 0/20. Job 69828 (adaptive curriculum) stalled at waypoint 20 and also scored 0/20.
- **Imitation of the scripted routes (job 69885):** 59% action accuracy and 0/5 exits.

These runs used a route and position-derived inputs. The project then moved to
human-equivalent observations.

### Classical subtask baselines (jobs 70606, 70610, 70615)

These runs gave a classical MLP the same compact observations, up to 524K decisions:

- **Skill-5 combat:** about 2 kills and 5/20 survived, but 98% of actions were ATTACK.
- **My Way Home:** 12/20 vests, using a goal-proximity signal from game state.
- **Take Cover dodging:** no better than random.

Outputs are in `subtask_baselines{,_extended}/`.

### Screen-only features without a learned encoder (jobs 70682–70983)

These policies saw eight fixed RGB sector summaries, optionally with a 4-value classical
travel or place memory. None was reliably better than random play on My Way Home. Across the
runs, held-out success ranged from 0/20 to 3/20, against 1/20 for random actions, and seed
20017 accounted for most successes:

- **Travel memory:** made no difference.
- **Private progress reward:** made no difference.
- **RGB place memory with "memory-bridge" gates in the circuit (106 params):** 1/20 greedy and
  3/20 sampled, against a classical MLP's 0/20 and 1/20 (Fisher p ≈ 0.6).

The classical MLP failed on the same inputs too, which pointed to the fixed features rather
than the policy model as the bottleneck. That led to the learned CNN encoder. Outputs are in
`human_nav_runs/`, `human_nav_progress_runs/` and `human_place_comparison/`.

## Known issues

- **`doom_nav_env.py`** still uses the old `raw_reward > 0` success test (see the
  success-detection bug above).
- **`human_e1m1_env.py`:** the STRAFE_RIGHT entry in `HumanE1M1Env.ACTIONS` has 6 elements
  instead of 7.
- **Shared code:** the PPO loop is copied across several training scripts, so fixes must be
  applied to each.
- **Older trainers:**
  - `clip_grad_norm_` covers actor and critic together, so critic gradients dominate.
  - They select "best" checkpoints by mean step reward, which is noisy.
- **`doom_env.py` and `doom_nav_env.py`:**
  - They mark time limits as `terminated`, so GAE does not bootstrap at timeouts.
  - `doom_env.py` treats every non-player, non-armour object as an enemy.
- **`DataReuploadingPQC`:** the final-layer RZ gates do not affect Z measurements, so 8–12
  parameters are redundant.
- **`e1m1_env.py`:**
  - It reuses the maze depth normalisation, which saturates beyond about 38 units.
  - Warping onto door waypoint 17 stalls the scripted follower.
- **`quantum_navigation_gameplay.mp4`** (earlier pipeline) is empty because of a device
  mismatch during recording.
