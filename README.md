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
(52–104 parameters). Privileged game information such as position, depth, object labels and
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
| Linear readout seed 2 | 91 | 1.79M | 100 | 100 | 6.00 | 0% | 100 | 30 min |
| **12-qubit circuit seed 0** | **74** | **4.35M** | **98** | **99** | **5.99** | **0%** | **96** | ~11 h |
| Always FORWARD | — | — | 27 | — | 0.49 | 73% | — | — |
| Random actions | — | — | 0 | — | 0.91 | 100% | — | — |

In this table, Kills, Died and HP left are for greedy play.

**What this shows so far**

- **Both tasks:** a circuit head trained end to end with a CNN encoder solves both tasks
  from the screen alone, and matches the classical heads in final success.
- **Learning speed:** the circuit needed more decisions to learn. On My Way Home it took
  2–3× as many as the linear readout. On the corridor it took 1.5–2.4× as many as the
  classical seeds. The 91-parameter linear readout ranged from 1.79M to beyond 3M decisions
  across its three seeds, so the circuit's 4.35M is slower than all six classical and linear
  runs, but with one circuit seed the gap cannot yet be separated from seed-to-seed variation.
- **Wall-clock cost:** simulating the circuit makes training about 9–10× slower per
  decision (165 vs 1,450 decisions/s).
- **No evidence the circuit is needed:** a linear readout does as well as the circuit on
  My Way Home (52 parameters), on the corridor (91 parameters) and on E1M1 by imitation (104
  parameters). The CNN does the
  perception, and any small head suffices.

Gameplay videos of one successful held-out episode per checkpoint are in `videos/`; see
[Gameplay videos](#gameplay-videos).

**E1M1** (the original level, 7 actions; reach the exit switch from the real start, skill 3).
Pure RL did not work, so these are **students distilled from a scripted teacher that reads
private state** (see [E1M1 by imitating a scripted teacher](#e1m1-by-imitating-a-scripted-teacher));
they see only the screen at test time:

| Head (same CNN encoder) | Head params | Exits, greedy | Exits, sampled | Wall time |
|---|---:|---:|---:|---:|
| Classical MLP | 1,096 | 72 | 77 | ~25 min |
| Linear readout | 104 | 77 | 72 | ~25 min |
| **12-qubit circuit** | **74** | **73** | **70** | 1h 53m |
| Scripted teacher (private state) | — | 84 | 84 | — |
| Pure RL, best of three attempts | 1,096 | 0 | 0 | ~2h each |
| Random actions | — | 0 | — | — |

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

| Head | Structure | Parameters (3 / 6 / 7 actions) |
|---|---|---:|
| `classical` (`ClassicalNavPolicy`) | Separate actor 12→32→A and critic 12→32→1, tanh hidden layers | 964 / 1,063 / 1,096 |
| `linear` (`LinearNavHead`) | One affine map 12 → A + 1 | 52 / 91 / 104 |
| `quantum` (`QuantumE1M1Policy`) | 12 qubits, 2 layers. Each layer applies RY(zᵢ·sᵢ), RZ(θ), RX(θ) to every qubit, then a ring of CNOTs. ⟨Z⟩ on qubits 0…A-1 gives the action logits and ⟨Z⟩ on qubit A gives the value, each times a learned gain. | 74 / 74 / 74 |

The circuit's 74 parameters are 48 rotation angles, 24 input scales, a logit gain and a value
gain. It is simulated with PennyLane `default.qubit` with torch backprop on the GPU.

### Frozen-perception experiment

`train_frozen_perception.py` tests whether the heads differ when perception cannot adapt
to the policy loss. It uses the existing E1M1 teacher and **DAgger**, rather than PPO.
Perception starts from random weights, learns only geometric targets, and is then frozen
once for all heads and all policy seeds. No action, value, reward, or success loss trains
the perception model. This tests the frozen-feature hypothesis; the earlier end-to-end
results alone do not establish why the linear head worked.

The geometry decoder predicts the existing 20 targets: eight depth sectors, eight enemy
visibility sectors, the door/switch ray, route sine/cosine, and route progress. Since these
do not fit individually into 12 coordinates, a `256→11` bottleneck learns geometry-only
latent angles. A twelfth angle records USE cooldown from the agent's **executed actions**,
which cannot reliably be inferred from four frames. The decoder is discarded for action
selection. Thus the inputs are compressed geometry predictions plus observable action
history, not 12 individually named privileged features. Neither the actor nor evaluation
receives the true geometry. Route information may be difficult to reconstruct from images;
validation geometry losses must be examined before interpreting weak task results.

| E1M1 head | Total actor/value parameters |
|---|---:|
| Linear, `12→8` | 104 |
| Small shared MLP, `12→3→8`, tanh | 71 |
| Existing classical MLP, separate actor/critic | 1,096 |
| Circuit, two re-uploading layers | 74 |
| Same circuit without CNOTs | 74 |
| Circuit, four re-uploading layers | 146 |

The 71-parameter MLP is a **near** parameter match to the two-layer circuit. Four layers
double the rotation/input-scale budget; their result is a capacity/depth comparison, not
an equal-parameter comparison. DAgger trains action logits only: value-only parameters
are retained for compatibility and included in these totals, but receive no value loss.
For example, E1M1 linear has 91 actor parameters plus 13 unused value parameters; the
small MLP has 67 parameters on its actor path, and the classical MLP actor has 647.
The earlier 91-parameter total is correct for the corridor's six-action actor/value head.
Removing CNOTs also removes cross-feature mixing: its seven actor outputs use only seven
input coordinates (43 actor-path parameters). It is a structural ablation, not a control
with identical effective input access, and a win over it alone cannot establish a benefit
specific to entanglement.

Protocol:

- Collect geometry pretraining and validation in separate episodes, using 75% teacher /
  25% random actions and uniform route starts. Select perception by geometry validation
  loss, then disable all its gradients and check its SHA-256 fingerprint after every fit.
- Give every head the same initial demonstrations. Each DAgger round collects the same
  state budget from **each** head, then adds their union to the dataset seen by **all**
  heads. Each seed starts again from the common initial demonstrations. This controls
  head-specific differences in visited training states within a seed.
- Cache the common angles, use identical minibatch orders and epoch counts, and optimize
  weighted action cross-entropy. Classical lr is `3e-4`, circuit lr is `3e-3`, following
  existing runs; this is a fixed-setting comparison, not a tuned optimum for every head.
- Select each head independently on real-start seeds 10000 onward. Test the selected
  checkpoint on real-start seeds 20000 onward, greedy and sampled. Report per-action
  accuracy on the separate geometry-validation trajectories as an additional diagnostic.
  The teacher can use hidden state beyond these features, so its labels need not be
  perfectly predictable by any head. This teacher does not label strafing actions, so this
  DAgger experiment cannot test the proposed strafe-clearance conjunctions.

The full preset uses policy seeds 0/1/2, 100k perception states, 10k validation states,
50k initial demonstrations, three DAgger rounds adding 10k states per head per round,
32 selection episodes and 100 final episodes per head/mode/seed. It needs roughly 6 GB
of host memory for frame replay plus training tensors. Checkpoints and full episode rows
are saved under the output directory; existing outputs are never overwritten.

```bash
# Requires torch, pennylane, vizdoom, gymnasium, numpy, opencv-python and matplotlib.
# Small end-to-end validation run; use a new output path when repeating.
PYTHON=python3 bash run_frozen_perception.sh pilot

# Full experiment on an allocated GPU (this does not request a GPU allocation).
PYTHON=python3 DEVICE=cuda bash run_frozen_perception.sh full

python3 -m unittest -v test_frozen_perception

python3 summarize_frozen_perception.py frozen_perception_runs/e1m1_full/results.json \
  --output frozen_perception_runs/e1m1_full/summary.json

# Recheck checkpoint integrity, teacher performance and geometry vs constant predictors.
python3 audit_frozen_perception.py frozen_perception_runs/e1m1_full --recollect-geometry
```

**Completed CPU pilot, 2026-11-03:** one policy seed, 4,096 geometry training states,
512 geometry-validation states, ten perception epochs, 2,048 initial demonstrations and
one shared DAgger round (3,584 total labeled states). Each fit used six epochs. Selection
used two episodes; final testing used four different real-start seeds in each mode.
These completed results cover only the CPU pilot. A larger GPU run was launched on
2026-11-03 with three policy seeds, all six heads, the full data/epoch budgets above,
and batches of 256 for every head. It tests whether the failures persist with more training;
its outcomes must be assessed separately from this pilot. See its
[launch configuration](frozen_perception_runs/e1m1_gpu_full_20261103/launch.json),
[completion status](frozen_perception_runs/e1m1_gpu_full_20261103/status.json), and
[training log](frozen_perception_runs/e1m1_gpu_full_20261103/run.log). The job writes
`summary.json` automatically after all three seeds finish.

All six heads scored **0/4 exits in both greedy and sampled play**; the teacher scored
**3/4** on those same final seeds. Selected-checkpoint action diagnostics on the 512
separate validation states were:

| Head | Action accuracy | Balanced action accuracy |
|---|---:|---:|
| Linear | 49.4% | 45.5% |
| Small MLP | 49.0% | 26.7% |
| Classical MLP | 41.4% | 42.3% |
| Two-layer circuit | 31.1% | 36.7% |
| Two-layer circuit without CNOTs | 52.5% | 22.2% |
| Four-layer circuit | 27.0% | 33.5% |

Balanced accuracy averages recall over the five action classes present in validation.
None of the selected heads correctly predicted USE on the seven validation USE states.
These low action scores and zero exits leave training and feature sufficiency unresolved;
they do **not** establish a ranking of converged heads. In particular, the higher raw
accuracy of the circuit without CNOTs is not an advantage in successful control.

The perception fingerprint stayed identical during all head fits. Recollecting the
geometry data reproduced validation losses exactly; perception beat constant training-mean
predictors in all four loss groups, but route-direction MSE improved only from 0.0942 to
0.0897. The pilot took 23.3 minutes on CPU. See the [summary and audit](frozen_perception_runs/e1m1_pilot/summary.json),
[full episode results](frozen_perception_runs/e1m1_pilot/results.json), and
[runtime versions](frozen_perception_runs/e1m1_pilot/versions.json).

A linear head can express some conjunctions, including binary AND via a thresholded
sum. The interesting question is whether the actual geometry-to-action boundaries need
more nonlinear capacity. Beating linear alone would support nonlinear readout capacity;
beating the small MLP and the circuit without CNOTs is needed to argue for a benefit from
this entangled ansatz. These are classically simulated circuits throughout.

### Multi-task perception and goal-conditioned imitation

`train_transfer.py` tests whether broader perception training improves transfer, before
revisiting quantum readouts. Its students use named semantic predictions and public
objective/action-history context; no route direction, global position, or route progress
is a perception target or policy input.

- **Training maps:** My Way Home, Deadly Corridor, and E1M1. The existing classical
  navigation/combat specialists label their own tasks; E1M1 uses the scripted teacher.
  All seven actions have one canonical meaning across tasks; unavailable actions are masked.
- **Shared perception:** 8 depth sectors, 8 enemy sectors, 8 goal sectors, an interaction
  ray, health, and ammunition. Goal identity is public context: pickup versus exit.
  Pickup targets use object-label buffers; exit targets use static wall-ray intersections,
  a geometric visibility proxy that does not fully model dynamic door occlusion.
- **Perception control:** `single` learns on 120k E1M1 states; `multi` learns on 40k states
  from each training map. Both receive 120k total samples, equal epochs, matching initial
  weights, and geometry-only losses. E1M1 multi-map samples are a random subset of the
  single-map data. No policy loss updates either encoder.
- **Policies:** a goal-conditioned GRU with 64 hidden units (21,767 parameters) and a
  feed-forward MLP with 411 hidden units (21,790 parameters). They receive the same 27
  semantic predictions, objective, available actions, executed USE cooldown, and previous
  action. GRU state resets at episode boundaries; training backpropagates complete episodes.
- **DAgger:** 50k initial teacher states per map, then three rounds. Each of the four
  policies contributes 10k states per map per round to a common union. Every policy sees
  identical labels and public context; only its frozen semantic predictions differ.
  Full defaults use policy seeds 0/1/2, six actor epochs per round, and an 80% teacher /
  20% random mixture during perception data collection.
- **Selection:** 32 real-start episodes on each training map, using seeds 10000 onward.
  Score is mean success rate plus 0.001 times mean held-out action accuracy on the training
  maps. Actor training and validation accuracy are measured on fixed checkpoints, making
  their gap interpretable. The E1M1 teacher tracks actual executed USE actions during DAgger.
- **Final evaluation:** 100 episodes per policy seed on each training map, then E1M2,
  using seeds 20000 onward. E1M2 supplies no training data or checkpoint-selection feedback.
  Both encoders are probed on the **same evaluation frames** every eight decisions; their
  paired semantic losses help separate representation errors from controller failures.

E1M2 is **one held-out layout**, evaluated in two variants: normal play, and all keys
granted at reset. The latter controls for key handling, which none of the training tasks
requires. It is an altered task and is reported separately. These results cannot establish
generalization across arbitrary Doom maps. The actor has no explicit map planner; the GRU
tests whether learned memory helps. Teachers and perception initialization are fixed across
policy seeds. This is a transfer experiment using classical controls, not a quantum comparison.

```bash
python3 -m unittest -v test_transfer

# Use a fresh output directory; the script does not allocate a GPU itself.
CUDA_VISIBLE_DEVICES=3 PYTHON=/path/to/cuda/python bash run_transfer.sh \
  --output transfer_runs/my_run

python3 summarize_transfer.py transfer_runs/my_run/results.json \
  --output transfer_runs/my_run/summary.json
```

The capped [smoke-run summary](transfer_runs/smoke_20261103/summary.json) validates the
training/evaluation flow only. Short smoke episodes do not measure successful transfer.
The full GPU study launched on 2026-11-03 is recorded in its
[configuration](transfer_runs/gpu_full_20261103/config.json),
[live status](transfer_runs/gpu_full_20261103/status.json), and
[training log](transfer_runs/gpu_full_20261103/run.log). Its supervisor writes a summary
after all seeds and teacher-reference evaluations finish.

### Auxiliary losses in the original end-to-end runs

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
5. **Linear-readout control (job 71605).** All three seeds finished at 98, 100 and 100 out of
   100 greedy. Seed 0 was still improving at the 3M-decision cap (30/32 selection) and
   averaged only 4.98 kills with 61 HP left. Seed 2 reached 32/32 at 1.79M decisions, the
   fastest of any run; seed 1 took 2.82M.

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

### E1M1 from the screen

The next task combines the two: follow a long route like My Way Home, fight like Deadly
Corridor, and use doors and the exit switch. `pixel_e1m1_env.py` is built and smoke-tested;
no policy has been trained on it yet.

- **Observation and actions:** the same 4 stacked 80×60 screens with HUD, and the six
  corridor actions plus USE (7 in total; the circuit measures 8 of its 12 qubits).
- **Rewards:** the corridor terms (−0.01 per step, +2 per kill, −0.05 per HP lost, −30 for
  death, +100 for the exit) plus route progress with weight 30.
- **Progress from a walking-distance field.** The route winds through rooms and doors, so
  straight-line distance would reward walking into walls. `e1m1_distance.py` runs a reverse
  Dijkstra from the exit over the route planner's grid, using the planner's own rules (wall
  clearance, step height, doors passable, nukage penalised), and saves it as
  `e1m1_distance.npz`. `field_progress_reward` pays for reducing that cost, with potential 1
  at the exit and 0 at the real start's distance (318.6 cost units). The field is read only
  inside the reward, like the corridor's position.
- **Auxiliary targets:** per-sector depth and monster visibility as before, plus one value
  for "USE would hit a door or the exit switch". That target casts rays along the heading
  (±8°, 72 units) like Doom's own use trace.
- **Curriculum starts:** `reset(options={"start_wp": k})` teleports to route waypoint k,
  facing the route within ±45°. The default is the real start, which is what evaluation
  uses. The trainer does not schedule these yet.
- **Success:** the episode ends before the 6,300-tic timeout while the player is alive,
  because the exit switch ends the level. Reported progress is the furthest route progress
  reached.

Smoke job 71707 checked the environment (`diagnose_pixel_e1m1.py`) and ran 16 PPO updates
with the classical head on CPU:

| Check | Result |
|---|---|
| Distance field | Falls monotonically along all 22 waypoints (303 → 1); 12,976 reachable cells |
| Warp starts (waypoints 0, 6, 12, 16, 21) | Land within 0.0 units, facing the route within ±27° |
| Scripted follower, skill 1 (uses private pose, so a test fixture, not a policy) | Exits from the real start on 3/3 seeds (601–769 decisions) and from waypoints 4, 15 and 19 |
| Random actions, skill 3 | Never exits; progress 0.12–0.37; 1/3 died |
| Always FORWARD, skill 3 | Never exits; stuck at progress 0.11 |
| Speed | About 1,100–1,200 decisions/s per env process, so the classical head is as fast as on the corridor |

Two findings from this smoke test:

- **USE toggles doors.** Pressing USE on a door that is already opening closes it again.
  My first scripted follower pressed USE every other step and so jammed the last door shut
  (3 of 6 runs timed out or died there). A policy has to learn to press USE once and then
  walk, and the follower now waits 25 decisions between presses.
- **The first "door ahead" test was too narrow.** It compared the bearing to the door
  line's midpoint, which fails when the player stands against a wide door. It is now the
  ray test above.

**Curriculum and warm start.** `train_pixel_nav.py --task e1m1` has three start modes
(`--curriculum`). `reverse` starts episodes at route waypoints counted back from the exit
(21, 19, 16, 12, 8, 4, 0): 50% at the current stage's waypoint, 30% at an earlier one and 20%
at the real start, advancing when 80% of the last 20 episodes from the current waypoint exit.
`uniform` starts 25% at the real start and otherwise at a random non-door waypoint, with no
stages. Door waypoints (5, 17, 20) are never start points, because teleporting into a closed
door sector traps the player. Evaluation always starts at the real start.
`--init-encoder` loads only the CNN encoder from another run.

**First attempt: reverse curriculum, classical head, 10M decisions (job 71710).** Warm-started
from the corridor encoder. It did not learn to play E1M1:

| Real-start result, 100 held-out seeds | Exits | Kills | Died | Progress |
|---|---:|---:|---:|---:|
| Classical, reverse curriculum, greedy (update 7250) | 0 | 0.00 | 1% | 0.25 |
| Classical, reverse curriculum, sampled | 0 | 0.79 | 18% | 0.26 |
| Random actions | 0 | 0.31 | 46% | 0.21 |
| Always FORWARD | 0 | 0.00 | 0% | 0.11 |

The curriculum passed stages 0–3 (waypoints 21, 19, 16, 12) by 1.8M decisions and advanced
to stage 4 (waypoint 8) at 2.4M. It never passed stage 4 in the next 7M decisions. Training
exits were about 10% of episodes throughout, and entropy only fell from 1.95 to 1.4.
Videos of the final policy (`videos_e1m1/`) show two failures:

- **Real start:** the policy never leaves the starting room. It strafes sideways against a
  wall for the whole episode, with 1,200 strafe actions against 70 FORWARD in the evaluation.
  Progress 0.25 is about the first door.
- **From waypoint 8:** in the dark corridor it faces a wall and presses ATTACK while imps
  throw fireballs from a distance, then dies. It never turns toward them.

A from-scratch run (job 71711, CPU, 230 decisions/s) was at stage 2 after 2M decisions, no
faster than the warm start, and was cancelled.

**Second and third attempts: uniform starts, then route auxiliary targets (jobs 71738,
71743).** The reverse curriculum might gate everything on its hardest stage, so job 71738
used uniform starts (`--curriculum uniform`). At 3M decisions the mean progress gain per
episode had fallen from 0.15 to 0.07 while entropy fell from 1.8 to 1.4, and real-start
evaluation was still 0/32 (cancelled). Job 71743 added two training-only auxiliary targets
from the walking-distance field: the sin and cos of the route direction relative to the
heading, and the progress along the route (`aux_route`, regression loss). The encoder learned
them from the screen (auxiliary route loss 0.19 → 0.05), but the policy did not use that:
at 3M decisions the gain was flat at about 0.07–0.10 per episode and real-start evaluation
was 0/32.

Job 71743 finished all 10M decisions with the same result. Real-start evaluation was 0/100
exits greedy and sampled (progress 0.15 and 0.22, against 0.21 for random actions and 0.11 for
always-forward), and every selection evaluation from update 500 on scored 0/32. Progress gain
per episode rose only from 0.09 to 0.12 over the run, entropy fell to 1.26, and the greedy
policy collapsed to pressing USE on 93% of steps (1,471 of 1,585 actions).

Videos of the sampled policy at update 3000 (`videos_e1m1_route/`) show undirected behaviour,
not a stuck curriculum: at the real start it turns on the spot and fires ATTACK until its ammo
runs down (47 → 10), from waypoint 12 it turns away from a zombieman while its health falls
from 85% to 18% with no kills, and from waypoint 8 it faces a blank wall. Learning the doors
and exit from the late waypoints worked under the reverse curriculum (stages 0–3 passed by
1.8M decisions), but E1M1 as a whole is not learned from reward alone at this scale. Skill
level is not a large lever: skills 1–3 all have the same two imps, and skill 3 adds two
zombiemen. Candidate next steps: imitating a scripted route follower that reads private
pose (a stronger use of privileged information than auxiliary targets), a higher input
resolution, and far more decisions.

### E1M1 by imitating a scripted teacher

Reward-only training did not learn E1M1 in three attempts, so the next experiment is a
different kind of result: **a screen-only student distilled from a teacher that reads
private state**. It is reported separately from the pure-RL results above.

**Teacher (`e1m1_teacher.py`).** It reads what the policy never sees: engine position and
heading, the walking-distance field (route direction), and door and exit-switch geometry
(the use ray). Monster positions come from the labels buffer, which matches what a player
sees. Each decision it (1) fires if a monster at least 12 px tall is near the screen centre
(every decision when 30 px or taller, otherwise every other one), (2) presses USE once when
a door or the exit switch is within 40 units ahead, with a 25-decision cooldown because USE
on an opening door closes it, (3) turns toward the route, then walks, with a short
turn-in-place when stuck. It never turns to chase a monster. Two earlier versions failed:
turning to face every visible monster, and standing to fire 20+ shots at an imp, both died
in nearly every episode near waypoint 12. The final teacher exits 84/100 at skill 3 from the
real start and 29/30 at skill 1.

**Student (`train_e1m1_imitation.py`).** The same screen-only network as the RL runs,
classical head, warm-started from the corridor encoder, with the same auxiliary targets
(depth, monster visibility, door ahead, route direction and progress). Training is DAgger:

1. Iteration 0 trains on 200K teacher-driven decisions.
2. Each later iteration lets the student drive (teacher mix 0.3, then 0.0 afterwards) for
   60K decisions, labels every state it visits with the teacher's action, and retrains on
   all data so far (about 680K states at the end).
3. Episodes start at the real start (25%) or a random non-door route waypoint, are cut at
   600 decisions, and the loss is cross-entropy with class weights `(1/frequency)^0.75`
   plus the auxiliary losses.

Selection uses 32 real-start seeds (10000-10031) and the final comparison uses 100
held-out seeds (20000-20099) at skill 3, with the teacher run on the same seeds.

| E1M1, skill 3, real start, 100 held-out seeds | Exits | Died | Kills | Progress |
|---|---:|---:|---:|---:|
| **Student v2, classical head, greedy** (job 71791) | **72** | 28% | 1.36 | 0.96 |
| Student v2, classical head, sampled | 77 | 23% | 1.14 | 0.97 |
| Student v2, linear head (104 params), greedy (job 71794) | 77 | 23% | 1.36 | 0.97 |
| Student v2, linear head, sampled | 72 | 27% | 1.26 | 0.94 |
| Student v2, 12-qubit circuit head (74 params), greedy (job 71796) | 73 | 27% | 1.55 | 0.95 |
| Student v2, circuit head, sampled | 70 | 30% | 1.23 | 0.96 |
| Teacher v2 (reference) | 84 | 16% | 1.11 | 0.98 |
| Student v1, greedy (job 71787) | 37 | 63% | 1.41 | 0.93 |
| Student v1, sampled | 83 | 17% | 1.30 | 0.97 |
| Teacher v1 (reference) | 89 | 11% | 1.14 | 0.98 |
| Pure RL, best attempt, greedy | 0 | 0% | 0.00 | 0.15 |
| Random actions | 0 | 46% | 0.31 | 0.21 |

The three heads are indistinguishable at this sample size: greedy 72 (classical), 77 (linear)
and 73 (circuit) exits out of 100, sampled 77, 72 and 70, against the teacher's 84. As on My Way
Home and the corridor, the CNN encoder does the perception and a small head suffices. The
circuit fits the teacher's labels less tightly (training accuracy 87% against 91-92%, and
cross-entropy 0.31 against 0.24 at the end), and took 1h 53m against about 25 minutes, with
the same 8 DAgger iterations. All three heads fail mostly at the waypoint-17 door (progress
about 0.82). The circuit run used 2-3 epochs per iteration instead of 3-6 to fit its time
budget, which makes the comparison slightly unfair to it.

Even the first iteration, trained on teacher demonstrations only, exited on 15 of 32
selection seeds, the first screen-only exits of E1M1 in this project. The student matches
the teacher's action about 91% of the time on its training data and 87% on states it visits
itself. Selection scores fluctuate between iterations (8-29 of 32 for v2), so the
checkpoint choice matters and one seed is not enough to rank iterations.

**Is DAgger needed? Behaviour cloning alone also clears E1M1.** Two ablations train the classical
head on teacher-driven demonstrations only (`--iterations 0`: the same teacher v2, loss, warm
start and 100-seed evaluation, no student rollouts):

| Classical head, 100 held-out seeds | Exits, greedy | Exits, sampled | Died, greedy | Kills, greedy |
|---|---:|---:|---:|---:|
| Behaviour cloning, 200K teacher states (job 71927) | 48 | 63 | 52% | 1.48 |
| Behaviour cloning, 680K teacher states (job 71928) | 58 | 78 | 42% | 1.30 |
| DAgger, 680K states in total (job 71791) | 72 | 77 | 28% | 1.36 |
| Scripted teacher | 84 | 84 | 16% | 1.11 |

So a screen-only classical network clears E1M1 on 48-58% of unseen seeds in greedy play, and
63-78% in sampled play, from teacher demonstrations alone. More demonstrations help (200K to
680K: +10 greedy, +15 sampled). At the same 680K-state budget DAgger improves greedy play
(72 against 58, a gap of about 2 standard errors) but not sampled play (77 against 78), so its
benefit here is modest and, with one seed each, not firmly established. Pure RL without any
teacher still scores 0.

**Why v1's greedy student was so much worse than its sampled one (37 vs 83).** v1's teacher
pressed USE once from up to 72 units away and then walked forward while the door opened,
so a closed door in view was labelled USE about 1 time in 4 and FORWARD otherwise. The
student's most likely action there was FORWARD, and in greedy play it never pressed USE
(4 times in about 500 decisions); it stood at the door under fire from the imp and died, at
progress 0.83 and 0.95, the doors at waypoints 17 and 20. Sampled play got through by
pressing USE by chance. v2's teacher presses USE only within 40 units, and rare actions
are weighted more in the loss, which moved greedy play from 37 to 72. Most of v2's
remaining greedy failures (22 of 28) are again at progress 0.82-0.88, the waypoint-17 door.

**What this does and does not show.** At test time the student sees only the four stacked
screens. But it learned from a teacher with map access, so this measures how well a
screen-only network can copy a route-and-combat expert, not whether reward alone can find
the behaviour. The results are one seed and one teacher. Videos of a winning episode for the
classical head (seed 20002) and the circuit head (seed 20003) are in `videos_e1m1_imitation/`.

### PPO fine-tuning of the imitation student

The imitation student only copies a rule-based teacher, so it never optimised the real reward
(it kills 1.4 of the six monsters on average, and fails at the waypoint-17 door). The next
experiment fine-tunes it with PPO on the environment reward alone, with no teacher
(`train_pixel_nav.py --task e1m1 --init-checkpoint <imitation best.pt>`, `run_e1m1_finetune.sh`).
Added for this: `--init-checkpoint` (load the whole model), `--value-warmup N` (first N updates
train only the critic, with the encoder's learning rate at 0 and no policy loss),
`--anchor-kl-coef` (KL penalty toward the starting policy), `--freeze-encoder`,
`--kill-weight`, and `--rich-critic`. The starting policy is evaluated first and kept as
`best.pt`, so a worse result cannot replace it. Selection uses 64 real-start seeds; episodes
start at the real start 60% of the time and at random route waypoints otherwise. The
starting policy scores 55/64 greedy on the selection seeds, with 1.48 kills and 14% deaths.

| Attempt (classical head, from imitation v2) | What happened on the 64 selection seeds |
|---|---|
| Naive PPO, encoder lr 5e-5 (job 71830) | Critic-only warm-up held 55/64. As soon as the policy updated it **collapsed to 0/64 within 100 updates** (progress 0.1, like always-forward). |
| Frozen encoder, KL anchor 1.0, head lr 5e-5 (71836) | Stable, but flat: 51-58/64, kills 1.4-1.5. KL to the start only 0.0001. |
| Frozen encoder, KL anchor 0.1, head lr 2e-4, entropy 0.01 (71838) | Stable but wandering: 45-56/64, kills 1.3-1.5. |
| Same, kill reward 8 instead of 2 (71841) | No change in kills (1.2-1.5); exits 43-52/64, slightly worse. |
| Same as the 0.1 anchor, with a critic on the 256 encoder features (71846) | Critic loss 0.05-0.27 against 0.4-0.6; 47-57/64 over 3M decisions, kills 1.1-1.4, no trend. The checkpoint with the best selection score (update 1450) is evaluated below. |

Findings:

- **Naive PPO destroys a cloned policy.** The imitation policy is nearly deterministic (entropy
  0.1-0.25), so its true advantages are tiny. PPO normalises them, which amplifies noise, and
  Adam takes steps of about the learning rate regardless of gradient size.
- **The critic on the 12-value bottleneck is poor.** Its loss stayed at 0.4-0.6 through
  warm-up and training, because the bottleneck of an imitation-trained policy keeps what the
  actions need, not what the value needs. A critic on the 256 encoder features reaches 0.05-0.27.
  Better advantages alone did not make the policy improve.
- **Fine-tuning gave at most a small gain in exits, and no gain in kills.** Four of the
  five runs never beat the starting policy on the selection seeds, and a larger kill reward did
  not increase kills. The last run's best-selected checkpoint (update 1450, picked from about
  60 noisy 64-seed evaluations) scored 82/100 greedy and 83/100 sampled on seeds 20000-20099,
  against 72 and 77 for the imitation student. Because that choice is noisy, both checkpoints
  were rerun on 300 fresh seeds (30000-30299, `evaluate_e1m1_checkpoints.py`):

  | 300 fresh seeds | Exits | Died | Kills |
  |---|---:|---:|---:|
  | Imitation student, greedy | 238 (79.3%) | 21% | 1.57 |
  | PPO fine-tuned, greedy | 255 (85.0%) | 15% | 1.28 |
  | Imitation student, sampled | 236 (78.7%) | 21% | 1.28 |
  | PPO fine-tuned, sampled | 243 (81.0%) | 19% | 1.29 |

  The greedy gap is +5.7 points (about 1.8 standard errors, unpaired) and the sampled gap is
  +2.3 points, so this is suggestive at best. Deaths fell a little in greedy play, but kills
  did not rise (greedy kills fell from 1.57 to 1.28).

Likely reasons, not yet tested: a near-deterministic policy rarely tries a different action
at the states where it fails, the frozen 12-value bottleneck limits what the head can change,
and killing is not needed to finish the level, so the reward gives little reason to fight.

### Is the E1M1 student memorising the level? Perturbation tests

E1M1 is one fixed level, and held-out seeds change only monster behaviour, so held-out success
cannot show whether the student follows what it sees or recalls a memorised route.
`evaluate_e1m1_robustness.py` (job 72275) perturbs the student's input or start state with no
retraining. 100 episodes per cell (seeds 40000-40099), cut at 800 decisions, skill 3, greedy
play (the sampled numbers are similar in the log):

| Exits out of 100 (greedy) | Classical | Linear | Circuit | Scripted teacher |
|---|---:|---:|---:|---:|
| Unmodified screens, real start | 76 | 82 | 73 | 82 |
| **HUD masked** (bottom 10 of 60 rows set to 0) | **0** | **0** | **0** | |
| Four stacked frames replaced by the newest | 66 | 0 | 47 | |
| Brightness x0.6 | 68 | 79 | **0** | |
| Brightness x1.4 | 75 | 72 | 58 | |
| Gaussian noise (std 25 of 255) | 77 | 73 | 65 | |
| **Random off-route start** (random reachable position and heading) | **25** | **30** | **30** | **92** |

- **Off-route starts:** all three students fall from about 75-82 to 25-30 exits when started at
  a random reachable position with a random heading, while the teacher gets 92 from the same
  starts. Only 21-29% of those episodes end in death; the rest wander until the cut-off. The
  students were trained only on the real start, 19 waypoint starts with a heading within
  45 degrees of the route, and their own trajectories, so they are lost anywhere else. The
  teacher follows the distance field and does not care where it starts.
- **HUD:** blanking the status bar takes every student from about 75 to exactly 0 exits, and
  they stop making progress instead of dying. Brightness and noise barely move the classical
  student (68-77), so this is not general fragility to a changed picture. The HUD (health, ammo,
  face) probably acts as a progress clock that correlates with where the player is on the route.
  The test cannot separate "uses the HUD digits" from "breaks on a black bar", and a human also
  sees the HUD, but it is a shortcut that will not carry to another level.
- **Motion cues and appearance:** the classical student barely needs the frame history (66 with
  one repeated frame). The circuit student is much more sensitive to appearance: at brightness
  x0.6 it makes 0 exits (8 when sampling), against 68 for the classical one. These are single
  seeds.
- **Reading:** the E1M1 students recognise places along one route, helped by the HUD, and
  fail away from it. That fits the failure to transfer to other tasks. It does not show
  that they use no visual information: the classical student tolerates noise and brightness.
  Candidate fixes inside E1M1 are DAgger with random off-route starts (the teacher can label
  any state, and scores 92 there) and HUD dropout during training.

### Does the circuit matter? Two ablations on E1M1

Two ablations use the circuit student's setup (teacher v2, the DAgger schedule, seed 0, 3 epochs in
round 0 and 2 per round after, head lr 3e-3) and the same 100 held-out seeds. `quantum_noent`
(job 72389) removes the CNOT ring, so each action's logit depends on one bottleneck value only
(checked in `test_circuit_ablations.py`: changing one input moves one logit). `quantum_frozen`
(job 72405) freezes the rotation angles and input scales at their random initial values, so
only the encoder, bottleneck and the two output gains train (2 trainable head parameters).

| E1M1, 100 held-out seeds | Trainable head params | Exits, greedy | Exits, sampled | Died, greedy | Kills, greedy | Final train. accuracy | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full 12-qubit circuit (job 71796) | 74 | 73 | 70 | 27% | 1.55 | 87% | 1 h 53 min |
| **No CNOTs** (72389) | 74 | 68 | 67 | 32% | 1.36 | 87% | ~1 h 50 min |
| **Frozen random circuit** (72405) | 2 | 52 | 66 | 48% | 1.88 | 83% | ~2 h |
| Classical MLP head | 1,096 | 72 | 77 | 28% | 1.36 | 91% | ~25 min |
| Linear head | 104 | 77 | 72 | 23% | 1.36 | 92% | ~25 min |
| Scripted teacher | -- | 84 | 84 | 16% | 1.11 | -- | -- |

- **Entanglement is not needed.** Without the CNOT ring the circuit scores 68 and 67 exits against 73
  and 70, with the same training accuracy (87%). A gap of 3-5 exits is inside the noise for 100 seeds
  (about 4-5 standard errors per cell). The encoder adapts to its head, so it can put the evidence for
  each action into that action's own qubit.
- **A random circuit nearly suffices.** With every angle frozen at random values, the student still
  clears E1M1 on 52 greedy and 66 sampled seeds. Training the angles adds about 16-21 greedy
  exits (52 to 68-73, roughly 3-4 standard errors) but nothing clear in sampled play (66 against
  67-70), and the fit improves from 83% to 87% accuracy (cross-entropy 0.43 against 0.30). So the
  trained angles help somewhat; most of the policy comes from the CNN encoder.
- **Caveats.** One seed each, and selection scores swing a lot between rounds (no-CNOT: 9-24 of 32;
  frozen: 0-17). The three circuit runs used 2-3 epochs per round against the classical heads'
  3-6, so the comparison is slightly unfair to the circuits. These tests show that neither entanglement nor
  trained angles is essential here; they do not rule out a benefit in a setting where the head has
  harder work to do.

### Zero-shot transfer of the E1M1 models

As a check, the four E1M1 policies (classical, linear and circuit imitation students, and the
PPO-fine-tuned student) were run on My Way Home and Deadly Corridor with **no retraining**
(`evaluate_cross_task.py`, job 71870). The observation is identical. The policy's 7 E1M1
logits are restricted to the target task's actions: My Way Home's forward, turn left and turn
right are E1M1 actions 0, 4 and 5; the corridor's six actions are E1M1 actions 0-5, with USE
dropped. Seeds 20000-20099.

| 100 held-out seeds | Greedy | Sampled | Dedicated model for comparison |
|---|---:|---:|---|
| **My Way Home**, random actions | 10 | | |
| E1M1 classical | 6 | 11 | |
| E1M1 linear | 8 | 20 | 99-100 greedy (classical, linear, circuit) |
| E1M1 circuit | 10 | 19 | |
| E1M1 fine-tuned | 7 | 13 | |
| **Deadly Corridor**, random / always-forward | 0 / 27 | | |
| E1M1 classical | 0 (98% died) | 0 (93% died) | |
| E1M1 linear | 0 (100% died) | 1 | 98-100 greedy |
| E1M1 circuit | 0 (100% died) | 0 | |
| E1M1 fine-tuned | 0 (99% died) | 0 | |

The E1M1 policies do not transfer. On My Way Home, greedy play is at the random level (6-10 of
100 against 10), and sampled play is somewhat above it for the linear and circuit heads
(20 and 19, against 10 for random; roughly p = 0.05 for 20 against 10, and not clearly
more than chance given these are among eight comparisons). In the corridor all four score 0-1
exits and die in 93-100% of episodes, below always-forward (27) and no better than random;
their mean kills (0.2-0.65) are below random's 0.91. The policies learned an E1M1-specific
mapping from screens to actions (a long winding route, doors, nukage, those textures), which does
not carry over to a different maze or to a corridor of enemies.

### Do the two small-task specialists play E1M1?

The original idea was to train specialists on the small tasks (navigation on My Way Home,
combat on Deadly Corridor) and distill them into an E1M1 model. Distilling means labelling E1M1
states with the specialists' actions, so it only works if they behave sensibly on E1M1 screens.
`evaluate_specialists_on_e1m1.py` (job 71875) ran them with no retraining on E1M1, skill 3,
real start, seeds 20000-20099: the navigation specialist's three actions map to E1M1 actions
0, 4, 5 and the combat specialist's six to 0-5. Modes were nav only, combat only, and an
**oracle gate** that uses the combat specialist when a monster is in the central four screen
sectors and the navigation specialist otherwise (the gate reads the environment's private
monster-visibility value, so it is an upper bound on composing the two skills).

| E1M1, 100 seeds | Exits | Died | Kills | Progress |
|---|---:|---:|---:|---:|
| Classical navigation specialist, greedy / sampled | 0 / 0 | 0% / 0% | 0.00 | 0.00 / 0.03 |
| Classical combat specialist, greedy / sampled | 0 / 0 | 0% / 2% | 0.00 | 0.02 / 0.03 |
| Classical oracle-gated mix, greedy / sampled | 0 / 0 | 0% / 0% | 0.00 | 0.00 / 0.03 |
| Circuit navigation specialist, greedy / sampled | 0 / 0 | 0% / 0% | 0.00 | 0.01 / 0.02 |
| Circuit combat specialist, greedy / sampled | 0 / 0 | 1% / 13% | 0.00 | 0.06 / 0.10 |
| Random actions / always-forward (for reference) | 0 / 0 | 46% / 0% | 0.31 / 0.00 | 0.21 / 0.11 |

The specialists do not leave the starting room: they time out at 1,575 decisions with almost no
progress and no kills. As a result no monster is ever in view, and the oracle gate fired on
0% of decisions, so the gated mix is the same as navigation only. The circuit gated runs did
not finish before the job's time limit; the ten completed settings are consistent. Labelling E1M1
states with these specialists would teach nothing. The specialists were trained on screens
of one maze and one corridor and have no route to follow in E1M1, which is the same
failure as the reverse direction in the zero-shot table above.

### Running and next steps

- **Next:** more seeds for all three heads, and fix the waypoint-17 door failures. Per-iteration
  selection on 32 seeds fluctuates (linear: 22-29 of 32), so a single seed cannot rank the
  heads or the iterations.
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
| `run_e1m1_imitation.sh` | E1M1 imitation (`MODEL=`, `TAG=`, `EXTRA=`); `run_record_e1m1_imitation.sh` records a winning episode |
| `run_pixel_smoke.sh`, `run_pixel_corridor_smoke.sh`, `run_pixel_e1m1_smoke.sh` | Short smoke tests (the E1M1 one runs on CPU and builds the distance field first) |

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
| `pixel_e1m1_env.py`, `pixel_e1m1.cfg` | E1M1 screen-only env: 7 actions, walking-distance progress, warp starts |
| `e1m1_distance.py` | Walking-distance field to the exit (`e1m1_distance.npz`) for the progress reward |
| `diagnose_pixel_e1m1.py` | E1M1 smoke checks: field, warps, scripted follower, baselines |
| `e1m1_teacher.py`, `test_e1m1_teacher.py` | Scripted E1M1 teacher (reads private state) and its evaluation |
| `train_e1m1_imitation.py` | DAgger imitation of the teacher from screens, then evaluation |
| `evaluate_e1m1_checkpoints.py`, `evaluate_cross_task.py` | Many-seed checkpoint comparison; E1M1 policies run zero-shot on the other tasks |
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
