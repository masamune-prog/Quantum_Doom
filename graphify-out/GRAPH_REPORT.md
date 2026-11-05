# Graph Report - Quantum_Doom  (2026-11-05)

## Corpus Check
- 234 files · ~780,938 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 353 file(s) not represented in the graph (top: .pt 144, .log 101, .err 93)

## Summary
- 1003 nodes · 2113 edges · 79 communities (73 shown, 6 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 209 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a970dfcd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pixel_nav.py
- run_human_nav_eval.sh
- human_nav_env.py
- train_subtask_baseline.py
- HumanNavigationEnv
- E1M1 Map Geometry
- Navigation Env (My Way Home)
- run_pixel_corridor_linear.sh
- run_pixel_e1m1_curriculum_smoke.sh
- run_e1m1_train.sh
- Data Re-uploading PQC
- E1M1 Map Figure
- run_pixel_corridor_quantum.sh
- Route Check Skill 3
- E1M1 Reference Route
- My Way Home Curve
- Learned screen encoder (CNN to 12-value bottleneck)
- Entanglement Ablation Curve
- Route Check Skill 1
- run_reward_smoke.sh
- Curriculum
- Deadly Corridor Curve
- E1M1 Sanity Figure
- run_pixel_corridor_quantum_resume.sh
- Radar Encoding Sanity Check
- venv_bin_activate
- run_pixel_e1m1_classical.sh
- numpy
- run_e1m1_finetune.sh
- PixelNavigationEnv
- Pixel Classical Job
- Pixel Corridor Classical Job
- train_transfer.py
- Pixel Linear Job
- Pixel Quantum Job
- Pixel Smoke Job
- validate_e1m1_route.py
- ClassicalNavPolicy
- run_e1m1_compare.sh
- run_e1m1_finetune_smoke.sh
- E1M1 Pretrain Job
- run_e1m1_route_check.sh
- train_human_nav.py
- run_e1m1_imitation.sh
- run_pixel_corridor_smoke.sh
- run_e1m1_starts.sh
- run_e1m1_reward_smoke.sh
- run_place_smoke.sh
- QuantumDoomEnv
- frozen_perception_models.py
- run_specialists_on_e1m1.sh
- CLAUDE.md
- .claude/CLAUDE.md
- run_human_nav.sh
- run_human_nav_progress.sh
- run_human_nav_smoke.sh
- run_place_comparison.sh
- torch
- QuantumE1M1Policy
- torch
- run_corridor.sh
- run_e1m1_benchmark.sh
- run_e1m1_cross_task.sh
- record_gameplay.py
- record_nav_agent.py
- HumanNavigationEnv
- train_frozen_perception.py
- train_transfer.py
- pretrain_e1m1.py
- QuantumE1M1Env
- PixelNavPolicy
- diagnose_pixel_aux.py
- Memory bridges (controlled rotations)
- run_e1m1_robustness.sh
- run_e1m1_unstick_check.sh
- run_frozen_perception.sh
- run_transfer.sh
- os
- run_e1m1_sanity.sh

## God Nodes (most connected - your core abstractions)
1. `DoomMap` - 31 edges
2. `PixelNavPolicy` - 29 edges
3. `HumanNavigationEnv` - 27 edges
4. `PixelE1M1Env` - 22 edges
5. `EnvPool` - 22 edges
6. `QuantumE1M1Env` - 21 edges
7. `PixelNavigationEnv` - 20 edges
8. `QuantumE1M1Policy` - 20 edges
9. `QuantumDoomEnv` - 19 edges
10. `TransferTests` - 18 edges

## Surprising Connections (you probably didn't know these)
- `12-qubit circuit head result (96/100)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `Memory bridges (controlled rotations)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `Success-detection bug` --references--> `PixelNavigationEnv`  [EXTRACTED]
  README.md → pixel_nav_env.py
- `Quantum Doom` --references--> `DoomActorCritic`  [EXTRACTED]
  README.md → models.py
- `Private progress and kill rewards` --references--> `HumanNavigationEnv`  [EXTRACTED]
  README.md → human_nav_env.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Policy architecture ablation (entangled PQC / product PQC / classical MLP)** — ablation_curve_quantum_entangled, ablation_curve_quantum_product, ablation_curve_classical_mlp [EXTRACTED 1.00]
- **E1M1 map-plan-validate-train pipeline** — readme_route_planner_dijkstra, readme_route_validation_scripted_follower, readme_e1m1_12_qubit_observation, readme_quantum_only_policy_value_model, readme_reverse_start_curriculum [EXTRACTED 1.00]
- **Quantum PPO navigation training setup** — nav_training_curve_quantum_pqc_agent, nav_training_curve_ppo, nav_training_curve_action_space, nav_training_curve_my_way_home [EXTRACTED 1.00]
- **Hand-crafted features to PQC actor with PPO/GAE** — readme_hand_crafted_angle_features, readme_data_reuploading_pqc, readme_classical_mlp_critic, readme_ppo_with_gae [EXTRACTED 1.00]
- **Start-to-exit navigation path** — e1m1_map_start, e1m1_map_central_courtyard, e1m1_map_zigzag_room, e1m1_map_exit_hall, e1m1_map_exit_switch [INFERRED 0.65]
- **On-route navigation obstacles** — e1m1_route_waypoint_path, e1m1_route_doors, e1m1_route_monsters [INFERRED 0.75]
- **Screen-to-Qubit Encoding Sanity Pipeline** — e1m1_sanity_screen_observation, e1m1_sanity_qubit_input_angles, e1m1_sanity_angle_encoding [INFERRED 0.75]
- **E1M1 Route Validation (map + waypoints + seed runs)** — e1m1_route_check_skill1_e1m1_map, e1m1_route_check_skill1_reference_waypoints, e1m1_route_check_skill1_seed_trajectories [INFERRED 0.85]
- **Route validation: waypoints vs seed trajectories on E1M1** — e1m1_route_check_skill3_waypoint_route, e1m1_route_check_skill3_seed_trajectories, e1m1_route_check_skill3_e1m1_map, e1m1_route_check_skill3_exit_switch [INFERRED 0.85]
- **Human-visible screen-only navigation experiments** — readme_human_visible_observation, readme_screen_only_navigation_memory, readme_rgb_place_memory, readme_matched_policy_comparison, readme_learned_screen_encoder, readme_circuit_head_result [INFERRED 0.85]
- **8-qubit PQC trained with PPO on Deadly Corridor** — corridor_training_curve_quantum_entangled_pqc, corridor_training_curve_ppo, corridor_training_curve_deadly_corridor [INFERRED 0.95]
- **Frame-to-Qubit Preprocessing Pipeline** — sanity_check_raw_rgb_frame, sanity_check_label_buffer, sanity_check_qubit_input_angles [INFERRED 0.95]

## Communities (79 total, 6 thin omitted)

### Community 0 - "train_pixel_nav.py"
Cohesion: 0.08
Nodes (17): load(), main(), Restricted, main(), load(), main(), run(), EnvPool (+9 more)

### Community 1 - "run_human_nav_eval.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_eval.sh script

### Community 2 - "human_nav_env.py"
Cohesion: 0.18
Nodes (3): goal_potential(), kill_reward(), progress_reward()

### Community 3 - "train_subtask_baseline.py"
Cohesion: 0.06
Nodes (27): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline.sh script, MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline_smoke.sh script (+19 more)

### Community 5 - "E1M1 Map Geometry"
Cohesion: 0.06
Nodes (22): build_field(), main(), DoomMap, skill_flag(), main(), point_segment_distances(), RoutePlanner, OffRouteStarts (+14 more)

### Community 6 - "Navigation Env (My Way Home)"
Cohesion: 0.13
Nodes (5): QuantumNavigationEnv, _convert_h264_if_available(), record_nav_episodes(), render_nav_hud(), NavigationTask

### Community 7 - "run_pixel_corridor_linear.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_linear.sh script

### Community 8 - "run_pixel_e1m1_curriculum_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_curriculum_smoke.sh script

### Community 9 - "run_e1m1_train.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_train.sh script

### Community 10 - "Data Re-uploading PQC"
Cohesion: 0.14
Nodes (10): DataReuploadingPQC, Classical MLP Critic, Data Re-uploading PQC, Entanglement ablation (basic.cfg), Hand-crafted angle features (depth + labels buffers), Known issues, PennyLane, PPO with GAE (+2 more)

### Community 11 - "E1M1 Map Figure"
Cohesion: 0.22
Nodes (12): Central Large Room / Courtyard, Doors (orange), Doom E1M1 Level (Hangar), Southern Exit Hall and Exit Room, Exit (green, ~2910,-4790), E1M1 Top-Down Map (skill 3), Items/Pickups (blue), Monsters (red, ~6 at skill 3) (+4 more)

### Community 12 - "run_pixel_corridor_quantum.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_quantum.sh script

### Community 13 - "Route Check Skill 3"
Cohesion: 0.31
Nodes (9): Doors on Route (waypoints 5, 16/17, 20), E1M1 Level Map (walls, doors, exit, monsters, items), Exit (green, ~2910,-4770), E1M1 Route Check (Skill 3) Map Figure, Monsters (red markers near route), Agent Trajectories Seeds 0-4, Doom Skill Level 3, START Position (~1060,-3620) (+1 more)

### Community 14 - "E1M1 Reference Route"
Cohesion: 0.22
Nodes (9): Doors on Route (orange), E1M1 Map (skill 3), Exit Switch (green), E1M1 Route Map Figure, Items (blue), Monsters (red), Start Position, Waypoint Route (22 waypoints, 0-21) (+1 more)

### Community 15 - "My Way Home Curve"
Cohesion: 0.22
Nodes (8): 7-Action Space (Corridor + USE), My Way Home 10-Qubit Quantum Agent Training Curve, Goal Success Rate (%), Mean Step Reward, My Way Home Scenario, PPO Update, Quantum PQC Agent (10 qubits, 4 layers, 7 actions), Smoothed Reward (window=5)

### Community 16 - "Learned screen encoder (CNN to 12-value bottleneck)"
Cohesion: 0.29
Nodes (6): Auxiliary depth and goal-visibility losses, Lample & Chaplot 2017, Mirowski et al. 2016 (auxiliary-task navigation), Potential-based reward shaping, Private progress and kill rewards, Screen-only navigation memory experiment

### Community 17 - "Entanglement Ablation Curve"
Cohesion: 0.31
Nodes (7): classical_mlp (Classical MLP baseline), Ablation Curve: Entangled PQC vs Unentangled vs Classical MLP (ViZDoom Aiming & Combat), Mean Step Reward, PPO Rollout Updates (x1024 frames), quantum_entangled (Entangled PQC policy), quantum_product (Unentangled product-state PQC), ViZDoom Aiming & Combat Scenario

### Community 18 - "Route Check Skill 1"
Cohesion: 0.36
Nodes (7): Doors on Route (orange, near waypoints 5, 16-17, 20), E1M1 Map Layout (walls, doors, exit, monsters, items), E1M1 Exit (green, bottom-right room), E1M1 Route Check (Skill 1) Map Figure, Reference Route Waypoints 0-21 (dashed purple), Agent Trajectories Seeds 0-4, Doom Skill Level 1

### Community 19 - "run_reward_smoke.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_reward_smoke.sh script

### Community 21 - "Deadly Corridor Curve"
Cohesion: 0.48
Nodes (6): Deadly Corridor: 8-Qubit Quantum Agent Training (figure), Deadly Corridor Scenario, Mean Step Reward, PPO Update, Quantum Entangled PQC (8 qubits, 4 layers), Smoothed Reward (window=5)

### Community 22 - "E1M1 Sanity Figure"
Cohesion: 0.33
Nodes (6): Agent View at Exit Switch, Angle Encoding Range [-pi, pi], E1M1 Exit Switch, E1M1 Sanity Check Figure, 12-Qubit Input Angles, 320x240 Screen Observation

### Community 23 - "run_pixel_corridor_quantum_resume.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_quantum_resume.sh script

### Community 24 - "Radar Encoding Sanity Check"
Cohesion: 0.38
Nodes (5): Sanity Check Figure (Frame -> Label Buffer -> 8-Qubit Angles), FOV Sector to Qubit Mapping (Sectors 0-7), Native ViZDoom Label Buffer (Zero CNN), 8-Qubit Input Angles (FOV Sector Encoding), Raw ViZDoom RGB Frame

### Community 25 - "venv_bin_activate"
Cohesion: 0.12
Nodes (11): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor_aim_probe.sh script, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_imitation_smoke.sh script, PYTHONUNBUFFERED, run_e1m1_teacher_test.sh script (+3 more)

### Community 26 - "run_pixel_e1m1_classical.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_classical.sh script

### Community 27 - "numpy"
Cohesion: 0.19
Nodes (5): build(), evaluate(), main(), summarize(), summarize()

### Community 28 - "run_e1m1_finetune.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_finetune.sh script

### Community 29 - "PixelNavigationEnv"
Cohesion: 0.06
Nodes (16): baseline(), check_field(), check_warps(), follow(), main(), DistanceField, _wrap_angle(), RouteTeacher (+8 more)

### Community 30 - "Pixel Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_classical.sh script

### Community 31 - "Pixel Corridor Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_classical.sh script

### Community 32 - "train_transfer.py"
Cohesion: 0.06
Nodes (19): TransferTests, action_metrics(), cache(), collect(), evaluate(), fit_actor(), PerceptionSubset, Replay (+11 more)

### Community 33 - "Pixel Linear Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_linear.sh script

### Community 34 - "Pixel Quantum Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_quantum.sh script

### Community 35 - "Pixel Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_smoke.sh script

### Community 36 - "validate_e1m1_route.py"
Cohesion: 0.25
Nodes (6): main(), main(), run_episode(), ScriptedFollower, main(), run()

### Community 37 - "ClassicalNavPolicy"
Cohesion: 0.12
Nodes (3): ClassicalNavPolicy, LinearNavHead, run_circuit_ablation_test.sh script

### Community 38 - "run_e1m1_compare.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_compare.sh script

### Community 39 - "run_e1m1_finetune_smoke.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_finetune_smoke.sh script

### Community 40 - "E1M1 Pretrain Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_pretrain.sh script

### Community 41 - "run_e1m1_route_check.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_route_check.sh script

### Community 42 - "train_human_nav.py"
Cohesion: 0.25
Nodes (5): HumanE1M1Env, run_place_smoke.sh script, evaluate(), make_env(), train()

### Community 43 - "run_e1m1_imitation.sh"
Cohesion: 0.40
Nodes (4): MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_imitation.sh script

### Community 44 - "run_pixel_corridor_smoke.sh"
Cohesion: 0.25
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_smoke.sh script

### Community 45 - "run_e1m1_starts.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_starts.sh script

### Community 46 - "run_e1m1_reward_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_reward_smoke.sh script

### Community 47 - "run_place_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED

### Community 48 - "QuantumDoomEnv"
Cohesion: 0.14
Nodes (4): QuantumDoomEnv, record_episodes(), CombatTask, DodgeTask

### Community 49 - "frozen_perception_models.py"
Cohesion: 0.14
Nodes (5): FrozenPolicy, GeometryPerception, make_head(), SharedMLP, FrozenPerceptionTests

### Community 50 - "run_specialists_on_e1m1.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_specialists_on_e1m1.sh script

### Community 53 - "run_human_nav.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav.sh script

### Community 54 - "run_human_nav_progress.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_progress.sh script

### Community 55 - "run_human_nav_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_smoke.sh script

### Community 56 - "run_place_comparison.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_place_comparison.sh script

### Community 57 - "torch"
Cohesion: 0.15
Nodes (4): collect(), Dataset, fit(), main()

### Community 58 - "QuantumE1M1Policy"
Cohesion: 0.15
Nodes (7): QuantumE1M1Policy, circuit(), 12-qubit circuit head result (96/100), Circuit imitation test, Quantum-only policy and value model, MPLCONFIGDIR, run_draw_circuit.sh script

### Community 59 - "torch"
Cohesion: 0.17
Nodes (8): DoomActorCritic, train_one_run(), make_corridor_env(), _init(), train_corridor(), make_nav_env(), _init(), train_navigation()

### Community 60 - "run_corridor.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor.sh script

### Community 61 - "run_e1m1_benchmark.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_benchmark.sh script

### Community 62 - "run_e1m1_cross_task.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_cross_task.sh script

### Community 63 - "record_gameplay.py"
Cohesion: 0.09
Nodes (16): caption(), default_targets(), describe(), main(), record(), MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED (+8 more)

### Community 64 - "record_nav_agent.py"
Cohesion: 0.10
Nodes (6): step(), render_radar_overlay(), MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_navigation.sh script

### Community 65 - "HumanNavigationEnv"
Cohesion: 0.26
Nodes (3): main(), run(), HumanNavigationEnv

### Community 66 - "train_frozen_perception.py"
Cohesion: 0.12
Nodes (14): audit(), fingerprint(), next_cooldown(), teacher_eval(), collect(), constant_geometry_metrics(), encode_replay(), evaluate() (+6 more)

### Community 67 - "train_transfer.py"
Cohesion: 0.17
Nodes (3): UniformStarts, fit_perception(), perception_metrics()

### Community 68 - "pretrain_e1m1.py"
Cohesion: 0.23
Nodes (7): collect(), main(), Reverse-start curriculum, Route validation (scripted follower), evaluate(), reset_at(), train()

### Community 70 - "PixelNavPolicy"
Cohesion: 0.18
Nodes (5): main(), perturb(), run(), PixelNavPolicy, load()

### Community 72 - "Memory bridges (controlled rotations)"
Cohesion: 0.50
Nodes (4): Action-based travel map, Matched classical vs bridged-quantum comparison, Memory bridges (controlled rotations), RGB place memory (VisualPlaceMemory)

### Community 73 - "run_e1m1_robustness.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_robustness.sh script

### Community 74 - "run_e1m1_unstick_check.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_unstick_check.sh script

### Community 77 - "run_frozen_perception.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_frozen_perception.sh script

### Community 78 - "run_transfer.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_transfer.sh script

### Community 79 - "os"
Cohesion: 0.24
Nodes (5): Classical subtask baselines, Deadly Corridor scenario, My Way Home scenario, Success-detection bug, Take Cover dodging task

### Community 81 - "run_e1m1_sanity.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_sanity.sh script

## Ambiguous Edges - Review These
- `Doors (orange)` → `Start-to-exit route requires traversing east then south through multiple doors`  [AMBIGUOUS]
  e1m1_map.png · relation: rationale_for
- `Exit (green, ~2910,-4770)` → `Agent Trajectories Seeds 0-4`  [AMBIGUOUS]
  e1m1_route_check_skill3.png · relation: conceptually_related_to
- `12-Qubit Input Angles` → `Empty Input-Angle Plot Finding`  [AMBIGUOUS]
  e1m1_sanity.png · relation: rationale_for

## Knowledge Gaps
- **189 isolated node(s):** `MKL_NUM_THREADS`, `MPLCONFIGDIR`, `OMP_NUM_THREADS`, `PYTHONUNBUFFERED`, `MKL_NUM_THREADS` (+184 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 409 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PixelNavigationEnv` connect `PixelNavigationEnv` to `train_pixel_nav.py`, `HumanNavigationEnv`, `human_nav_env.py`, `train_transfer.py`, `E1M1 Map Geometry`, `diagnose_pixel_aux.py`, `os`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `DoomMap` (e.g. with `QuantumE1M1Env` and `RoutePlanner`) actually correct?**
  _`DoomMap` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `MKL_NUM_THREADS`, `MPLCONFIGDIR`, `OMP_NUM_THREADS` to the rest of the system?**
  _189 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pixel_nav.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07878787878787878 - nodes in this community are weakly interconnected._
- **Why does `QuantumE1M1Policy` connect `QuantumE1M1Policy` to `HumanNavigationEnv`, `pretrain_e1m1.py`, `ClassicalNavPolicy`, `PixelNavPolicy`, `Memory bridges (controlled rotations)`, `train_human_nav.py`, `frozen_perception_models.py`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Doors (orange)` and `Start-to-exit route requires traversing east then south through multiple doors`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Exit (green, ~2910,-4770)` and `Agent Trajectories Seeds 0-4`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._