# Graph Report - Quantum_Doom  (2026-11-12)

## Corpus Check
- 376 files · ~3,790,877 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 511 file(s) not represented in the graph (top: .pt 230, .log 165, .err 93)

## Summary
- 1388 nodes · 3062 edges · 114 communities (83 shown, 31 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 226 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0be96ac9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EnvPool
- RoutePlanner
- frozen_perception_models.py
- run_subtask_baselines.sh
- HumanNavigationEnv
- RoutePlanner
- Navigation Env (My Way Home)
- run_pixel_corridor_linear.sh
- PixelE1M1Env
- QuantumE1M1Policy
- Data Re-uploading PQC
- E1M1 Map Figure
- run_pixel_corridor_quantum.sh
- Route Check Skill 3
- E1M1 Reference Route
- My Way Home Curve
- QuantumDoomEnv
- Entanglement Ablation Curve
- Route Check Skill 1
- numpy
- Curriculum
- Deadly Corridor Curve
- run_dodge_baseline_smoke.sh
- run_pixel_corridor_quantum_resume.sh
- Radar Encoding Sanity Check
- venv_bin_activate
- run_pixel_e1m1_classical.sh
- HumanNavigationEnv
- run_e1m1_finetune.sh
- pixel_e1m1_env.py
- Pixel Classical Job
- Pixel Corridor Classical Job
- train_transfer.py
- Pixel Linear Job
- Pixel Quantum Job
- Pixel Smoke Job
- QuantumE1M1Env
- QuantumE1M1Policy
- run_e1m1_compare.sh
- PixelE1M1Env
- run_subtask_baselines_extended.sh
- draw_circuit.py
- train_human_nav.py
- run_e1m1_imitation.sh
- run_pixel_corridor_smoke.sh
- replication/pixel_nav_models.py
- run_e1m1_reward_smoke.sh
- train_subtask_baseline.py
- pixel_nav_env.py
- train_frozen_perception.py
- replication/train_pixel_nav.py
- CLAUDE.md
- .claude/CLAUDE.md
- run_human_nav.sh
- run_human_nav_progress.sh
- run_human_nav_smoke.sh
- run_place_comparison.sh
- torch
- matplotlib_pyplot
- train_e1m1_imitation.py
- record_agent.py
- e1m1_distance.py
- torch
- record_gameplay.py
- train_canny_e1m1.py
- HumanNavigationEnv
- train_pixel_nav.py
- run_e1m1_cross_task.sh
- human_nav_env.py
- PixelNavPolicy
- DoomMap
- probe_frozen_heads.py
- run_dodge_baseline.sh
- run_specialists_on_e1m1.sh
- run_subtask_baseline_smoke.sh
- Classical subtask baselines
- run_e1m1_finetune_smoke.sh
- run_frozen_perception.sh
- run_transfer.sh
- json
- replication/evaluate_e1m1_robustness.py
- launch.sh
- progress_reward
- Training Quantum Neural Networks to play Doom: replication package
- replication/record_gameplay.py
- DistanceField
- launch20.sh
- replication/diagnose_pixel_e1m1.py
- replication/evaluate_cross_task.py
- EnvPool
- common.sh
- Curriculum
- evaluate_specialists_on_e1m1.py
- .__init__
- RouteTeacher
- EpisodeTracker
- run_pixel_e1m1_smoke.sh
- DistanceField
- run_pixel_e1m1_curriculum_smoke.sh
- run_subtask_baselines_extended_smoke.sh
- test_circuit_ablations.py
- run_corridor_aim_probe.sh
- run_e1m1_imitation_smoke.sh
- .things_for_skill
- 00_smoke_test.sh
- 01_pilot_my_way_home.sh
- 02_pilot_deadly_corridor.sh
- 03_e1m1_ppo.sh
- 04_e1m1_teacher.sh
- 05_e1m1_imitation.sh
- 06_e1m1_finetune_300_seeds.sh
- 07_e1m1_memorisation.sh
- 08_transfer.sh
- 09_figures_and_videos.sh

## God Nodes (most connected - your core abstractions)
1. `PixelNavPolicy` - 42 edges
2. `EnvPool` - 40 edges
3. `DoomMap` - 38 edges
4. `PixelE1M1Env` - 27 edges
5. `HumanNavigationEnv` - 24 edges
6. `QuantumE1M1Policy` - 21 edges
7. `PixelCorridorEnv` - 21 edges
8. `QuantumDoomEnv` - 19 edges
9. `PixelNavigationEnv` - 19 edges
10. `RoutePlanner` - 19 edges

## Surprising Connections (you probably didn't know these)
- `12-qubit circuit head result (96/100)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `check_warps()` --calls--> `_wrap_angle()`  [EXTRACTED]
  replication/diagnose_pixel_e1m1.py → e1m1_env.py
- `follow()` --calls--> `_wrap_angle()`  [EXTRACTED]
  replication/diagnose_pixel_e1m1.py → e1m1_env.py
- `main()` --calls--> `PixelE1M1Env`  [EXTRACTED]
  replication/diagnose_pixel_e1m1.py → pixel_e1m1_env.py
- `main()` --calls--> `DoomMap`  [EXTRACTED]
  replication/e1m1_distance.py → e1m1_map.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Policy architecture ablation (entangled PQC / product PQC / classical MLP)** — ablation_curve_quantum_entangled, ablation_curve_quantum_product, ablation_curve_classical_mlp [EXTRACTED 1.00]
- **E1M1 map-plan-validate-train pipeline** — readme_route_planner_dijkstra, readme_route_validation_scripted_follower, readme_e1m1_12_qubit_observation, readme_quantum_only_policy_value_model, readme_reverse_start_curriculum [EXTRACTED 1.00]
- **Quantum PPO navigation training setup** — nav_training_curve_quantum_pqc_agent, nav_training_curve_ppo, nav_training_curve_action_space, nav_training_curve_my_way_home [EXTRACTED 1.00]
- **Hand-crafted features to PQC actor with PPO/GAE** — readme_hand_crafted_angle_features, readme_data_reuploading_pqc, readme_classical_mlp_critic, readme_ppo_with_gae [EXTRACTED 1.00]
- **Start-to-exit navigation path** — e1m1_map_start, e1m1_map_central_courtyard, e1m1_map_zigzag_room, e1m1_map_exit_hall, e1m1_map_exit_switch [INFERRED 0.65]
- **On-route navigation obstacles** — e1m1_route_waypoint_path, e1m1_route_doors, e1m1_route_monsters [INFERRED 0.75]
- **E1M1 Route Validation (map + waypoints + seed runs)** — e1m1_route_check_skill1_e1m1_map, e1m1_route_check_skill1_reference_waypoints, e1m1_route_check_skill1_seed_trajectories [INFERRED 0.85]
- **Route validation: waypoints vs seed trajectories on E1M1** — e1m1_route_check_skill3_waypoint_route, e1m1_route_check_skill3_seed_trajectories, e1m1_route_check_skill3_e1m1_map, e1m1_route_check_skill3_exit_switch [INFERRED 0.85]
- **Human-visible screen-only navigation experiments** — readme_human_visible_observation, readme_screen_only_navigation_memory, readme_rgb_place_memory, readme_matched_policy_comparison, readme_learned_screen_encoder, readme_circuit_head_result [INFERRED 0.85]
- **8-qubit PQC trained with PPO on Deadly Corridor** — corridor_training_curve_quantum_entangled_pqc, corridor_training_curve_ppo, corridor_training_curve_deadly_corridor [INFERRED 0.95]
- **Frame-to-Qubit Preprocessing Pipeline** — sanity_check_raw_rgb_frame, sanity_check_label_buffer, sanity_check_qubit_input_angles [INFERRED 0.95]

## Communities (114 total, 31 thin omitted)

### Community 0 - "EnvPool"
Cohesion: 0.16
Nodes (10): main(), main(), perturb(), run(), EnvPool, train(), train(), reset_all() (+2 more)

### Community 1 - "RoutePlanner"
Cohesion: 0.08
Nodes (6): QuantumE1M1Env, _wrap_angle(), DoomMap, skill_flag(), point_segment_distances(), RoutePlanner

### Community 2 - "frozen_perception_models.py"
Cohesion: 0.15
Nodes (3): ClassicalNavPolicy, LinearNavHead, build()

### Community 3 - "run_subtask_baselines.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_subtask_baselines.sh script

### Community 6 - "Navigation Env (My Way Home)"
Cohesion: 0.08
Nodes (11): QuantumNavigationEnv, _convert_h264_if_available(), record_nav_episodes(), render_nav_hud(), MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_navigation.sh script (+3 more)

### Community 7 - "run_pixel_corridor_linear.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_linear.sh script

### Community 8 - "PixelE1M1Env"
Cohesion: 0.11
Nodes (5): RouteTeacher, PixelCorridorEnv, PixelE1M1Env, make_pixel_env(), PixelNavigationEnv

### Community 9 - "QuantumE1M1Policy"
Cohesion: 0.12
Nodes (15): QuantumE1M1Policy, circuit(), Action-based travel map, Auxiliary depth and goal-visibility losses, 12-qubit circuit head result (96/100), Freinberger et al. 2024 (hybrid Atari), Lample & Chaplot 2017, Learned screen encoder (CNN to 12-value bottleneck) (+7 more)

### Community 10 - "Data Re-uploading PQC"
Cohesion: 0.11
Nodes (18): Circuit imitation test, Classical MLP Critic, Data Re-uploading PQC, Shareware doom1.wad v1.9, E1M1 (Hangar), E1M1 pipeline, Entanglement ablation (basic.cfg), Hand-crafted angle features (depth + labels buffers) (+10 more)

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

### Community 16 - "QuantumDoomEnv"
Cohesion: 0.13
Nodes (6): QuantumDoomEnv, record_episodes(), CombatTask, DodgeTask, make_corridor_env(), _init()

### Community 17 - "Entanglement Ablation Curve"
Cohesion: 0.31
Nodes (7): classical_mlp (Classical MLP baseline), Ablation Curve: Entangled PQC vs Unentangled vs Classical MLP (ViZDoom Aiming & Combat), Mean Step Reward, PPO Rollout Updates (x1024 frames), quantum_entangled (Entangled PQC policy), quantum_product (Unentangled product-state PQC), ViZDoom Aiming & Combat Scenario

### Community 18 - "Route Check Skill 1"
Cohesion: 0.36
Nodes (7): Doors on Route (orange, near waypoints 5, 16-17, 20), E1M1 Map Layout (walls, doors, exit, monsters, items), E1M1 Exit (green, bottom-right room), E1M1 Route Check (Skill 1) Map Figure, Reference Route Waypoints 0-21 (dashed purple), Agent Trajectories Seeds 0-4, Doom Skill Level 1

### Community 19 - "numpy"
Cohesion: 0.12
Nodes (6): Success-detection bug, _point_segment_distance(), field_progress_reward(), goal_potential(), kill_reward(), progress_reward()

### Community 21 - "Deadly Corridor Curve"
Cohesion: 0.48
Nodes (6): Deadly Corridor: 8-Qubit Quantum Agent Training (figure), Deadly Corridor Scenario, Mean Step Reward, PPO Update, Quantum Entangled PQC (8 qubits, 4 layers), Smoothed Reward (window=5)

### Community 22 - "run_dodge_baseline_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline_smoke.sh script

### Community 23 - "run_pixel_corridor_quantum_resume.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_quantum_resume.sh script

### Community 24 - "Radar Encoding Sanity Check"
Cohesion: 0.38
Nodes (5): Sanity Check Figure (Frame -> Label Buffer -> 8-Qubit Angles), FOV Sector to Qubit Mapping (Sectors 0-7), Native ViZDoom Label Buffer (Zero CNN), 8-Qubit Input Angles (FOV Sector Encoding), Raw ViZDoom RGB Frame

### Community 25 - "venv_bin_activate"
Cohesion: 0.14
Nodes (9): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_robustness.sh script, PYTHONUNBUFFERED (+1 more)

### Community 26 - "run_pixel_e1m1_classical.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_classical.sh script

### Community 27 - "HumanNavigationEnv"
Cohesion: 0.11
Nodes (3): HumanNavigationEnv, VisualPlaceMemory, VisualTravelMap

### Community 28 - "run_e1m1_finetune.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_finetune.sh script

### Community 29 - "pixel_e1m1_env.py"
Cohesion: 0.16
Nodes (7): baseline(), check_field(), check_warps(), follow(), main(), _point_segment_distance(), _wrap_angle()

### Community 30 - "Pixel Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_classical.sh script

### Community 31 - "Pixel Corridor Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_classical.sh script

### Community 32 - "train_transfer.py"
Cohesion: 0.05
Nodes (22): next_cooldown(), TransferTests, action_metrics(), cache(), collect(), evaluate(), fit_actor(), fit_perception() (+14 more)

### Community 33 - "Pixel Linear Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_linear.sh script

### Community 34 - "Pixel Quantum Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_quantum.sh script

### Community 35 - "Pixel Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_smoke.sh script

### Community 36 - "QuantumE1M1Env"
Cohesion: 0.05
Nodes (29): QuantumE1M1Env, E1M1 12-qubit observation, Reverse-start curriculum, Route planner (grid Dijkstra), Route validation (scripted follower), MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED (+21 more)

### Community 37 - "QuantumE1M1Policy"
Cohesion: 0.17
Nodes (5): check(), test_matches_pennylane(), test_param_count(), _cnot_ring_permutation(), WideCircuitPolicy

### Community 38 - "run_e1m1_compare.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_compare.sh script

### Community 40 - "run_subtask_baselines_extended.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_subtask_baselines_extended.sh script

### Community 41 - "draw_circuit.py"
Cohesion: 0.18
Nodes (3): QuantumE1M1Policy, circuit(), run_draw_circuit.sh script

### Community 42 - "train_human_nav.py"
Cohesion: 0.39
Nodes (4): HumanE1M1Env, evaluate(), make_env(), train()

### Community 43 - "run_e1m1_imitation.sh"
Cohesion: 0.40
Nodes (4): MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_imitation.sh script

### Community 44 - "run_pixel_corridor_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_smoke.sh script

### Community 45 - "replication/pixel_nav_models.py"
Cohesion: 0.14
Nodes (3): ClassicalNavPolicy, LinearNavHead, PixelNavPolicy

### Community 46 - "run_e1m1_reward_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_reward_smoke.sh script

### Community 47 - "train_subtask_baseline.py"
Cohesion: 0.15
Nodes (6): DataReuploadingPQC, DoomActorCritic, gae_step_dones(), checkpoint_score(), evaluate(), train()

### Community 48 - "pixel_nav_env.py"
Cohesion: 0.13
Nodes (9): CannyCorridorEnv, CannyEnemyCorridorEnv, CannyE1M1Env, PixelCorridorEnv, make_pixel_env(), PixelNavigationEnv, _worker(), collect() (+1 more)

### Community 49 - "train_frozen_perception.py"
Cohesion: 0.06
Nodes (23): audit(), fingerprint(), FrozenPolicy, GeometryPerception, fit(), main(), inputs(), make_head() (+15 more)

### Community 50 - "replication/train_pixel_nav.py"
Cohesion: 0.15
Nodes (7): evaluate(), summary(), train(), reset_all(), reset_env(), train_seed(), UniformStarts

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
Cohesion: 0.20
Nodes (9): compute_gae(), EpisodeTracker, final_info(), make_vector_env(), ppo_update(), timeout_bootstrap(), train_one_run(), train_corridor() (+1 more)

### Community 59 - "train_e1m1_imitation.py"
Cohesion: 0.10
Nodes (9): collect(), Dataset, fit(), main(), teacher_eval(), collect(), fit(), main() (+1 more)

### Community 61 - "e1m1_distance.py"
Cohesion: 0.17
Nodes (5): build_field(), main(), build_field(), main(), main()

### Community 62 - "torch"
Cohesion: 0.19
Nodes (5): compute_gae(), final_info(), gae_step_dones(), ppo_update(), timeout_bootstrap()

### Community 63 - "record_gameplay.py"
Cohesion: 0.09
Nodes (16): caption(), default_targets(), describe(), main(), record(), MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED (+8 more)

### Community 64 - "train_canny_e1m1.py"
Cohesion: 0.15
Nodes (7): build_head(), collect(), Data, FeaturePolicy, run(), fit_heads(), heldout_metrics()

### Community 66 - "train_pixel_nav.py"
Cohesion: 0.15
Nodes (6): load(), main(), Restricted, main(), evaluate(), summary()

### Community 67 - "run_e1m1_cross_task.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_cross_task.sh script

### Community 68 - "human_nav_env.py"
Cohesion: 0.20
Nodes (3): enemy_features(), enemy_mask(), screen_features()

### Community 69 - "PixelNavPolicy"
Cohesion: 0.22
Nodes (5): PixelNavPolicy, load(), load(), main(), run()

### Community 71 - "probe_frozen_heads.py"
Cohesion: 0.21
Nodes (6): make_head(), SharedMLP, build(), evaluate(), head_name(), main()

### Community 72 - "run_dodge_baseline.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline.sh script

### Community 73 - "run_specialists_on_e1m1.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_specialists_on_e1m1.sh script

### Community 74 - "run_subtask_baseline_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_subtask_baseline_smoke.sh script

### Community 75 - "Classical subtask baselines"
Cohesion: 0.50
Nodes (4): Classical subtask baselines, Deadly Corridor scenario, My Way Home scenario, Take Cover dodging task

### Community 76 - "run_e1m1_finetune_smoke.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_finetune_smoke.sh script

### Community 77 - "run_frozen_perception.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_frozen_perception.sh script

### Community 78 - "run_transfer.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_transfer.sh script

### Community 80 - "replication/evaluate_e1m1_robustness.py"
Cohesion: 0.27
Nodes (4): main(), OffRouteStarts, perturb(), run()

### Community 81 - "launch.sh"
Cohesion: 0.67
Nodes (3): job(), OMP_NUM_THREADS, launch.sh script

### Community 82 - "progress_reward"
Cohesion: 0.22
Nodes (4): field_progress_reward(), goal_potential(), kill_reward(), progress_reward()

### Community 83 - "Training Quantum Neural Networks to play Doom: replication package"
Cohesion: 0.22
Nodes (8): 1. Setup, 2. Fast path: check the reported numbers without training, 3. Full replication from scratch, 4. Protocol (shared by all experiments), 5. Files, 6. Notes and known differences, Training Quantum Neural Networks to play Doom: replication package, What to expect

### Community 84 - "replication/record_gameplay.py"
Cohesion: 0.33
Nodes (5): caption(), default_targets(), describe(), main(), record()

### Community 87 - "replication/diagnose_pixel_e1m1.py"
Cohesion: 0.39
Nodes (5): baseline(), check_field(), check_warps(), follow(), main()

### Community 88 - "replication/evaluate_cross_task.py"
Cohesion: 0.32
Nodes (3): load(), main(), Restricted

### Community 90 - "common.sh"
Cohesion: 0.25
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, common.sh script

### Community 92 - "evaluate_specialists_on_e1m1.py"
Cohesion: 0.38
Nodes (3): load(), main(), run()

### Community 96 - "run_pixel_e1m1_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_smoke.sh script

### Community 98 - "run_pixel_e1m1_curriculum_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_curriculum_smoke.sh script

### Community 99 - "run_subtask_baselines_extended_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_subtask_baselines_extended_smoke.sh script

### Community 101 - "run_corridor_aim_probe.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor_aim_probe.sh script

### Community 102 - "run_e1m1_imitation_smoke.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_imitation_smoke.sh script

## Ambiguous Edges - Review These
- `Doors (orange)` → `Start-to-exit route requires traversing east then south through multiple doors`  [AMBIGUOUS]
  e1m1_map.png · relation: rationale_for
- `Exit (green, ~2910,-4770)` → `Agent Trajectories Seeds 0-4`  [AMBIGUOUS]
  e1m1_route_check_skill3.png · relation: conceptually_related_to

## Knowledge Gaps
- **175 isolated node(s):** `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `MPLCONFIGDIR`, `MKL_NUM_THREADS` (+170 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 568 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PixelNavPolicy` connect `PixelNavPolicy` to `EnvPool`, `train_transfer.py`, `train_pixel_nav.py`, `frozen_perception_models.py`, `test_circuit_ablations.py`, `QuantumE1M1Policy`, `pixel_nav_env.py`, `replication/evaluate_e1m1_robustness.py`, `replication/train_pixel_nav.py`, `replication/record_gameplay.py`, `replication/evaluate_cross_task.py`, `train_e1m1_imitation.py`, `evaluate_specialists_on_e1m1.py`, `record_gameplay.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `PixelNavPolicy` (e.g. with `QuantumE1M1Policy` and `ClassicalNavPolicy`) actually correct?**
  _`PixelNavPolicy` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` to the rest of the system?**
  _175 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `RoutePlanner` be split into smaller, more focused modules?**
  _Cohesion score 0.07716701902748414 - nodes in this community are weakly interconnected._
- **Why does `RoutePlanner` connect `RoutePlanner` to `e1m1_distance.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Doors (orange)` and `Start-to-exit route requires traversing east then south through multiple doors`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Exit (green, ~2910,-4770)` and `Agent Trajectories Seeds 0-4`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._