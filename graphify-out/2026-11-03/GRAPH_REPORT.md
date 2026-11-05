# Graph Report - Quantum_Doom  (2026-11-03)

## Corpus Check
- 182 files · ~225,819 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 328 file(s) not represented in the graph (top: .pt 136, .log 95, .err 87)

## Summary
- 877 nodes · 1732 edges · 74 communities (61 shown, 13 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 177 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f6f060ae`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pixel_nav.py
- run_human_nav_eval.sh
- QuantumDoomEnv
- train_subtask_baseline.py
- VisualTravelMap
- E1M1 Map Geometry
- Navigation Env (My Way Home)
- run_pixel_corridor_linear.sh
- numpy
- train_e1m1_imitation.py
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
- run_corridor_aim_probe.sh
- run_pixel_e1m1_classical.sh
- train_frozen_perception.py
- run_e1m1_finetune.sh
- PixelNavigationEnv
- Pixel Classical Job
- Pixel Corridor Classical Job
- probe_corridor_aim.py
- Pixel Linear Job
- Pixel Quantum Job
- Pixel Smoke Job
- QuantumE1M1Env
- pixel_nav_models.py
- run_e1m1_compare.sh
- run_e1m1_finetune_smoke.sh
- test_frozen_perception.py
- UniformStarts
- train_human_nav.py
- run_e1m1_imitation.sh
- run_pixel_corridor_smoke.sh
- QuantumE1M1Policy
- run_e1m1_reward_smoke.sh
- run_place_smoke.sh
- train_corridor.py
- run_e1m1_imitation_smoke.sh
- HumanNavigationEnv
- PixelE1M1Env
- reward_shaping.py
- run_human_nav.sh
- run_human_nav_progress.sh
- run_human_nav_smoke.sh
- run_place_comparison.sh
- DoomActorCritic
- diagnose_pixel_e1m1.py
- RouteTeacher
- record_agent.py
- run_frozen_perception.sh
- run_e1m1_cross_task.sh
- record_gameplay.py
- record_nav_agent.py
- run_pixel_e1m1_curriculum_smoke.sh
- diagnose_pixel_aux.py
- run_e1m1_sanity.sh
- run_specialists_on_e1m1.sh
- run_e1m1_teacher_test.sh
- run_human_nav_diagnostic.sh
- CLAUDE.md
- .claude/CLAUDE.md
- run_pixel_success_check.sh script

## God Nodes (most connected - your core abstractions)
1. `HumanNavigationEnv` - 27 edges
2. `DoomMap` - 25 edges
3. `PixelNavPolicy` - 24 edges
4. `QuantumE1M1Env` - 21 edges
5. `EnvPool` - 20 edges
6. `QuantumDoomEnv` - 19 edges
7. `QuantumE1M1Policy` - 19 edges
8. `PixelE1M1Env` - 19 edges
9. `RoutePlanner` - 17 edges
10. `DoomActorCritic` - 17 edges

## Surprising Connections (you probably didn't know these)
- `12-qubit circuit head result (96/100)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `Memory bridges (controlled rotations)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `Private progress and kill rewards` --references--> `HumanNavigationEnv`  [EXTRACTED]
  README.md → human_nav_env.py
- `Success-detection bug` --references--> `HumanNavigationEnv`  [EXTRACTED]
  README.md → human_nav_env.py
- `Data Re-uploading PQC` --references--> `DataReuploadingPQC`  [EXTRACTED]
  README.md → models.py

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

## Communities (74 total, 13 thin omitted)

### Community 0 - "train_pixel_nav.py"
Cohesion: 0.09
Nodes (15): load(), main(), Restricted, main(), load(), main(), run(), EnvPool (+7 more)

### Community 1 - "run_human_nav_eval.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_eval.sh script

### Community 2 - "QuantumDoomEnv"
Cohesion: 0.13
Nodes (6): QuantumDoomEnv, record_episodes(), CombatTask, DodgeTask, make_corridor_env(), _init()

### Community 3 - "train_subtask_baseline.py"
Cohesion: 0.06
Nodes (27): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline.sh script, MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline_smoke.sh script (+19 more)

### Community 5 - "E1M1 Map Geometry"
Cohesion: 0.07
Nodes (15): build_field(), DistanceField, main(), DoomMap, skill_flag(), main(), point_segment_distances(), RoutePlanner (+7 more)

### Community 6 - "Navigation Env (My Way Home)"
Cohesion: 0.18
Nodes (4): QuantumNavigationEnv, record_nav_episodes(), NavigationTask, _init()

### Community 7 - "run_pixel_corridor_linear.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_linear.sh script

### Community 9 - "train_e1m1_imitation.py"
Cohesion: 0.16
Nodes (4): collect(), Dataset, fit(), main()

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

### Community 16 - "Learned screen encoder (CNN to 12-value bottleneck)"
Cohesion: 0.18
Nodes (10): Action-based travel map, Auxiliary depth and goal-visibility losses, 12-qubit circuit head result (96/100), Freinberger et al. 2024 (hybrid Atari), Lample & Chaplot 2017, Learned screen encoder (CNN to 12-value bottleneck), Mirowski et al. 2016 (auxiliary-task navigation), Potential-based reward shaping (+2 more)

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

### Community 25 - "run_corridor_aim_probe.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor_aim_probe.sh script

### Community 26 - "run_pixel_e1m1_classical.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_classical.sh script

### Community 27 - "train_frozen_perception.py"
Cohesion: 0.15
Nodes (12): audit(), fingerprint(), teacher_eval(), constant_geometry_metrics(), encode_replay(), evaluate(), fit_heads(), geometry_metrics() (+4 more)

### Community 28 - "run_e1m1_finetune.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_finetune.sh script

### Community 29 - "PixelNavigationEnv"
Cohesion: 0.18
Nodes (6): PixelCorridorEnv, make_pixel_env(), PixelNavigationEnv, _worker(), collect(), Success-detection bug

### Community 30 - "Pixel Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_classical.sh script

### Community 31 - "Pixel Corridor Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_classical.sh script

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
Nodes (36): main(), QuantumE1M1Env, collect(), main(), Reverse-start curriculum, Route validation (scripted follower), MKL_NUM_THREADS, OMP_NUM_THREADS (+28 more)

### Community 37 - "pixel_nav_models.py"
Cohesion: 0.12
Nodes (3): SharedMLP, ClassicalNavPolicy, LinearNavHead

### Community 38 - "run_e1m1_compare.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_compare.sh script

### Community 39 - "run_e1m1_finetune_smoke.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_finetune_smoke.sh script

### Community 40 - "test_frozen_perception.py"
Cohesion: 0.13
Nodes (5): FrozenPolicy, GeometryPerception, make_head(), summarize(), FrozenPerceptionTests

### Community 41 - "UniformStarts"
Cohesion: 0.15
Nodes (3): next_cooldown(), collect(), UniformStarts

### Community 42 - "train_human_nav.py"
Cohesion: 0.39
Nodes (4): HumanE1M1Env, evaluate(), make_env(), train()

### Community 43 - "run_e1m1_imitation.sh"
Cohesion: 0.40
Nodes (4): MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_imitation.sh script

### Community 44 - "run_pixel_corridor_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_smoke.sh script

### Community 45 - "QuantumE1M1Policy"
Cohesion: 0.20
Nodes (5): QuantumE1M1Policy, circuit(), Matched classical vs bridged-quantum comparison, Memory bridges (controlled rotations), RGB place memory (VisualPlaceMemory)

### Community 46 - "run_e1m1_reward_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_reward_smoke.sh script

### Community 47 - "run_place_smoke.sh"
Cohesion: 0.25
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_place_smoke.sh script

### Community 48 - "train_corridor.py"
Cohesion: 0.19
Nodes (8): Classical subtask baselines, Deadly Corridor scenario, My Way Home scenario, Take Cover dodging task, train_one_run(), train_corridor(), make_nav_env(), train_navigation()

### Community 49 - "run_e1m1_imitation_smoke.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_imitation_smoke.sh script

### Community 50 - "HumanNavigationEnv"
Cohesion: 0.26
Nodes (3): main(), run(), HumanNavigationEnv

### Community 52 - "reward_shaping.py"
Cohesion: 0.21
Nodes (4): field_progress_reward(), goal_potential(), kill_reward(), progress_reward()

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

### Community 58 - "diagnose_pixel_e1m1.py"
Cohesion: 0.39
Nodes (6): baseline(), check_field(), check_warps(), follow(), main(), _wrap_angle()

### Community 60 - "record_agent.py"
Cohesion: 0.17
Nodes (5): render_radar_overlay(), MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor.sh script

### Community 61 - "run_frozen_perception.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_frozen_perception.sh script

### Community 62 - "run_e1m1_cross_task.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_cross_task.sh script

### Community 63 - "record_gameplay.py"
Cohesion: 0.09
Nodes (16): caption(), default_targets(), describe(), main(), record(), MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED (+8 more)

### Community 64 - "record_nav_agent.py"
Cohesion: 0.14
Nodes (6): _convert_h264_if_available(), render_nav_hud(), MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_navigation.sh script

### Community 65 - "run_pixel_e1m1_curriculum_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_e1m1_curriculum_smoke.sh script

### Community 67 - "run_e1m1_sanity.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_sanity.sh script

### Community 68 - "run_specialists_on_e1m1.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_specialists_on_e1m1.sh script

## Ambiguous Edges - Review These
- `Doors (orange)` → `Start-to-exit route requires traversing east then south through multiple doors`  [AMBIGUOUS]
  e1m1_map.png · relation: rationale_for
- `Exit (green, ~2910,-4770)` → `Agent Trajectories Seeds 0-4`  [AMBIGUOUS]
  e1m1_route_check_skill3.png · relation: conceptually_related_to
- `12-Qubit Input Angles` → `Empty Input-Angle Plot Finding`  [AMBIGUOUS]
  e1m1_sanity.png · relation: rationale_for

## Knowledge Gaps
- **183 isolated node(s):** `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `PYTHONUNBUFFERED`, `OMP_NUM_THREADS` (+178 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 366 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HumanNavigationEnv` connect `HumanNavigationEnv` to `VisualTravelMap`, `numpy`, `train_human_nav.py`, `run_place_smoke.sh`, `Learned screen encoder (CNN to 12-value bottleneck)`, `reward_shaping.py`, `PixelNavigationEnv`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `HumanNavigationEnv` (e.g. with `PixelCorridorEnv` and `PixelNavigationEnv`) actually correct?**
  _`HumanNavigationEnv` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` to the rest of the system?**
  _183 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pixel_nav.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09191583610188261 - nodes in this community are weakly interconnected._
- **Why does `PixelE1M1Env` connect `PixelE1M1Env` to `E1M1 Map Geometry`, `numpy`, `diagnose_pixel_e1m1.py`, `RouteTeacher`, `PixelNavigationEnv`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Doors (orange)` and `Start-to-exit route requires traversing east then south through multiple doors`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Exit (green, ~2910,-4770)` and `Agent Trajectories Seeds 0-4`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._