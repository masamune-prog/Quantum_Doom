# Graph Report - quantum_doom  (2026-11-02)

## Corpus Check
- 135 files · ~135,867 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 219 file(s) not represented in the graph (top: .pt 106, .log 53, .err 52)

## Summary
- 654 nodes · 1240 edges · 53 communities (46 shown, 7 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 147 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- record_gameplay.py
- E1M1 Env & Diagnostics
- QuantumDoomEnv
- Dodge Baseline Jobs
- Human-Equivalent Nav Env
- E1M1 Map Geometry
- Navigation Env (My Way Home)
- run_pixel_corridor_linear.sh
- numpy
- Learned screen encoder (CNN to 12-value bottleneck)
- README Project Concepts
- E1M1 Map Figure
- run_pixel_corridor_quantum.sh
- Route Check Skill 3
- E1M1 Reference Route
- My Way Home Curve
- Pixel Nav Env Pool
- Entanglement Ablation Curve
- Route Check Skill 1
- venv_bin_activate
- run_human_nav.sh
- Deadly Corridor Curve
- E1M1 Sanity Figure
- run_pixel_corridor_quantum_resume.sh
- Radar Encoding Sanity Check
- E1M1 Reward Smoke Job
- HumanNavigationEnv
- Human Nav Eval Job
- pixel_nav_models.py
- PixelNavPolicy
- Pixel Classical Job
- Pixel Corridor Classical Job
- PixelNavigationEnv
- Pixel Linear Job
- Pixel Quantum Job
- Pixel Smoke Job
- run_e1m1_starts.sh
- run_e1m1_sanity.sh
- Corridor Train Job
- E1M1 Benchmark Job
- E1M1 Pretrain Job
- run_navigation.sh
- run_human_nav_smoke.sh
- Graphify Project Rules
- run_pixel_corridor_smoke.sh
- E1M1 Route Check Job
- LinearNavHead
- run_place_smoke.sh
- E1M1 Unstick Job
- run_record_gameplay.sh
- Pixel Nav Evaluation
- run_corridor_aim_probe.sh
- run_human_nav_diagnostic.sh

## God Nodes (most connected - your core abstractions)
1. `HumanNavigationEnv` - 27 edges
2. `QuantumE1M1Env` - 21 edges
3. `DoomMap` - 20 edges
4. `QuantumDoomEnv` - 19 edges
5. `QuantumE1M1Policy` - 17 edges
6. `DoomActorCritic` - 17 edges
7. `QuantumNavigationEnv` - 16 edges
8. `PixelNavigationEnv` - 16 edges
9. `RoutePlanner` - 15 edges
10. `PixelNavPolicy` - 15 edges

## Surprising Connections (you probably didn't know these)
- `12-qubit circuit head result (96/100)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `Memory bridges (controlled rotations)` --references--> `QuantumE1M1Policy`  [INFERRED]
  README.md → e1m1_quantum_model.py
- `Success-detection bug` --references--> `HumanNavigationEnv`  [EXTRACTED]
  README.md → human_nav_env.py
- `Quantum Doom` --references--> `DoomActorCritic`  [EXTRACTED]
  README.md → models.py
- `graphify skill` --conceptually_related_to--> `graphify knowledge graph (graphify-out/)`  [INFERRED]
  .claude/CLAUDE.md → CLAUDE.md

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

## Communities (53 total, 7 thin omitted)

### Community 0 - "record_gameplay.py"
Cohesion: 0.20
Nodes (7): make_pixel_env(), _worker(), caption(), default_targets(), describe(), main(), record()

### Community 1 - "E1M1 Env & Diagnostics"
Cohesion: 0.07
Nodes (26): main(), _point_segment_distance(), QuantumE1M1Env, _wrap_angle(), QuantumE1M1Policy, circuit(), main(), collect() (+18 more)

### Community 2 - "QuantumDoomEnv"
Cohesion: 0.11
Nodes (7): QuantumDoomEnv, record_episodes(), render_radar_overlay(), CombatTask, DodgeTask, make_corridor_env(), _init()

### Community 3 - "Dodge Baseline Jobs"
Cohesion: 0.06
Nodes (27): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline.sh script, MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline_smoke.sh script (+19 more)

### Community 5 - "E1M1 Map Geometry"
Cohesion: 0.13
Nodes (5): DoomMap, skill_flag(), main(), point_segment_distances(), RoutePlanner

### Community 6 - "Navigation Env (My Way Home)"
Cohesion: 0.12
Nodes (7): QuantumNavigationEnv, _convert_h264_if_available(), record_nav_episodes(), render_nav_hud(), NavigationTask, make_nav_env(), _init()

### Community 7 - "run_pixel_corridor_linear.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_linear.sh script

### Community 8 - "numpy"
Cohesion: 0.07
Nodes (13): DoomActorCritic, Classical subtask baselines, Deadly Corridor scenario, My Way Home scenario, Take Cover dodging task, goal_potential(), kill_reward(), progress_reward() (+5 more)

### Community 9 - "Learned screen encoder (CNN to 12-value bottleneck)"
Cohesion: 0.33
Nodes (6): Auxiliary depth and goal-visibility losses, 12-qubit circuit head result (96/100), Freinberger et al. 2024 (hybrid Atari), Lample & Chaplot 2017, Learned screen encoder (CNN to 12-value bottleneck), Mirowski et al. 2016 (auxiliary-task navigation)

### Community 10 - "README Project Concepts"
Cohesion: 0.08
Nodes (19): DataReuploadingPQC, Circuit imitation test, Classical MLP Critic, Data Re-uploading PQC, Shareware doom1.wad v1.9, E1M1 (Hangar), E1M1 pipeline, Entanglement ablation (basic.cfg) (+11 more)

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

### Community 16 - "Pixel Nav Env Pool"
Cohesion: 0.31
Nodes (3): EnvPool, summary(), train()

### Community 17 - "Entanglement Ablation Curve"
Cohesion: 0.31
Nodes (7): classical_mlp (Classical MLP baseline), Ablation Curve: Entangled PQC vs Unentangled vs Classical MLP (ViZDoom Aiming & Combat), Mean Step Reward, PPO Rollout Updates (x1024 frames), quantum_entangled (Entangled PQC policy), quantum_product (Unentangled product-state PQC), ViZDoom Aiming & Combat Scenario

### Community 18 - "Route Check Skill 1"
Cohesion: 0.36
Nodes (7): Doors on Route (orange, near waypoints 5, 16-17, 20), E1M1 Map Layout (walls, doors, exit, monsters, items), E1M1 Exit (green, bottom-right room), E1M1 Route Check (Skill 1) Map Figure, Reference Route Waypoints 0-21 (dashed purple), Agent Trajectories Seeds 0-4, Doom Skill Level 1

### Community 19 - "venv_bin_activate"
Cohesion: 0.22
Nodes (5): run_pixel_aux_check.sh script, run_pixel_success_check.sh script, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_reward_smoke.sh script

### Community 20 - "run_human_nav.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav.sh script

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

### Community 25 - "E1M1 Reward Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_reward_smoke.sh script

### Community 26 - "HumanNavigationEnv"
Cohesion: 0.07
Nodes (24): run(), HumanE1M1Env, HumanNavigationEnv, Action-based travel map, Matched classical vs bridged-quantum comparison, Memory bridges (controlled rotations), Potential-based reward shaping, Private progress and kill rewards (+16 more)

### Community 27 - "Human Nav Eval Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_eval.sh script

### Community 30 - "Pixel Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_classical.sh script

### Community 31 - "Pixel Corridor Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_classical.sh script

### Community 32 - "PixelNavigationEnv"
Cohesion: 0.19
Nodes (5): PixelCorridorEnv, PixelNavigationEnv, collect(), enemy_bin(), Success-detection bug

### Community 33 - "Pixel Linear Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_linear.sh script

### Community 34 - "Pixel Quantum Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_quantum.sh script

### Community 35 - "Pixel Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_smoke.sh script

### Community 36 - "run_e1m1_starts.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_starts.sh script

### Community 37 - "run_e1m1_sanity.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_sanity.sh script

### Community 38 - "Corridor Train Job"
Cohesion: 0.50
Nodes (3): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED

### Community 39 - "E1M1 Benchmark Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_benchmark.sh script

### Community 40 - "E1M1 Pretrain Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_pretrain.sh script

### Community 41 - "run_navigation.sh"
Cohesion: 0.50
Nodes (3): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED

### Community 42 - "run_human_nav_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_smoke.sh script

### Community 44 - "run_pixel_corridor_smoke.sh"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_smoke.sh script

### Community 45 - "E1M1 Route Check Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_route_check.sh script

### Community 47 - "run_place_smoke.sh"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED

### Community 48 - "E1M1 Unstick Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_unstick_check.sh script

### Community 49 - "run_record_gameplay.sh"
Cohesion: 0.40
Nodes (4): MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_record_gameplay.sh script

### Community 51 - "run_corridor_aim_probe.sh"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor_aim_probe.sh script

## Ambiguous Edges - Review These
- `Doors (orange)` → `Start-to-exit route requires traversing east then south through multiple doors`  [AMBIGUOUS]
  e1m1_map.png · relation: rationale_for
- `Exit (green, ~2910,-4770)` → `Agent Trajectories Seeds 0-4`  [AMBIGUOUS]
  e1m1_route_check_skill3.png · relation: conceptually_related_to
- `12-Qubit Input Angles` → `Empty Input-Angle Plot Finding`  [AMBIGUOUS]
  e1m1_sanity.png · relation: rationale_for

## Knowledge Gaps
- **147 isolated node(s):** `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `PYTHONUNBUFFERED`, `OMP_NUM_THREADS` (+142 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 266 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HumanNavigationEnv` connect `HumanNavigationEnv` to `numpy`, `E1M1 Env & Diagnostics`, `PixelNavigationEnv`, `Human-Equivalent Nav Env`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `HumanNavigationEnv` (e.g. with `PixelCorridorEnv` and `PixelNavigationEnv`) actually correct?**
  _`HumanNavigationEnv` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` to the rest of the system?**
  _147 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `E1M1 Env & Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.06826923076923076 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `E1M1 Map Geometry` to `numpy`, `E1M1 Env & Diagnostics`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Doors (orange)` and `Start-to-exit route requires traversing east then south through multiple doors`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Exit (green, ~2910,-4770)` and `Agent Trajectories Seeds 0-4`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._