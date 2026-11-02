# Graph Report - quantum_doom  (2026-11-01)

## Corpus Check
- 129 files · ~91,507 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 168 file(s) not represented in the graph (top: .pt 83, .log 39, .err 38)

## Summary
- 609 nodes · 1158 edges · 51 communities (46 shown, 5 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 139 edges (avg confidence: 0.76)
- Token cost: 414,804 input · 0 output

## Community Hubs (Navigation)
- E1M1 Quantum Policy
- E1M1 Env & Diagnostics
- Nav Diagnostics Tooling
- Dodge Baseline Jobs
- Human-Equivalent Nav Env
- E1M1 Map Geometry
- Navigation Env (My Way Home)
- Actor-Critic & Training
- Pixel Corridor Env
- Combat Radar Env
- README Project Concepts
- E1M1 Map Figure
- Gameplay Recording
- Route Check Skill 3
- E1M1 Reference Route
- My Way Home Curve
- Pixel Nav Env Pool
- Entanglement Ablation Curve
- Route Check Skill 1
- Diagnostic Job Scripts
- Place Memory Smoke
- Deadly Corridor Curve
- E1M1 Sanity Figure
- Data Re-uploading PQC
- Radar Encoding Sanity Check
- E1M1 Reward Smoke Job
- Human Nav Train Job
- Human Nav Eval Job
- Human Nav Progress Job
- Human Nav Smoke Job
- Pixel Classical Job
- Pixel Corridor Classical Job
- Pixel Corridor Smoke Job
- Pixel Linear Job
- Pixel Quantum Job
- Pixel Smoke Job
- Place Comparison Job
- Combat Subtask Env
- Corridor Train Job
- E1M1 Benchmark Job
- E1M1 Pretrain Job
- E1M1 Train Job
- Navigation Train Job
- Graphify Project Rules
- Benchmark Scenarios
- E1M1 Route Check Job
- E1M1 Sanity Job
- E1M1 Starts Job
- E1M1 Unstick Job
- Reward Smoke Job
- Pixel Nav Evaluation

## God Nodes (most connected - your core abstractions)
1. `HumanNavigationEnv` - 25 edges
2. `QuantumE1M1Env` - 21 edges
3. `DoomMap` - 20 edges
4. `QuantumDoomEnv` - 19 edges
5. `QuantumE1M1Policy` - 17 edges
6. `DoomActorCritic` - 17 edges
7. `QuantumNavigationEnv` - 16 edges
8. `PixelNavigationEnv` - 16 edges
9. `RoutePlanner` - 15 edges
10. `VisualTravelMap` - 14 edges

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
- **E1M1 map-plan-validate-train pipeline** — readme_route_planner_dijkstra, readme_route_validation_scripted_follower, readme_e1m1_12_qubit_observation, readme_quantum_only_policy_value_model, readme_reverse_start_curriculum [EXTRACTED 1.00]
- **Hand-crafted features to PQC actor with PPO/GAE** — readme_hand_crafted_angle_features, readme_data_reuploading_pqc, readme_classical_mlp_critic, readme_ppo_with_gae [EXTRACTED 1.00]
- **Human-visible screen-only navigation experiments** — readme_human_visible_observation, readme_screen_only_navigation_memory, readme_rgb_place_memory, readme_matched_policy_comparison, readme_learned_screen_encoder, readme_circuit_head_result [INFERRED 0.85]
- **Policy architecture ablation (entangled PQC / product PQC / classical MLP)** — ablation_curve_quantum_entangled, ablation_curve_quantum_product, ablation_curve_classical_mlp [EXTRACTED 1.00]
- **8-qubit PQC trained with PPO on Deadly Corridor** — corridor_training_curve_quantum_entangled_pqc, corridor_training_curve_ppo, corridor_training_curve_deadly_corridor [INFERRED 0.95]
- **Start-to-exit navigation path** — e1m1_map_start, e1m1_map_central_courtyard, e1m1_map_zigzag_room, e1m1_map_exit_hall, e1m1_map_exit_switch [INFERRED 0.65]
- **On-route navigation obstacles** — e1m1_route_waypoint_path, e1m1_route_doors, e1m1_route_monsters [INFERRED 0.75]
- **E1M1 Route Validation (map + waypoints + seed runs)** — e1m1_route_check_skill1_e1m1_map, e1m1_route_check_skill1_reference_waypoints, e1m1_route_check_skill1_seed_trajectories [INFERRED 0.85]
- **Route validation: waypoints vs seed trajectories on E1M1** — e1m1_route_check_skill3_waypoint_route, e1m1_route_check_skill3_seed_trajectories, e1m1_route_check_skill3_e1m1_map, e1m1_route_check_skill3_exit_switch [INFERRED 0.85]
- **Screen-to-Qubit Encoding Sanity Pipeline** — e1m1_sanity_screen_observation, e1m1_sanity_qubit_input_angles, e1m1_sanity_angle_encoding [INFERRED 0.75]
- **Quantum PPO navigation training setup** — nav_training_curve_quantum_pqc_agent, nav_training_curve_ppo, nav_training_curve_action_space, nav_training_curve_my_way_home [EXTRACTED 1.00]
- **Frame-to-Qubit Preprocessing Pipeline** — sanity_check_raw_rgb_frame, sanity_check_label_buffer, sanity_check_qubit_input_angles [INFERRED 0.95]

## Communities (51 total, 5 thin omitted)

### Community 0 - "E1M1 Quantum Policy"
Cohesion: 0.06
Nodes (23): QuantumE1M1Policy, circuit(), main(), run(), ClassicalNavPolicy, LinearNavHead, PixelNavPolicy, Action-based travel map (+15 more)

### Community 1 - "E1M1 Env & Diagnostics"
Cohesion: 0.10
Nodes (19): main(), _point_segment_distance(), QuantumE1M1Env, _wrap_angle(), collect(), main(), E1M1 12-qubit observation, Reverse-start curriculum (+11 more)

### Community 2 - "Nav Diagnostics Tooling"
Cohesion: 0.10
Nodes (3): goal_potential(), kill_reward(), progress_reward()

### Community 3 - "Dodge Baseline Jobs"
Cohesion: 0.06
Nodes (27): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline.sh script, MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_dodge_baseline_smoke.sh script (+19 more)

### Community 4 - "Human-Equivalent Nav Env"
Cohesion: 0.13
Nodes (4): HumanE1M1Env, HumanNavigationEnv, VisualPlaceMemory, VisualTravelMap

### Community 5 - "E1M1 Map Geometry"
Cohesion: 0.13
Nodes (5): DoomMap, skill_flag(), main(), point_segment_distances(), RoutePlanner

### Community 6 - "Navigation Env (My Way Home)"
Cohesion: 0.13
Nodes (6): QuantumNavigationEnv, _convert_h264_if_available(), record_nav_episodes(), render_nav_hud(), NavigationTask, _init()

### Community 7 - "Actor-Critic & Training"
Cohesion: 0.19
Nodes (7): DoomActorCritic, train_one_run(), make_corridor_env(), _init(), train_corridor(), make_nav_env(), train_navigation()

### Community 8 - "Pixel Corridor Env"
Cohesion: 0.19
Nodes (5): PixelCorridorEnv, make_pixel_env(), PixelNavigationEnv, _worker(), Success-detection bug

### Community 9 - "Combat Radar Env"
Cohesion: 0.18
Nodes (3): QuantumDoomEnv, record_episodes(), DodgeTask

### Community 10 - "README Project Concepts"
Cohesion: 0.11
Nodes (18): Circuit imitation test, Classical MLP Critic, Data Re-uploading PQC, Shareware doom1.wad v1.9, E1M1 (Hangar), E1M1 pipeline, Entanglement ablation (basic.cfg), Hand-crafted angle features (depth + labels buffers) (+10 more)

### Community 11 - "E1M1 Map Figure"
Cohesion: 0.22
Nodes (12): Central Large Room / Courtyard, Doors (orange), Doom E1M1 Level (Hangar), Southern Exit Hall and Exit Room, Exit (green, ~2910,-4790), E1M1 Top-Down Map (skill 3), Items/Pickups (blue), Monsters (red, ~6 at skill 3) (+4 more)

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

### Community 19 - "Diagnostic Job Scripts"
Cohesion: 0.25
Nodes (4): OMP_NUM_THREADS, run_human_nav_diagnostic.sh script, run_pixel_aux_check.sh script, run_pixel_success_check.sh script

### Community 20 - "Place Memory Smoke"
Cohesion: 0.25
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_place_smoke.sh script

### Community 21 - "Deadly Corridor Curve"
Cohesion: 0.48
Nodes (6): Deadly Corridor: 8-Qubit Quantum Agent Training (figure), Deadly Corridor Scenario, Mean Step Reward, PPO Update, Quantum Entangled PQC (8 qubits, 4 layers), Smoothed Reward (window=5)

### Community 22 - "E1M1 Sanity Figure"
Cohesion: 0.33
Nodes (6): Agent View at Exit Switch, Angle Encoding Range [-pi, pi], E1M1 Exit Switch, E1M1 Sanity Check Figure, 12-Qubit Input Angles, 320x240 Screen Observation

### Community 24 - "Radar Encoding Sanity Check"
Cohesion: 0.38
Nodes (5): Sanity Check Figure (Frame -> Label Buffer -> 8-Qubit Angles), FOV Sector to Qubit Mapping (Sectors 0-7), Native ViZDoom Label Buffer (Zero CNN), 8-Qubit Input Angles (FOV Sector Encoding), Raw ViZDoom RGB Frame

### Community 25 - "E1M1 Reward Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_reward_smoke.sh script

### Community 26 - "Human Nav Train Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav.sh script

### Community 27 - "Human Nav Eval Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_eval.sh script

### Community 28 - "Human Nav Progress Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_progress.sh script

### Community 29 - "Human Nav Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_human_nav_smoke.sh script

### Community 30 - "Pixel Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_classical.sh script

### Community 31 - "Pixel Corridor Classical Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_classical.sh script

### Community 32 - "Pixel Corridor Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_corridor_smoke.sh script

### Community 33 - "Pixel Linear Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_linear.sh script

### Community 34 - "Pixel Quantum Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_quantum.sh script

### Community 35 - "Pixel Smoke Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_pixel_smoke.sh script

### Community 36 - "Place Comparison Job"
Cohesion: 0.33
Nodes (5): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_place_comparison.sh script

### Community 38 - "Corridor Train Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_corridor.sh script

### Community 39 - "E1M1 Benchmark Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_benchmark.sh script

### Community 40 - "E1M1 Pretrain Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_pretrain.sh script

### Community 41 - "E1M1 Train Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONUNBUFFERED

### Community 42 - "Navigation Train Job"
Cohesion: 0.40
Nodes (4): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTHONUNBUFFERED, run_navigation.sh script

### Community 44 - "Benchmark Scenarios"
Cohesion: 0.50
Nodes (4): Classical subtask baselines, Deadly Corridor scenario, My Way Home scenario, Take Cover dodging task

### Community 45 - "E1M1 Route Check Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_route_check.sh script

### Community 46 - "E1M1 Sanity Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_sanity.sh script

### Community 47 - "E1M1 Starts Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_starts.sh script

### Community 48 - "E1M1 Unstick Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_e1m1_unstick_check.sh script

### Community 49 - "Reward Smoke Job"
Cohesion: 0.50
Nodes (3): OMP_NUM_THREADS, PYTHONUNBUFFERED, run_reward_smoke.sh script

## Ambiguous Edges - Review These
- `Doors (orange)` → `Start-to-exit route requires traversing east then south through multiple doors`  [AMBIGUOUS]
  e1m1_map.png · relation: rationale_for
- `Agent Trajectories Seeds 0-4` → `Exit (green, ~2910,-4770)`  [AMBIGUOUS]
  e1m1_route_check_skill3.png · relation: conceptually_related_to
- `12-Qubit Input Angles` → `Empty Input-Angle Plot Finding`  [AMBIGUOUS]
  e1m1_sanity.png · relation: rationale_for

## Knowledge Gaps
- **130 isolated node(s):** `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `PYTHONUNBUFFERED`, `OMP_NUM_THREADS` (+125 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 241 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HumanNavigationEnv` connect `Human-Equivalent Nav Env` to `E1M1 Quantum Policy`, `Pixel Corridor Env`, `Nav Diagnostics Tooling`, `Place Memory Smoke`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **What connects `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` to the rest of the system?**
  _130 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `E1M1 Quantum Policy` be split into smaller, more focused modules?**
  _Cohesion score 0.0593990216631726 - nodes in this community are weakly interconnected._
- **Why does `DoomMap` connect `E1M1 Map Geometry` to `E1M1 Env & Diagnostics`, `Nav Diagnostics Tooling`, `Human-Equivalent Nav Env`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Should `E1M1 Env & Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.09693877551020408 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Doors (orange)` and `Start-to-exit route requires traversing east then south through multiple doors`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Agent Trajectories Seeds 0-4` and `Exit (green, ~2910,-4770)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._