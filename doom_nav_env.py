import os
import vizdoom as vzd
import gymnasium as gym
import numpy as np


class QuantumNavigationEnv(gym.Env):
    """
    Fair First-Person Gymnasium environment for multi-room maze navigation and full Doom levels.
    
    Supports scaled qubit representations (e.g. 10 or 12 qubits):
    
    10-Qubit Configuration (Default):
      - Qubits 0-7: 8 horizontal FOV visual sectors (11.25° per sector).
        * Negative [-pi, 0]: Corridor clearance / depth (close wall = -pi, deep passage = 0).
        * Positive [0.4*pi, pi]: Visual beacon ONLY when armor/item is directly visible in FOV.
      - Qubit 8: Obstacle / Wall Collision Warning (-pi if blocked into wall, 0 if free).
      - Qubit 9: Local Proximity (0.0 when far away, scales up only inside room < 350 units).
      
    12-Qubit Configuration:
      - Qubits 0-7: 8 horizontal FOV visual clearance sectors.
      - Qubit 8: Visible Target Angle in FOV ([-pi, pi], 0 if not visible).
      - Qubit 9: Monster / Hazard Threat Detector ([0, pi]).
      - Qubit 10: Obstacle / Wall Collision Warning (-pi if blocked).
      - Qubit 11: Local Proximity (Near-field item awareness).
      
    Action Space (7 discrete actions matching Deadly Corridor + USE):
      0: MOVE_FORWARD
      1: ATTACK
      2: MOVE_LEFT (strafe left)
      3: MOVE_RIGHT (strafe right)
      4: TURN_LEFT
      5: TURN_RIGHT
      6: USE (open doors, activate switches)
    """

    # Buttons: [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, USE]
    NAV_ACTIONS = [
        # 0: MOVE_FORWARD
        [0, 0, 0, 1, 0, 0, 0],
        # 1: ATTACK
        [0, 0, 1, 0, 0, 0, 0],
        # 2: MOVE_LEFT
        [1, 0, 0, 0, 0, 0, 0],
        # 3: MOVE_RIGHT
        [0, 1, 0, 0, 0, 0, 0],
        # 4: TURN_LEFT
        [0, 0, 0, 0, 1, 0, 0],
        # 5: TURN_RIGHT
        [0, 0, 0, 0, 0, 1, 0],
        # 6: USE
        [0, 0, 0, 0, 0, 0, 1],
    ]

    ACTION_NAMES = [
        "MOVE_FORWARD", "ATTACK", "MOVE_LEFT", "MOVE_RIGHT",
        "TURN_LEFT", "TURN_RIGHT", "USE",
    ]

    def __init__(
        self,
        scenario: str = "my_way_home.cfg",
        n_qubits: int = 10,
        frame_skip: int = 4,
        custom_cfg_path: str = None,
        grid_cell_size: float = 64.0,
        nearby_threshold: float = 350.0,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.frame_skip = frame_skip
        self.grid_cell_size = grid_cell_size
        self.nearby_threshold = nearby_threshold

        self.game = vzd.DoomGame()
        if custom_cfg_path and os.path.exists(custom_cfg_path):
            self.game.load_config(custom_cfg_path)
        else:
            self.game.load_config(os.path.join(vzd.scenarios_path, scenario))

        # Ensure WAD path resolves correctly
        default_wad = os.path.join(vzd.scenarios_path, "my_way_home.wad")
        if os.path.exists(default_wad):
            self.game.set_doom_scenario_path(default_wad)

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_window_visible(False)

        self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        self.game.add_available_game_variable(vzd.GameVariable.ANGLE)
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.init()

        self.actions = self.NAV_ACTIONS
        self.n_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(n_qubits,), dtype=np.float32
        )

        self.last_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.visited_cells = set()
        self.goal_pos = np.array([1040.0, -352.0], dtype=np.float32)
        self.last_pos = None

    def _update_goal_position(self, state):
        """Identify ground truth goal position for termination and nearby radius check."""
        if state and state.objects:
            for obj in state.objects:
                if "Armor" in obj.name or "Vest" in obj.name:
                    self.goal_pos = np.array([obj.position_x, obj.position_y], dtype=np.float32)
                    return

    def _extract_features(self, state, is_stuck: bool = False) -> np.ndarray:
        features = np.zeros(self.n_qubits, dtype=np.float32)
        if state is None:
            return features

        self.last_rgb = state.screen_buffer
        depth = state.depth_buffer      # (240, 320) uint8
        labels = state.labels_buffer    # (240, 320) uint8

        armor_label_ids = {
            l.value for l in state.labels
            if "Armor" in l.object_name or "Vest" in l.object_name
        }
        enemy_label_ids = {
            l.value for l in state.labels
            if l.object_name not in ("DoomPlayer", "GreenArmor") and "Marine" not in l.object_name and "Dead" not in l.object_name
        }

        # 8 High-Resolution FOV Visual Sectors
        n_visual_sectors = 8 if self.n_qubits >= 10 else 6
        h, w = depth.shape
        mid_band = slice(h // 3, 2 * h // 3)
        sector_w = w // n_visual_sectors

        target_sector_center = None

        for i in range(n_visual_sectors):
            col = slice(i * sector_w, (i + 1) * sector_w)
            sec_depth = float(np.mean(depth[mid_band, col]))
            # 0 = wall right at face, 1 = deep passage
            norm_clearance = np.clip((sec_depth - 3.0) / 35.0, 0.0, 1.0)

            armor_ratio = 0.0
            if armor_label_ids:
                sec_labels = labels[mid_band, col]
                armor_mask = np.isin(sec_labels, list(armor_label_ids))
                armor_ratio = float(np.mean(armor_mask))

            if armor_ratio > 0.01:
                # Target in sight! Strong positive visual rotation
                features[i] = np.pi * np.clip(0.4 + 2.0 * armor_ratio, 0.4, 1.0)
                target_sector_center = (i + 0.5) / n_visual_sectors
            else:
                # Open hallway clearance: [-pi, 0] (-pi = blocked wall, 0 = wide open path)
                features[i] = -np.pi * (1.0 - norm_clearance)

        # Distance to goal for fair near-field check
        px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        dist = np.hypot(self.goal_pos[0] - px, self.goal_pos[1] - py)

        nearby_signal = 0.0
        if dist < self.nearby_threshold:
            proximity = (self.nearby_threshold - dist) / self.nearby_threshold
            nearby_signal = np.pi * float(proximity)

        collision_signal = -np.pi if is_stuck else 0.0

        if self.n_qubits == 10:
            # Qubit 8: Collision / Wall Contact Warning
            features[8] = collision_signal
            # Qubit 9: Local Proximity (Nearby detection ONLY)
            features[9] = nearby_signal
        elif self.n_qubits >= 12:
            # Qubit 8: Visible target horizontal offset [-pi, pi] (0 if centered or not visible)
            if target_sector_center is not None:
                features[8] = np.pi * (2.0 * target_sector_center - 1.0)
            else:
                features[8] = 0.0

            # Qubit 9: Threat / Monster Presence Ratio in FOV
            if enemy_label_ids:
                enemy_mask = np.isin(labels[mid_band, :], list(enemy_label_ids))
                features[9] = np.pi * np.clip(float(np.mean(enemy_mask)) * 5.0, 0.0, 1.0)
            else:
                features[9] = 0.0

            # Qubit 10: Collision Warning
            features[10] = collision_signal
            # Qubit 11: Local Proximity
            features[11] = nearby_signal

        return features

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.visited_cells.clear()

        state = self.game.get_state()
        self._update_goal_position(state)

        px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        self.last_pos = np.array([px, py], dtype=np.float32)

        cell = (int(px // self.grid_cell_size), int(py // self.grid_cell_size))
        self.visited_cells.add(cell)

        obs = self._extract_features(state, is_stuck=False)
        return obs, {}

    def step(self, action: int):
        self.game.make_action(self.actions[action], self.frame_skip)
        ended = self.game.is_episode_finished()
        # A time limit is a truncation, not a terminal state: PPO bootstraps through it.
        truncated = bool(ended and self.game.is_episode_timeout_reached())
        terminated = ended and not truncated
        # The vest's +1 can land on the tic before the episode ends, so this step's own
        # reward may be slightly negative; test the episode total instead.
        success = bool(terminated and self.game.get_total_reward() > 0.5)
        state = None if ended else self.game.get_state()

        reward = 0.0
        is_stuck = False

        if not ended and state is not None:
            px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
            curr_pos = np.array([px, py], dtype=np.float32)

            move_dist = np.linalg.norm(curr_pos - self.last_pos) if self.last_pos is not None else 0.0
            self.last_pos = curr_pos

            # Forward locomotion check (action 0)
            if action == 0 and move_dist < 2.0:
                is_stuck = True
                reward -= 0.1
            elif action == 0:
                reward += 0.05

            # Intrinsic Exploration / Room Discovery Bonus
            cell = (int(px // self.grid_cell_size), int(py // self.grid_cell_size))
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                reward += 0.8

            reward -= 0.01

        # Goal reward: Armor collected / level exit
        if success:
            reward += 100.0

        obs = self._extract_features(state, is_stuck=is_stuck)
        return obs, reward, terminated, truncated, {"success": success}

    def close(self):
        self.game.close()
