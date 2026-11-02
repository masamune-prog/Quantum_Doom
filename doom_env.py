import os
import vizdoom as vzd
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class QuantumDoomEnv(gym.Env):
    """
    Custom Gymnasium wrapper around native ViZDoom that enables Depth & Label buffers
    BEFORE init() and extracts an N-qubit spatial radar vector in [-pi, pi]^N.

    Supports both basic.cfg (3 actions) and deadly_corridor (composite actions).
    """
    # Pre-defined decoupled action set for deadly corridor
    # Buttons: [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
    CORRIDOR_ACTIONS = [
        # 0: MOVE_FORWARD (advance down corridor toward vest)
        [0, 0, 0, 1, 0, 0],
        # 1: ATTACK (stationary shooting / hold aim and fire)
        [0, 0, 1, 0, 0, 0],
        # 2: MOVE_LEFT (strafe left / dodge without charging forward)
        [1, 0, 0, 0, 0, 0],
        # 3: MOVE_RIGHT (strafe right / dodge without charging forward)
        [0, 1, 0, 0, 0, 0],
        # 4: TURN_LEFT (aim left toward alcove enemies)
        [0, 0, 0, 0, 1, 0],
        # 5: TURN_RIGHT (aim right toward alcove enemies)
        [0, 0, 0, 0, 0, 1],
    ]

    def __init__(self, scenario: str = "basic.cfg", n_qubits: int = 8, frame_skip: int = 4,
                 use_composite_actions: bool = False, custom_cfg_path: str = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.frame_skip = frame_skip
        self.use_composite_actions = use_composite_actions

        self.game = vzd.DoomGame()

        # Load config: either from custom path or from ViZDoom's built-in scenarios
        if custom_cfg_path and os.path.exists(custom_cfg_path):
            self.game.load_config(custom_cfg_path)
        else:
            self.game.load_config(os.path.join(vzd.scenarios_path, scenario))

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        # CRITICAL: Enable buffers BEFORE calling game.init()
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_window_visible(False)
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        self.game.init()

        # Setup actions
        if use_composite_actions:
            self.actions = self.CORRIDOR_ACTIONS
            self.n_actions = len(self.actions)
        else:
            # Discrete actions: one-hot button vectors (basic.cfg style)
            self.n_actions = self.game.get_available_buttons_size()
            self.actions = np.eye(self.n_actions, dtype=bool).tolist()

        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(n_qubits,), dtype=np.float32
        )
        self.last_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.last_health = 100.0
        self.last_killcount = 0.0
        self.last_pos_x = 0.0
        self.max_pos_x = 0.0

    def _extract_radar(self, state) -> np.ndarray:
        if state is None:
            return np.zeros(self.n_qubits, dtype=np.float32)

        self.last_rgb = state.screen_buffer
        depth = state.depth_buffer      # (240, 320) uint8
        labels = state.labels_buffer    # (240, 320) uint8

        h, w = depth.shape
        mid_band = slice(h // 3, 2 * h // 3)
        sector_w = w // self.n_qubits
        features = np.zeros(self.n_qubits, dtype=np.float32)

        # Filter specifically for Monster/Enemy bounding pixels (exclude player and goal vest)
        enemy_ids = {
            l.value for l in state.labels
            if l.object_name not in ("DoomPlayer", "GreenArmor") and "Marine" not in l.object_name
        }

        for i in range(self.n_qubits):
            col = slice(i * sector_w, (i + 1) * sector_w)
            sec_depth = np.mean(depth[mid_band, col]) / 255.0

            # Check fraction of sector occupied by an enemy sprite
            sec_labels = labels[mid_band, col]
            enemy_mask = np.isin(sec_labels, list(enemy_ids))
            enemy_ratio = np.mean(enemy_mask)

            if enemy_ratio > 0.02:
                # Positive rotation [0.3*pi, pi]: Monster detected in sector i!
                features[i] = np.pi * np.clip(0.3 + 2.0 * enemy_ratio, 0.3, 1.0)
            else:
                # Negative rotation [-pi, 0]: Wall distance only
                features[i] = -np.pi * sec_depth

        return features

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.last_health = 100.0
        self.last_killcount = 0.0
        self.last_pos_x = 0.0
        self.max_pos_x = 0.0
        state = self.game.get_state()
        return self._extract_radar(state), {}

    def step(self, action: int):
        reward = self.game.make_action(self.actions[action], self.frame_skip)
        terminated = self.game.is_episode_finished()
        state = None if terminated else self.game.get_state()
        obs = self._extract_radar(state)

        # For corridor: combat and survival reward shaping
        if self.use_composite_actions:
            try:
                health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
                killcount = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                pos_x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            except Exception:
                health = self.last_health
                killcount = self.last_killcount
                pos_x = getattr(self, "last_pos_x", 0.0)

            health_delta = health - self.last_health
            kill_delta = killcount - self.last_killcount
            self.last_health = health
            self.last_killcount = killcount

            # Combat & survival shaping:
            # +200 bonus per enemy eliminated
            # -1.0 penalty per HP of damage taken
            reward += kill_delta * 200.0 + health_delta * 1.0

            # Forward advancement bonus: reward pushing to new furthest distance
            if not hasattr(self, "max_pos_x"):
                self.max_pos_x = 0.0
            if pos_x > self.max_pos_x:
                reward += (pos_x - self.max_pos_x) * 1.0
                self.max_pos_x = pos_x

            # Anti-camping penalty: discourage stalling in the starting spawn room (X < 150)
            if pos_x < 150.0:
                reward -= 0.5

            self.last_pos_x = pos_x

        return obs, float(reward), terminated, False, {}

    def close(self):
        self.game.close()


if __name__ == "__main__":
    env = QuantumDoomEnv(n_qubits=8)
    obs, _ = env.reset(seed=42)
    state = env.game.get_state()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(state.screen_buffer)
    axes[0].set_title("1. Raw ViZDoom RGB Frame")
    axes[1].imshow(state.labels_buffer, cmap="magma")
    axes[1].set_title("2. Native ViZDoom Label Buffer (Zero CNN)")
    colors = ["#ff4b4b" if val > 0 else "#3b82f6" for val in obs]
    axes[2].bar(range(8), obs, color=colors)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylim(-np.pi, np.pi)
    axes[2].set_title("3. 8-Qubit Input Angles (Red = Monster Sector)")
    axes[2].set_xlabel("Qubit Index (FOV Sector 0 to 7)")
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150)
    print("Saved sanity_check.png! Radar angles:", np.round(obs, 2))
    env.close()
