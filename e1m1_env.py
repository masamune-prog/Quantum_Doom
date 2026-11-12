"""
Gymnasium environment for the original Doom E1M1 (Hangar) from the shareware doom1.wad.

Goal: reach and press the exit switch without dying.

12-qubit observation, all angles in [-pi, pi]:
  - Qubits 0-7:  8 horizontal FOV sectors.
                 * Positive [0.3*pi, pi]: a monster is visible in that sector.
                 * Negative [-pi, 0]:     wall clearance (-pi = wall at face, 0 = open),
                   log-scaled over 3-120 depth units (depth_scale="legacy": linear 3-38).
  - Qubit 8:     Bearing to the next route waypoint relative to view direction
                 (0 = straight ahead, +/- = turn left / right).
  - Qubit 9:     Proximity to the next waypoint (0 = far, pi = on top of it).
  - Qubit 10:    Health (-pi = 0 HP, pi = 100+ HP).
  - Qubit 11:    Interaction cue: pi if the exit switch is right ahead, pi/2 for a door,
                 0 otherwise.

Waypoints come from the classical route planner (e1m1_waypoints.json); without them
qubits 8-9 are zero and no progress reward is given. After the last waypoint, qubits
8-9 point at the exit switch.

Action space (same 7 actions as the navigation env):
  0: MOVE_FORWARD  1: ATTACK  2: MOVE_LEFT  3: MOVE_RIGHT
  4: TURN_LEFT     5: TURN_RIGHT  6: USE
"""

import json
import os
import numpy as np
import gymnasium as gym
import vizdoom as vzd

from e1m1_map import DoomMap, WAD_PATH, MONSTERS

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "e1m1_quantum.cfg")
# Route waypoints inside door sectors. Teleporting onto one while the door is shut traps
# the player in the door sector, so they are never valid start points.
DOOR_WAYPOINTS = (5, 17, 20)
WAYPOINTS_PATH = os.path.join(HERE, "e1m1_waypoints.json")

MONSTER_NAMES = set(MONSTERS.values())


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class QuantumE1M1Env(gym.Env):
    # Buttons: [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, USE]
    ACTIONS = [
        [0, 0, 0, 1, 0, 0, 0],  # 0: MOVE_FORWARD
        [0, 0, 1, 0, 0, 0, 0],  # 1: ATTACK
        [1, 0, 0, 0, 0, 0, 0],  # 2: MOVE_LEFT
        [0, 1, 0, 0, 0, 0, 0],  # 3: MOVE_RIGHT
        [0, 0, 0, 0, 1, 0, 0],  # 4: TURN_LEFT
        [0, 0, 0, 0, 0, 1, 0],  # 5: TURN_RIGHT
        [0, 0, 0, 0, 0, 0, 1],  # 6: USE
    ]
    ACTION_NAMES = ["MOVE_FORWARD", "ATTACK", "MOVE_LEFT", "MOVE_RIGHT",
                    "TURN_LEFT", "TURN_RIGHT", "USE"]

    def __init__(
        self,
        skill: int = 3,
        frame_skip: int = 4,
        waypoints=None,
        waypoint_radius: float = 64.0,
        interact_range: float = 96.0,
        render_rgb: bool = False,
        cheats: bool = False,
        depth_scale: str = "log",
    ):
        super().__init__()
        if depth_scale not in ("log", "legacy"):
            raise ValueError(f"depth_scale must be 'log' or 'legacy', not {depth_scale!r}")
        # "legacy" is the original linear 3-38 unit range (saturated on ~9% of E1M1 sectors);
        # keep it only to reproduce runs trained with it.
        self.depth_scale = depth_scale
        self.n_qubits = 12
        self.frame_skip = frame_skip
        self.waypoint_radius = waypoint_radius
        self.interact_range = interact_range

        if waypoints is None and os.path.exists(WAYPOINTS_PATH):
            with open(WAYPOINTS_PATH) as f:
                waypoints = json.load(f)["waypoints"]
        self.waypoints = np.array(waypoints, dtype=np.float32) if waypoints else None

        self.doom_map = DoomMap(WAD_PATH, "E1M1")
        self.exit_segments = [(l["p1"], l["p2"]) for l in self.doom_map.exit_lines()]
        self.door_segments = [(l["p1"], l["p2"]) for l in self.doom_map.door_lines()]
        self.exit_mid = 0.5 * (self.exit_segments[0][0] + self.exit_segments[0][1])

        self.game = vzd.DoomGame()
        self.game.load_config(CFG_PATH)
        self.game.set_doom_game_path(WAD_PATH)
        self.game.set_doom_map("E1M1")
        self.game.set_doom_skill(skill)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_window_visible(False)
        if cheats:
            # Enables console commands such as "warp x y" (used for curriculum start points)
            self.game.add_game_args("+sv_cheats 1")
        self.game.init()

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.n_qubits,), dtype=np.float32
        )
        self.render_rgb = render_rgb
        self.last_rgb = np.zeros((240, 320, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ helpers
    def _player(self):
        x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        angle = np.deg2rad(self.game.get_game_variable(vzd.GameVariable.ANGLE))
        return np.array([x, y], dtype=np.float32), float(angle)

    def _segment_ahead(self, pos, angle, segments) -> bool:
        """True if any segment is within interact range and roughly in front of the player."""
        for a, b in segments:
            if _point_segment_distance(pos, a, b) > self.interact_range:
                continue
            mid = 0.5 * (a + b)
            bearing = _wrap_angle(np.arctan2(mid[1] - pos[1], mid[0] - pos[0]) - angle)
            if abs(bearing) < np.deg2rad(35):
                return True
        return False

    def _extract_features(self, state) -> np.ndarray:
        features = np.zeros(self.n_qubits, dtype=np.float32)
        if state is None:
            return features
        if self.render_rgb:
            self.last_rgb = state.screen_buffer

        depth = state.depth_buffer
        labels = state.labels_buffer
        monster_ids = [l.value for l in state.labels if l.object_name in MONSTER_NAMES]

        h, w = depth.shape
        mid_band = slice(h // 3, 2 * h // 3)
        sector_w = w // 8
        for i in range(8):
            col = slice(i * sector_w, (i + 1) * sector_w)
            monster_ratio = 0.0
            if monster_ids:
                monster_ratio = float(np.mean(np.isin(labels[mid_band, col], monster_ids)))
            if monster_ratio > 0.02:
                features[i] = np.pi * np.clip(0.3 + 2.0 * monster_ratio, 0.3, 1.0)
            else:
                clearance = self._clearance(float(np.mean(depth[mid_band, col])))
                features[i] = -np.pi * (1.0 - clearance)

        pos, angle = self._player()
        if self.waypoints is not None:
            # After the last waypoint, point at the exit switch so the agent turns to face it
            target = self.waypoints[self.wp_idx] if self.wp_idx < len(self.waypoints) else self.exit_mid
            delta = target - pos
            # Screen x grows to the right, Doom angles grow counter-clockwise: +bearing = target on the left
            features[8] = _wrap_angle(np.arctan2(delta[1], delta[0]) - angle)
            features[9] = np.pi * (1.0 - min(float(np.linalg.norm(delta)), 1024.0) / 1024.0)

        health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        features[10] = np.pi * np.clip(2.0 * health / 100.0 - 1.0, -1.0, 1.0)

        if self._segment_ahead(pos, angle, self.exit_segments):
            features[11] = np.pi
        elif self._segment_ahead(pos, angle, self.door_segments):
            features[11] = np.pi / 2

        return features

    def _clearance(self, depth: float) -> float:
        """Sector clearance in [0, 1] from mean depth-buffer value."""
        if self.depth_scale == "legacy":
            return float(np.clip((depth - 3.0) / 35.0, 0.0, 1.0))
        # Log scale up to 120 units (~max seen in E1M1): fine near walls, unsaturated far away.
        return float(np.clip(np.log1p(max(depth - 3.0, 0.0)) / np.log1p(117.0), 0.0, 1.0))

    def warp(self, x: float, y: float):
        """Teleport the player (requires cheats=True). Resets waypoint tracking to the new spot."""
        self.game.send_game_command(f"warp {x} {y}")
        self.game.advance_action(1)
        pos, _ = self._player()
        if self.waypoints is not None:
            self.wp_idx = int(np.argmin(np.linalg.norm(self.waypoints - pos, axis=1)))
            self.last_wp_dist = self._waypoint_distance(pos)
        return self._extract_features(self.game.get_state())

    def _waypoint_distance(self, pos) -> float:
        return float(np.linalg.norm(self.waypoints[self.wp_idx] - pos))

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.wp_idx = 0
        self.last_health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        self.last_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        pos, _ = self._player()
        self.last_wp_dist = self._waypoint_distance(pos) if self.waypoints is not None else 0.0
        return self._extract_features(self.game.get_state()), {}

    def step(self, action: int):
        self.game.make_action(self.ACTIONS[action], self.frame_skip)
        finished = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        timeout = finished and self.game.is_episode_timeout_reached()
        # ViZDoom ends the episode when the level exit is triggered
        exit_reached = finished and not dead and not timeout

        reward = -0.01  # living cost

        if not finished:
            pos, _ = self._player()
            health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
            kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)

            # Damage taken is the main survival signal; healing is not rewarded
            reward += 0.05 * min(health - self.last_health, 0.0)
            reward += 2.0 * (kills - self.last_kills)
            self.last_health, self.last_kills = health, kills

            if self.waypoints is not None and self.wp_idx < len(self.waypoints):
                dist = self._waypoint_distance(pos)
                reward += np.clip((self.last_wp_dist - dist) / 64.0, -1.0, 1.0)
                if dist < self.waypoint_radius:
                    reward += 5.0
                    self.wp_idx += 1
                    if self.wp_idx < len(self.waypoints):
                        dist = self._waypoint_distance(pos)
                self.last_wp_dist = dist

        if dead:
            reward -= 50.0
        if exit_reached:
            reward += 100.0

        state = None if finished else self.game.get_state()
        obs = self._extract_features(state)
        info = {
            "exit_reached": exit_reached,
            "dead": dead,
            "waypoint_idx": self.wp_idx,
            "health": self.last_health,
            "kills": self.last_kills,
        }
        return obs, float(reward), bool(finished and not timeout), bool(timeout), info

    def close(self):
        self.game.close()
