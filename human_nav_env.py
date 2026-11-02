"""Screen-only navigation with action/visual-motion map memory.

Policy observations and map updates use only RGB frames and actions. Engine pose,
goal coordinates, and kill count are private training-reward inputs only.
"""

import os

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd

from reward_shaping import progress_reward, kill_reward


HERE = os.path.dirname(os.path.abspath(__file__))


class VisualTravelMap:
    """Coarse egocentric dead-reckoning map; never reads engine pose or geometry."""

    def __init__(self, forward_cell_actions=3, turn_radians=np.deg2rad(14.0)):
        self.forward_cell_actions = forward_cell_actions
        self.turn_radians = turn_radians
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.visits = {(0, 0): 1}
        self.blocked = set()
        self.last_new = False

    def cell(self, x=None, y=None):
        return (round(self.x if x is None else x), round(self.y if y is None else y))

    def neighbor(self, offset=0.0):
        angle = self.heading + offset
        return self.cell(self.x + np.cos(angle), self.y + np.sin(angle))

    def step(self, action, visible_motion):
        self.last_new = False
        if action == 1:
            self.heading += self.turn_radians
        elif action == 2:
            self.heading -= self.turn_radians
        elif action == 0:
            if visible_motion:
                distance = 1.0 / self.forward_cell_actions
                self.x += distance * np.cos(self.heading)
                self.y += distance * np.sin(self.heading)
                cell = self.cell()
                self.last_new = cell not in self.visits
                self.visits[cell] = self.visits.get(cell, 0) + 1
            else:
                self.blocked.add((self.cell(), round(self.heading / (np.pi / 4)) % 8))

    def step_strafe(self, direction, visible_motion):
        """Update the travel map for a side-step; direction is +1 left, -1 right."""
        self.last_new = False
        offset = direction * np.pi / 2
        if visible_motion:
            distance = 1.0 / self.forward_cell_actions
            self.x += distance * np.cos(self.heading + offset)
            self.y += distance * np.sin(self.heading + offset)
            cell = self.cell()
            self.last_new = cell not in self.visits
            self.visits[cell] = self.visits.get(cell, 0) + 1
        else:
            self.blocked.add((self.cell(), round((self.heading + offset) / (np.pi / 4)) % 8))

    def features(self):
        # 0 = unexplored; positive = repeatedly visited; negative = attempted blockage.
        values = []
        for offset in (0.0, np.pi / 2, -np.pi / 2):
            direction = round((self.heading + offset) / (np.pi / 4)) % 8
            if (self.cell(), direction) in self.blocked:
                values.append(-np.pi)
            else:
                visits = self.visits.get(self.neighbor(offset), 0)
                values.append(np.pi * min(visits, 4) / 4)
        values.append(np.pi * min(self.visits.get(self.cell(), 0), 8) / 8)
        return np.asarray(values, dtype=np.float32)


class VisualPlaceMemory(VisualTravelMap):
    """Travel map with RGB keyframes for recognizing revisited viewpoints."""

    def reset(self):
        super().reset()
        self.keyframes = []
        self.recorded_views = set()
        self.edges = set()
        self.tick = 0
        self.familiarity = 0.0

    @staticmethod
    def _descriptor(view):
        return cv2.resize(view, (16, 10), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def observe(self, view):
        """Match old views, then store a keyframe for a new place or heading."""
        descriptor = self._descriptor(view)
        heading_bin = round(self.heading / (np.pi / 4)) % 8
        position = np.array([self.x, self.y], dtype=np.float32)
        best = None
        best_error = float("inf")
        for frame in self.keyframes:
            if self.tick - frame["tick"] < 12:
                continue
            heading_delta = abs(heading_bin - frame["heading_bin"])
            if min(heading_delta, 8 - heading_delta) > 1:
                continue
            error = float(np.mean(np.abs(descriptor - frame["descriptor"])))
            if error < best_error:
                best, best_error = frame, error
        self.familiarity = float(np.clip((0.08 - best_error) / 0.08, 0.0, 1.0)) if best else 0.0
        if best is not None:
            estimated_distance = float(np.linalg.norm(position - best["position"]))
            if (best_error < 0.05 and estimated_distance <= 3.0) or best_error < 0.022:
                self.x, self.y = map(float, best["position"])
                cell = self.cell()
                self.visits[cell] = self.visits.get(cell, 0) + 1
                self.last_new = False
        view_key = (self.cell(), heading_bin)
        if view_key not in self.recorded_views:
            self.keyframes.append({"descriptor": descriptor, "position": np.array([self.x, self.y]),
                                   "heading_bin": heading_bin, "tick": self.tick})
            self.recorded_views.add(view_key)
            if len(self.keyframes) > 256:
                self.keyframes.pop(0)

    def step(self, action, visible_motion, view=None):
        previous_cell = self.cell()
        super().step(action, visible_motion)
        self.tick += 1
        if view is not None:
            self.observe(view)
        cell = self.cell()
        if cell != previous_cell:
            self.edges.add(tuple(sorted((previous_cell, cell))))

    def features(self):
        values = super().features()
        values[3] = np.pi * self.familiarity
        return values


class HumanNavigationEnv(gym.Env):
    """My Way Home with RGB-derived features and optional classical travel map."""

    ACTION_NAMES = ("FORWARD", "TURN_LEFT", "TURN_RIGHT")
    ACTIONS = (
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    )

    def __init__(self, memory=True, frame_skip=4, memory_kind="travel"):
        super().__init__()
        self.memory = memory
        if memory_kind not in ("travel", "place"):
            raise ValueError("memory_kind must be 'travel' or 'place'")
        self.memory_kind = memory_kind
        self.frame_skip = frame_skip
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(HERE, "my_way_home_quantum.cfg"))
        self.game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "my_way_home.wad"))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(True)
        self.game.set_window_visible(False)
        # Pose and object data are private to reward calculation. _observe and
        # VisualTravelMap never read them.
        self.game.clear_available_game_variables()
        for variable in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                         vzd.GameVariable.KILLCOUNT):
            self.game.add_available_game_variable(variable)
        self.game.set_objects_info_enabled(True)
        self.game.init()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-np.pi, np.pi, shape=(12,), dtype=np.float32)
        self.travel_map = VisualPlaceMemory() if memory_kind == "place" else VisualTravelMap()
        self.previous_view = None
        self.last_rgb = None
        self.goal_pos = None
        self.last_reward_position = None
        self.last_kills = 0.0

    def _reward_position(self):
        return np.array([
            self.game.get_game_variable(vzd.GameVariable.POSITION_X),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
        ], dtype=np.float32)

    @staticmethod
    def _goal_from_objects(state):
        for obj in state.objects or ():
            if "Armor" in obj.name or "Vest" in obj.name:
                return np.array([obj.position_x, obj.position_y], dtype=np.float32)
        raise RuntimeError("Goal item was absent from ViZDoom object information")

    @staticmethod
    def _view(rgb):
        # Upper screen avoids weapon animation and the visible HUD.
        return cv2.resize(rgb[:165], (80, 40), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _visual_features(rgb):
        view = HumanNavigationEnv._view(rgb)
        gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 35, 95)
        features = []
        for i in range(8):
            sector = view[:, 10 * i:10 * (i + 1)]
            red, green, blue = (sector[:, :, j].astype(np.float32) for j in range(3))
            green_ratio = np.mean((green > 45) & (green > 1.4 * red) & (green > 1.2 * blue))
            edge_ratio = np.mean(edges[:, 10 * i:10 * (i + 1)] > 0)
            brightness = np.mean(gray[:, 10 * i:10 * (i + 1)]) / 255.0
            # Green is an image-color cue, never an object label.
            feature = (0.55 * brightness + 0.45 * min(edge_ratio * 3, 1)) * 2 - 1
            if green_ratio > 0.025:
                feature = min(1.0, 0.4 + 4 * green_ratio)
            features.append(np.pi * feature)
        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _forward_motion(previous, current):
        # Compare visible scenery, ignoring the gun and HUD. Threshold catches
        # blocked forward actions while allowing normal lighting changes.
        diff = np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32)))
        return diff > 2.5

    def _observe(self, rgb):
        visual = self._visual_features(rgb)
        memory = self.travel_map.features() if self.memory else np.zeros(4, dtype=np.float32)
        return np.concatenate((visual, memory)).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self.travel_map.reset()
        state = self.game.get_state()
        self.goal_pos = self._goal_from_objects(state)
        self.last_reward_position = self._reward_position()
        self.last_kills = float(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        self.last_rgb = state.screen_buffer.copy()
        self.previous_view = self._view(self.last_rgb)
        if self.memory_kind == "place":
            self.travel_map.observe(self.previous_view)
        return self._observe(self.last_rgb), {}

    def step(self, action):
        action = int(action)
        self.game.make_action(self.ACTIONS[action], self.frame_skip)
        ended = self.game.is_episode_finished()
        timeout = bool(ended and self.game.is_episode_timeout_reached())
        # The +1 pickup can land on the tic before the episode ends, so the final
        # action's own reward may be negative; use the episode total instead.
        success = bool(ended and not timeout and self.game.get_total_reward() > 0.5)
        reward = -0.01 + (100.0 if success else 0.0)
        position = self._reward_position()
        kills = float(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        reward += progress_reward(self.last_reward_position, position, self.goal_pos,
                                  distance_scale=1024.0, terminal=ended)
        reward += kill_reward(self.last_kills, kills)
        self.last_reward_position, self.last_kills = position, kills
        if not ended:
            self.last_rgb = self.game.get_state().screen_buffer.copy()
            current_view = self._view(self.last_rgb)
            moving = self._forward_motion(self.previous_view, current_view) if action == 0 else False
            if self.memory_kind == "place":
                self.travel_map.step(action, moving, current_view)
            else:
                self.travel_map.step(action, moving)
            self.previous_view = current_view
            if action == 0:
                reward += 0.05 if self.travel_map.last_new else (-0.03 if not moving else 0.0)
            obs = self._observe(self.last_rgb)
        else:
            obs = np.zeros(12, dtype=np.float32)
        return obs, reward, ended and not timeout, timeout, {"success": success}

    def close(self):
        self.game.close()
