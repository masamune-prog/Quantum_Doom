"""E1M1 with stacked screen frames as the only policy observation.

Combines the two earlier screen-only tasks: route-following like My Way Home and
combat like Deadly Corridor, plus USE for doors and the exit switch. Pose, health,
kill count, the depth and labels buffers and the map geometry are private to the
reward, episode statistics, curriculum starts and auxiliary targets.

Progress is paid from a precomputed walking-distance field (e1m1_distance.py), not
straight-line distance, because the route winds through rooms and doors.

reset(options={"start_wp": k}) teleports to route waypoint k (training curriculum,
needs cheats); the default is the real start, which is what evaluation uses.
"""

import json
import os
from collections import deque

import gymnasium as gym
import numpy as np
import vizdoom as vzd

from e1m1_distance import DistanceField, WAYPOINTS_PATH
from e1m1_env import DOOR_WAYPOINTS, _wrap_angle
from e1m1_map import MONSTERS, WAD_PATH, DoomMap
from human_nav_env import HERE
from pixel_corridor_env import PixelCorridorEnv
from pixel_nav_env import FRAME_SHAPE, PixelNavigationEnv
from e1m1_teacher import RouteTeacher
from reward_shaping import field_progress_reward, kill_reward

AUX_SIZE_E1M1 = 20  # 8 depth, 8 monster, 1 "door or exit switch ahead", then 3 route targets
AUX_ROUTE = 3       # sin and cos of the route direction relative to the heading, route progress
INTERACT_RANGE = 72.0  # Doom's use range is 64 units from the player's centre
INTERACT_RAY_OFFSETS = np.deg2rad([-8.0, 0.0, 8.0])


class PixelE1M1Env(PixelCorridorEnv):
    ACTION_NAMES = ("FORWARD", "ATTACK", "STRAFE_LEFT", "STRAFE_RIGHT",
                    "TURN_LEFT", "TURN_RIGHT", "USE")
    # Buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, USE.
    ACTIONS = (
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    )

    def __init__(self, frame_skip=4, frames=4, skill=3, progress_weight=30.0,
                 death_penalty=30.0, teacher=False, kill_weight=2.0):
        gym.Env.__init__(self)
        self.progress_weight = progress_weight
        self.death_penalty = death_penalty
        self.kill_weight = kill_weight
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frames)
        with open(WAYPOINTS_PATH) as handle:
            route = json.load(handle)
        self.waypoints = np.array(route["waypoints"], dtype=np.float32)
        self.goal_pos = np.array(route["goal"], dtype=np.float32)
        self.field = DistanceField()
        doom_map = DoomMap(WAD_PATH, "E1M1")
        exit_line = doom_map.exit_lines()[0]
        self.exit_mid = 0.5 * (exit_line["p1"] + exit_line["p2"])
        self.interact_segments = [(l["p1"], l["p2"]) for l in
                                  doom_map.door_lines() + doom_map.exit_lines()]
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(HERE, "pixel_e1m1.cfg"))
        self.game.set_doom_game_path(WAD_PATH)
        self.game.set_doom_map("E1M1")
        self.game.set_doom_skill(skill)
        self.game.set_window_visible(False)
        # Teleport commands for curriculum starts; the policy cannot use them.
        self.game.add_game_args("+sv_cheats 1")
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.clear_available_game_variables()
        for variable in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                         vzd.GameVariable.ANGLE, vzd.GameVariable.HEALTH,
                         vzd.GameVariable.KILLCOUNT):
            self.game.add_available_game_variable(variable)
        self.game.init()
        # Imitation only: info["teacher"] carries the scripted teacher's action for the
        # state just returned. It reads private state and is never part of the observation.
        self.teacher = RouteTeacher(self) if teacher else None
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, shape=(3 * frames, *FRAME_SHAPE),
                                                dtype=np.uint8)

    def _pose(self):
        return (self._reward_position(),
                float(np.deg2rad(self._var(vzd.GameVariable.ANGLE))))

    def _pose_vector(self):
        """Training/probe only: (x, y, angle in radians); never part of the observation."""
        position, angle = self._pose()
        return np.array([position[0], position[1], angle], dtype=np.float32)

    def _route_cost(self, position):
        """Walking distance to the exit; keeps the last value if the map has no cell here."""
        cost = self.field(position)
        return self.route_cost if cost is None else cost

    def _interact_ahead(self, position, angle, max_range=INTERACT_RANGE):
        """True if USE would hit a door or the exit switch, like Doom's own use trace.

        Casts rays along the heading (and 8 degrees either side) up to INTERACT_RANGE.
        A midpoint-bearing test fails when the player stands against a wide door.
        """
        for offset in INTERACT_RAY_OFFSETS:
            heading = angle + offset
            direction = np.array([np.cos(heading), np.sin(heading)])
            for a, b in self.interact_segments:
                edge = b - a
                denom = direction[0] * edge[1] - direction[1] * edge[0]
                if abs(denom) < 1e-6:
                    continue
                delta = a - position
                t = (delta[0] * edge[1] - delta[1] * edge[0]) / denom
                u = (delta[0] * direction[1] - delta[1] * direction[0]) / denom
                if 0.0 <= t <= max_range and 0.0 <= u <= 1.0:
                    return True
        return False

    def _route_targets(self, position, angle):
        """Training-only: where the route leads from here, and how far along it we are."""
        h = 16.0
        slope = np.zeros(2)
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = h
            ahead, behind = self.field(position + step), self.field(position - step)
            if ahead is not None and behind is not None:
                slope[axis] = ahead - behind
        if np.linalg.norm(slope) < 1e-6:
            direction = np.zeros(2)
        else:
            downhill = _wrap_angle(np.arctan2(-slope[1], -slope[0]) - angle)
            direction = np.array([np.sin(downhill), np.cos(downhill)])
        progress = np.clip(1 - self.route_cost / self.field.start_distance, 0, 1)
        return np.append(direction, progress)

    def _aux(self, state):
        position, angle = self._pose()
        base = PixelCorridorEnv._aux_targets(state)
        return np.concatenate([base, [float(self._interact_ahead(position, angle))],
                               self._route_targets(position, angle)]).astype(np.float32)

    def _warp_to(self, index, rng):
        """Teleport to waypoint ``index`` facing roughly along the route (±45 degrees)."""
        x, y = self.waypoints[index]
        ahead = self.waypoints[index + 1] if index + 1 < len(self.waypoints) else self.exit_mid
        self._warp_xy(x, y, np.arctan2(ahead[1] - y, ahead[0] - x)
                      + rng.uniform(-np.pi / 4, np.pi / 4))

    def _warp_xy(self, x, y, desired):
        """Teleport to (x, y) and turn to the absolute heading ``desired`` (radians)."""
        self.game.send_game_command(f"warp {x} {y}")
        self.game.advance_action(1)
        for _ in range(120):
            error = _wrap_angle(desired - self._pose()[1])
            if abs(error) < np.deg2rad(4) or self.game.is_episode_finished():
                break
            self.game.make_action(self.ACTIONS[4 if error > 0 else 5], 1)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        start_wp = (options or {}).get("start_wp")
        start_xy = (options or {}).get("start_xy")  # robustness tests: any position + heading
        if start_xy is not None:
            self._warp_xy(start_xy[0], start_xy[1], (options or {})["start_heading"])
        elif start_wp is not None and start_wp >= 0:
            if int(start_wp) in DOOR_WAYPOINTS:
                raise ValueError(f"start_wp {start_wp} is a door waypoint {DOOR_WAYPOINTS}")
            self._warp_to(int(start_wp), np.random.default_rng(seed))
        state = self.game.get_state()
        position, _ = self._pose()
        self.route_cost = self.field.start_distance
        self.route_cost = self._route_cost(position)
        self.best_cost = self.route_cost
        self.start_progress = float(np.clip(1 - self.route_cost / self.field.start_distance, 0, 1))
        self.last_reward_position = position
        self.start_pos = position
        self.health = self._var(vzd.GameVariable.HEALTH)
        self.kills = self._var(vzd.GameVariable.KILLCOUNT)
        self.start_kills = self.kills
        frame = self._frame(state.screen_buffer)
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        info = {"aux": self._aux(state), "pose": self._pose_vector()}
        if self.teacher is not None:
            self.teacher.reset()
            info["teacher"] = self.teacher.action(state)
        return self._observe(), info

    def step(self, action):
        self.game.make_action(self.ACTIONS[int(action)], self.frame_skip)
        ended = self.game.is_episode_finished()
        timeout = bool(ended and self.game.is_episode_timeout_reached())
        dead = bool(self.game.is_player_dead())
        # The exit switch ends the level, so a non-timeout ending while alive is an exit.
        success = bool(ended and not timeout and not dead)
        position, _ = self._pose()
        health = self._var(vzd.GameVariable.HEALTH)
        kills = self._var(vzd.GameVariable.KILLCOUNT)
        cost = self._route_cost(position)
        reward = -0.01 + (100.0 if success else 0.0) - (self.death_penalty if dead else 0.0)
        reward -= 0.05 * max(0.0, self.health - health)
        reward += kill_reward(self.kills, kills, weight=self.kill_weight)
        reward += field_progress_reward(self.route_cost, cost,
                                        distance_scale=self.field.start_distance,
                                        weight=self.progress_weight, terminal=ended)
        self.route_cost, self.health, self.kills = cost, health, kills
        self.best_cost = min(self.best_cost, cost)
        info = {"success": success}
        if ended:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info["aux"] = np.zeros(AUX_SIZE_E1M1, dtype=np.float32)
            info["pose"] = np.zeros(3, dtype=np.float32)
            # progress is the furthest route progress reached, relative to the real start
            info["episode"] = {
                "kills": kills - self.start_kills, "health": max(0.0, health), "dead": dead,
                "start_progress": self.start_progress,
                "progress": float(np.clip(1 - self.best_cost / self.field.start_distance,
                                          0, 1))}
        else:
            state = self.game.get_state()
            self.frames.append(self._frame(state.screen_buffer))
            obs, info["aux"] = self._observe(), self._aux(state)
            info["pose"] = self._pose_vector()
            if self.teacher is not None:
                info["teacher"] = self.teacher.action(state)
        return obs, reward, ended and not timeout, timeout, info
