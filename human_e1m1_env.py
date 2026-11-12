"""E1M1 with screen-only policy observations and private progress/kill rewards."""

import numpy as np
import vizdoom as vzd
import gymnasium as gym

from e1m1_map import DoomMap, WAD_PATH
from human_nav_env import HumanNavigationEnv, VisualTravelMap, HERE
from reward_shaping import progress_reward, kill_reward


class HumanE1M1Env(HumanNavigationEnv):
    """The actor sees RGB-derived features and optional action/visual map only."""

    ACTION_NAMES = ("FORWARD", "ATTACK", "STRAFE_LEFT", "STRAFE_RIGHT",
                    "TURN_LEFT", "TURN_RIGHT", "USE")
    ACTIONS = (
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    )

    def __init__(self, memory=True, frame_skip=4, skill=3):
        gym.Env.__init__(self)
        self.memory = memory
        self.frame_skip = frame_skip
        doom_map = DoomMap(WAD_PATH, "E1M1")
        exit_line = doom_map.exit_lines()[0]
        self.goal_pos = 0.5 * (exit_line["p1"] + exit_line["p2"])
        self.game = vzd.DoomGame()
        self.game.load_config(f"{HERE}/e1m1_quantum.cfg")
        self.game.set_doom_game_path(WAD_PATH)
        self.game.set_doom_map("E1M1")
        self.game.set_doom_skill(skill)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(True)
        self.game.set_window_visible(False)
        self.game.clear_available_game_variables()
        for variable in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                         vzd.GameVariable.KILLCOUNT, vzd.GameVariable.HEALTH):
            self.game.add_available_game_variable(variable)
        self.game.init()
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(-np.pi, np.pi, shape=(12,), dtype=np.float32)
        self.travel_map = VisualTravelMap()
        self.last_rgb = None
        self.previous_view = None
        self.last_reward_position = None
        self.last_kills = 0.0
        self.last_health = 100.0

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self.travel_map.reset()
        self.last_reward_position = self._reward_position()
        self.last_kills = float(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        self.last_health = float(self.game.get_game_variable(vzd.GameVariable.HEALTH))
        self.last_rgb = self.game.get_state().screen_buffer.copy()
        self.previous_view = self._view(self.last_rgb)
        return self._observe(self.last_rgb), {}

    def step(self, action):
        action = int(action)
        self.game.make_action(self.ACTIONS[action], self.frame_skip)
        ended = self.game.is_episode_finished()
        timeout = bool(ended and self.game.is_episode_timeout_reached())
        dead = bool(self.game.is_player_dead())
        success = bool(ended and not timeout and not dead)
        position = self._reward_position()
        kills = float(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        health = float(self.game.get_game_variable(vzd.GameVariable.HEALTH))
        reward = (-0.01 + (100.0 if success else 0.0) - (50.0 if dead else 0.0)
                  - 0.05 * max(0.0, self.last_health - health))
        reward += progress_reward(self.last_reward_position, position, self.goal_pos,
                                  distance_scale=4096.0, terminal=ended)
        reward += kill_reward(self.last_kills, kills)
        self.last_reward_position, self.last_kills, self.last_health = position, kills, health
        if not ended:
            self.last_rgb = self.game.get_state().screen_buffer.copy()
            current_view = self._view(self.last_rgb)
            moving = (self._forward_motion(self.previous_view, current_view)
                      if action in (0, 2, 3) else False)
            if action == 0:
                self.travel_map.step(0, moving)
            elif action == 2:
                self.travel_map.step_strafe(+1, moving)
            elif action == 3:
                self.travel_map.step_strafe(-1, moving)
            elif action == 4:
                self.travel_map.step(1, False)
            elif action == 5:
                self.travel_map.step(2, False)
            self.previous_view = current_view
            if action in (0, 2, 3):
                reward += 0.05 if self.travel_map.last_new else (-0.03 if not moving else 0.0)
            obs = self._observe(self.last_rgb)
        else:
            obs = np.zeros(12, dtype=np.float32)
        return obs, float(reward), ended and not timeout, timeout, {"success": success, "dead": dead}
