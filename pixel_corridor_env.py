"""Deadly Corridor (skill 3) with stacked screen frames as the only policy observation.

Reward is computed here once: kills, damage, death, potential-based progress
toward the vest and a success bonus. The scenario's own x-progress reward is
ignored. Pose, health, kill count, the depth buffer and the labels buffer are
private to the reward, episode statistics and auxiliary targets.
"""

import os
from collections import deque

import gymnasium as gym
import numpy as np
import vizdoom as vzd

from e1m1_map import MONSTERS
from human_nav_env import HERE, HumanNavigationEnv
from pixel_nav_env import AUX_SIZE, FRAME_SHAPE, N_SECTORS, PixelNavigationEnv
from reward_shaping import kill_reward, progress_reward

MONSTER_NAMES = set(MONSTERS.values())


class PixelCorridorEnv(PixelNavigationEnv):
    ACTION_NAMES = ("FORWARD", "ATTACK", "STRAFE_LEFT", "STRAFE_RIGHT",
                    "TURN_LEFT", "TURN_RIGHT")
    # Buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT.
    ACTIONS = (
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    )

    def __init__(self, frame_skip=4, frames=4, skill=3, progress_weight=30.0,
                 death_penalty=30.0, timeout_penalty=0.0, progress_mode="potential"):
        gym.Env.__init__(self)
        # Job 71183 used 5 and 50: the agent camped at the start, killing nearby
        # enemies and never advancing. A blind charge still scores below camping.
        # "potential" progress is potential-based shaping that returns to zero at episode
        # end, so it sums to ~0 per episode and never changes which policy is best.
        # "max" pays progress_weight times each new furthest fraction of the start-to-vest
        # distance and keeps it, so advancing is worth reward even if the agent dies.
        # timeout_penalty makes camping until the time limit cost as much as dying.
        if progress_mode not in ("potential", "max"):
            raise ValueError(progress_mode)
        self.progress_weight = progress_weight
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        self.progress_mode = progress_mode
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frames)
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(HERE, "pixel_corridor.cfg"))
        self.game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "deadly_corridor.wad"))
        self.game.set_doom_skill(skill)
        self.game.set_window_visible(False)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.clear_available_game_variables()
        for variable in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                         vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT):
            self.game.add_available_game_variable(variable)
        self.game.init()
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, shape=(3 * frames, *FRAME_SHAPE),
                                                dtype=np.uint8)

    def _var(self, variable):
        return float(self.game.get_game_variable(variable))

    @staticmethod
    def _aux_targets(state):
        """Per-sector depth and monster visibility, for auxiliary losses only."""
        depth = state.depth_buffer
        labels = state.labels_buffer
        rows = slice(depth.shape[0] // 3, 2 * depth.shape[0] // 3)
        monster_ids = [l.value for l in state.labels if l.object_name in MONSTER_NAMES]
        width = depth.shape[1] // N_SECTORS
        near, enemy = [], []
        for i in range(N_SECTORS):
            cols = slice(i * width, (i + 1) * width)
            near.append(min(1.0, np.log1p(np.mean(depth[rows, cols])) / np.log(65.0)))
            enemy.append(float(np.isin(labels[:, cols], monster_ids).any()) if monster_ids else 0.0)
        return np.asarray(near + enemy, dtype=np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        state = self.game.get_state()
        self.goal_pos = HumanNavigationEnv._goal_from_objects(state)
        self.start_pos = self._reward_position()
        self.last_reward_position = self.start_pos
        self.health = self._var(vzd.GameVariable.HEALTH)
        self.kills = self._var(vzd.GameVariable.KILLCOUNT)
        self.start_kills = self.kills
        self.best_progress = 0.0
        frame = self._frame(state.screen_buffer)
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return self._observe(), {"aux": self._aux_targets(state)}

    def step(self, action):
        self.game.make_action(self.ACTIONS[int(action)], self.frame_skip)
        ended = self.game.is_episode_finished()
        timeout = bool(ended and self.game.is_episode_timeout_reached())
        dead = bool(self.game.is_player_dead())
        # The vest script exits the level, so a non-timeout ending while alive is a pickup.
        success = bool(ended and not timeout and not dead)
        position = self._reward_position()
        health = self._var(vzd.GameVariable.HEALTH)
        kills = self._var(vzd.GameVariable.KILLCOUNT)
        reward = -0.01 + (100.0 if success else 0.0) - (self.death_penalty if dead else 0.0)
        reward -= self.timeout_penalty if timeout else 0.0
        reward -= 0.05 * max(0.0, self.health - health)
        reward += kill_reward(self.kills, kills)
        if self.progress_mode == "max":
            start = np.linalg.norm(self.start_pos - self.goal_pos)
            progress = float(np.clip(1 - np.linalg.norm(position - self.goal_pos) / start, 0, 1))
            if success:
                progress = 1.0
            reward += self.progress_weight * max(0.0, progress - self.best_progress)
            self.best_progress = max(self.best_progress, progress)
        else:
            reward += progress_reward(self.last_reward_position, position, self.goal_pos,
                                      distance_scale=1500.0, weight=self.progress_weight,
                                      terminal=ended)
        self.last_reward_position, self.health, self.kills = position, health, kills
        info = {"success": success}
        if ended:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info["aux"] = np.zeros(AUX_SIZE, dtype=np.float32)
            start = np.linalg.norm(self.start_pos - self.goal_pos)
            info["episode"] = {
                "kills": kills - self.start_kills, "health": max(0.0, health), "dead": dead,
                "progress": float(np.clip(1 - np.linalg.norm(position - self.goal_pos) / start,
                                          0, 1))}
        else:
            state = self.game.get_state()
            self.frames.append(self._frame(state.screen_buffer))
            obs, info["aux"] = self._observe(), self._aux_targets(state)
        return obs, reward, ended and not timeout, timeout, info
