"""My Way Home with stacked screen frames as the only policy observation.

The policy sees the rendered RGB screen (HUD included). Engine pose, the goal
position, the depth buffer and the labels buffer are read only for the private
training reward and for auxiliary prediction targets returned in ``info``.
"""

import multiprocessing as mp
import os
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd

from human_nav_env import HERE, HumanNavigationEnv
from reward_shaping import progress_reward

FRAME_SHAPE = (60, 80)
N_SECTORS = 8
AUX_SIZE = 2 * N_SECTORS


class PixelNavigationEnv(gym.Env):
    ACTION_NAMES = HumanNavigationEnv.ACTION_NAMES
    ACTIONS = HumanNavigationEnv.ACTIONS

    def __init__(self, frame_skip=4, frames=4):
        super().__init__()
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frames)
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(HERE, "my_way_home_quantum.cfg"))
        self.game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "my_way_home.wad"))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(True)
        self.game.set_window_visible(False)
        # Private to reward and auxiliary targets; never part of the observation.
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.clear_available_game_variables()
        for variable in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y):
            self.game.add_available_game_variable(variable)
        self.game.init()
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, shape=(3 * frames, *FRAME_SHAPE),
                                                dtype=np.uint8)
        self.goal_pos = None
        self.last_reward_position = None

    def _reward_position(self):
        return np.array([self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                         self.game.get_game_variable(vzd.GameVariable.POSITION_Y)],
                        dtype=np.float32)

    @staticmethod
    def _frame(rgb):
        small = cv2.resize(rgb, FRAME_SHAPE[::-1], interpolation=cv2.INTER_AREA)
        return np.transpose(small, (2, 0, 1))

    @staticmethod
    def _aux_targets(state):
        """Per-sector depth and goal visibility, for auxiliary losses only."""
        depth = state.depth_buffer
        labels = state.labels_buffer
        rows = slice(depth.shape[0] // 3, 2 * depth.shape[0] // 3)
        goal_ids = [l.value for l in state.labels
                    if "Armor" in l.object_name or "Vest" in l.object_name]
        width = depth.shape[1] // N_SECTORS
        near, goal = [], []
        for i in range(N_SECTORS):
            cols = slice(i * width, (i + 1) * width)
            # Raw My Way Home depth is roughly 0-45; log scaling spreads it over [0, 1].
            near.append(min(1.0, np.log1p(np.mean(depth[rows, cols])) / np.log(65.0)))
            goal.append(float(np.isin(labels[:, cols], goal_ids).any()) if goal_ids else 0.0)
        return np.asarray(near + goal, dtype=np.float32)

    def _observe(self):
        return np.concatenate(self.frames, axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        state = self.game.get_state()
        self.goal_pos = HumanNavigationEnv._goal_from_objects(state)
        self.last_reward_position = self._reward_position()
        frame = self._frame(state.screen_buffer)
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return self._observe(), {"aux": self._aux_targets(state)}

    def step(self, action):
        self.game.make_action(self.ACTIONS[int(action)], self.frame_skip)
        ended = self.game.is_episode_finished()
        timeout = bool(ended and self.game.is_episode_timeout_reached())
        # The +1 pickup can land on the tic before the episode ends, so the final
        # action's own reward may be negative; use the episode total instead.
        success = bool(ended and not timeout and self.game.get_total_reward() > 0.5)
        position = self._reward_position()
        reward = -0.01 + (100.0 if success else 0.0)
        reward += progress_reward(self.last_reward_position, position, self.goal_pos,
                                  distance_scale=1024.0, terminal=ended)
        self.last_reward_position = position
        if ended:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            aux = np.zeros(AUX_SIZE, dtype=np.float32)
        else:
            state = self.game.get_state()
            self.frames.append(self._frame(state.screen_buffer))
            obs, aux = self._observe(), self._aux_targets(state)
        return obs, reward, ended and not timeout, timeout, {"success": success, "aux": aux}

    def close(self):
        self.game.close()


def make_pixel_env(task="navigation", **kwargs):
    if task == "corridor":
        from pixel_corridor_env import PixelCorridorEnv
        return PixelCorridorEnv(**kwargs)
    if task == "e1m1":
        from pixel_e1m1_env import PixelE1M1Env
        return PixelE1M1Env(**kwargs)
    return PixelNavigationEnv(**kwargs)


def _worker(conn, task, kwargs):
    env = make_pixel_env(task, **kwargs)
    try:
        while True:
            command, arg = conn.recv()
            if command == "reset":
                conn.send(env.reset(seed=arg[0], options=arg[1]))
            elif command == "step":
                conn.send(env.step(arg))
            else:
                break
    finally:
        env.close()
        conn.close()


class EnvPool:
    """One ViZDoom process per environment; the caller chooses every reset seed."""

    def __init__(self, n, task="navigation", **kwargs):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, task, kwargs), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self.n = n

    def reset(self, ids, seeds, options=None):
        """``options`` (e.g. {"start_wp": 5} for E1M1) goes to every env in this call."""
        for i, seed in zip(ids, seeds):
            self.conns[i].send(("reset", (int(seed), options)))
        return [self.conns[i].recv() for i in ids]

    def step(self, ids, actions):
        for i, action in zip(ids, actions):
            self.conns[i].send(("step", int(action)))
        return [self.conns[i].recv() for i in ids]

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
