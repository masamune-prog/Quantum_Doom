"""Shared semantic targets, public objective/action context, and unseen-map evaluation.

Private geometry is used exclusively for labels and episode diagnostics. Policies see
RGB, their objective, available actions, and their own executed-action history.
E1M2 is reserved for final evaluation; the all-keys variant controls for a new mechanic.
"""

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vzd

from e1m1_map import EXIT_SPECIALS, DoomMap, WAD_PATH
from pixel_corridor_env import PixelCorridorEnv
from pixel_e1m1_env import PixelE1M1Env
from pixel_nav_env import PixelNavigationEnv

TRAIN_MAPS = ("navigation", "corridor", "e1m1")
TEST_MAPS = ("e1m2_keys", "e1m2")
ACTION_MAPS = {"navigation": (0, 4, 5), "corridor": tuple(range(6)),
               "e1m1": tuple(range(7)), "e1m2": tuple(range(7)),
               "e1m2_keys": tuple(range(7))}
N_TARGETS = 27  # 8 depth, 8 enemy, 8 goal, door/use ray, health, ammunition


def context(task, previous=7, cooldown=0):
    goal = [float(task in TRAIN_MAPS[:2]), float(task not in TRAIN_MAPS[:2])]
    available = [float(a in ACTION_MAPS[task]) for a in range(7)]
    history = np.eye(8, dtype=np.float32)[previous]
    return np.concatenate((goal, available, [cooldown / 25.0], history)).astype(np.float32)


def ray_hits(position, angles, lines):
    """Nearest intersecting line per ray. All computations stay private to targets."""
    if not lines:
        return np.full(len(angles), -1), np.full(len(angles), np.inf)
    a = np.array([line["p1"] for line in lines])
    edge = np.array([line["p2"] - line["p1"] for line in lines])
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    delta = a - position
    denom = direction[:, 0, None] * edge[None, :, 1] - direction[:, 1, None] * edge[None, :, 0]
    safe = np.where(np.abs(denom) > 1e-8, denom, 1.0)
    distance = (delta[:, 0] * edge[:, 1] - delta[:, 1] * edge[:, 0])[None, :] / safe
    along = (delta[None, :, 0] * direction[:, 1, None] -
             delta[None, :, 1] * direction[:, 0, None]) / safe
    valid = (np.abs(denom) > 1e-8) & (distance >= 0) & (along >= 0) & (along <= 1)
    distances = np.where(valid, distance, np.inf)
    index = distances.argmin(1)
    nearest = distances[np.arange(len(angles)), index]
    return np.where(np.isfinite(nearest), index, -1), nearest


class UnseenExitEnv(PixelNavigationEnv):
    ACTIONS = PixelE1M1Env.ACTIONS

    def __init__(self, keys=False):
        gym.Env.__init__(self)
        self.keys, self.frame_skip = keys, 4
        self.frames = deque(maxlen=4)
        self.game = vzd.DoomGame()
        self.game.load_config(str(Path(__file__).parent / "pixel_e1m1.cfg"))
        self.game.set_doom_game_path(WAD_PATH)
        self.game.set_doom_map("E1M2")
        self.game.set_doom_skill(3)
        self.game.set_window_visible(False)
        self.game.add_game_args("+sv_cheats 1")
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.init()
        self.observation_space = gym.spaces.Box(0, 255, (12, 60, 80), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(7)
        geometry = DoomMap(WAD_PATH, "E1M2")
        line = geometry.exit_lines()[0]
        self.goal = (line["p1"] + line["p2"]) / 2

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.game.set_seed(int(seed or 0))
        self.game.new_episode()
        before_keys = self.game.get_state().screen_buffer.copy()
        if self.keys:
            self.game.send_game_command("give keys")
            self.game.advance_action(1)
        state = self.game.get_state()
        h, w = state.screen_buffer.shape[:2]
        key_region = (slice(int(.85 * h), h), slice(int(.70 * w), int(.85 * w)))
        self.key_hud_changed = bool(np.any(before_keys[key_region] != state.screen_buffer[key_region]))
        frame = self._frame(state.screen_buffer)
        self.frames.clear()
        self.frames.extend([frame] * 4)
        self.start_distance = np.linalg.norm(self._reward_position() - self.goal)
        self.best_progress = 0.0
        return self._observe(), {"keys_granted": self.keys, "key_hud_changed": self.key_hud_changed}

    def step(self, action):
        self.game.make_action(self.ACTIONS[int(action)], 4)
        ended = self.game.is_episode_finished()
        timeout = ended and self.game.is_episode_timeout_reached()
        dead = self.game.is_player_dead()
        self.best_progress = max(self.best_progress, float(np.clip(
            1 - np.linalg.norm(self._reward_position() - self.goal) /
            max(self.start_distance, 1), 0, 1)))
        info = {"success": bool(ended and not timeout and not dead)}
        if ended:
            info["episode"] = {"dead": bool(dead), "progress": self.best_progress}
            return np.zeros((12, 60, 80), np.uint8), 0.0, bool(not timeout), bool(timeout), info
        self.frames.append(self._frame(self.game.get_state().screen_buffer))
        return self._observe(), 0.0, False, False, info


class TransferEnv:
    """Canonical seven-action interface, common semantic labels, and optional teacher."""

    def __init__(self, task, teacher=False):
        self.task = task
        if task == "navigation":
            self.env = PixelNavigationEnv()
        elif task == "corridor":
            self.env = PixelCorridorEnv()
        elif task == "e1m1":
            self.env = PixelE1M1Env(teacher=teacher)
        elif task in TEST_MAPS:
            if teacher:
                raise ValueError("Held-out map cannot supply training labels")
            self.env = UnseenExitEnv(keys=task.endswith("_keys"))
        else:
            raise ValueError(task)
        self.geometry = None
        if task not in TRAIN_MAPS[:2]:
            self.geometry = DoomMap(WAD_PATH, "E1M1" if task == "e1m1" else "E1M2")
            self.walls = [line for line in self.geometry.lines if not line["two_sided"]]
            self.interactions = self.geometry.door_lines() + self.geometry.exit_lines()

    def targets(self):
        state = self.env.game.get_state()
        depth = PixelNavigationEnv._aux_targets(state)[:8]
        enemies = PixelCorridorEnv._aux_targets(state)[8:]
        door = 0.0
        if self.geometry is None:
            goal = PixelNavigationEnv._aux_targets(state)[8:]
        else:
            position = self.env._reward_position()
            angle = np.deg2rad(self.env.game.get_game_variable(vzd.GameVariable.ANGLE))
            # Screen columns go from left (+angle) to right (-angle).
            indices, _ = ray_hits(position, angle + np.deg2rad(np.linspace(39.375, -39.375, 8)),
                                   self.walls)
            goal = np.array([float(i >= 0 and self.walls[i]["special"] in EXIT_SPECIALS)
                             for i in indices])
            _, distances = ray_hits(position, angle + np.deg2rad([-8, 0, 8]), self.interactions)
            door = float((distances <= 72).any())
        health = self.env.game.get_game_variable(vzd.GameVariable.HEALTH)
        ammo = self.env.game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        return np.concatenate((depth, enemies, goal, [door, np.clip(health / 200, 0, 1),
                                                       np.clip(ammo / 200, 0, 1)])).astype(np.float32)

    def reset(self, seed=None, options=None):
        image, info = self.env.reset(seed=seed, options=options)
        self.last_executed_use = -10_000
        if self.task == "e1m1" and self.env.teacher is not None:
            self.env.teacher.last_use = self.last_executed_use
        info["semantic"] = self.targets()
        return image, info

    def step(self, action):
        allowed = ACTION_MAPS[self.task]
        if int(action) not in allowed:
            raise ValueError(f"Action {action} is unavailable in {self.task}")
        if self.task == "e1m1" and self.env.teacher is not None:
            if action == 6:
                self.last_executed_use = self.env.teacher.step
            self.env.teacher.last_use = self.last_executed_use
        image, reward, term, trunc, info = self.env.step(allowed.index(int(action)))
        if self.task == "e1m1" and self.env.teacher is not None:
            self.env.teacher.last_use = self.last_executed_use
        info["semantic"] = np.zeros(N_TARGETS, np.float32) if term or trunc else self.targets()
        return image, reward, term, trunc, info

    def close(self):
        self.env.close()


def worker(conn, task, teacher):
    env = TransferEnv(task, teacher)
    try:
        while True:
            command, argument = conn.recv()
            if command == "reset":
                conn.send(env.reset(seed=argument[0], options=argument[1]))
            elif command == "step":
                conn.send(env.step(argument))
            else:
                break
    finally:
        env.close()
        conn.close()


class TransferPool:
    def __init__(self, n, task, teacher=False):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.n, self.conns, self.procs = n, [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, task, teacher), daemon=True)
            process.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(process)

    def reset(self, ids, seeds, options=None):
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
        for process in self.procs:
            process.join(timeout=10)
