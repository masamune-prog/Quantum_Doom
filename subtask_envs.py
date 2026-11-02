"""Task-specific observations and outcome metrics for classical/quantum comparisons."""

import os

import gymnasium as gym
import numpy as np
import vizdoom as vzd

from doom_env import QuantumDoomEnv
from doom_nav_env import QuantumNavigationEnv

HERE = os.path.dirname(os.path.abspath(__file__))


class CombatTask(gym.Env):
    """Kill enemies and survive in skill-5 Deadly Corridor.

    Observation: eight visual angles, health, and damage on the last action.
    The forward-progress reward from the original wrapper is discarded.
    """

    ACTION_NAMES = ["FORWARD", "ATTACK", "STRAFE_LEFT", "STRAFE_RIGHT",
                    "TURN_LEFT", "TURN_RIGHT", "FORWARD_ATTACK"]

    def __init__(self):
        super().__init__()
        self.base = QuantumDoomEnv(
            n_qubits=8, frame_skip=4, use_composite_actions=True,
            custom_cfg_path=os.path.join(HERE, "combat_baseline.cfg"),
        )
        self.base.actions.append([0, 0, 1, 1, 0, 0])
        self.action_space = gym.spaces.Discrete(len(self.ACTION_NAMES))
        self.observation_space = gym.spaces.Box(-np.pi, np.pi, shape=(10,), dtype=np.float32)
        self.health = 100.0
        self.kills = 0.0
        self.damage_taken = 0.0

    def _features(self, radar, damage):
        health_angle = np.pi * np.clip(2 * self.health / 100 - 1, -1, 1)
        damage_angle = np.pi * np.clip(damage / 20, 0, 1)
        return np.concatenate([radar, [health_angle, damage_angle]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        radar, _ = self.base.reset(seed=seed)
        self.health = float(self.base.game.get_game_variable(vzd.GameVariable.HEALTH))
        self.kills = float(self.base.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        self.damage_taken = 0.0
        return self._features(radar, 0.0), {}

    def step(self, action):
        old_health, old_kills = self.health, self.kills
        radar, _, ended, _, _ = self.base.step(int(action))
        self.health = float(self.base.game.get_game_variable(vzd.GameVariable.HEALTH))
        self.kills = float(self.base.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        damage = max(0.0, old_health - self.health)
        self.damage_taken += damage
        kill_delta = max(0.0, self.kills - old_kills)
        dead = bool(self.base.game.is_player_dead())
        timeout = bool(ended and self.base.game.is_episode_timeout_reached())
        # Reward only combat outcomes; no vest or x-position bonus.
        reward = 5.0 * kill_delta - 0.1 * damage - 0.01 - 10.0 * dead
        info = {"kills": self.kills, "health": self.health, "damage_taken": self.damage_taken,
                "dead": dead, "survived": ended and not dead}
        return self._features(radar, damage), float(reward), bool(ended and not timeout), timeout, info

    def close(self):
        self.base.close()


class NavigationTask(gym.Env):
    """Find and collect the vest in My Way Home using three movement actions."""

    ACTION_NAMES = ["FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    ACTION_MAP = [0, 4, 5]

    def __init__(self):
        super().__init__()
        self.base = QuantumNavigationEnv(
            n_qubits=10, frame_skip=4,
            custom_cfg_path=os.path.join(HERE, "my_way_home_quantum.cfg"),
        )
        self.action_space = gym.spaces.Discrete(len(self.ACTION_NAMES))
        self.observation_space = self.base.observation_space

    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed)

    def step(self, action):
        obs, reward, ended, _, _ = self.base.step(self.ACTION_MAP[int(action)])
        timeout = bool(ended and self.base.game.is_episode_timeout_reached())
        success = bool(ended and not timeout and reward >= 100.0)
        return obs, reward, bool(ended and not timeout), timeout, {"success": success}

    def close(self):
        self.base.close()


class DodgeTask(gym.Env):
    """Avoid projectiles in Take Cover; shooting is unavailable in this scenario."""

    ACTION_NAMES = ["MOVE_LEFT", "MOVE_RIGHT"]

    def __init__(self):
        super().__init__()
        self.base = QuantumDoomEnv(n_qubits=8, frame_skip=4, scenario="take_cover.cfg")
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-np.pi, np.pi, shape=(10,), dtype=np.float32)
        self.health = 100.0
        self.damage_taken = 0.0

    def _features(self, radar, damage):
        health_angle = np.pi * np.clip(2 * self.health / 100 - 1, -1, 1)
        damage_angle = np.pi * np.clip(damage / 20, 0, 1)
        return np.concatenate([radar, [health_angle, damage_angle]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        radar, _ = self.base.reset(seed=seed)
        self.health = float(self.base.game.get_game_variable(vzd.GameVariable.HEALTH))
        self.damage_taken = 0.0
        return self._features(radar, 0.0), {}

    def step(self, action):
        old_health = self.health
        radar, _, ended, _, _ = self.base.step(int(action))
        self.health = float(self.base.game.get_game_variable(vzd.GameVariable.HEALTH))
        damage = max(0.0, old_health - self.health)
        self.damage_taken += damage
        dead = bool(self.base.game.is_player_dead())
        timeout = bool(ended and self.base.game.is_episode_timeout_reached())
        reward = 0.05 - 0.1 * damage - 10.0 * dead
        info = {"health": self.health, "damage_taken": self.damage_taken, "dead": dead}
        return self._features(radar, damage), float(reward), bool(ended and not timeout), timeout, info

    def close(self):
        self.base.close()


TASKS = {"combat": CombatTask, "navigation": NavigationTask, "dodge": DodgeTask}
