"""Deadly Corridor whose observation is only the current screen's 8 hand-made features.

Same features as canny_e1m1_env.py: per screen sector, brightness + Canny edge density
mapped to [-1, 1], overridden by a green-pixel cue (the vest is green armour), times pi;
padded with zeros to 12. No CNN, frame stack or memory. Rewards, success and episode
statistics are PixelCorridorEnv's, so results compare directly with pixel_corridor_runs/.
"""

import os

import gymnasium as gym
import numpy as np

from canny_e1m1_env import N_FEATURES, screen_features
from pixel_corridor_env import PixelCorridorEnv


class CannyCorridorEnv(PixelCorridorEnv):
    def __init__(self, **kwargs):
        kwargs["frames"] = 1
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(-np.pi, np.pi, shape=(N_FEATURES,),
                                                dtype=np.float32)
        self._rgb = None

    def _frame(self, rgb):
        self._rgb = rgb
        return np.zeros(1, dtype=np.uint8)  # the frame stack is unused

    def _observe(self):
        return screen_features(self._rgb)


# Enemy-colour mask: a 15-bit RGB lookup table (32 levels per channel). An entry is on when
# more than half of the training pixels with that colour belonged to a living enemy
# (Zombieman, ShotgunGuy, ChaingunGuy) in the labels buffer. The labels buffer was used
# once, offline, to build the table (2,380 frames from seeds 1000-1799, rows above the
# HUD); at run time the mask reads only the screen colours. Held out on 2 of 8 collection
# batches: pixel precision 0.77, recall 0.42; per-sector enemy fraction correlates 0.88
# with the true fraction.
ENEMY_LUT = np.unpackbits(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "corridor_enemy_lut.npy"))).astype(bool)
N_ENEMY_FEATURES = 16
ENEMY_FULL = 0.1  # a sector whose crop is 10% enemy pixels reads as fully occupied


def enemy_mask(rgb):
    crop = int(round(rgb.shape[0] * 165 / 240))
    q = rgb[:crop].astype(np.int32) >> 3
    return ENEMY_LUT[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]]


def enemy_features(rgb):
    """The 8 Canny/green sector values followed by 8 enemy-mask densities, all in [-pi, pi]."""
    mask = enemy_mask(rgb)
    width = mask.shape[1] // 8
    density = mask[:, :8 * width].reshape(mask.shape[0], 8, width).mean(axis=(0, 2))
    features = np.empty(N_ENEMY_FEATURES, dtype=np.float32)
    features[:8] = screen_features(rgb)[:8]
    features[8:] = np.pi * (2 * np.minimum(1.0, density / ENEMY_FULL) - 1)
    return features


class CannyEnemyCorridorEnv(CannyCorridorEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(-np.pi, np.pi, shape=(N_ENEMY_FEATURES,),
                                                dtype=np.float32)

    def _observe(self):
        return enemy_features(self._rgb)
