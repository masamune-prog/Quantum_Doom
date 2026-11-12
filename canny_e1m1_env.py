"""E1M1 whose observation is only the 8 hand-made screen features of the current frame.

No CNN and no frame stack: each step returns HumanNavigationEnv's eight sector values
(brightness + Canny edge density mapped to [-1, 1], overridden by a green-pixel cue),
times pi, computed from the current RGB screen only. Indices 8-11 are zero so the vector
fits the 12-qubit circuit unchanged. Rewards, teacher labels, starts and episode info come
from PixelE1M1Env (private state is used only for the teacher and for rewards).

The game renders at 160x120 (the teacher's on-screen thresholds assume it). The original
features crop rows [:165] of a 320x240 screen and resize to 80x40; here the same fraction
of the screen (rows [:83]) is resized to the same 80x40 view.
"""

import cv2
import gymnasium as gym
import numpy as np

from human_nav_env import HumanNavigationEnv
from pixel_e1m1_env import PixelE1M1Env

N_FEATURES = 12
VISUAL = 8

# One env per process: OpenCV otherwise starts a thread per core in every worker and
# a few dozen workers oversubscribe the machine (load average > 300 with 96 workers).
cv2.setNumThreads(1)


def screen_features(rgb):
    crop = int(round(rgb.shape[0] * 165 / 240))
    view = cv2.resize(rgb[:crop], (80, 40), interpolation=cv2.INTER_AREA)
    features = np.zeros(N_FEATURES, dtype=np.float32)
    features[:VISUAL] = HumanNavigationEnv._view_features(view)
    return features


class CannyE1M1Env(PixelE1M1Env):
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
