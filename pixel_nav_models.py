"""Shared CNN screen encoder with matched classical and circuit policy heads.

Both heads read the same 12-value bottleneck in [-pi, pi]. The auxiliary head
predicts per-sector depth and goal visibility from the encoder features; it is
used only as a training loss and never affects action selection.
"""

import math

import torch
from torch import nn
from torch.distributions import Categorical

from e1m1_quantum_model import QuantumE1M1Policy
from human_nav_models import ClassicalNavPolicy
from pixel_nav_env import AUX_SIZE, N_SECTORS


class LinearNavHead(nn.Module):
    """Control for the circuit head: one affine map from the bottleneck (52 parameters)."""

    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.linear = nn.Linear(n_inputs, n_actions + 1)

    def forward(self, z):
        out = self.linear(z)
        return out[..., :self.n_actions], out[..., self.n_actions]


class PixelNavPolicy(nn.Module):
    def __init__(self, head="classical", in_channels=12, n_actions=3, bottleneck=12,
                 layers=2, aux_pos_weight=20.0):
        super().__init__()
        self.aux_pos_weight = aux_pos_weight
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 6, 256), nn.ReLU())
        self.bottleneck = nn.Linear(256, bottleneck)
        self.aux = nn.Linear(256, AUX_SIZE)
        if head == "classical":
            self.head = ClassicalNavPolicy(n_inputs=bottleneck, n_actions=n_actions)
        elif head == "linear":
            self.head = LinearNavHead(bottleneck, n_actions)
        else:
            self.head = QuantumE1M1Policy(n_qubits=bottleneck, n_layers=layers,
                                          n_actions=n_actions)

    def encode(self, obs):
        features = self.encoder(obs.float() / 255.0)
        return features, math.pi * torch.tanh(self.bottleneck(features))

    def forward(self, obs):
        return self.head(self.encode(obs)[1])

    def act(self, obs, action=None):
        features, z = self.encode(obs)
        logits, value = self.head(z)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value, features

    def aux_loss(self, features, targets):
        prediction = self.aux(features)
        depth = nn.functional.mse_loss(torch.sigmoid(prediction[:, :N_SECTORS]),
                                       targets[:, :N_SECTORS])
        # Visibility targets are sparse (the My Way Home vest: ~1% of sectors).
        goal = nn.functional.binary_cross_entropy_with_logits(
            prediction[:, N_SECTORS:], targets[:, N_SECTORS:],
            pos_weight=torch.tensor(self.aux_pos_weight, device=prediction.device))
        return depth, goal
