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
                 layers=2, aux_pos_weight=20.0, aux_size=AUX_SIZE, aux_route=0,
                 rich_critic=False):
        super().__init__()
        self.aux_pos_weight = aux_pos_weight
        self.aux_route = aux_route
        self.rich_critic = rich_critic
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 6, 256), nn.ReLU())
        self.bottleneck = nn.Linear(256, bottleneck)
        self.aux = nn.Linear(256, aux_size)
        # Optional critic on the full 256 encoder features (detached, so it cannot change the
        # policy). The 12-value bottleneck of an imitation-trained policy may lack what the
        # value needs, such as route progress and danger.
        self.critic256 = (nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
                          if rich_critic else None)
        if head == "classical":
            self.head = ClassicalNavPolicy(n_inputs=bottleneck, n_actions=n_actions)
        elif head == "linear":
            self.head = LinearNavHead(bottleneck, n_actions)
        elif head in ("quantum", "quantum_noent", "quantum_frozen"):
            # quantum_noent: CNOT ring removed. quantum_frozen: rotation angles and input scales
            # frozen at their random initial values (only the two output gains still train).
            self.head = QuantumE1M1Policy(n_qubits=bottleneck, n_layers=layers,
                                          n_actions=n_actions, entangle=head != "quantum_noent")
            if head == "quantum_frozen":
                self.head.theta.requires_grad_(False)
                self.head.input_scale.requires_grad_(False)
        else:
            raise ValueError(f"unknown head {head!r}")

    def encode(self, obs):
        features = self.encoder(obs.float() / 255.0)
        return features, math.pi * torch.tanh(self.bottleneck(features))

    def forward(self, obs):
        features, z = self.encode(obs)
        logits, value = self.head(z)
        return logits, self.value_of(features, value)

    def value_of(self, features, head_value):
        if self.critic256 is None:
            return head_value
        return self.critic256(features.detach()).squeeze(-1)

    def act(self, obs, action=None):
        features, z = self.encode(obs)
        logits, value = self.head(z)
        value = self.value_of(features, value)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value, features

    def aux_loss(self, features, targets):
        prediction = self.aux(features)
        end = prediction.shape[1] - self.aux_route
        depth = nn.functional.mse_loss(torch.sigmoid(prediction[:, :N_SECTORS]),
                                       targets[:, :N_SECTORS])
        # Visibility targets are sparse (the My Way Home vest: ~1% of sectors).
        goal = nn.functional.binary_cross_entropy_with_logits(
            prediction[:, N_SECTORS:end], targets[:, N_SECTORS:end],
            pos_weight=torch.tensor(self.aux_pos_weight, device=prediction.device))
        # Route targets (E1M1): direction to walk and progress along the route, by regression.
        route = (nn.functional.mse_loss(prediction[:, end:], targets[:, end:])
                 if self.aux_route else prediction.new_zeros(()))
        return depth, goal, route
