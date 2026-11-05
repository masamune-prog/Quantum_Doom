"""Geometry-only perception and frozen E1M1 head comparisons.

Eleven learned angles reconstruct all 20 existing privileged geometric targets.
The twelfth angle is an observable USE cooldown, computed from executed actions.
The decoder is trained only during perception pretraining and is never a policy input.
"""

import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from e1m1_quantum_model import QuantumE1M1Policy
from human_nav_models import ClassicalNavPolicy

N_ACTIONS = 7
USE = 6
COOLDOWN = 25
HEADS = ("linear", "small_mlp", "classical", "quantum2", "product2", "quantum4")


def next_cooldown(remaining, action):
    """Remaining cooldown at the next observation; only the executed action matters."""
    return COOLDOWN if action == USE else max(0, remaining - 1)


def fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GeometryPerception(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 6, 256), nn.ReLU())
        self.bottleneck = nn.Linear(256, 11)
        self.decoder = nn.Linear(12, 20)

    def forward(self, images, cooldown):
        latent = torch.tanh(self.bottleneck(self.encoder(images.float() / 255.0)))
        memory = 2 * cooldown.float().reshape(-1, 1) / COOLDOWN - 1
        return math.pi * torch.cat((latent, memory), dim=-1)

    def geometry_loss(self, images, cooldown, targets):
        prediction = self.decoder(self(images, cooldown) / math.pi)
        depth = F.mse_loss(prediction[:, :8].sigmoid(), targets[:, :8])
        visibility = F.binary_cross_entropy_with_logits(
            prediction[:, 8:17], targets[:, 8:17],
            pos_weight=prediction.new_tensor(5.0))
        direction = F.mse_loss(prediction[:, 17:19].tanh(), targets[:, 17:19])
        progress = F.mse_loss(prediction[:, 19].sigmoid(), targets[:, 19])
        return {"depth": depth, "visibility_door": visibility,
                "direction": direction, "progress": progress}

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self


class SharedMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12, hidden), nn.Tanh(),
                                     nn.Linear(hidden, N_ACTIONS + 1))

    def forward(self, angles):
        output = self.network(angles)
        return output[:, :N_ACTIONS], output[:, N_ACTIONS]


def make_head(name):
    if name == "linear":
        # SharedMLP without a hidden activation: exactly 104 actor/value parameters.
        from pixel_nav_models import LinearNavHead
        return LinearNavHead(12, N_ACTIONS)
    if name == "small_mlp":
        return SharedMLP(3)  # 71 params; 12 -> 5 -> 8 would have 113.
    if name == "classical":
        return ClassicalNavPolicy(n_inputs=12, n_actions=N_ACTIONS)
    if name not in HEADS:
        raise ValueError(f"Unknown head: {name}")
    return QuantumE1M1Policy(n_layers=4 if name == "quantum4" else 2,
                             n_actions=N_ACTIONS, entangle=name != "product2")


class FrozenPolicy(nn.Module):
    def __init__(self, perception, head):
        super().__init__()
        self.perception = perception.freeze()
        self.head = head

    def forward(self, images, cooldown):
        with torch.no_grad():
            angles = self.perception(images, cooldown)
        return self.head(angles)
