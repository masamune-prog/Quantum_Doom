"""Named semantic perception and objective-conditioned policies with optional memory."""

import torch
from torch import nn
import torch.nn.functional as F

from frozen_perception_models import GeometryPerception


class SemanticPerception(nn.Module):
    """Shared RGB encoder; every output has the same meaning on every map.

    Objective is public context (pickup versus exit), never a map identifier.
    No route direction, global position, route progress, or teacher action is an input.
    """

    def __init__(self):
        super().__init__()
        self.encoder = GeometryPerception().encoder
        self.predictor = nn.Linear(258, 27)

    def forward(self, images, goal):
        features = self.encoder(images.float() / 255.0)
        return self.predictor(torch.cat((features, goal), -1))

    def features(self, images, goal):
        return self(images, goal).sigmoid()

    def losses(self, images, goal, target):
        prediction = self(images, goal)
        return {
            "depth": F.mse_loss(prediction[:, :8].sigmoid(), target[:, :8]),
            "enemy": F.binary_cross_entropy_with_logits(prediction[:, 8:16], target[:, 8:16],
                                                         pos_weight=prediction.new_tensor(5.0)),
            "goal": F.binary_cross_entropy_with_logits(prediction[:, 16:24], target[:, 16:24],
                                                        pos_weight=prediction.new_tensor(10.0)),
            "door": F.binary_cross_entropy_with_logits(prediction[:, 24], target[:, 24],
                                                        pos_weight=prediction.new_tensor(5.0)),
            "hud": F.mse_loss(prediction[:, 25:].sigmoid(), target[:, 25:])}

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


class GoalActor(nn.Module):
    def __init__(self, kind="gru", hidden=64):
        super().__init__()
        self.kind = kind
        self.core = (nn.GRU(45, hidden, batch_first=True) if kind == "gru" else
                     nn.Sequential(nn.Linear(45, hidden), nn.Tanh()))
        self.output = nn.Linear(hidden, 7)

    def forward(self, inputs, state=None):
        if self.kind == "gru":
            features, state = self.core(inputs, state)
        else:
            features = self.core(inputs)
        logits = self.output(features)
        # Input is [27 semantics, goal(2), available(7), cooldown(1), previous(8)].
        allowed = inputs[..., 29:36] > 0.5
        return logits.masked_fill(~allowed, -1e9), state


class TransferPolicy(nn.Module):
    def __init__(self, perception, actor):
        super().__init__()
        self.perception = perception.freeze()
        self.actor = actor

    def step(self, images, public_context, state=None):
        with torch.no_grad():
            semantic = self.perception.features(images, public_context[:, :2])
        inputs = torch.cat((semantic, public_context), -1).unsqueeze(1)
        logits, state = self.actor(inputs, state)
        return logits[:, 0], state
