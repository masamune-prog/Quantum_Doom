"""Classical reference policy with the same observation and action interface."""

import torch
from torch import nn
from torch.distributions import Categorical


class ClassicalNavPolicy(nn.Module):
    def __init__(self, n_inputs=12, n_actions=3, hidden=32):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh(),
                                   nn.Linear(hidden, n_actions))
        self.critic = nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def act(self, obs, action=None):
        logits, value = self(obs)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value
