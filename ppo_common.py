"""PPO pieces shared by the gymnasium-vector trainers (navigation, corridor, ablation).

- Vector envs auto-reset on the SAME step, so the returned observation is already the
  next episode's first frame and the ended episode's last frame is in info["final_obs"].
- Time limits (truncated, not terminated) are bootstrapped: gamma * V(final_obs) is added
  to that step's reward before the episode boundary cuts the GAE recursion.
- Actor and critic gradients are clipped separately, so critic gradients cannot crowd out
  the actor's share of a joint norm.
"""

from collections import deque

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import numpy as np
import torch
from torch import nn


def make_vector_env(env_fns, asynchronous=True):
    cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)


def final_info(infos, key, n_envs, default=0.0):
    """Per-env value of `key` from the info of episodes that ended on this step."""
    values = np.full(n_envs, default, dtype=np.float64)
    final = infos.get("final_info", {})
    if key in final:
        mask = final.get(f"_{key}", np.ones(n_envs, bool))
        values[mask] = np.asarray(final[key], dtype=np.float64)[mask]
    return values


@torch.no_grad()
def timeout_bootstrap(agent, infos, terminated, truncated, gamma, device):
    """gamma * V(final_obs) for envs cut off by a time limit, zero elsewhere."""
    bonus = torch.zeros(len(truncated), device=device)
    cut = np.flatnonzero(np.logical_and(truncated, np.logical_not(terminated)))
    if len(cut):
        final_obs = np.stack([infos["final_obs"][i] for i in cut])
        values = agent.get_value(torch.as_tensor(final_obs, dtype=torch.float32, device=device))
        bonus[torch.as_tensor(cut, device=device)] = gamma * values.flatten()
    return bonus


@torch.no_grad()
def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    """dones[t] marks that obs[t] starts a new episode (the previous step ended one)."""
    n_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(n_steps)):
        nextnonterminal = 1.0 - (next_done if t == n_steps - 1 else dones[t + 1])
        nextvalues = next_value if t == n_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages, advantages + values


@torch.no_grad()
def gae_step_dones(rewards, values, dones, next_value, gamma, gae_lambda):
    """GAE where dones[t] marks that step t itself ended an episode (EnvPool trainers)."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t]
        following = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * following * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


def ppo_update(agent, optimizer, obs, actions, logprobs, advantages, returns, *, n_epochs,
               minibatch_size, clip_coef, ent_coef, vf_coef, max_grad_norm):
    obs_dim = obs.shape[2:]
    b_obs, b_act = obs.reshape((-1, *obs_dim)), actions.reshape(-1)
    b_logp, b_adv, b_ret = logprobs.reshape(-1), advantages.reshape(-1), returns.reshape(-1)
    inds = np.arange(len(b_act))
    for _ in range(n_epochs):
        np.random.shuffle(inds)
        for start in range(0, len(inds), minibatch_size):
            mb = inds[start:start + minibatch_size]
            _, newlogprob, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
            ratio = (newlogprob - b_logp[mb]).exp()
            mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
            pg_loss = torch.max(-mb_adv * ratio,
                                -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
            v_loss = 0.5 * ((newval.view(-1) - b_ret[mb]) ** 2).mean()
            loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(agent.critic.parameters(), max_grad_norm)
            optimizer.step()
    return {"pg_loss": pg_loss.item(), "v_loss": v_loss.item(),
            "entropy": entropy.mean().item()}


class EpisodeTracker:
    """Undiscounted per-episode returns and successes over a rolling window of episodes."""

    def __init__(self, n_envs, window=100):
        self.running = np.zeros(n_envs)
        self.returns, self.successes = deque(maxlen=window), deque(maxlen=window)

    def step(self, rewards, done, success=None):
        self.running += rewards
        for i in np.flatnonzero(done):
            self.returns.append(self.running[i])
            self.successes.append(float(success[i]) if success is not None else 0.0)
            self.running[i] = 0.0

    def mean_return(self):
        return float(np.mean(self.returns)) if self.returns else float("nan")

    def success_rate(self):
        return float(np.mean(self.successes)) if self.successes else float("nan")
