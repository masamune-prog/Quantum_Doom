"""
Train an 8-qubit Quantum RL agent on ViZDoom Deadly Corridor.

The agent must navigate a long corridor filled with enemies shooting fireballs,
dodging and returning fire to reach a vest at the end of the corridor.

Uses 6 decoupled actions mapped to the 8-qubit Data Re-Uploading PQC:
  0: FORWARD      1: ATTACK       2: MOVE_LEFT
  3: MOVE_RIGHT   4: TURN_LEFT    5: TURN_RIGHT

Architecture:
  - Actor: 8-qubit, 4-layer data re-uploading PQC (99 + 3 extra β params for 6 actions)
  - Critic: Classical 2-layer MLP (8→32→1)
  - PPO with GAE
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from doom_env import QuantumDoomEnv
from models import DoomActorCritic


def make_corridor_env(idx: int, seed: int = 0):
    """Factory for deadly corridor environments."""
    def _init():
        cfg_path = os.path.join(os.path.dirname(__file__), "deadly_corridor_quantum.cfg")
        env = QuantumDoomEnv(
            n_qubits=8,
            frame_skip=4,
            use_composite_actions=True,
            custom_cfg_path=cfg_path,
        )
        return env
    return _init


def train_corridor(
    seed: int = 42,
    total_steps: int = 131072,  # 128K frames (~4x more than basic)
    num_envs: int = 16,         # Parallel environments matching CPU cores
    n_steps: int = 128,         # 16 envs * 128 steps = 2048 batch size
    n_actions: int = 6,
    lr_actor: float = 0.008,
    lr_critic: float = 0.003,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,     # Higher entropy for exploration in corridor
    vf_coef: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 4,
    minibatch_size: int = 256,
    max_grad_norm: float = 0.5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Create vectorized environments across multiple processes to max out all CPU cores
    envs = gym.vector.AsyncVectorEnv(
        [make_corridor_env(i, seed) for i in range(num_envs)]
    )

    # Agent with 6 actions for corridor composite action space
    agent = DoomActorCritic(
        n_qubits=8, n_layers=4, n_actions=n_actions,
        mode="quantum_entangled"
    ).to(device)

    n_actor_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"=== Deadly Corridor Training (Seed {seed}) ===")
    print(f"  Actor params: {n_actor_params}")
    print(f"  Total params: {n_total_params}")
    print(f"  Device: {device}")
    print(f"  Actions: {n_actions} composite")
    print(f"  Total frames: {total_steps:,}")

    optimizer = optim.Adam([
        {"params": agent.actor.parameters(), "lr": lr_actor},
        {"params": agent.critic.parameters(), "lr": lr_critic},
    ])

    # Rollout buffers
    obs_buf = torch.zeros((n_steps, num_envs, 8), device=device)
    act_buf = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((n_steps, num_envs), device=device)
    rew_buf = torch.zeros((n_steps, num_envs), device=device)
    done_buf = torch.zeros((n_steps, num_envs), device=device)
    val_buf = torch.zeros((n_steps, num_envs), device=device)

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    batch_size = num_envs * n_steps
    num_updates = total_steps // batch_size
    reward_history = []
    best_reward = -float("inf")

    print(f"  Batch size: {batch_size}, Updates: {num_updates}\n")

    for update in range(1, num_updates + 1):
        # Rollout phase
        episode_rewards = []
        for step in range(n_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                val_buf[step] = value.flatten()

            act_buf[step] = action
            logp_buf[step] = logprob

            raw_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Normalize rewards (death_penalty=100 is very large)
            rew_buf[step] = torch.tensor(reward / 100.0, dtype=torch.float32, device=device)
            next_obs = torch.tensor(raw_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

        # Compute GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rew_buf, device=device)
            lastgaelam = 0
            for t in reversed(range(n_steps)):
                nextnonterminal = 1.0 - (next_done if t == n_steps - 1 else done_buf[t + 1])
                nextvalues = next_value if t == n_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # PPO update phase
        b_obs = obs_buf.reshape((-1, 8))
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlogprob - b_logp[mb]).exp()

                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newval.view(-1) - b_ret[mb]) ** 2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        mean_step_rew = rew_buf.mean().item() * 100.0
        reward_history.append(mean_step_rew)

        if mean_step_rew > best_reward:
            best_reward = mean_step_rew
            torch.save(agent.state_dict(), "checkpoint_corridor_best.pt")

        if update % 2 == 0 or update == 1:
            print(
                f"Update {update:03d}/{num_updates} | "
                f"Mean R: {mean_step_rew:+7.2f} | "
                f"Best R: {best_reward:+7.2f} | "
                f"PG: {pg_loss.item():.4f} | "
                f"VF: {v_loss.item():.4f} | "
                f"Ent: {entropy.mean().item():.4f}"
            )

    envs.close()

    # Save final checkpoint
    torch.save(agent.state_dict(), f"checkpoint_corridor_seed{seed}.pt")
    print(f"\nSaved checkpoint_corridor_seed{seed}.pt")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, linewidth=2, color="#ef4444", label="Quantum Entangled PQC (8q, 4L)")
    if len(reward_history) > 5:
        window = min(5, len(reward_history))
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(reward_history)), smoothed, "--", linewidth=2,
                 color="#3b82f6", label=f"Smoothed (window={window})")
    plt.xlabel("PPO Update")
    plt.ylabel("Mean Step Reward")
    plt.title("Deadly Corridor: 8-Qubit Quantum Agent Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("corridor_training_curve.png", dpi=150)
    print("Saved corridor_training_curve.png")

    # Save raw results
    with open("corridor_results.json", "w") as f:
        json.dump({"reward_history": reward_history, "seed": seed, "n_actor_params": n_actor_params}, f)

    return reward_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    args = parser.parse_args()

    train_corridor(
        seed=args.seed,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        n_steps=args.n_steps,
    )
