"""
Train an N-qubit Quantum RL agent on ViZDoom My Way Home (Multi-Room Labyrinth Navigation).

Action Space: 7 decoupled actions matching Deadly Corridor + USE:
  0: FORWARD, 1: ATTACK, 2: MOVE_LEFT, 3: MOVE_RIGHT, 4: TURN_LEFT, 5: TURN_RIGHT, 6: USE

State Representation (10 or 12 qubits, fair first-person vision):
  10 Qubits (Default):
    - Qubits 0-7: 8 horizontal FOV visual sectors (11.25° per sector)
    - Qubit 8: Obstacle / collision warning
    - Qubit 9: Local proximity (silent 0.0 until nearby)
  12 Qubits:
    - Qubits 0-7: 8 horizontal FOV visual sectors
    - Qubit 8: Visible target horizontal offset [-pi, pi]
    - Qubit 9: Monster / hazard threat ratio [0, pi]
    - Qubit 10: Obstacle / collision warning
    - Qubit 11: Local proximity

Architecture:
  - Actor: N-qubit, 4-layer Data Re-Uploading PQC (linear head -> 7 actions)
  - Critic: Classical 2-layer MLP (N -> 32 -> 1)
  - PPO with GAE
"""

import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from doom_nav_env import QuantumNavigationEnv
from models import DoomActorCritic


def make_nav_env(idx: int, seed: int = 0, n_qubits: int = 10):
    """Factory for multi-room maze navigation environments."""
    def _init():
        cfg_path = os.path.join(os.path.dirname(__file__), "my_way_home_quantum.cfg")
        env = QuantumNavigationEnv(
            n_qubits=n_qubits,
            frame_skip=4,
            custom_cfg_path=cfg_path,
            grid_cell_size=64.0,
        )
        return env
    return _init


def train_navigation(
    seed: int = 42,
    total_steps: int = 131072,
    num_envs: int = 16,
    n_steps: int = 128,
    n_actions: int = 7,
    n_qubits: int = 10,
    lr_actor: float = 0.008,
    lr_critic: float = 0.003,
    clip_coef: float = 0.2,
    ent_coef: float = 0.02,
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

    envs = gym.vector.AsyncVectorEnv(
        [make_nav_env(i, seed, n_qubits=n_qubits) for i in range(num_envs)]
    )

    agent = DoomActorCritic(
        n_qubits=n_qubits, n_layers=4, n_actions=n_actions,
        mode="quantum_entangled"
    ).to(device)

    n_actor_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"=== Multi-Room Maze Navigation Training (Seed {seed}) ===")
    print(f"  Qubits: {n_qubits} (Fair First-Person Vision + Proprioception)")
    print(f"  Actor params: {n_actor_params}")
    print(f"  Total params: {n_total_params}")
    print(f"  Device: {device}")
    print(f"  Actions: {n_actions} (FORWARD, ATTACK, MOVE_L, MOVE_R, TURN_L, TURN_R, USE)")
    print(f"  Total frames: {total_steps:,}")

    optimizer = optim.Adam([
        {"params": agent.actor.parameters(), "lr": lr_actor},
        {"params": agent.critic.parameters(), "lr": lr_critic},
    ])

    # Rollout buffers
    obs_buf = torch.zeros((n_steps, num_envs, n_qubits), device=device)
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
    success_history = []
    best_reward = -float("inf")

    print(f"  Batch size: {batch_size}, Updates: {num_updates}\n")

    for update in range(1, num_updates + 1):
        goals_reached = 0
        episodes_completed = 0

        # Rollout phase
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

            for r, d in zip(reward, done):
                if d:
                    episodes_completed += 1
                    if r > 50.0:
                        goals_reached += 1

            rew_buf[step] = torch.tensor(reward / 10.0, dtype=torch.float32, device=device)
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
        b_obs = obs_buf.reshape((-1, n_qubits))
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

        mean_step_rew = rew_buf.mean().item() * 10.0
        success_rate = (goals_reached / max(1, episodes_completed)) * 100.0 if episodes_completed > 0 else 0.0
        reward_history.append(mean_step_rew)
        success_history.append(success_rate)

        if mean_step_rew > best_reward:
            best_reward = mean_step_rew
            torch.save(agent.state_dict(), "checkpoint_nav_best.pt")

        if update % 2 == 0 or update == 1:
            print(
                f"Update {update:03d}/{num_updates} | "
                f"Mean R: {mean_step_rew:+6.2f} | "
                f"Best R: {best_reward:+6.2f} | "
                f"Success: {success_rate:4.1f}% | "
                f"PG: {pg_loss.item():.4f} | "
                f"VF: {v_loss.item():.4f} | "
                f"Ent: {entropy.mean().item():.4f}"
            )

    envs.close()

    torch.save(agent.state_dict(), f"checkpoint_nav_seed{seed}.pt")
    print(f"\nSaved checkpoint_nav_seed{seed}.pt and checkpoint_nav_best.pt")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(reward_history, linewidth=2, color="#3b82f6", label=f"Quantum PQC ({n_qubits}q, 4L, {n_actions} acts)")
    if len(reward_history) > 5:
        window = min(5, len(reward_history))
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(reward_history)), smoothed, "--", linewidth=2,
                 color="#ef4444", label=f"Smoothed (window={window})")
    ax1.set_ylabel("Mean Step Reward")
    ax1.set_title(f"My Way Home: {n_qubits}-Qubit Quantum Agent (7 Actions: Corridor + USE)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(success_history, linewidth=2, color="#10b981", label="Goal Success Rate (%)")
    ax2.set_xlabel("PPO Update")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("nav_training_curve.png", dpi=150)
    print("Saved nav_training_curve.png")

    with open("nav_results.json", "w") as f:
        json.dump({
            "reward_history": reward_history,
            "success_history": success_history,
            "seed": seed,
            "n_qubits": n_qubits,
            "n_actions": n_actions,
            "n_actor_params": n_actor_params,
        }, f)

    return reward_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-actions", type=int, default=7)
    parser.add_argument("--n-qubits", type=int, default=10)
    args = parser.parse_args()

    train_navigation(
        seed=args.seed,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        n_steps=args.n_steps,
        n_actions=args.n_actions,
        n_qubits=args.n_qubits,
    )
