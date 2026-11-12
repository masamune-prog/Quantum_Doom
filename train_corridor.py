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
from ppo_common import (EpisodeTracker, compute_gae, final_info, make_vector_env,
                        ppo_update, timeout_bootstrap)


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
    envs = make_vector_env(
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
    success_history = []
    tracker = EpisodeTracker(num_envs)
    best_score = (-float("inf"), -float("inf"))

    print(f"  Batch size: {batch_size}, Updates: {num_updates}\n")

    for update in range(1, num_updates + 1):
        step_reward_sum = 0.0
        for step in range(n_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logprob

            raw_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            tracker.step(reward, done)
            step_reward_sum += float(np.sum(reward))

            rew_buf[step] = (torch.tensor(reward / 100.0, dtype=torch.float32, device=device)
                             + timeout_bootstrap(agent, infos, terminated, truncated, gamma, device))
            next_obs = torch.tensor(raw_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value, next_done,
                                          gamma, gae_lambda)
        losses = ppo_update(agent, optimizer, obs_buf, act_buf, logp_buf, advantages, returns,
                            n_epochs=n_epochs, minibatch_size=minibatch_size,
                            clip_coef=clip_coef, ent_coef=ent_coef, vf_coef=vf_coef,
                            max_grad_norm=max_grad_norm)

        mean_step_rew = step_reward_sum / batch_size
        success_rate = 100.0 * tracker.success_rate()
        reward_history.append(mean_step_rew)
        success_history.append(success_rate)

        # Best = highest mean return over the last 100 completed episodes.
        if tracker.returns:
            score = (tracker.mean_return(), 0.0)
            if score > best_score:
                best_score = score
                torch.save(agent.state_dict(), "checkpoint_corridor_best.pt")

        if update % 2 == 0 or update == 1:
            print(
                f"Update {update:03d}/{num_updates} | "
                f"Mean step R: {mean_step_rew:+7.2f} | "
                f"Episode R (last 100): {tracker.mean_return():+8.2f} | "
                f"PG: {losses['pg_loss']:.4f} | "
                f"VF: {losses['v_loss']:.4f} | "
                f"Ent: {losses['entropy']:.4f}"
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
