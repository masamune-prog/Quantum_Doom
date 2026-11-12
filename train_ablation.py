import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from doom_env import QuantumDoomEnv
from models import DoomActorCritic
from ppo_common import compute_gae, make_vector_env, ppo_update, timeout_bootstrap

def train_one_run(mode: str, seed: int = 0, total_steps: int = 32768, num_envs: int = 8, n_steps: int = 128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    envs = make_vector_env([
        lambda i=i: QuantumDoomEnv(n_qubits=8, frame_skip=4) for i in range(num_envs)
    ], asynchronous=False)
    gamma, gae_lambda = 0.99, 0.95
    agent = DoomActorCritic(n_qubits=8, n_layers=4, n_actions=3, mode=mode).to(device)
    n_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    print(f"=== Running {mode} (Seed {seed}) | Actor Params: {n_params} | Device: {device} ===")

    optimizer = optim.Adam([
        {"params": agent.actor.parameters(), "lr": 0.01 if "quantum" in mode else 0.002},
        {"params": agent.critic.parameters(), "lr": 0.003}
    ])

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
            step_reward_sum += float(np.sum(reward))
            rew_buf[step] = (torch.tensor(reward / 100.0, dtype=torch.float32, device=device)
                             + timeout_bootstrap(agent, infos, terminated, truncated, gamma, device))

            next_obs = torch.tensor(raw_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value, next_done,
                                          gamma, gae_lambda)
        ppo_update(agent, optimizer, obs_buf, act_buf, logp_buf, advantages, returns,
                   n_epochs=4, minibatch_size=128, clip_coef=0.2, ent_coef=0.005, vf_coef=0.5,
                   max_grad_norm=0.5)

        mean_step_rew = step_reward_sum / batch_size
        reward_history.append(mean_step_rew)
        if update % 4 == 0:
            print(f"Update {update:02d}/{num_updates} | Mean Step Reward: {mean_step_rew:+.2f}")

    envs.close()
    torch.save(agent.state_dict(), f"checkpoint_{mode}_seed{seed}.pt")
    return reward_history

if __name__ == "__main__":
    modes = ["quantum_entangled", "quantum_product", "classical_mlp"]
    results = {}
    for m in modes:
        results[m] = train_one_run(mode=m, seed=42, total_steps=32768)

    with open("ablation_results.json", "w") as f:
        json.dump(results, f)

    plt.figure(figsize=(8, 5))
    for m, curve in results.items():
        plt.plot(curve, label=m, linewidth=2)
    plt.xlabel("PPO Rollout Updates (x1024 frames)")
    plt.ylabel("Mean Step Reward")
    plt.title("ViZDoom Aiming & Combat: Entangled PQC vs. Unentangled vs. Classical MLP")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("ablation_curve.png", dpi=150)
    print("Saved ablation_curve.png!")
