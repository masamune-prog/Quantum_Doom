import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from doom_env import QuantumDoomEnv
from models import DoomActorCritic

def train_one_run(mode: str, seed: int = 0, total_steps: int = 32768, num_envs: int = 8, n_steps: int = 128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    envs = gym.vector.SyncVectorEnv([
        lambda i=i: QuantumDoomEnv(n_qubits=8, frame_skip=4) for i in range(num_envs)
    ])
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
                delta = rew_buf[t] + 0.99 * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # Flatten batch: (num_envs * n_steps, ...)
        b_obs = obs_buf.reshape((-1, 8))
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for _ in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, 128):
                mb = b_inds[start:start + 128]
                _, newlogprob, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlogprob - b_logp[mb]).exp()

                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newval.view(-1) - b_ret[mb]) ** 2).mean()
                loss = pg_loss - 0.005 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        mean_step_rew = (rew_buf.mean().item()) * 100.0
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
