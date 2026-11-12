"""PPO classical MLP baselines on the two quantum-policy subtasks."""

import argparse
from collections import Counter
import json
import os
import time

import numpy as np
import torch

from models import DoomActorCritic
from subtask_envs import TASKS
from ppo_common import gae_step_dones


@torch.no_grad()
def evaluate(model, task, device, episodes, seed, policy="greedy"):
    env = TASKS[task]()
    outcomes = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            actions = Counter()
            rng = np.random.default_rng(seed + ep)
            info = {}
            for step in range(1, 1601):
                if policy == "random":
                    action = int(rng.integers(env.action_space.n))
                elif policy == "forward":
                    action = 0
                elif policy == "attack":
                    action = 1
                elif policy == "forward_attack":
                    action = 6
                elif policy == "left":
                    action = 0
                elif policy == "right":
                    action = 1
                else:
                    x = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                    logits = model.actor(x)
                    action = (int(torch.distributions.Categorical(logits=logits).sample().item())
                              if policy == "sampled" else int(logits.argmax(-1).item()))
                actions[action] += 1
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                if term or trunc:
                    break
            row = {"seed": seed + ep, "steps": step, "return": round(total, 3),
                   "actions": dict(actions), **info}
            outcomes.append(row)
    finally:
        env.close()
    if task == "navigation":
        summary = {"success_rate": np.mean([r["success"] for r in outcomes]).item(),
                   "mean_steps": np.mean([r["steps"] for r in outcomes]).item()}
    elif task == "dodge":
        summary = {"mean_steps_alive": np.mean([r["steps"] for r in outcomes]).item(),
                   "mean_damage": np.mean([r["damage_taken"] for r in outcomes]).item(),
                   "survival_rate": np.mean([not r["dead"] for r in outcomes]).item()}
    else:
        summary = {"mean_kills": np.mean([r["kills"] for r in outcomes]).item(),
                   "mean_damage": np.mean([r["damage_taken"] for r in outcomes]).item(),
                   "mean_steps_alive": np.mean([r["steps"] for r in outcomes]).item(),
                   "survival_rate": np.mean([not r["dead"] for r in outcomes]).item(),
                   "kill_and_survive_rate": np.mean([r["kills"] >= 2 and not r["dead"]
                                                     for r in outcomes]).item()}
    return {"policy": policy, "task": task, "episodes": outcomes, "summary": summary}


def checkpoint_score(task, result):
    metrics = result["summary"]
    if task == "navigation":
        return metrics["success_rate"]
    if task == "dodge":
        return metrics["mean_steps_alive"] - 0.2 * metrics["mean_damage"]
    return (metrics["mean_kills"] + 0.005 * metrics["mean_steps_alive"]
            - 0.01 * metrics["mean_damage"])


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = [TASKS[args.task]() for _ in range(args.num_envs)]
    sample_env = envs[0]
    n_obs = sample_env.observation_space.shape[0]
    n_actions = sample_env.action_space.n
    model = DoomActorCritic(n_qubits=n_obs, n_actions=n_actions, mode="classical_mlp").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = os.path.join(args.output, args.task)
    os.makedirs(output, exist_ok=True)
    history = []
    best_score = -float("inf")
    updates = args.total_steps // (args.num_envs * args.steps)
    first_update = 0
    if args.resume_from:
        source = os.path.join(args.resume_from, args.task)
        saved = torch.load(os.path.join(source, "latest.pt"), map_location=device,
                           weights_only=False)
        if saved["args"]["num_envs"] != args.num_envs or saved["args"]["steps"] != args.steps:
            raise ValueError("Resume requires the same environment count and rollout length")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = list(saved["history"])
        first_update = saved["update"]
        if first_update >= updates:
            raise ValueError("New total-steps must exceed the completed training steps")
        previous_model = {k: v.clone() for k, v in model.state_dict().items()}
        previous_best = torch.load(os.path.join(source, "best.pt"), map_location=device,
                                   weights_only=False)
        model.load_state_dict(previous_best["model"])
        previous_eval = evaluate(model, args.task, device, args.eval_episodes,
                                 seed=10000, policy="greedy")
        best_score = checkpoint_score(args.task, previous_eval)
        torch.save({**previous_best, "score": best_score}, os.path.join(output, "best.pt"))
        model.load_state_dict(previous_model)
        torch.manual_seed(args.seed + first_update)
        np.random.seed(args.seed + first_update)
        print(f"resumed task={args.task} from update={first_update} "
              f"previous_best_score={best_score:.3f}", flush=True)
    obs = np.stack([env.reset(seed=args.seed + first_update * 100000 + i)[0]
                    for i, env in enumerate(envs)])
    start_time = time.monotonic()
    print(f"task={args.task} device={device} obs={n_obs} actions={n_actions} "
          f"actor_params={sum(p.numel() for p in model.actor.parameters())} "
          f"critic_params={sum(p.numel() for p in model.critic.parameters())} "
          f"updates={updates} envs={args.num_envs} steps={args.steps}", flush=True)
    try:
        for update in range(first_update + 1, updates + 1):
            ob, ac, lp, rw, dn, va = [], [], [], [], [], []
            episodes_done, successes, kills = 0, 0, 0.0
            for t in range(args.steps):
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, logp, _, value = model.get_action_and_value(x)
                actions = action.cpu().numpy()
                new_obs, rewards, dones = [], [], []
                for i, env in enumerate(envs):
                    o, reward, term, trunc, info = env.step(int(actions[i]))
                    done = term or trunc
                    episodes_done += int(done)
                    if done:
                        successes += int(info.get("success", False))
                        kills += float(info.get("kills", 0.0))
                        o, _ = env.reset(seed=args.seed + update * 100000 + t * args.num_envs + i)
                    new_obs.append(o)
                    rewards.append(reward / args.reward_scale)
                    dones.append(done)
                ob.append(x)
                ac.append(action)
                lp.append(logp)
                rw.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
                dn.append(torch.as_tensor(dones, dtype=torch.float32, device=device))
                va.append(value.flatten())
                obs = np.stack(new_obs)

            with torch.no_grad():
                next_value = model.get_value(torch.as_tensor(obs, dtype=torch.float32,
                                                             device=device)).flatten()
            rewards = torch.stack(rw)
            dones = torch.stack(dn)
            values = torch.stack(va)
            advantages, returns = gae_step_dones(rewards, values, dones, next_value,
                                                 args.gamma, args.gae_lambda)
            b_obs = torch.stack(ob).reshape(-1, n_obs)
            b_act = torch.stack(ac).reshape(-1)
            b_logp = torch.stack(lp).reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            n = len(b_act)
            for _ in range(args.epochs):
                for ids in torch.randperm(n, device=device).split(args.minibatch):
                    _, new_logp, entropy, new_value = model.get_action_and_value(
                        b_obs[ids], b_act[ids])
                    ratio = (new_logp - b_logp[ids]).exp()
                    adv = (b_adv[ids] - b_adv[ids].mean()) / (b_adv[ids].std(unbiased=False) + 1e-8)
                    policy_loss = torch.maximum(-adv * ratio,
                        -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    value_loss = 0.5 * (new_value.flatten() - b_ret[ids]).square().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            row = {"update": update, "step_reward": float(rewards.mean() * args.reward_scale),
                   "episodes": episodes_done, "successes": successes, "episode_kills": kills,
                   "seconds": round(time.monotonic() - start_time, 1)}
            history.append(row)
            if update % args.log_every == 0 or update == 1:
                print(json.dumps(row), flush=True)
            if update % args.eval_every == 0 or update == updates:
                result = evaluate(model, args.task, device, args.eval_episodes,
                                  seed=10000, policy="greedy")
                score = checkpoint_score(args.task, result)
                print(f"eval update={update} score={score:.3f} summary={result['summary']}", flush=True)
                if score > best_score:
                    best_score = score
                    torch.save({"model": model.state_dict(), "update": update,
                                "task": args.task, "obs": n_obs, "actions": n_actions,
                                "score": score},
                               os.path.join(output, "best.pt"))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "update": update, "history": history, "args": vars(args)},
                       os.path.join(output, "latest.pt"))
            if time.monotonic() - start_time > args.max_seconds:
                print("Training time budget reached", flush=True)
                break
    finally:
        for env in envs:
            env.close()

    best = torch.load(os.path.join(output, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    policies = ["greedy", "sampled", "random"]
    if args.task == "combat":
        policies.extend(["forward", "attack", "forward_attack"])
    elif args.task == "navigation":
        policies.append("forward")
    else:
        policies.extend(["left", "right"])
    results = {policy: evaluate(model, args.task, device, args.final_episodes,
                                seed=20000, policy=policy) for policy in policies}
    with open(os.path.join(output, "results.json"), "w") as f:
        json.dump({"task": args.task, "best_update": best["update"],
                   "training_steps": update * args.num_envs * args.steps,
                   "results": results}, f, indent=2)
    summaries = {policy: result["summary"] for policy, result in results.items()}
    print(f"FINAL task={args.task} best_update={best['update']} "
          f"summaries={json.dumps(summaries)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-steps", type=int, default=262144)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--final-episodes", type=int, default=20)
    p.add_argument("--max-seconds", type=int, default=9000)
    p.add_argument("--output", default="subtask_baselines")
    p.add_argument("--resume-from", default=None,
                   help="Root directory containing a prior task/latest.pt and task/best.pt")
    train(p.parse_args())
