"""Matched PPO navigation runs: RGB-only quantum actor/critic, with or without map input."""

import argparse
from collections import Counter
import json
import os
import time

import numpy as np
import torch

from e1m1_quantum_model import QuantumE1M1Policy
from human_nav_env import HumanNavigationEnv
from human_e1m1_env import HumanE1M1Env
from human_nav_models import ClassicalNavPolicy

ENVS = {"navigation": HumanNavigationEnv, "e1m1": HumanE1M1Env}


def make_env(task, memory, memory_kind):
    if task == "navigation":
        return HumanNavigationEnv(memory=memory, memory_kind=memory_kind)
    return HumanE1M1Env(memory=memory)


@torch.no_grad()
def evaluate(model, device, task, memory, memory_kind, episodes, seed, sampled=False):
    env = make_env(task, memory, memory_kind)
    rows = []
    try:
        for ep in range(episodes):
            if sampled:
                torch.manual_seed(seed + ep)
            obs, _ = env.reset(seed=seed + ep)
            counts = Counter()
            info = {"success": False}
            for step in range(1, 1601 if task == "e1m1" else 526):
                logits, _ = model(torch.as_tensor(obs[None], dtype=torch.float32, device=device))
                if sampled:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
                else:
                    action = int(logits.argmax(-1).item())
                counts[action] += 1
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            rows.append({"seed": seed + ep, "success": bool(info["success"]),
                         "steps": step, "actions": dict(counts),
                         "visited_cells": len(env.travel_map.visits)})
    finally:
        env.close()
    return {"successes": sum(r["success"] for r in rows), "episodes": len(rows),
            "mean_steps": float(np.mean([r["steps"] for r in rows])), "rows": rows}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    memory = args.memory == "on"
    envs = [make_env(args.task, memory, args.memory_kind) for _ in range(args.num_envs)]
    if args.model == "classical_mlp":
        model = ClassicalNavPolicy(n_inputs=12, n_actions=envs[0].action_space.n).to(device)
    else:
        model = QuantumE1M1Policy(n_qubits=12, n_layers=args.layers,
                                  n_actions=envs[0].action_space.n,
                                  memory_bridges=args.model == "quantum_bridged").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.output, exist_ok=True)
    best_score = -1.0
    history = []
    obs = np.stack([e.reset(seed=args.seed + i)[0] for i, e in enumerate(envs)])
    start = time.monotonic()
    print(f"task={args.task} memory={args.memory} memory_kind={args.memory_kind} "
          f"model={args.model} device={device} envs={args.num_envs} "
          f"steps={args.steps} updates={args.updates} parameters={sum(p.numel() for p in model.parameters())}",
          flush=True)
    completed = 0
    try:
        for update in range(1, args.updates + 1):
            ob, ac, lp, rw, dn, va = [], [], [], [], [], []
            episodes_done = successes = 0
            for t in range(args.steps):
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, logp, _, value = model.act(x)
                actions = action.cpu().numpy()
                new_obs, rewards, dones = [], [], []
                for i, env in enumerate(envs):
                    o, reward, term, trunc, info = env.step(int(actions[i]))
                    done = term or trunc
                    episodes_done += int(done)
                    successes += int(info["success"])
                    if done:
                        o, _ = env.reset(seed=args.seed + update * 100000 + t * args.num_envs + i)
                    new_obs.append(o)
                    rewards.append(reward / args.reward_scale)
                    dones.append(done)
                ob.append(x)
                ac.append(action)
                lp.append(logp)
                rw.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
                dn.append(torch.as_tensor(dones, dtype=torch.float32, device=device))
                va.append(value)
                obs = np.stack(new_obs)
            with torch.no_grad():
                _, next_value = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
            rewards, dones, values = torch.stack(rw), torch.stack(dn), torch.stack(va)
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.steps)):
                nonterminal = 1.0 - dones[t]
                following = next_value if t == args.steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * following * nonterminal - values[t]
                gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
                advantages[t] = gae
            returns = advantages + values
            b_obs = torch.stack(ob).reshape(-1, 12)
            b_act = torch.stack(ac).reshape(-1)
            b_logp = torch.stack(lp).reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            n = len(b_act)
            for _ in range(args.epochs):
                for ids in torch.randperm(n, device=device).split(args.minibatch):
                    _, new_logp, entropy, new_value = model.act(b_obs[ids], b_act[ids])
                    ratio = (new_logp - b_logp[ids]).exp()
                    adv = (b_adv[ids] - b_adv[ids].mean()) / (b_adv[ids].std(unbiased=False) + 1e-8)
                    policy_loss = torch.maximum(-adv * ratio,
                        -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    value_loss = 0.5 * (new_value - b_ret[ids]).square().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            completed = update
            row = {"update": update, "episodes": episodes_done, "successes": successes,
                   "reward": float(rewards.mean().item() * args.reward_scale),
                   "seconds": round(time.monotonic() - start, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "update": update, "history": history, "args": vars(args)},
                       os.path.join(args.output, "latest.pt"))
            if update % args.eval_every == 0 or update == args.updates:
                result = evaluate(model, device, args.task, memory, args.memory_kind,
                                  args.eval_episodes, 10000)
                score = result["successes"] / result["episodes"]
                print(f"eval memory={args.memory} update={update} score={score:.3f}", flush=True)
                if score >= best_score:
                    best_score = score
                    torch.save({"model": model.state_dict(), "update": update, "score": score},
                               os.path.join(args.output, "best.pt"))
            if time.monotonic() - start > args.max_seconds:
                print("time budget reached", flush=True)
                break
    finally:
        for env in envs:
            env.close()
    best_path = os.path.join(args.output, "best.pt")
    if not os.path.exists(best_path):
        torch.save({"model": model.state_dict(), "update": completed, "score": None}, best_path)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final = evaluate(model, device, args.task, memory, args.memory_kind,
                     args.final_episodes, 20000)
    final_sampled = evaluate(model, device, args.task, memory, args.memory_kind, args.final_episodes,
                             20000, sampled=True)
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump({"task": args.task, "memory": args.memory, "memory_kind": args.memory_kind,
                   "model_type": args.model, "completed_updates": completed,
                   "training_decisions": completed * args.num_envs * args.steps,
                   "best_update": best["update"], "best_selection_score": best["score"],
                   "evaluation": final, "sampled_evaluation": final_sampled}, f, indent=2)
    print(f"FINAL memory={args.memory} success={final['successes']}/{final['episodes']} "
          f"sampled={final_sampled['successes']}/{final_sampled['episodes']} "
          f"best_update={best['update']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=tuple(ENVS), default="navigation")
    p.add_argument("--memory", choices=("on", "off"), required=True)
    p.add_argument("--memory-kind", choices=("travel", "place"), default="travel")
    p.add_argument("--model", choices=("quantum_base", "quantum_bridged", "classical_mlp"),
                   default="quantum_base")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--final-episodes", type=int, default=20)
    p.add_argument("--max-seconds", type=int, default=9000)
    p.add_argument("--output", required=True)
    train(p.parse_args())
