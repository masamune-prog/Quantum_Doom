"""PPO with reverse-start curriculum for the circuit-only E1M1 agent."""

import argparse
from collections import deque
import json
import os
import time

import numpy as np
import torch

from e1m1_env import QuantumE1M1Env
from e1m1_quantum_model import QuantumE1M1Policy


def reset_at(env, seed, start_wp):
    obs, _ = env.reset(seed=seed)
    if start_wp:
        target = env.waypoints[start_wp]
        obs = env.warp(*target)
        pos, _ = env._player()
        if np.linalg.norm(pos - target) > 96:
            raise RuntimeError(f"Curriculum warp failed: requested {target}, got {pos}")
    return obs


@torch.no_grad()
def evaluate(model, env, device, episodes=20, max_steps=1600, seed=10000):
    results = []
    for ep in range(episodes):
        obs = reset_at(env, seed + ep, 0)
        total = 0.0
        for step in range(max_steps):
            logits, _ = model(torch.as_tensor(obs[None], device=device))
            obs, reward, term, trunc, info = env.step(int(logits.argmax(-1).item()))
            total += reward
            if term or trunc:
                break
        results.append({"exit_reached": bool(info["exit_reached"]), "dead": bool(info["dead"]),
                        "waypoint_idx": int(info["waypoint_idx"]), "health": float(info["health"]),
                        "steps": step + 1, "return": total})
    return results


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PennyLane default.qubit creates its initial state using Torch's default
    # device; it must match the CUDA inputs and trainable gate parameters.
    torch.set_default_device(device)
    envs = [QuantumE1M1Env(skill=args.skill, cheats=True) for _ in range(args.num_envs)]
    eval_env = QuantumE1M1Env(skill=args.skill)
    model = QuantumE1M1Policy(n_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.output, exist_ok=True)
    history = []
    first_update = 0
    stage_index = 0
    stage_updates = 0
    stage_outcomes = deque(maxlen=args.stage_window)
    checkpoint = os.path.join(args.output, "latest.pt")
    if args.resume and os.path.exists(checkpoint):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        first_update = saved["update"]
        if args.adaptive:
            stage_index = saved["stage_index"]
            stage_updates = saved["stage_updates"]
            stage_outcomes.extend(saved["stage_outcomes"])

    def stage_for(update):
        return args.starts[min((update - 1) * len(args.starts) // args.updates,
                               len(args.starts) - 1)]

    current_stage = args.starts[stage_index] if args.adaptive else stage_for(first_update + 1)
    obs = np.stack([reset_at(env, args.seed + i, current_stage) for i, env in enumerate(envs)])
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters())} "
          f"updates={args.updates} envs={args.num_envs} steps={args.steps} "
          f"curriculum={args.starts} adaptive={args.adaptive}", flush=True)
    start_time = time.monotonic()
    try:
        for update in range(first_update + 1, args.updates + 1):
            stage = args.starts[stage_index] if args.adaptive else stage_for(update)
            if stage != current_stage:
                current_stage = stage
                obs = np.stack([reset_at(env, args.seed + update * 1000 + i, stage)
                                for i, env in enumerate(envs)])
            ob, ac, lp, rw, dn, va = [], [], [], [], [], []
            exits = 0
            episode_count = 0
            max_wp = 0
            for t in range(args.steps):
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, logp, _, value = model.act(x)
                actions = action.cpu().numpy()
                new_obs, rewards, dones = [], [], []
                for i, env in enumerate(envs):
                    o, reward, term, trunc, info = env.step(int(actions[i]))
                    done = term or trunc
                    exits += int(info["exit_reached"])
                    max_wp = max(max_wp, info["waypoint_idx"])
                    episode_count += int(done)
                    if done and args.adaptive:
                        stage_outcomes.append(bool(info["exit_reached"]))
                    rewards.append(reward / args.reward_scale)
                    dones.append(done)
                    if done:
                        o = reset_at(env, args.seed + update * 100000 + t * args.num_envs + i, stage)
                    new_obs.append(o)
                ob.append(x)
                ac.append(action)
                lp.append(logp)
                rw.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
                dn.append(torch.as_tensor(dones, dtype=torch.float32, device=device))
                va.append(value)
                obs = np.stack(new_obs)

            with torch.no_grad():
                _, next_value = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
            rewards = torch.stack(rw)
            dones = torch.stack(dn)
            values = torch.stack(va)
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
            row = {"update": update, "stage": stage, "mean_step_reward": float(rewards.mean() * args.reward_scale),
                   "episodes": episode_count, "exits": exits, "max_waypoint": int(max_wp),
                   "seconds": round(time.monotonic() - start_time, 1)}
            stage_complete = False
            stage_failed = False
            if args.adaptive:
                stage_updates += 1
                row["stage_updates"] = stage_updates
                row["recent_exits"] = sum(stage_outcomes)
                row["recent_episodes"] = len(stage_outcomes)
                ready = (stage_updates >= args.min_stage_updates
                         and len(stage_outcomes) >= args.min_stage_episodes
                         and sum(stage_outcomes) >= args.min_stage_exits)
                if ready:
                    stage_index += 1
                    stage_complete = stage_index == len(args.starts)
                    stage_updates = 0
                    stage_outcomes.clear()
                    if not stage_complete:
                        print(f"Advancing curriculum to waypoint {args.starts[stage_index]}", flush=True)
                elif stage_updates >= args.max_stage_updates:
                    stage_failed = True
            history.append(row)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history, "update": update, "args": vars(args),
                        "stage_index": stage_index, "stage_updates": stage_updates,
                        "stage_outcomes": list(stage_outcomes)}, checkpoint)
            print(json.dumps(row), flush=True)
            if stage_complete:
                print("All curriculum stages passed; running final evaluation", flush=True)
                break
            if stage_failed:
                print(f"Stage {stage} did not meet exit criterion after "
                      f"{args.max_stage_updates} updates; running final evaluation", flush=True)
                break
            if time.monotonic() - start_time > args.max_seconds:
                print("Training time budget reached; running final evaluation", flush=True)
                break
        result = evaluate(model, eval_env, device, episodes=args.eval_episodes)
        with open(os.path.join(args.output, "evaluation.json"), "w") as f:
            json.dump({"skill": args.skill, "seed": 10000, "episodes": result,
                       "wins": sum(r["exit_reached"] for r in result)}, f, indent=2)
        print(f"Full-start greedy wins: {sum(r['exit_reached'] for r in result)}/{len(result)}", flush=True)
    finally:
        for env in envs:
            env.close()
        eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skill", type=int, default=3)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--starts", type=int, nargs="+", default=[21, 20, 19, 16, 12, 6, 0])
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--min-stage-updates", type=int, default=40)
    p.add_argument("--max-stage-updates", type=int, default=400)
    p.add_argument("--stage-window", type=int, default=20)
    p.add_argument("--min-stage-episodes", type=int, default=10)
    p.add_argument("--min-stage-exits", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--max-seconds", type=int, default=18000)
    p.add_argument("--output", default="e1m1_run")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())
