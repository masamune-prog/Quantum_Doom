"""PPO on Deadly Corridor with a fully quantum policy on hand-made screen features (no CNN).

Observation: the current screen's 8 sector features (brightness + Canny edge density in
[-1, 1], green override, times pi; canny_corridor_env.py), padded to 12. With
--features enemy, 8 enemy-colour-mask densities follow (16 inputs; use --qubits 16). For --model
quantum the 12-qubit circuit is the whole agent: <Z> on qubits 0-5 are the action logits
and <Z> on qubit 6 is the value, each times a learned gain (wide_circuit.py, same circuit
as QuantumE1M1Policy). linear and classical heads on the same features are controls.

PPO settings, rewards, selection seeds (10000+) and final seeds (20000+) match the CNN
corridor runs (run_pixel_corridor_*.sh), so pixel_corridor_runs/ is the reference.
"""

import argparse
from collections import Counter
import json
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical

from human_nav_models import ClassicalNavPolicy
from pixel_nav_env import EnvPool
from pixel_nav_models import LinearNavHead
from ppo_common import gae_step_dones
from train_pixel_nav import FINAL_SEED, SELECTION_SEED, evaluate, summary
from wide_circuit import WideCircuitPolicy

N_ACTIONS = 6


TASKS = {"canny": ("corridor_canny", 12), "enemy": ("corridor_canny_enemy", 16)}


def build(args):
    n_inputs = TASKS[args.features][1]
    if args.model == "quantum":
        return WideCircuitPolicy(args.qubits, args.layers, n_inputs=n_inputs,
                                 n_actions=N_ACTIONS, input_scale_init=args.input_scale)
    if args.model == "linear":
        return LinearNavHead(n_inputs, N_ACTIONS)
    return ClassicalNavPolicy(n_inputs=n_inputs, n_actions=N_ACTIONS)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "results.json")):
        raise FileExistsError(f"Use a new output directory: {args.output}")
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    model = build(args).to(device)
    lr = args.lr if args.lr is not None else (3e-3 if args.model == "quantum" else 2.5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={params} lr={lr} decisions={args.decisions}", flush=True)
    env_kwargs = {}
    if args.reward == "advance":
        env_kwargs = {"progress_mode": "max", "progress_weight": 60.0, "timeout_penalty": 30.0}
    pool = EnvPool(args.num_envs, task=TASKS[args.features][0], **env_kwargs)
    episode_counter = 0

    def reset_all():
        nonlocal episode_counter
        out = []
        for i in range(args.num_envs):
            episode_counter += 1
            out.append(pool.reset([i], [1_000_000 * (args.seed + 1) + episode_counter])[0][0])
        return np.stack(out)

    ids = list(range(args.num_envs))
    updates = args.decisions // (args.num_envs * args.steps)
    best_score, history = -1.0, []
    obs = reset_all()
    start = time.monotonic()
    completed = 0
    try:
        for update in range(1, updates + 1):
            ob, ac, lp, rw, dn, va = [], [], [], [], [], []
            episodes = successes = 0
            action_counts = Counter()
            for _ in range(args.steps):
                x = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    logits, value = model(x)
                    dist = Categorical(logits=logits.float())
                    action = dist.sample()
                actions = action.tolist()
                action_counts.update(actions)
                new_obs, rewards, dones = [], [], []
                for i, (o, reward, term, trunc, info) in enumerate(pool.step(ids, actions)):
                    done = term or trunc
                    episodes += int(done)
                    successes += int(info["success"])
                    if done:
                        episode_counter += 1
                        o = pool.reset([i], [1_000_000 * (args.seed + 1) + episode_counter])[0][0]
                    new_obs.append(o)
                    rewards.append(reward / args.reward_scale)
                    dones.append(float(done))
                ob.append(x)
                ac.append(action)
                lp.append(dist.log_prob(action))
                va.append(value.float())
                rw.append(torch.tensor(rewards, device=device))
                dn.append(torch.tensor(dones, device=device))
                obs = np.stack(new_obs)
            with torch.no_grad():
                _, next_value = model(torch.as_tensor(obs, device=device))
            rewards_t, dones_t, values = torch.stack(rw), torch.stack(dn), torch.stack(va)
            advantages, returns = gae_step_dones(rewards_t, values, dones_t, next_value.float(),
                                                 args.gamma, args.gae_lambda)
            b_obs, b_act = torch.stack(ob).flatten(0, 1), torch.stack(ac).flatten()
            b_logp, b_adv, b_ret = torch.stack(lp).flatten(), advantages.flatten(), returns.flatten()
            stats = Counter()
            for _ in range(args.epochs):
                for mb in torch.randperm(len(b_act), device=device).split(args.minibatch):
                    logits, new_value = model(b_obs[mb])
                    dist = Categorical(logits=logits.float())
                    new_logp, entropy = dist.log_prob(b_act[mb]), dist.entropy()
                    ratio = (new_logp - b_logp[mb]).exp()
                    adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std(unbiased=False) + 1e-8)
                    policy_loss = torch.maximum(-adv * ratio,
                                                -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    value_loss = 0.5 * (new_value.float() - b_ret[mb]).square().mean()
                    loss = (policy_loss + args.value_coef * value_loss
                            - args.entropy_coef * entropy.mean())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    stats.update({"policy": policy_loss.item(), "value": value_loss.item(),
                                  "entropy": entropy.mean().item(), "n": 1})
            completed = update
            n = stats.pop("n")
            total = sum(action_counts.values())
            row = {"update": update, "decisions": update * args.num_envs * args.steps,
                   "episodes": episodes, "successes": successes,
                   "reward": float(rewards_t.mean().item() * args.reward_scale),
                   **{k: round(v / n, 4) for k, v in stats.items()},
                   "forward_share": round(action_counts[0] / total, 3),
                   "attack_share": round(action_counts[1] / total, 3),
                   "seconds": round(time.monotonic() - start, 1)}
            history.append(row)
            if update % args.log_every == 0:
                print(json.dumps(row), flush=True)
            if update % args.eval_every == 0 or update == updates:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "update": update, "history": history, "args": vars(args)},
                           os.path.join(args.output, "latest.pt"))
                result = evaluate(model, pool, device,
                                  range(SELECTION_SEED, SELECTION_SEED + args.eval_episodes),
                                  "greedy", N_ACTIONS)
                score = result["successes"] / result["episodes"] + 0.01 * result["mean_progress"]
                print(f"eval update={update} decisions={row['decisions']} "
                      f"selection {summary(result)}", flush=True)
                if score >= best_score:
                    best_score = score
                    torch.save({"model": model.state_dict(), "update": update, "score": score},
                               os.path.join(args.output, "best.pt"))
                obs = reset_all()
            if time.monotonic() - start > args.max_seconds:
                print("time budget reached", flush=True)
                break
        best = torch.load(os.path.join(args.output, "best.pt"), map_location=device)
        model.load_state_dict(best["model"])
        final_seeds = range(FINAL_SEED, FINAL_SEED + args.final_episodes)
        results = {"task": TASKS[args.features][0], "model_type": args.model, "seed": args.seed,
                   "parameters": params, "completed_updates": completed,
                   "training_decisions": completed * args.num_envs * args.steps,
                   "training_seconds": round(time.monotonic() - start, 1),
                   "best_update": best["update"], "best_selection_score": best["score"],
                   "history": history,
                   "greedy": evaluate(model, pool, device, final_seeds, "greedy", N_ACTIONS),
                   "sampled": evaluate(model, pool, device, final_seeds, "sampled", N_ACTIONS,
                                       rng_seed=args.seed)}
        if args.baselines:
            results["random"] = evaluate(model, pool, device, final_seeds, "random", N_ACTIONS)
            results["forward"] = evaluate(model, pool, device, final_seeds, "forward", N_ACTIONS)
    finally:
        pool.close()
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for key in ("greedy", "sampled", "random", "forward"):
        if key in results:
            print(f"FINAL model={args.model} seed={args.seed} best_update={best['update']} "
                  f"{summary(results[key])}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("quantum", "linear", "classical"), required=True)
    p.add_argument("--features", choices=tuple(TASKS), default="canny",
                   help="canny: 8 sector values; enemy: plus 8 enemy-colour-mask densities")
    p.add_argument("--reward", choices=("default", "advance"), default="default",
                   help="advance: keep +60 x furthest progress, and -30 on timeout as on death")
    p.add_argument("--qubits", type=int, default=12)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--decisions", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help="default 3e-3 quantum, 2.5e-4 classical")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--final-episodes", type=int, default=100)
    p.add_argument("--baselines", action="store_true", help="also evaluate random and FORWARD")
    p.add_argument("--max-seconds", type=int, default=6 * 3600)
    p.add_argument("--output", required=True)
    train(p.parse_args())
