"""PPO on My Way Home from stacked screen frames, with training-only auxiliary losses.

The classical and quantum runs share the CNN encoder and 12-value bottleneck;
only the policy/value head differs. Evaluation uses held-out spawn seeds and
reports results with and without seed 20017, which random actions also solve.
"""

import argparse
import copy
from collections import Counter, deque
import json
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical

from pixel_nav_env import EnvPool
from pixel_nav_models import PixelNavPolicy

EASY_SEED = 20017
SELECTION_SEED = 10000
FINAL_SEED = 20000
N_ACTIONS = {"navigation": 3, "corridor": 6, "e1m1": 7}
# Monsters appear in far more sectors than the vest does.
AUX_POS_WEIGHT = {"navigation": 20.0, "corridor": 5.0, "e1m1": 5.0}
AUX_SIZES = {"e1m1": 20}  # others use pixel_nav_env.AUX_SIZE
AUX_ROUTE = {"e1m1": 3}   # trailing regression targets in the auxiliary vector


class Curriculum:
    """Reverse-start curriculum for E1M1: begin near the exit, move back as success rises.

    Each episode starts at the current stage's route waypoint (50%), an earlier stage's
    waypoint (30%) or the real start (20%). The stage advances when 80% of the last 20
    episodes started at the current waypoint reached the exit. After the last stage every
    episode starts at the real start, which is also what evaluation uses. Stage starts
    avoid the door waypoints (5, 17, 20): teleporting into a closed door sector.
    """

    STAGES = (21, 19, 16, 12, 8, 4, 0)

    def __init__(self, window=20, threshold=0.8, real_fraction=0.2, current_fraction=0.5):
        self.window, self.threshold = window, threshold
        self.real_fraction, self.current_fraction = real_fraction, current_fraction
        self.stage, self.recent = 0, deque(maxlen=window)

    @property
    def finished(self):
        return self.stage >= len(self.STAGES)

    def sample(self, rng):
        """Start waypoint for the next episode, or -1 for the real start."""
        draw = rng.random()
        if self.finished or draw < self.real_fraction:
            return -1
        if self.stage == 0 or draw < self.real_fraction + self.current_fraction:
            return self.STAGES[self.stage]
        return self.STAGES[int(rng.integers(self.stage))]

    def record(self, start, success):
        if not self.finished and start == self.STAGES[self.stage]:
            self.recent.append(bool(success))

    def maybe_advance(self):
        if (not self.finished and len(self.recent) == self.window
                and sum(self.recent) >= self.threshold * self.window):
            self.stage += 1
            self.recent.clear()
            return True
        return False


class UniformStarts:
    """E1M1 start sampler without stages: the real start, or any non-door route waypoint.

    The reverse curriculum gates all progress on its hardest stage; this gives every
    route segment practice from the first update. Evaluation still starts at the real
    start, so the policy has to chain the segments.
    """

    STARTS = tuple(k for k in range(22) if k not in (5, 17, 20))
    stage, finished = 0, False

    def __init__(self, real_fraction=0.25):
        self.real_fraction = real_fraction

    def sample(self, rng):
        if rng.random() < self.real_fraction:
            return -1
        return int(self.STARTS[int(rng.integers(len(self.STARTS)))])

    def record(self, start, success):
        pass

    def maybe_advance(self):
        return False


@torch.no_grad()
def evaluate(model, pool, device, seeds, mode, n_actions, rng_seed=0):
    """Run one episode per seed; mode is greedy, sampled, random or forward (action 0)."""
    generator = torch.Generator(device=device).manual_seed(rng_seed)
    pending = list(seeds)
    active, obs, rows = {}, {}, []

    def start(i):
        seed = pending.pop(0)
        obs[i] = pool.reset([i], [seed])[0][0]
        active[i] = {"seed": seed, "steps": 0, "actions": Counter(),
                     "rng": np.random.default_rng(seed)}

    for i in range(min(pool.n, len(pending))):
        start(i)
    while active:
        ids = sorted(active)
        if mode == "forward":
            actions = [0] * len(ids)
        elif mode == "random":
            actions = [int(active[i]["rng"].integers(n_actions))
                       for i in ids]
        else:
            x = torch.as_tensor(np.stack([obs[i] for i in ids]), device=device)
            logits, _ = model(x)
            if mode == "sampled":
                probs = torch.softmax(logits.float(), -1)
                actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1).tolist()
            else:
                actions = logits.argmax(-1).tolist()
        for i, action, (o, _, term, trunc, info) in zip(ids, actions, pool.step(ids, actions)):
            episode = active[i]
            episode["steps"] += 1
            episode["actions"][int(action)] += 1
            obs[i] = o
            if term or trunc:
                rows.append({"seed": episode["seed"], "success": bool(info["success"]),
                             "steps": episode["steps"], "actions": dict(episode["actions"]),
                             **info.get("episode", {})})
                del active[i]
                if pending:
                    start(i)
    rows.sort(key=lambda r: r["seed"])
    hard = [r for r in rows if r["seed"] != EASY_SEED]
    return {"mode": mode, "successes": sum(r["success"] for r in rows), "episodes": len(rows),
            "successes_without_easy_seed": sum(r["success"] for r in hard),
            "episodes_without_easy_seed": len(hard),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            **{f"mean_{k}": float(np.mean([r[k] for r in rows]))
               for k in ("kills", "health", "dead", "progress") if k in rows[0]},
            "rows": rows}


def summary(result):
    return (f"{result['mode']} {result['successes']}/{result['episodes']} "
            f"(excluding {EASY_SEED}: {result['successes_without_easy_seed']}/"
            f"{result['episodes_without_easy_seed']})"
            + "".join(f" {k}={result[k]:.2f}" for k in
                      ("mean_kills", "mean_dead", "mean_progress") if k in result))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env_kwargs = {"kill_weight": args.kill_weight} if args.kill_weight is not None else {}
    pool = EnvPool(args.num_envs, task=args.task, **env_kwargs)
    n_actions = N_ACTIONS[args.task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PennyLane allocates the circuit's initial state on the default device.
    torch.set_default_device(device)
    model = PixelNavPolicy(head=args.model, n_actions=n_actions, layers=args.layers,
                           aux_pos_weight=AUX_POS_WEIGHT[args.task],
                           rich_critic=args.rich_critic,
                           **({"aux_size": AUX_SIZES[args.task],
                               "aux_route": AUX_ROUTE[args.task]} if args.task in AUX_SIZES else {})
                           ).to(device)
    if args.init_encoder:
        # Warm start: reuse another run's CNN encoder (not its bottleneck, head or aux head).
        source = torch.load(args.init_encoder, map_location=device, weights_only=False)["model"]
        encoder = {k: v for k, v in source.items() if k.startswith("encoder.")}
        model.load_state_dict(encoder, strict=False)
        print(f"loaded {len(encoder)} encoder tensors from {args.init_encoder}", flush=True)
    if args.init_checkpoint:
        # Fine-tune a whole trained model (encoder, head, auxiliary head), e.g. an
        # imitation student, instead of starting from scratch.
        source = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing = model.load_state_dict(source["model"], strict=not args.rich_critic)
        print(f"loaded full model from {args.init_checkpoint}"
              + (f" (new: {missing.missing_keys})" if args.rich_critic else ""), flush=True)
    # Frozen copy of the starting policy: a KL penalty toward it keeps fine-tuning close to
    # the imitation policy, whose advantages are tiny and easily swamped by noise.
    anchor = None
    if args.init_checkpoint and args.anchor_kl_coef > 0:
        anchor = copy.deepcopy(model).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
    head_params = list(model.head.parameters())
    if model.critic256 is not None:
        head_params += list(model.critic256.parameters())  # trains with the head, not the encoder
    head_ids = {id(p) for p in head_params}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
        {"params": head_params, "lr": args.head_lr or args.lr}], eps=1e-5)
    os.makedirs(args.output, exist_ok=True)
    print(f"task={args.task} model={args.model} seed={args.seed} device={device} envs={args.num_envs} "
          f"steps={args.steps} decisions={args.decisions} aux_coef={args.aux_coef} "
          f"parameters={sum(p.numel() for p in model.parameters())} "
          f"head_parameters={sum(p.numel() for p in head_params)}", flush=True)

    episode_counter = 0
    curriculum = (Curriculum() if args.curriculum == "reverse"
                  else UniformStarts(real_fraction=args.real_fraction)
                  if args.curriculum == "uniform" else None)
    if args.task != "e1m1":
        curriculum = None
    curriculum_rng = np.random.default_rng(args.seed)
    starts = {}

    def train_seed():
        nonlocal episode_counter
        episode_counter += 1
        return 1_000_000 * (args.seed + 1) + episode_counter

    def reset_env(i):
        options = None
        if curriculum is not None:
            starts[i] = curriculum.sample(curriculum_rng)
            options = {"start_wp": starts[i]}
        return pool.reset([i], [train_seed()], options)[0]

    def reset_all():
        return [reset_env(i) for i in range(args.num_envs)]

    updates = args.decisions // (args.num_envs * args.steps)
    best_score, history, completed = -1.0, [], 0
    if args.resume:
        # Continue a run that hit its time budget; --decisions stays the total target.
        checkpoint = torch.load(os.path.join(args.output, "latest.pt"), map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed, history = checkpoint["update"], checkpoint["history"]
        if curriculum is not None and checkpoint.get("curriculum"):
            curriculum.stage = checkpoint["curriculum"]["stage"]
        best = torch.load(os.path.join(args.output, "best.pt"), map_location=device,
                          weights_only=False)
        best_score = best["score"] if best["score"] is not None else -1.0
        # Keep training spawn seeds disjoint from the first segment's.
        episode_counter = 100_000 * (1 + completed // 1000)
        print(f"resumed update={completed} best_update={best['update']} "
              f"best_score={best_score:.3f}", flush=True)
    if args.init_checkpoint and not args.resume:
        result = evaluate(model, pool, device,
                          range(SELECTION_SEED, SELECTION_SEED + args.eval_episodes),
                          "greedy", n_actions)
        best_score = (result["successes"] / result["episodes"]
                      + 0.01 * result.get("mean_progress", 0.0))
        print(f"eval update=0 decisions=0 selection {summary(result)} (starting policy)",
              flush=True)
        torch.save({"model": model.state_dict(), "update": 0, "score": best_score},
                   os.path.join(args.output, "best.pt"))
    ids = list(range(args.num_envs))
    resets = reset_all()
    obs = np.stack([o for o, _ in resets])
    aux = np.stack([info["aux"] for _, info in resets])
    start = time.monotonic()
    try:
        for update in range(completed + 1, updates + 1):
            # Value warm-up: only the critic learns (encoder lr 0, no policy or entropy loss),
            # so an untrained critic cannot wreck an imitation-trained policy.
            warmup = update <= args.value_warmup
            optimizer.param_groups[0]["lr"] = 0.0 if (warmup or args.freeze_encoder) else args.lr
            optimizer.param_groups[1]["lr"] = (args.warmup_head_lr if warmup
                                               else (args.head_lr or args.lr))
            policy_scale = 0.0 if warmup else 1.0
            ob, ax, ac, lp, rw, dn, va = [], [], [], [], [], [], []
            episodes = successes = 0
            gains = []
            for _ in range(args.steps):
                x = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    action, logp, _, value, _ = model.act(x)
                actions = action.tolist()
                new_obs, new_aux, rewards, dones = [], [], [], []
                for i, (o, reward, term, trunc, info) in enumerate(pool.step(ids, actions)):
                    done = term or trunc
                    episodes += int(done)
                    successes += int(info["success"])
                    if done:
                        if curriculum is not None:
                            curriculum.record(starts[i], info["success"])
                        episode_info = info.get("episode", {})
                        if "start_progress" in episode_info:
                            gains.append(episode_info["progress"] - episode_info["start_progress"])
                        o, reset_info = reset_env(i)
                        info = reset_info
                    new_obs.append(o)
                    new_aux.append(info["aux"])
                    rewards.append(reward / args.reward_scale)
                    dones.append(float(done))
                ob.append(x)
                ax.append(torch.as_tensor(aux, device=device))
                ac.append(action)
                lp.append(logp)
                va.append(value.float())
                rw.append(torch.tensor(rewards, device=device))
                dn.append(torch.tensor(dones, device=device))
                obs, aux = np.stack(new_obs), np.stack(new_aux)
            with torch.no_grad():
                _, next_value = model(torch.as_tensor(obs, device=device))
            rewards, dones, values = torch.stack(rw), torch.stack(dn), torch.stack(va)
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.steps)):
                nonterminal = 1.0 - dones[t]
                following = next_value.float() if t == args.steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * following * nonterminal - values[t]
                gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
                advantages[t] = gae
            returns = advantages + values
            b_obs = torch.stack(ob).flatten(0, 1)
            b_aux = torch.stack(ax).flatten(0, 1)
            b_act = torch.stack(ac).flatten()
            b_logp = torch.stack(lp).flatten()
            b_adv = advantages.flatten()
            b_ret = returns.flatten()
            stats = Counter()
            for _ in range(args.epochs):
                for mb in torch.randperm(len(b_act), device=device).split(args.minibatch):
                    features, z = model.encode(b_obs[mb])
                    logits, new_value = model.head(z)
                    new_value = model.value_of(features, new_value)
                    distribution = Categorical(logits=logits)
                    new_logp, entropy = distribution.log_prob(b_act[mb]), distribution.entropy()
                    ratio = (new_logp - b_logp[mb]).exp()
                    adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std(unbiased=False) + 1e-8)
                    policy_loss = torch.maximum(-adv * ratio,
                        -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    value_loss = 0.5 * (new_value.float() - b_ret[mb]).square().mean()
                    depth_loss, goal_loss, route_loss = model.aux_loss(features, b_aux[mb])
                    anchor_kl = logits.new_zeros(())
                    if anchor is not None:
                        with torch.no_grad():
                            anchor_logits, _ = anchor.head(anchor.encode(b_obs[mb])[1])
                        anchor_p = torch.softmax(anchor_logits.float(), -1)
                        anchor_kl = (anchor_p * (torch.log_softmax(anchor_logits.float(), -1)
                                                 - torch.log_softmax(logits.float(), -1))
                                     ).sum(-1).mean()
                    loss = (policy_scale * policy_loss + args.value_coef * value_loss
                            - policy_scale * args.entropy_coef * entropy.mean()
                            + policy_scale * args.anchor_kl_coef * anchor_kl
                            + args.aux_coef * (depth_loss + goal_loss + route_loss))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    stats.update({"policy": policy_loss.item(), "value": value_loss.item(),
                                  "entropy": entropy.mean().item(), "depth": depth_loss.item(),
                                  "goal": goal_loss.item(), "route": route_loss.item(),
                                  "kl": float(anchor_kl), "n": 1})
            completed = update
            advanced = curriculum.maybe_advance() if curriculum is not None else False
            n = stats.pop("n")
            row = {"update": update, "decisions": update * args.num_envs * args.steps,
                   "episodes": episodes, "successes": successes,
                   "reward": float(rewards.mean().item() * args.reward_scale),
                   **{k: round(v / n, 4) for k, v in stats.items()},
                   **({"stage": curriculum.stage} if curriculum is not None else {}),
                   **({"gain": round(float(np.mean(gains)), 3)} if gains else {}),
                   "seconds": round(time.monotonic() - start, 1)}
            history.append(row)
            if advanced:
                print(f"curriculum advanced to stage {curriculum.stage} at update {update}"
                      + ("" if curriculum.finished else
                         f" (start waypoint {curriculum.STAGES[curriculum.stage]})"), flush=True)
            if update % args.log_every == 0:
                print(json.dumps(row), flush=True)
            if update % args.eval_every == 0 or update == updates:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "update": update, "history": history, "args": vars(args),
                            "curriculum": ({"stage": curriculum.stage} if curriculum else None)},
                           os.path.join(args.output, "latest.pt"))
                result = evaluate(model, pool, device,
                                  range(SELECTION_SEED, SELECTION_SEED + args.eval_episodes),
                                  "greedy", n_actions)
                # Mean progress breaks ties while no episode succeeds.
                score = (result["successes"] / result["episodes"]
                         + 0.01 * result.get("mean_progress", 0.0))
                print(f"eval update={update} decisions={row['decisions']} "
                      f"selection {summary(result)}", flush=True)
                if score >= best_score:
                    best_score = score
                    torch.save({"model": model.state_dict(), "update": update, "score": score},
                               os.path.join(args.output, "best.pt"))
                # Evaluation consumed the pool; restart training episodes.
                resets = reset_all()
                obs = np.stack([o for o, _ in resets])
                aux = np.stack([info["aux"] for _, info in resets])
            if time.monotonic() - start > args.max_seconds:
                print("time budget reached", flush=True)
                break

        best = torch.load(os.path.join(args.output, "best.pt"), map_location=device,
                          weights_only=False)
        model.load_state_dict(best["model"])
        model.eval()
        final_seeds = range(FINAL_SEED, FINAL_SEED + args.final_episodes)
        results = {"task": args.task, "model_type": args.model, "seed": args.seed,
                   "completed_updates": completed,
                   "training_decisions": completed * args.num_envs * args.steps,
                   "training_seconds": round(time.monotonic() - start, 1),
                   "best_update": best["update"], "best_selection_score": best["score"],
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "head_parameters": sum(p.numel() for p in head_params),
                   "greedy": evaluate(model, pool, device, final_seeds, "greedy", n_actions),
                   "sampled": evaluate(model, pool, device, final_seeds, "sampled", n_actions,
                                       rng_seed=args.seed)}
        if args.random_baseline:
            results["random"] = evaluate(model, pool, device, final_seeds, "random", n_actions)
            if args.task in ("corridor", "e1m1"):
                # Running straight at the vest sometimes survives; a policy must beat it.
                results["forward"] = evaluate(model, pool, device, final_seeds, "forward",
                                              n_actions)
    finally:
        pool.close()
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for key in ("greedy", "sampled", "random", "forward"):
        if key in results:
            print(f"FINAL model={args.model} seed={args.seed} best_update={best['update']} "
                  f"{summary(results[key])}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=tuple(N_ACTIONS), default="navigation")
    p.add_argument("--model", choices=("classical", "linear", "quantum"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decisions", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--aux-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--final-episodes", type=int, default=100)
    p.add_argument("--random-baseline", action="store_true")
    p.add_argument("--max-seconds", type=int, default=18000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--kill-weight", type=float, default=None,
                   help="E1M1: reward per kill (default 2.0)")
    p.add_argument("--init-checkpoint", default=None,
                   help="checkpoint of a whole trained model to fine-tune (e.g. imitation)")
    p.add_argument("--value-warmup", type=int, default=0,
                   help="first N updates train only the critic")
    p.add_argument("--warmup-head-lr", type=float, default=1e-3)
    p.add_argument("--anchor-kl-coef", type=float, default=0.0,
                   help="penalty KL(starting policy || policy); needs --init-checkpoint")
    p.add_argument("--rich-critic", action="store_true",
                   help="critic on the 256 encoder features instead of the 12-value bottleneck")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train only the head (encoder and bottleneck learning rate 0)")
    p.add_argument("--real-fraction", type=float, default=0.25,
                   help="uniform curriculum: share of episodes starting at the real start")
    p.add_argument("--init-encoder", default=None,
                   help="checkpoint whose CNN encoder initialises this run (warm start)")
    p.add_argument("--curriculum", choices=("reverse", "uniform", "none"), default="reverse",
                   help="E1M1 start states: reverse stages, uniform over the route, or real start")
    p.add_argument("--output", required=True)
    train(p.parse_args())
