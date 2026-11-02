"""Record greedy gameplay videos of saved pixel-policy checkpoints.

Each checkpoint plays held-out evaluation seeds (20000+) in order until its first
success, and only that episode is saved. Play is exactly as
in evaluation: the policy acts every 4 tics on the 80x60 stacked screens. The
video shows every game tic of the native 160x120 screen, upscaled 4x, at 35 fps,
with a caption strip (run, seed, step, action and, for the corridor, kills).

Usage: python record_gameplay.py [--episodes 20] [--out videos] [run_dir_or_pt ...]
With no paths, every run directory with a results.json is recorded.
"""

import argparse
import glob
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import vizdoom as vzd

from pixel_nav_env import make_pixel_env
from pixel_nav_models import PixelNavPolicy

N_ACTIONS = {"navigation": 3, "corridor": 6}
SCALE, FPS, CAPTION = 4, 35, 56


def default_targets():
    targets = []
    for root, task in (("pixel_nav_runs", "navigation"), ("pixel_corridor_runs", "corridor")):
        for run in sorted(glob.glob(os.path.join(root, "*_seed*"))):
            if os.path.exists(os.path.join(run, "results.json")):
                targets.append(os.path.join(run, "best.pt"))
        targets += sorted(glob.glob(os.path.join(root, "*_seed*", "best_*.pt")))
    return targets


def describe(path):
    run_dir = os.path.dirname(path) if path.endswith(".pt") else path
    path = path if path.endswith(".pt") else os.path.join(path, "best.pt")
    run = os.path.basename(run_dir)
    task = "corridor" if "corridor" in run_dir else "navigation"
    head = run.split("_seed")[0]
    suffix = os.path.basename(path)[len("best"):-len(".pt")]
    return path, task, head, f"{'corridor' if task == 'corridor' else 'mywayhome'}_{run}{suffix}"


def caption(frame, lines):
    big = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    strip = np.zeros((CAPTION, big.shape[1], 3), dtype=np.uint8)
    for row, text in enumerate(lines):
        cv2.putText(strip, text, (8, 22 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([big, strip], axis=0)


def record(path, task, head, name, seeds, out_dir):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = PixelNavPolicy(head=head, n_actions=N_ACTIONS[task])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    env = make_pixel_env(task)
    game, tic_frames = env.game, []

    # Same game logic as make_action(action, 4), but keeps the screen of every tic.
    def make_action(action, tics):
        game.set_action(action)
        for _ in range(tics):
            game.advance_action(1)
            if game.is_episode_finished():
                break
            tic_frames.append(game.get_state().screen_buffer.copy())

    class TicRecorder:
        def __getattr__(self, attr):
            return make_action if attr == "make_action" else getattr(game, attr)

    env.game = TicRecorder()
    has_kills = task == "corridor"
    summary = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        tic_frames[:] = [game.get_state().screen_buffer.copy()]
        start_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT) if has_kills else 0
        step, done, info, last, video = 0, False, {}, None, []
        title = f"{name}  update {checkpoint.get('update', '?')}  seed {seed}"
        while not done:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(obs[None]))
            action = int(logits.argmax(-1))
            step += 1
            tic_frames.clear()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            kills = (game.get_game_variable(vzd.GameVariable.KILLCOUNT) - start_kills
                     if has_kills and not done else info.get("episode", {}).get("kills", 0))
            status = f"step {step}  {env.ACTION_NAMES[action]}"
            if has_kills:
                status += f"  kills {int(kills)}"
            for frame in tic_frames:
                last = caption(frame, [title, status])
                video.append(last)
        result = "VEST REACHED" if info["success"] else (
            "DIED" if info.get("episode", {}).get("dead") else "TIMEOUT")
        if last is not None:
            end = last.copy()
            cv2.putText(end, result, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 255, 0) if info["success"] else (255, 60, 60), 3, cv2.LINE_AA)
            video += [end] * FPS
        summary.append({"seed": seed, "success": bool(info["success"]), "steps": step,
                        **info.get("episode", {})})
        print(name, summary[-1], flush=True)
        if info["success"]:
            imageio.mimwrite(os.path.join(out_dir, f"{name}.mp4"), video, fps=FPS,
                             codec="libx264", quality=8, macro_block_size=8)
            break
    env.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--episodes", type=int, default=20,
                        help="seeds to try; the first successful episode is saved")
    parser.add_argument("--first-seed", type=int, default=20000)
    parser.add_argument("--out", default="videos")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    results = {}
    for target in args.paths or default_targets():
        path, task, head, name = describe(target)
        results[name] = {"checkpoint": path, "episodes": record(path, task, head, name, seeds,
                                                                 args.out)}
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=float)


if __name__ == "__main__":
    main()
