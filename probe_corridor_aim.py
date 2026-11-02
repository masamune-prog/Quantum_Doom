"""Does each corridor policy turn toward enemies and fire only when they are centred?

Frames come from one shared behaviour policy (classical seed 0, sampled, with 25%
random actions) so every checkpoint is scored on identical inputs. The labels
buffer gives the nearest monster's horizontal offset from the crosshair; it is
used only to bin frames, never as policy input.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from pixel_corridor_env import MONSTER_NAMES, PixelCorridorEnv
from pixel_nav_models import PixelNavPolicy

torch.set_default_device("cpu")
NAMES = PixelCorridorEnv.ACTION_NAMES
CHECKPOINTS = {
    "classical seed0 (2.30M)": ("classical", "pixel_corridor_runs/classical_seed0/best.pt"),
    "quantum seed0 (2.82M)": ("quantum", "pixel_corridor_runs/quantum_seed0/best_2.8M.pt"),
    "quantum seed0 (best, resumed)": ("quantum", "aim_probe/quantum_best_snapshot.pt"),
    "quantum seed0 (latest, resumed)": ("quantum", "aim_probe/quantum_latest_snapshot.pt"),
}
# Extra checkpoints as name=head:path, e.g. "quantum final=quantum:aim_probe/q.pt".
for spec in sys.argv[1:]:
    name, rest = spec.split("=", 1)
    head, path = rest.split(":", 1)
    CHECKPOINTS[name] = (head, path)


def load(head, path):
    model = PixelNavPolicy(head=head, n_actions=len(NAMES), aux_pos_weight=5.0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("update")


def enemy_bin(state, width):
    """Bin by the horizontal offset of the largest visible monster from screen centre."""
    monsters = [l for l in state.labels if l.object_name in MONSTER_NAMES]
    if not monsters:
        return "none", None
    target = max(monsters, key=lambda l: l.width * l.height)
    offset = (target.x + target.width / 2) / width - 0.5
    if abs(offset) < 0.06:
        return "centre", offset
    return ("left" if offset < 0 else "right"), offset


def collect(behaviour, episodes=40, epsilon=0.25):
    env = PixelCorridorEnv()
    rng = np.random.default_rng(0)
    frames, bins, offsets = [], [], []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=30000 + episode)
            for _ in range(525):
                state = env.game.get_state()
                label, offset = enemy_bin(state, state.screen_buffer.shape[1])
                frames.append(obs)
                bins.append(label)
                offsets.append(offset)
                if rng.random() < epsilon:
                    action = int(rng.integers(len(NAMES)))
                else:
                    with torch.no_grad():
                        logits, _ = behaviour(torch.as_tensor(obs[None]))
                    action = int(torch.distributions.Categorical(logits=logits).sample())
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
    finally:
        env.close()
    return np.stack(frames), np.array(bins), offsets


torch.manual_seed(0)
behaviour, _ = load(*CHECKPOINTS["classical seed0 (2.30M)"])
frames, bins, offsets = collect(behaviour)
print("frames per bin", dict(Counter(bins.tolist())), flush=True)

report = {}
for name, (head, path) in CHECKPOINTS.items():
    model, update = load(head, path)
    probs = []
    with torch.no_grad():
        for chunk in np.array_split(frames, max(1, len(frames) // 256)):
            logits, _ = model(torch.as_tensor(chunk))
            probs.append(torch.softmax(logits.float(), -1).numpy())
    probs = np.concatenate(probs)
    greedy = probs.argmax(1)
    rows = {}
    for b in ("left", "centre", "right", "none"):
        mask = bins == b
        if not mask.any():
            continue
        rows[b] = {"frames": int(mask.sum()),
                   "mean_prob": {n: round(float(probs[mask, i].mean()), 3) for i, n in enumerate(NAMES)},
                   "greedy_share": {n: round(float((greedy[mask] == i).mean()), 3) for i, n in enumerate(NAMES)}}
    turn_l, turn_r, attack = NAMES.index("TURN_LEFT"), NAMES.index("TURN_RIGHT"), NAMES.index("ATTACK")
    # Positive when the policy turns toward the enemy: left turns for left enemies, right for right.
    toward = ((probs[bins == "left", turn_l] - probs[bins == "left", turn_r]).mean()
              + (probs[bins == "right", turn_r] - probs[bins == "right", turn_l]).mean()) / 2
    # Positive when attack is likelier with a centred enemy than an off-centre one.
    aim_fire = probs[bins == "centre", attack].mean() - probs[np.isin(bins, ["left", "right"]), attack].mean()
    report[name] = {"update": update, "turn_toward_enemy": round(float(toward), 3),
                    "attack_centre_minus_offcentre": round(float(aim_fire), 3),
                    "attack_with_no_enemy": round(float(probs[bins == "none", attack].mean()), 3),
                    "bins": rows}
    print(f"\n{name} update={update} turn_toward={toward:+.3f} "
          f"attack(centre-offcentre)={aim_fire:+.3f} attack|none={report[name]['attack_with_no_enemy']:.3f}")
    for b, row in rows.items():
        print(f"  {b:6s} n={row['frames']:5d} probs " +
              " ".join(f"{n}={v:.2f}" for n, v in row["mean_prob"].items()), flush=True)

with open(os.environ.get("PROBE_OUT", "aim_probe/results.json"), "w") as f:
    json.dump(report, f, indent=2)
