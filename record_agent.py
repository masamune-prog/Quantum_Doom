"""
Record a trained Quantum RL agent playing ViZDoom.
Produces an MP4 video with an informative HUD overlay showing:
 - Raw RGB game frame
 - 8-qubit radar visualization
 - Current action & cumulative reward

Supports both basic.cfg (3 actions) and deadly_corridor (6 composite actions).
"""

import argparse
import os
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from doom_env import QuantumDoomEnv
from models import DoomActorCritic


# Action name lookups
BASIC_ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
CORRIDOR_ACTIONS = [
    "FORWARD", "ATTACK", "MOVE_LEFT",
    "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT",
]
ACTION_COLORS_6 = ["#3b82f6", "#8b5cf6", "#ef4444", "#f59e0b", "#10b981", "#6366f1"]


def render_radar_overlay(obs: np.ndarray, action: int, step_reward: float,
                         cum_reward: float, step: int, health: float = 100.0,
                         action_names: list = None, width: int = 400, height: int = 240) -> np.ndarray:
    """Render a radar bar chart + info panel as a NumPy RGB image."""
    if action_names is None:
        action_names = BASIC_ACTIONS

    fig, (ax_radar, ax_info) = plt.subplots(
        2, 1, figsize=(width / 100, height / 100), dpi=100,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0f172a")

    # Radar bar chart
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in obs]
    ax_radar.bar(range(len(obs)), obs, color=colors, width=0.7, edgecolor="white", linewidth=0.5)
    ax_radar.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax_radar.set_ylim(-np.pi, np.pi)
    ax_radar.set_xlim(-0.5, len(obs) - 0.5)
    ax_radar.set_xticks(range(len(obs)))
    ax_radar.set_xticklabels([f"q{i}" for i in range(len(obs))], fontsize=7, color="white")
    ax_radar.set_ylabel("angle", fontsize=7, color="white")
    ax_radar.set_title("8-Qubit Spatial Radar", fontsize=9, color="white", fontweight="bold")
    ax_radar.tick_params(colors="white", labelsize=6)
    ax_radar.set_facecolor("#1e293b")
    for spine in ax_radar.spines.values():
        spine.set_color("#334155")

    # Info panel
    ax_info.axis("off")
    act_name = action_names[action] if 0 <= action < len(action_names) else "?"
    n_act = len(action_names)
    act_color = ACTION_COLORS_6[action % len(ACTION_COLORS_6)]

    info_text = (
        f"Step:{step:>4d} | "
        f"Act:{act_name} | "
        f"R:{step_reward:+.0f} | "
        f"Σ:{cum_reward:+.0f} | "
        f"HP:{health:.0f}"
    )
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                 ha="center", va="center", fontsize=7, color=act_color,
                 fontfamily="monospace", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor=act_color, alpha=0.9))

    plt.tight_layout(pad=0.3)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]  # drop alpha -> RGB
    plt.close(fig)

    return cv2.resize(img, (width, height))


def record_episodes(
    checkpoint_path: str,
    output_path: str = "quantum_agent_gameplay.mp4",
    scenario: str = "basic.cfg",
    n_episodes: int = 5,
    max_steps_per_episode: int = 2000,
    fps: int = 15,
    greedy: bool = False,
    corridor: bool = False,
    custom_cfg_path: str = None,
):
    """Load a trained agent and record gameplay video."""
    import vizdoom as vzd

    device = torch.device("cpu")  # CPU is fine for inference
    torch.set_default_device(device)

    n_actions = 6 if corridor else 3

    # --- Load agent ---
    agent = DoomActorCritic(n_qubits=8, n_layers=4, n_actions=n_actions, mode="quantum_entangled")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    try:
        agent.load_state_dict(state_dict)
    except Exception:
        agent.load_state_dict(state_dict, strict=False)
    agent.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    n_params = sum(p.numel() for p in agent.actor.parameters())
    print(f"Actor parameters: {n_params}")

    # --- Setup environment ---
    env = QuantumDoomEnv(
        scenario=scenario,
        n_qubits=8,
        frame_skip=4,
        use_composite_actions=corridor,
        custom_cfg_path=custom_cfg_path,
    )

    action_names = CORRIDOR_ACTIONS if corridor else BASIC_ACTIONS

    # --- Video writer ---
    game_h, game_w = 240, 320
    radar_w = 400
    canvas_w = game_w + radar_w
    canvas_h = game_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    total_frames = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=42 + ep)
        cum_reward = 0.0
        step = 0
        done = False

        while not done and step < max_steps_per_episode:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits = agent.actor(obs_t)
                if greedy:
                    action = logits.argmax(dim=-1).item()
                else:
                    from torch.distributions import Categorical
                    action = Categorical(logits=logits).sample().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cum_reward += reward
            step += 1

            # Get health for display
            health = 0.0
            if not done and corridor:
                try:
                    health = env.game.get_game_variable(vzd.GameVariable.HEALTH)
                except:
                    health = 0.0
            elif not done:
                health = 100.0

            # --- Compose frame ---
            rgb = env.last_rgb  # (240, 320, 3) RGB
            radar_img = render_radar_overlay(
                obs, action, reward, cum_reward, step,
                health=health, action_names=action_names,
                width=radar_w, height=game_h,
            )

            # Combine game frame + radar panel side by side
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:game_h, :game_w] = rgb
            canvas[:game_h, game_w:] = radar_img

            # Write as BGR for OpenCV
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            total_frames += 1

        print(f"Episode {ep}/{n_episodes} — Steps: {step}, Total Reward: {cum_reward:+.1f}")

    writer.release()
    env.close()

    duration = total_frames / fps
    print(f"\nSaved {total_frames} frames ({duration:.1f}s) → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record trained Quantum Doom agent gameplay")
    parser.add_argument("--checkpoint", default="checkpoint_quantum_entangled_seed42.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", default="quantum_agent_gameplay.mp4",
                        help="Output video file path")
    parser.add_argument("--scenario", default="basic.cfg",
                        help="ViZDoom scenario config")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to record")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=15,
                        help="Video frames per second")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy action selection instead of sampling")
    parser.add_argument("--corridor", action="store_true",
                        help="Use deadly corridor mode with composite actions")
    parser.add_argument("--custom-cfg", default=None,
                        help="Path to custom ViZDoom config file")
    args = parser.parse_args()

    record_episodes(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        scenario=args.scenario,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        fps=args.fps,
        greedy=args.greedy,
        corridor=args.corridor,
        custom_cfg_path=args.custom_cfg,
    )
