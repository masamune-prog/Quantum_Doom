"""
Record a trained Quantum RL navigation agent on ViZDoom My Way Home.
Produces an MP4 video with a fair HUD overlay showing:
 - Raw RGB game frame
 - 8-Sector FOV depth & visual beacon radar
 - Collision warning & local nearby proximity cue
 - Current action (7 actions: Corridor + USE) & cumulative reward
"""

import argparse
import os
import shutil
import subprocess
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from doom_nav_env import QuantumNavigationEnv
from models import DoomActorCritic


NAV_ACTIONS = [
    "FORWARD", "ATTACK", "MOVE_LEFT", "MOVE_RIGHT",
    "TURN_LEFT", "TURN_RIGHT", "USE",
]
NAV_ACTION_COLORS = [
    "#3b82f6", "#ef4444", "#8b5cf6", "#06b6d4",
    "#f59e0b", "#10b981", "#ec4899",
]


def render_nav_hud(
    obs: np.ndarray,
    action: int,
    step_reward: float,
    cum_reward: float,
    step: int,
    n_qubits: int = 10,
    width: int = 400,
    height: int = 240,
) -> np.ndarray:
    """Render a fair N-qubit navigation HUD panel as a NumPy RGB image."""
    fig, (ax_radar, ax_info) = plt.subplots(
        2, 1, figsize=(width / 100, height / 100), dpi=100,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0f172a")

    # Radar bar chart
    bar_colors = []
    n_visual = 8 if n_qubits >= 10 else 6
    for i, v in enumerate(obs):
        if i < n_visual:
            bar_colors.append("#10b981" if v > 0 else "#3b82f6")  # Green for visible armor, blue for corridor clearance
        elif n_qubits == 10:
            if i == 8:
                bar_colors.append("#ef4444" if v < -0.1 else "#334155")  # Red if wall collision detected
            else:
                bar_colors.append("#ec4899" if v > 0.05 else "#334155")  # Pink if nearby cue active
        elif n_qubits == 12:
            if i == 8:
                bar_colors.append("#10b981" if abs(v) > 0.01 else "#334155") # Target lock
            elif i == 9:
                bar_colors.append("#ef4444" if v > 0.01 else "#334155") # Threat
            elif i == 10:
                bar_colors.append("#f59e0b" if v < -0.1 else "#334155") # Stuck
            else:
                bar_colors.append("#ec4899" if v > 0.05 else "#334155") # Nearby

    ax_radar.bar(range(len(obs)), obs, color=bar_colors, width=0.7, edgecolor="white", linewidth=0.5)
    ax_radar.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax_radar.set_ylim(-np.pi, np.pi)
    ax_radar.set_xlim(-0.5, len(obs) - 0.5)

    if n_qubits == 10:
        labels = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "stuck", "near"]
    elif n_qubits == 12:
        labels = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "targ", "threat", "stuck", "near"]
    else:
        labels = [f"q{i}" for i in range(len(obs))]

    ax_radar.set_xticks(range(len(obs)))
    ax_radar.set_xticklabels(labels, fontsize=6, color="white")
    ax_radar.set_ylabel("angle", fontsize=7, color="white")
    ax_radar.set_title(f"Fair {n_qubits}-Qubit Vision & Sensory Radar", fontsize=9, color="white", fontweight="bold")
    ax_radar.tick_params(colors="white", labelsize=6)
    ax_radar.set_facecolor("#1e293b")
    for spine in ax_radar.spines.values():
        spine.set_color("#334155")

    # Info panel
    ax_info.axis("off")
    act_name = NAV_ACTIONS[action] if 0 <= action < len(NAV_ACTIONS) else "?"
    act_color = NAV_ACTION_COLORS[action % len(NAV_ACTION_COLORS)]

    info_text = (
        f"Step:{step:>4d} | "
        f"Act:{act_name:<11s} | "
        f"R:{step_reward:+.1f} | "
        f"Σ:{cum_reward:+.0f}"
    )
    ax_info.text(
        0.5, 0.5, info_text, transform=ax_info.transAxes,
        ha="center", va="center", fontsize=7, color=act_color,
        fontfamily="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor=act_color, alpha=0.9),
    )

    plt.tight_layout(pad=0.3)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]
    plt.close(fig)

    return cv2.resize(img, (width, height))


def _convert_h264_if_available(video_path: str):
    """Re-encode video to H.264 (yuv420p) for universal browser and quicktime compatibility."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return
    temp_path = video_path.replace(".mp4", "_h264.mp4")
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        temp_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.replace(temp_path, video_path)
            print(f"Re-encoded video to H.264 (yuv420p) for universal playback.")
    except Exception as e:
        print(f"H.264 re-encoding skipped: {e}")


def record_nav_episodes(
    checkpoint_path: str,
    output_path: str = "quantum_navigation_gameplay.mp4",
    n_episodes: int = 3,
    max_steps_per_episode: int = 500,
    fps: int = 15,
    greedy: bool = True,
    n_qubits: int = 10,
    n_actions: int = 7,
    device: str = "cpu",
):
    dev = torch.device(device)
    torch.set_default_device(dev)

    cfg_path = os.path.join(os.path.dirname(__file__), "my_way_home_quantum.cfg")
    env = QuantumNavigationEnv(
        n_qubits=n_qubits,
        frame_skip=4,
        custom_cfg_path=cfg_path,
    )

    agent = DoomActorCritic(
        n_qubits=n_qubits, n_layers=4, n_actions=n_actions,
        mode="quantum_entangled",
    ).to(dev)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        try:
            agent.load_state_dict(state_dict)
        except Exception:
            agent.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found! Recording with initialized weights.")

    agent.eval()

    h, w = 240, 320
    hud_w = 400
    canvas_w = w + hud_w
    canvas_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    print(f"Recording {n_episodes} episodes to {output_path} ({canvas_w}x{canvas_h} @ {fps}fps, device={dev})...")

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep * 100 + 42)
            done = False
            step = 0
            cum_reward = 0.0

            while not done and step < max_steps_per_episode:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                with torch.no_grad():
                    logits = agent.actor(obs_tensor)
                    if greedy:
                        action = logits.argmax(dim=-1).item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                cum_reward += reward
                step += 1

                game_frame = cv2.cvtColor(env.last_rgb, cv2.COLOR_RGB2BGR)
                hud_panel = render_nav_hud(
                    obs, action, reward, cum_reward, step, n_qubits=n_qubits, width=hud_w, height=canvas_h
                )
                hud_panel = cv2.cvtColor(hud_panel, cv2.COLOR_RGB2BGR)

                combined_frame = np.hstack([game_frame, hud_panel])
                out.write(combined_frame)

            print(f"  Episode {ep + 1}/{n_episodes} finished: steps={step}, cum_reward={cum_reward:.1f}")
    finally:
        out.release()
        env.close()

    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"Video saved to {output_path} ({size_kb:.1f} KB)")
        _convert_h264_if_available(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_nav_best.pt")
    parser.add_argument("--output", type=str, default="quantum_navigation_gameplay.mp4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--n-actions", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for inference (default: cpu)")
    args = parser.parse_args()

    record_nav_episodes(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        n_episodes=args.episodes,
        fps=args.fps,
        greedy=args.greedy,
        n_qubits=args.n_qubits,
        n_actions=args.n_actions,
        device=args.device,
    )
