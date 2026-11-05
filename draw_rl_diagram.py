"""Diagram of the RL training process for the E1M1 hybrid agent (matplotlib; no torch needed)."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "diagrams"
GRAY, BLUE, GREEN, RED, ORANGE = "#ececec", "#dbe9f6", "#e1f0dc", "#fbe3e0", "#fdeccf"
DARK, REDLINE, GREENLINE = "#333333", "#c0392b", "#2e8b57"

fig, ax = plt.subplots(figsize=(15, 9.2))
ax.set_xlim(0, 100)
ax.set_ylim(6, 68)
ax.axis("off")


def box(x0, y0, x1, y1, title, body="", fc="white", ec=DARK, ls="-", lw=1.3, tfs=10, bfs=8.4,
        tcolor=DARK, rounding=0.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    cx = (x0 + x1) / 2
    ax.text(cx, y1 - 1.0, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=tcolor)
    if body:
        ax.text(cx, y1 - 3.5, body, ha="center", va="top", fontsize=bfs, color=DARK, linespacing=1.35)


def arrow(p0, p1, color=DARK, ls="-", lw=1.6, scale=15):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=scale, color=color,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def line(xs, ys, color=DARK, ls="-", lw=1.6):
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, solid_capstyle="butt")


ax.text(50, 67, "Reinforcement-learning loop of the E1M1 hybrid agent (PPO)", ha="center", va="center",
        fontsize=13.5, fontweight="bold")

# ---------------- acting loop (top band) ----------------
box(1, 50, 14, 60, "Environment", "ViZDoom E1M1\nskill 3\n1,575 decisions max\n(decision = 4 tics)", GRAY)
box(1, 40, 14, 49, "Private state", "position, health, kills,\ndepth & labels, map\n(training only)", RED,
    REDLINE, "--", 1.3, tcolor=REDLINE, bfs=7.8)
box(17, 46, 31, 58, "Observation o_t", "screen 160×120 RGB\n(HUD visible) → 80×60\nstack last 4 frames\n12 × 60 × 80", BLUE)

ax.add_patch(FancyBboxPatch((33, 43), 50, 18, boxstyle="round,pad=0,rounding_size=1.0", fc="#f4f9fe",
                            ec="#5b8fc7", lw=1.4, ls="-"))
ax.text(34, 60.3, "Policy network θ  (screen in → action out)", ha="left", va="center", fontsize=9.2,
        color="#2b5f9e", fontweight="bold")
box(34, 46, 48, 58, "CNN encoder", "3 conv layers\n(32, 64, 64 filters)\n+ linear → 256 features\n487,840 params", BLUE)
box(51, 46, 64, 58, "Bottleneck", "12 angles\nz = π·tanh(W f)\n∈ [−π, π]¹²\n3,084 params", BLUE)
box(67, 46, 82, 58, "Head (3 variants)", "classical MLP  (1,096)\nlinear  (104)\n12-qubit circuit  (74)\nall read the same z", BLUE)
box(85, 46, 99, 58, "Outputs", "action logits → π(a | z)\n7 actions\nvalue V(z)\n(the critic)", BLUE)

arrow((14, 54), (17, 54))
arrow((31, 52), (34, 52))
arrow((48, 52), (51, 52))
arrow((64, 52), (67, 52))
arrow((82, 52), (85, 52))

# action goes back to the environment
line([92, 92], [58, 63], DARK)
line([92, 7.5], [63, 63], DARK)
arrow((7.5, 63), (7.5, 60))
ax.text(50, 64.1, "action a_t ~ π(· | z_t): forward, attack, strafe left/right, turn left/right, use"
        "   (held for 4 game tics)", ha="center", va="bottom", fontsize=8.8, color=DARK)

# ---------------- training-only information ----------------
box(16, 30, 61, 41, "What the policy sees, and what only training sees",
    "Policy input: the 4 stacked screens (HUD included), nothing else.\n"
    "Training only, never a policy input: private state → reward r_t and auxiliary targets\n"
    "(depth, monster visibility, door ahead, route direction & progress);\n"
    "on E1M1 also a scripted teacher's action labels.", "#fffdf5", "#b8a64a", "-", 1.1, tfs=9.2, bfs=7.9)

# legend
box(62, 30, 77, 41, "Legend", "", "white", "#999999", "-", 1.0, tfs=9)
line([63, 66], [37.4, 37.4], DARK, "-", 1.8)
ax.text(67, 37.4, "data flow", va="center", fontsize=8)
line([63, 66], [35.4, 35.4], REDLINE, "--", 1.8)
ax.text(67, 35.4, "training-only", va="center", fontsize=8)
line([63, 66], [33.4, 33.4], GREENLINE, "-", 3.2)
ax.text(67, 33.4, "weight update", va="center", fontsize=8)
ax.text(63.2, 31.4, "①–⑥ order of one iteration", va="center", fontsize=7.6)

# reward + aux targets from the private state to the buffer; experience from the policy to the buffer
arrow((5, 40), (5, 19), REDLINE, "--", 1.8)
ax.text(5.8, 30, "r_t, done,\naux targets", ha="left", va="center", fontsize=7.6, color=REDLINE, linespacing=1.25)
line([92, 92], [46, 27], DARK)
line([92, 12], [27, 27], DARK)
arrow((12, 27), (12, 19))
ax.text(40, 27.7, "experience: (o_t, a_t, log π, V)", ha="center", va="bottom", fontsize=8)

# ---------------- learner (bottom band) ----------------
box(1, 8, 18, 19, "Rollout buffer", "8 envs × 128 steps\n= 1,024 decisions\n(o, a, r, log π, V,\naux targets)", GREEN)
box(22, 8, 38, 19, "Advantages (GAE)", "γ = 0.99,  λ = 0.95\nδ = r + γV′ − V\ncritic V is the baseline", GREEN)
box(42, 8, 66, 19, "PPO loss", "L = L_policy (clip ε = 0.1)\n+ 0.5 · L_value − 0.01 · entropy\n+ 0.5 · L_aux", GREEN)
box(70, 8, 84, 19, "Adam update", "4 epochs, minibatch 256\nlr 2.5e-4 (circuit 3e-3)\ngradient-norm clip 0.5", GREEN)
box(87, 8, 99, 19, "Checkpoint", "every 250 updates:\ngreedy on 32 seeds,\nkeep the best", GREEN)
for x0, x1 in ((18, 22), (38, 42), (66, 70), (84, 87)):
    arrow((x0, 13.5), (x1, 13.5))
arrow((79, 19), (79, 43), GREENLINE, "-", 3.4, 17)
ax.text(80, 35.5, "updated weights θ", ha="left", va="center", fontsize=8.6, color=GREENLINE,
        fontweight="bold")
ax.text(20, 20.2, "PPO learner", ha="left", va="bottom", fontsize=9.2, color="#2e6b3a", fontweight="bold")

# ---------------- numbered steps ----------------
def step(x, y, n):
    ax.add_patch(plt.Circle((x, y), 1.25, fc="#222222", ec="white", lw=1.2, zorder=10))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            zorder=11)


step(17.2, 58.2, 1)    # observe
step(67.2, 58.2, 2)    # policy computes logits and value
step(92, 61, 3)        # sample an action and send it to the environment
step(1.2, 60.2, 4)     # environment steps and computes r_t
step(1.2, 19.2, 5)     # store transitions, compute advantages
step(70.2, 19.2, 6)    # PPO loss, Adam update

for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{OUT}/rl_process.{ext}", dpi=200, bbox_inches="tight")
print("saved")
