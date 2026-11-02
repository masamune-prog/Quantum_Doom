"""Why do some non-timeout episodes end without a counted success?"""
import numpy as np
import torch
import vizdoom as vzd
from pixel_nav_env import PixelNavigationEnv
from pixel_nav_models import PixelNavPolicy

torch.set_default_device("cpu")
model = PixelNavPolicy("classical")
model.load_state_dict(torch.load("pixel_nav_runs/classical_seed0/best.pt", map_location="cpu",
                                 weights_only=False)["model"])
env = PixelNavigationEnv()
try:
    for seed in range(20000, 20012):
        obs, _ = env.reset(seed=seed)
        for step in range(1, 526):
            with torch.no_grad():
                action = int(model(torch.as_tensor(obs[None]))[0].argmax())
            before = env.last_reward_position.copy()
            raw = env.game.make_action(env.ACTIONS[action], env.frame_skip)
            if env.game.is_episode_finished():
                pos = np.array([env.game.get_game_variable(vzd.GameVariable.POSITION_X),
                                env.game.get_game_variable(vzd.GameVariable.POSITION_Y)])
                print(f"seed={seed} step={step} raw={raw:.4f} total={env.game.get_total_reward():.4f} "
                      f"timeout={env.game.is_episode_timeout_reached()} "
                      f"dead={env.game.is_player_dead()} dist_goal={np.linalg.norm(pos - env.goal_pos):.1f} "
                      f"dist_before={np.linalg.norm(before - env.goal_pos):.1f}", flush=True)
                break
            env.last_reward_position = env._reward_position()
            state = env.game.get_state()
            env.frames.append(env._frame(state.screen_buffer))
            obs = env._observe()
finally:
    env.close()
