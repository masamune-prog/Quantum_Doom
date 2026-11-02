"""Read-only development diagnostic: compare screen odometry to engine pose."""
import numpy as np
import vizdoom as vzd
from human_nav_env import HumanNavigationEnv

env=HumanNavigationEnv(memory=True)
try:
 env.reset(seed=42)
 def pose():
  return (env.game.get_game_variable(vzd.GameVariable.POSITION_X),
          env.game.get_game_variable(vzd.GameVariable.POSITION_Y),
          env.game.get_game_variable(vzd.GameVariable.ANGLE))
 before=pose()
 print('initial pose',before,flush=True)
 for a in [0]*8+[1]*9+[0]*8+[2]*9:
  previous=pose()
  obs,reward,term,trunc,info=env.step(a)
  current=pose()
  distance=np.hypot(current[0]-previous[0],current[1]-previous[1])
  print(f'action={a} distance={distance:.1f} angle_delta={(current[2]-previous[2]+180)%360-180:.1f} '
        f'map_cell={env.travel_map.cell()} new={env.travel_map.last_new} reward={reward:.2f}',flush=True)
  if term or trunc:break
finally:
 env.close()
