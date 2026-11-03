"""Scripted E1M1 teacher for imitation. It reads PRIVATE state; a student never does.

Used only to label states with an action during training. The student policy still
sees nothing but stacked screens. What the teacher reads that a human cannot:
engine position and heading, the walking-distance field to the exit (route
direction), and door/exit-switch geometry (the use ray). Monster positions on the
screen come from the labels buffer, which matches what a player sees.

Priority each decision: (1) shoot a monster that is already near the screen centre,
(2) press USE once at a door or the exit switch, (3) turn toward the route, then walk.
It never turns to chase a monster, and it alternates shooting with moving along the route:
standing still to duel kills it (an imp has 60 HP and the pistol spreads when held down),
while walking on and firing in passing does not.
"""

import numpy as np

from e1m1_env import _wrap_angle
from e1m1_map import MONSTERS

MONSTER_NAMES = set(MONSTERS.values())
FORWARD, ATTACK, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT, USE = range(7)
SCREEN_WIDTH = 160


class RouteTeacher:
    def __init__(self, env, use_cooldown=25, use_range=40.0, turn_degrees=15.0, engage_height=12, close_height=30,
                 aim_tolerance=0.25, stuck_steps=30):
        self.env = env
        self.use_cooldown, self.engage_height = use_cooldown, engage_height
        self.close_height, self.use_range = close_height, use_range
        self.turn, self.aim_tolerance, self.stuck_steps = np.deg2rad(turn_degrees), aim_tolerance, stuck_steps
        self.reset()

    def reset(self):
        self.step, self.last_use = 0, -10_000
        self.anchor, self.anchor_step, self.escape, self.escape_turn = None, 0, 0, TURN_LEFT

    def _monster(self, state):
        seen = [l for l in state.labels
                if l.object_name in MONSTER_NAMES and l.height >= self.engage_height]
        return max(seen, key=lambda l: l.height) if seen else None

    def action(self, state):
        env = self.env
        self.step += 1
        position, angle = env._pose()

        monster = self._monster(state)
        if monster is not None:
            offset = ((monster.x + monster.width / 2) - SCREEN_WIDTH / 2) / (SCREEN_WIDTH / 2)
            # Fire every decision when it is close, otherwise every other one, moving in between.
            if abs(offset) <= self.aim_tolerance and (
                    monster.height >= self.close_height or self.step % 2 == 0):
                return ATTACK

        # Press USE only when close, so a closed door in view is almost always labelled USE
        # (from far away it would be labelled FORWARD on most steps, and a greedy student
        # would never press it).
        if (env._interact_ahead(position, angle, self.use_range)
                and self.step - self.last_use > self.use_cooldown):
            self.last_use = self.step
            return USE

        # Not making headway for a while (wall, corner): turn in place for a few steps.
        if self.anchor is None or np.linalg.norm(position - self.anchor) > 4.0:
            self.anchor, self.anchor_step = position.copy(), self.step
        elif self.step - self.anchor_step > self.stuck_steps and self.step - self.last_use > 8:
            self.escape, self.escape_turn = 6, int(np.random.choice([TURN_LEFT, TURN_RIGHT]))
            self.anchor_step = self.step
        if self.escape > 0:
            self.escape -= 1
            return self.escape_turn

        if env.route_cost < 6.0:                       # at the exit room: face the switch
            delta = env.exit_mid - position
            error = _wrap_angle(np.arctan2(delta[1], delta[0]) - angle)
        else:
            sin, cos, _ = env._route_targets(position, angle)
            error = np.arctan2(sin, cos)
        if abs(error) > self.turn:
            return TURN_LEFT if error > 0 else TURN_RIGHT
        return FORWARD
