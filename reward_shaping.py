"""Private training rewards; none of these values are policy observations."""

import numpy as np


def goal_potential(position, goal, distance_scale):
    """Bounded closeness to an endpoint, using coordinates only inside the reward."""
    distance = float(np.linalg.norm(np.asarray(position) - np.asarray(goal)))
    return max(0.0, 1.0 - distance / distance_scale)


def progress_reward(previous_position, position, goal, *, distance_scale,
                    weight=5.0, gamma=0.99, terminal=False):
    """Potential-based distance shaping with zero potential at episode end."""
    before = goal_potential(previous_position, goal, distance_scale)
    after = 0.0 if terminal else goal_potential(position, goal, distance_scale)
    return weight * (gamma * after - before)


def kill_reward(previous_kills, kills, weight=2.0):
    """Pay only for newly recorded kills, including on a terminal action."""
    return weight * max(0.0, float(kills) - float(previous_kills))


def field_progress_reward(previous_distance, distance, *, distance_scale, weight=30.0,
                          gamma=0.99, terminal=False):
    """Potential-based shaping from a precomputed walking distance to the goal.

    ``distance`` comes from a path-cost field, so a wall between the player and the
    goal does not count as progress. Potential is 1 at the goal, 0 at
    ``distance_scale`` or farther, and 0 at episode end.
    """
    before = max(0.0, 1.0 - float(previous_distance) / distance_scale)
    after = 0.0 if terminal else max(0.0, 1.0 - float(distance) / distance_scale)
    return weight * (gamma * after - before)
