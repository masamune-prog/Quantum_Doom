"""Walking-distance field to the E1M1 exit, for the private training reward.

Reverse Dijkstra from the exit over the route planner's grid, using the planner's
own movement rules and costs (wall clearance, step height, door sectors passable,
nukage heavily penalised). The result is the cost of the cheapest route from every
cell to the exit, so progress follows the winding route instead of straight-line
distance. It is used only inside the environment's reward, never as policy input.

    python e1m1_distance.py        # writes e1m1_distance.npz
"""

import heapq
import json
import os

import numpy as np

from e1m1_map import DoomMap
from e1m1_planner import CELL, RoutePlanner

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_PATH = os.path.join(HERE, "e1m1_distance.npz")
WAYPOINTS_PATH = os.path.join(HERE, "e1m1_waypoints.json")


def build_field(planner, goal_xy):
    """Return a (nx, ny) array of route cost to the goal; inf where unreachable."""
    goal = planner.nearest_free_cell(goal_xy)
    steps = [(di, dj, np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    dist = np.full((planner.nx, planner.ny), np.inf)
    dist[goal] = 0.0
    heap = [(0.0, goal)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist[n]:
            continue
        for di, dj, length in steps:
            c = (n[0] + di, n[1] + dj)
            # Predecessor c must be able to step onto n; the edge costs what the
            # forward planner charges for entering n.
            if not planner._cell_ok(c) or not planner._move_ok(c, n):
                continue
            nd = d + length * planner._cell_cost(n)
            if nd < dist[c]:
                dist[c] = nd
                heapq.heappush(heap, (nd, c))
    return dist


class DistanceField:
    """Bilinear lookup of the route cost to the exit (in planner cost units)."""

    def __init__(self, path=FIELD_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing; run `python e1m1_distance.py` first")
        data = np.load(path)
        self.dist, self.origin = data["dist"], data["origin"]
        self.start_distance = float(data["start_distance"])

    def __call__(self, xy):
        """Route cost from a map position, or None if no nearby cell can reach the exit."""
        f = (np.asarray(xy, dtype=np.float64) - self.origin) / CELL - 0.5
        i0, j0 = int(np.floor(f[0])), int(np.floor(f[1]))
        total = weight_sum = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                i, j = i0 + di, j0 + dj
                if not (0 <= i < self.dist.shape[0] and 0 <= j < self.dist.shape[1]):
                    continue
                value = self.dist[i, j]
                if not np.isfinite(value):
                    continue
                w = (1 - abs(f[0] - i)) * (1 - abs(f[1] - j))
                total += w * value
                weight_sum += w
        return total / weight_sum if weight_sum > 1e-6 else None


def main():
    with open(WAYPOINTS_PATH) as handle:
        route = json.load(handle)
    planner = RoutePlanner(DoomMap())
    dist = build_field(planner, route["goal"])
    start = DoomMap().player_start()[0]
    start_distance = float(dist[planner.nearest_free_cell(start)])
    np.savez(FIELD_PATH, dist=dist, origin=planner.origin, start_distance=start_distance)
    reachable = int(np.isfinite(dist).sum())
    print(f"grid {planner.nx}x{planner.ny}, {reachable} reachable cells, "
          f"start distance {start_distance:.1f}, saved {FIELD_PATH}")


if __name__ == "__main__":
    main()
