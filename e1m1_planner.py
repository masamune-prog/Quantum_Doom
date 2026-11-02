"""
Classical route planner for E1M1: start -> exit switch, as a list of waypoints.

Rasterises the WAD map onto a grid, works out which sector each cell is in,
and runs Dijkstra with Doom movement rules:
  - cells closer than the player radius to a solid wall are blocked,
  - steps up of more than 24 units and gaps lower than the player are blocked
    (closed doors are allowed: the agent opens them with USE),
  - damaging floors (nukage) are allowed but heavily penalised,
  - hugging walls is mildly penalised so the path keeps to the middle of corridors.
The cell path is then pruned to straight-line segments and written to e1m1_waypoints.json.
"""

import heapq
import json
import os
import numpy as np

from e1m1_map import DoomMap

HERE = os.path.dirname(os.path.abspath(__file__))

CELL = 16.0
PLAYER_RADIUS = 16.0
MAX_STEP_UP = 24.0
PLAYER_HEIGHT = 56.0
DAMAGE_SPECIALS = {4, 5, 7, 11, 16}
IMPASSABLE_FLAG = 1


def point_segment_distances(points, a, b):
    """Distance from each point (N, 2) to each segment (M, 2)->(M, 2): returns (N, M)."""
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-6)[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


class RoutePlanner:
    def __init__(self, doom_map: DoomMap):
        self.m = doom_map
        lines = self.m.lines
        self.a = np.array([l["p1"] for l in lines], dtype=np.float64)
        self.b = np.array([l["p2"] for l in lines], dtype=np.float64)
        self.front = np.array([-1 if l["front_sector"] is None else l["front_sector"] for l in lines])
        self.back = np.array([-1 if l["back_sector"] is None else l["back_sector"] for l in lines])

        # Keep this a boolean mask: an integer array makes NumPy index rows 0/1
        # repeatedly instead of selecting the actual solid linedefs.
        solid = np.array([not l["two_sided"] or bool(l["flags"] & IMPASSABLE_FLAG)
                          for l in lines], dtype=bool)
        self.solid_a, self.solid_b = self.a[solid], self.b[solid]

        lo = self.m.vertices.min(axis=0) - 2 * CELL
        hi = self.m.vertices.max(axis=0) + 2 * CELL
        self.origin = lo
        self.nx = int(np.ceil((hi[0] - lo[0]) / CELL))
        self.ny = int(np.ceil((hi[1] - lo[1]) / CELL))
        xs = lo[0] + (np.arange(self.nx) + 0.5) * CELL
        ys = lo[1] + (np.arange(self.ny) + 0.5) * CELL
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.centers = np.stack([gx.ravel(), gy.ravel()], axis=1)

        self.sector = self._sector_of(self.centers).reshape(self.nx, self.ny)
        self.clearance = self._clearance(self.centers).reshape(self.nx, self.ny)

        self.door_sectors = set()
        for l in self.m.door_lines():
            for s in (l["front_sector"], l["back_sector"]):
                if s is not None:
                    sec = self.m.sectors[s]
                    if sec["ceil_h"] - sec["floor_h"] < PLAYER_HEIGHT:
                        self.door_sectors.add(s)

    def _sector_of(self, points, chunk=4096):
        """Cast a ray towards +x; the first line hit and which side we are on give the sector."""
        out = np.full(len(points), -1, dtype=np.int64)
        ax, ay, bx, by = self.a[:, 0], self.a[:, 1], self.b[:, 0], self.b[:, 1]
        for s in range(0, len(points), chunk):
            p = points[s:s + chunk]
            px, py = p[:, 0:1], p[:, 1:2]
            crosses = ((ay <= py) & (py < by)) | ((by <= py) & (py < ay))
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = ax + (py - ay) * (bx - ax) / (by - ay)
            xi = np.where(crosses & (xi > px), xi, np.inf)
            idx = np.argmin(xi, axis=1)
            hit = np.isfinite(xi[np.arange(len(p)), idx])
            # Doom's front side is to the right of v1 -> v2 (y axis points up)
            cross = (bx[idx] - ax[idx]) * (p[:, 1] - ay[idx]) - (by[idx] - ay[idx]) * (p[:, 0] - ax[idx])
            sec = np.where(cross < 0, self.front[idx], self.back[idx])
            out[s:s + chunk] = np.where(hit, sec, -1)
        return out

    def _clearance(self, points, chunk=4096):
        out = np.empty(len(points))
        for s in range(0, len(points), chunk):
            out[s:s + chunk] = point_segment_distances(points[s:s + chunk], self.solid_a, self.solid_b).min(axis=1)
        return out

    def to_cell(self, xy):
        c = ((np.asarray(xy) - self.origin) // CELL).astype(int)
        return int(c[0]), int(c[1])

    def to_xy(self, cell):
        return self.origin + (np.array(cell) + 0.5) * CELL

    def _cell_ok(self, c):
        i, j = c
        return (0 <= i < self.nx and 0 <= j < self.ny
                and self.sector[i, j] >= 0 and self.clearance[i, j] >= PLAYER_RADIUS)

    def _move_ok(self, c0, c1):
        s0, s1 = self.m.sectors[self.sector[c0]], self.m.sectors[self.sector[c1]]
        if self.sector[c1] in self.door_sectors:
            return True
        if s1["floor_h"] - s0["floor_h"] > MAX_STEP_UP:
            return False
        return s1["ceil_h"] - max(s0["floor_h"], s1["floor_h"]) >= PLAYER_HEIGHT

    def _cell_cost(self, c):
        sec = self.m.sectors[self.sector[c]]
        cost = 1.0
        if sec["special"] in DAMAGE_SPECIALS:
            cost += 30.0
        cost += 2.0 * max(0.0, 48.0 - self.clearance[c]) / 48.0
        return cost

    def nearest_free_cell(self, xy):
        c = self.to_cell(xy)
        if self._cell_ok(c):
            return c
        ok = (self.sector >= 0) & (self.clearance >= PLAYER_RADIUS)
        cand = np.argwhere(ok)
        return tuple(cand[np.argmin(np.linalg.norm(cand - np.array(c), axis=1))])

    def plan(self, start_xy, goal_xy):
        start, goal = self.nearest_free_cell(start_xy), self.nearest_free_cell(goal_xy)
        steps = [(di, dj, np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        dist = {start: 0.0}
        parent = {}
        heap = [(0.0, start)]
        while heap:
            d, c = heapq.heappop(heap)
            if c == goal:
                break
            if d > dist[c]:
                continue
            for di, dj, length in steps:
                n = (c[0] + di, c[1] + dj)
                if not self._cell_ok(n) or not self._move_ok(c, n):
                    continue
                nd = d + length * self._cell_cost(n)
                if nd < dist.get(n, np.inf):
                    dist[n] = nd
                    parent[n] = c
                    heapq.heappush(heap, (nd, n))
        if goal not in parent and goal != start:
            raise RuntimeError("No route found from start to exit")

        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        return path[::-1]

    def _line_ok(self, c0, c1):
        """Straight segment between two cells stays on cheap, passable cells."""
        n = int(max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))) * 2 + 1
        prev = c0
        for t in np.linspace(0.0, 1.0, n):
            c = (int(round(c0[0] + t * (c1[0] - c0[0]))), int(round(c0[1] + t * (c1[1] - c0[1]))))
            if c == prev:
                continue
            if not self._cell_ok(c) or not self._move_ok(prev, c) or self._cell_cost(c) > 3.0:
                return False
            prev = c
        return True

    def simplify(self, path, max_spacing=256.0, min_spacing=48.0):
        """Prune the cell path to straight segments that stay in the planned corridor.
        The first cell of every door is always kept (the agent must USE it there)."""
        in_door = [self.sector[c] in self.door_sectors for c in path]
        must_keep = {k for k in range(1, len(path)) if in_door[k] and not in_door[k - 1]}
        must_keep.add(len(path) - 1)

        keep = [0]
        i = 0
        while i < len(path) - 1:
            j = i + 1
            while (j not in must_keep
                   and np.hypot(*np.subtract(path[j + 1], path[i])) * CELL <= max_spacing
                   and self._line_ok(path[i], path[j + 1])):
                j += 1
            keep.append(j)
            i = j

        # Merge waypoints that are too close together, never dropping a door or the goal
        merged = [keep[0]]
        for k in keep[1:]:
            close = np.hypot(*np.subtract(path[k], path[merged[-1]])) * CELL < min_spacing
            if close and merged[-1] not in must_keep and merged[-1] != 0:
                merged[-1] = k
            else:
                merged.append(k)
        return [path[k] for k in merged]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=int, default=3)
    parser.add_argument("--max-spacing", type=float, default=256.0)
    args = parser.parse_args()

    m = DoomMap()
    planner = RoutePlanner(m)
    start, _ = m.player_start()

    # Stand in front of the exit switch (it is on the west wall of the exit room)
    exit_line = m.exit_lines()[0]
    mid = 0.5 * (exit_line["p1"] + exit_line["p2"])
    normal = np.array([exit_line["p1"][1] - exit_line["p2"][1], exit_line["p2"][0] - exit_line["p1"][0]])
    normal = normal / np.linalg.norm(normal)
    # Stand on whichever side of the switch line is inside the map
    goal = mid + 40.0 * normal
    if planner._sector_of(goal[None])[0] < 0:
        goal = mid - 40.0 * normal

    path = planner.plan(start, goal)
    cells = planner.simplify(path, args.max_spacing)
    waypoints = [planner.to_xy(c).round(1).tolist() for c in cells[1:]]
    waypoints[-1] = goal.round(1).tolist()

    door_wps = [i for i, c in enumerate(cells[1:]) if planner.sector[c] in planner.door_sectors]
    nukage_cells = sum(m.sectors[planner.sector[c]]["special"] in DAMAGE_SPECIALS for c in path)
    length = sum(np.hypot(*np.subtract(path[k + 1], path[k])) for k in range(len(path) - 1)) * CELL

    print(f"Grid {planner.nx}x{planner.ny}, route length {length:.0f} units, "
          f"{len(waypoints)} waypoints, door waypoints {door_wps}, nukage cells on route {nukage_cells}")
    for i, w in enumerate(waypoints):
        print(f"  {i:2d}: {w}")

    out = {"waypoints": waypoints, "door_waypoints": door_wps, "goal": goal.round(1).tolist(),
           "route_length": round(float(length), 1)}
    with open(os.path.join(HERE, "e1m1_waypoints.json"), "w") as f:
        json.dump(out, f, indent=2)
    m.plot(os.path.join(HERE, "e1m1_route.png"), skill=args.skill, waypoints=waypoints)
    print("Saved e1m1_waypoints.json and e1m1_route.png")


if __name__ == "__main__":
    main()
