"""
Minimal Doom WAD map reader for E1M1 (Hangar).

Parses the raw map lumps (THINGS, LINEDEFS, SIDEDEFS, VERTEXES, SECTORS) so the
environment and the route planner know where the player starts, where doors and
the exit switch are, and which monsters/items exist at a given skill level.
Coordinates are Doom map units, identical to ViZDoom's POSITION_X / POSITION_Y.
"""

import os
import struct
import numpy as np

WAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wads", "doom1.wad")

MAP_LUMPS = ["THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
             "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP"]

# Linedef specials
EXIT_SPECIALS = {11, 52, 197}           # S1 exit switch, W1 exit, G1 exit
SECRET_EXIT_SPECIALS = {51, 124, 198}
DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}  # manual (USE) doors

# Thing types
PLAYER_START = 1
MONSTERS = {3004: "ZombieMan", 9: "ShotgunGuy", 3001: "DoomImp", 3002: "Demon",
            58: "Spectre", 3006: "LostSoul", 3005: "Cacodemon", 3003: "BaronOfHell"}
ITEMS = {2011: "Stimpack", 2012: "Medikit", 2014: "HealthBonus", 2015: "ArmorBonus",
         2018: "GreenArmor", 2019: "BlueArmor", 2001: "Shotgun", 2007: "Clip",
         2008: "Shell", 2048: "ClipBox", 2049: "ShellBox", 2035: "ExplosiveBarrel"}

# Thing flag bits: which skills the thing appears on
SKILL_EASY, SKILL_MEDIUM, SKILL_HARD, MULTIPLAYER_ONLY = 1, 2, 4, 16


def skill_flag(skill: int) -> int:
    """ViZDoom doom_skill 1-5 -> THINGS flag bit."""
    if skill <= 2:
        return SKILL_EASY
    if skill == 3:
        return SKILL_MEDIUM
    return SKILL_HARD


class DoomMap:
    def __init__(self, wad_path: str = WAD_PATH, map_name: str = "E1M1"):
        with open(wad_path, "rb") as f:
            data = f.read()
        ident, n_lumps, dir_ofs = struct.unpack_from("<4sii", data, 0)
        if ident not in (b"IWAD", b"PWAD"):
            raise ValueError(f"{wad_path} is not a WAD file")

        directory = []
        for i in range(n_lumps):
            pos, size, name = struct.unpack_from("<ii8s", data, dir_ofs + 16 * i)
            directory.append((name.rstrip(b"\0").decode("ascii"), pos, size))

        names = [d[0] for d in directory]
        if map_name not in names:
            raise ValueError(f"{map_name} not found in {wad_path}")
        base = names.index(map_name)
        lumps = {}
        for name, pos, size in directory[base + 1: base + 1 + len(MAP_LUMPS)]:
            lumps[name] = data[pos:pos + size]

        self.vertices = np.array(
            [struct.unpack_from("<hh", lumps["VERTEXES"], o)
             for o in range(0, len(lumps["VERTEXES"]), 4)], dtype=np.float32)

        self.sectors = []
        for o in range(0, len(lumps["SECTORS"]), 26):
            floor_h, ceil_h, floor_tex, ceil_tex, light, special, tag = \
                struct.unpack_from("<hh8s8shhh", lumps["SECTORS"], o)
            self.sectors.append({
                "floor_h": floor_h, "ceil_h": ceil_h,
                "floor_tex": floor_tex.rstrip(b"\0").decode("ascii"),
                "special": special, "tag": tag,
            })

        sidedef_sector = [
            struct.unpack_from("<hh8s8s8sh", lumps["SIDEDEFS"], o)[5]
            for o in range(0, len(lumps["SIDEDEFS"]), 30)
        ]

        self.lines = []
        for o in range(0, len(lumps["LINEDEFS"]), 14):
            v1, v2, flags, special, tag, right, left = \
                struct.unpack_from("<HHhhhHH", lumps["LINEDEFS"], o)
            self.lines.append({
                "p1": self.vertices[v1], "p2": self.vertices[v2],
                "flags": flags, "special": special, "tag": tag,
                "two_sided": left != 0xFFFF,
                "front_sector": sidedef_sector[right] if right != 0xFFFF else None,
                "back_sector": sidedef_sector[left] if left != 0xFFFF else None,
            })

        self.things = []
        for o in range(0, len(lumps["THINGS"]), 10):
            x, y, angle, ttype, flags = struct.unpack_from("<hhhhh", lumps["THINGS"], o)
            self.things.append({"x": x, "y": y, "angle": angle, "type": ttype, "flags": flags})

    def things_for_skill(self, skill: int):
        bit = skill_flag(skill)
        return [t for t in self.things
                if t["flags"] & bit and not t["flags"] & MULTIPLAYER_ONLY]

    def player_start(self):
        t = next(t for t in self.things if t["type"] == PLAYER_START)
        return np.array([t["x"], t["y"]], dtype=np.float32), float(t["angle"])

    def lines_with(self, specials):
        return [l for l in self.lines if l["special"] in specials]

    def exit_lines(self):
        return self.lines_with(EXIT_SPECIALS)

    def door_lines(self):
        return self.lines_with(DOOR_SPECIALS)

    def plot(self, out_path: str, skill: int = 3, waypoints=None, trajectories=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 12))
        for l in self.lines:
            (x1, y1), (x2, y2) = l["p1"], l["p2"]
            if l["special"] in EXIT_SPECIALS:
                ax.plot([x1, x2], [y1, y2], color="#16a34a", linewidth=4, zorder=3)
            elif l["special"] in DOOR_SPECIALS:
                ax.plot([x1, x2], [y1, y2], color="#f97316", linewidth=3, zorder=3)
            elif not l["two_sided"]:
                ax.plot([x1, x2], [y1, y2], color="#111827", linewidth=1)
            else:
                ax.plot([x1, x2], [y1, y2], color="#9ca3af", linewidth=0.5)

        for t in self.things_for_skill(skill):
            if t["type"] in MONSTERS:
                ax.scatter(t["x"], t["y"], color="#dc2626", s=30, zorder=4)
            elif t["type"] in ITEMS:
                ax.scatter(t["x"], t["y"], color="#2563eb", s=12, marker="s", zorder=4)

        start, angle = self.player_start()
        ax.scatter(*start, color="#7c3aed", s=120, marker="*", zorder=5)
        ax.annotate("START", start, textcoords="offset points", xytext=(6, 6))

        if waypoints:
            wp = np.array(waypoints)
            ax.plot(wp[:, 0], wp[:, 1], "--o", color="#7c3aed", markersize=4, zorder=5)
            for i, (x, y) in enumerate(waypoints):
                ax.annotate(str(i), (x, y), fontsize=7, textcoords="offset points", xytext=(3, 3))

        for label, traj in (trajectories or {}).items():
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.2, alpha=0.8, label=label, zorder=6)
        if trajectories:
            ax.legend(loc="upper left", fontsize=8)

        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"E1M1 (skill {skill}): black=walls, orange=doors, green=exit, "
                     f"red=monsters, blue=items, star=start")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=int, default=3)
    parser.add_argument("--out", default="e1m1_map.png")
    args = parser.parse_args()

    m = DoomMap()
    start, angle = m.player_start()
    print(f"Vertices: {len(m.vertices)}, Lines: {len(m.lines)}, Sectors: {len(m.sectors)}")
    print(f"Player start: {start}, angle {angle}")
    for l in m.exit_lines():
        print(f"Exit line (special {l['special']}): {l['p1']} -> {l['p2']}")
    print(f"Door lines: {len(m.door_lines())}")
    things = m.things_for_skill(args.skill)
    print(f"Monsters (skill {args.skill}):",
          dict(Counter(MONSTERS[t['type']] for t in things if t['type'] in MONSTERS)))
    print(f"Items (skill {args.skill}):",
          dict(Counter(ITEMS[t['type']] for t in things if t['type'] in ITEMS)))
    nukage = [i for i, s in enumerate(m.sectors) if s["special"] in (5, 7, 16, 4, 11)]
    print(f"Damaging sectors: {len(nukage)}")
    m.plot(args.out, skill=args.skill)
    print(f"Saved {args.out}")
