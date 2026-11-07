"""
Check that the NPCs can be talked to on the geodata: from the floor within ``TALK`` units of each NPC point (a
merchant stands behind his counter, the player talks across it), a flood over the cells by the server's movement
rules (open directions, steps of at most ``MAX_STEP``) must get ``AWAY`` units from it - out of the house. Prints the NPCs that stay shut in (a floor patch without a way out) and writes a CSV with, for every
reachable NPC, a start point about ``START`` units away from which a test character can walk to it.

Usage (from ``l2_agent/``)::

    python -m geo.reach --geodata ../geodata_work/stazis ../geodata_work/fixed [--files Giran/GiranNPCs.xml ...]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path

import numpy as np

from . import l2j
from .scan import NPC_DZ, load_overlay, npc_points

MAX_STEP = 40
AWAY = 1200
#: the server's Npc.INTERACTION_DISTANCE (250), a little less
TALK = 230
START = 900
DIRECTIONS = ((l2j.EAST, 1, 0, l2j.WEST), (l2j.WEST, -1, 0, l2j.EAST), (l2j.SOUTH, 0, 1, l2j.NORTH), (l2j.NORTH, 0, -1, l2j.SOUTH))


def layer(r: l2j.Region, cx: int, cy: int, z: int) -> tuple[int, int] | None:
    k = int(r.layers[cx, cy])
    if not k:
        return None
    i = int(np.argmin(np.abs(r.height[cx, cy, :k] - z)))
    return int(r.height[cx, cy, i]), int(r.nswe[cx, cy, i])


def flood(r: l2j.Region, x: int, y: int, z: int, limit: int = AWAY) -> dict[tuple[int, int], int]:
    """Cells (with their height) reachable from the world point within ``limit`` units, in one region."""
    start = r.cell_of(x, y)
    reach = TALK // 16
    seen: dict[tuple[int, int], int] = {}
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            c = (start[0] + dx, start[1] + dy)
            if math.hypot(dx * 16, dy * 16) > TALK or not (0 <= c[0] < l2j.REGION_CELLS and 0 <= c[1] < l2j.REGION_CELLS):
                continue
            here = layer(r, *c, z)
            if here is not None and abs(here[0] - z) <= NPC_DZ:
                seen[c] = here[0]
    if not seen:
        return {}
    queue = deque(seen)
    cells = limit // 16 + 2
    while queue:
        c = queue.popleft()
        h = seen[c]
        bits = layer(r, *c, h)[1]
        for bit, dx, dy, opp in DIRECTIONS:
            n = (c[0] + dx, c[1] + dy)
            if not bits & bit or n in seen or abs(n[0] - start[0]) > cells or abs(n[1] - start[1]) > cells:
                continue
            if not (0 <= n[0] < l2j.REGION_CELLS and 0 <= n[1] < l2j.REGION_CELLS):
                continue
            there = layer(r, *n, h)
            if there is None or not there[1] & opp or abs(there[0] - h) > MAX_STEP:
                continue
            seen[n] = there[0]
            queue.append(n)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geodata", nargs="+", required=True)
    ap.add_argument("--files", nargs="*", help="only these spawn files")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "geodata_work" / "report" / "reach_npcs.csv"))
    args = ap.parse_args()
    regions: dict[str, l2j.Region | None] = {}
    rows, shut = [], []
    for rel, npc_id, x, y, z in npc_points():
        if args.files and rel not in args.files:
            continue
        key = "%d_%d" % l2j.region_of(x, y)
        if key not in regions:
            regions[key] = load_overlay(args.geodata, key)
        r = regions[key]
        if r is None:
            continue
        cells = flood(r, x, y, z)
        ox, oy = r.origin
        far = max((math.hypot(ox + c[0] * 16 + 8 - x, oy + c[1] * 16 + 8 - y) for c in cells), default=0.0)
        start = None
        if far >= START:
            # the reachable cell nearest to START units away: where a test character starts its walk
            c = min(cells, key=lambda c: abs(math.hypot(ox + c[0] * 16 + 8 - x, oy + c[1] * 16 + 8 - y) - START))
            start = (ox + c[0] * 16 + 8, oy + c[1] * 16 + 8, cells[c])
        rows.append((rel, npc_id, x, y, z, len(cells), int(far), *(start or ("", "", ""))))
        if far < AWAY:
            shut.append((rel, npc_id, x, y, z, len(cells), int(far)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spawn_file", "npc_id", "x", "y", "z", "cells", "farthest", "start_x", "start_y", "start_z"])
        w.writerows(rows)
    print(f"{len(rows)} NPC points; shut in (no way {AWAY} units out): {len(shut)}")
    for s in shut[:60]:
        print("  ", *s)


if __name__ == "__main__":
    main()
