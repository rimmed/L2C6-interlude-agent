"""
Add the floor layers a spawn file's territories lack, from the client's BSP floors (rooms and caves: the
reliable kind; mesh tops and terrain are ambiguous outdoors) (``repair``), region by region, writing the result to ``geodata_work/fixed`` (starting from a fixed region when there is one).

``--npcs`` does the same from the fixed spawn points of a spawn file's NPCs that have no layer near their z
(a merchant inside a house whose geodata has only the roof): from the client floor at the NPC's height (BSP or
a mesh: town houses are meshes), the floor is flooded within ``NPC_REACH`` cells and its storey and gets layers.

Usage (from ``l2_agent/``)::

    python -m geo.fix Gludio/Wasteland.xml [more spawn files ...]
    python -m geo.fix --npcs TalkingIsland/TalkingIslandNPCs.xml
    python -m geo.fix --interiors Giran/GiranNPCs.xml   # the town's houses, from their doorways
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import l2j, repair, scene
from .scan import NPC_DZ, inside, npc_points, territories

ROOT = Path(__file__).resolve().parents[2]
STAZIS = ROOT / "geodata_work" / "stazis"
FIXED = ROOT / "geodata_work" / "fixed"
MARGIN = 64
#: cells beyond a territory's polygon the added floor may reach (to join the layers around it)
REACH = 2
#: cells around an NPC point its floor is flooded through (a house, a hall)
NPC_REACH = 48
#: how far the client floor under an NPC may lie from the NPC's height to seed the flood
NPC_SEED_DZ = 24


def load_region(reg: str) -> l2j.Region:
    path = FIXED / f"{reg}.l2j"
    return l2j.read(path if path.exists() else STAZIS / f"{reg}.l2j")


def fix_spawn_file(spawn_file: str) -> list[dict]:
    """Repair the regions of one spawn file; returns a report entry per touched region."""
    terrs = [t for t in territories() if t[0] == spawn_file]
    by_region = defaultdict(list)
    for t in terrs:
        xs = [p[0] for p in t[2]]
        ys = [p[1] for p in t[2]]
        by_region["%d_%d" % l2j.region_of(int(np.mean(xs)), int(np.mean(ys)))].append(t)
    reports = []
    for reg, ts in sorted(by_region.items()):
        if not (scene.CLIENT / "maps" / f"{reg}.unr").exists() or not (STAZIS / f"{reg}.l2j").exists():
            continue
        g = load_region(reg)
        ox, oy = g.origin
        cells: dict[tuple[int, int], int] = {}
        seeds_total = 0
        for _rel, idx, pts, lo, hi in ts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = max(((min(xs) - MARGIN) // 16) * 16, ox), max(((min(ys) - MARGIN) // 16) * 16, oy)
            x1, y1 = min(max(xs) + MARGIN, ox + l2j.REGION_SIZE), min(max(ys) + MARGIN, oy + l2j.REGION_SIZE)
            nx, ny = (x1 - x0) // 16, (y1 - y0) // 16
            if nx <= 0 or ny <= 0 or nx * ny > 600_000:
                continue
            # seeds: cells inside the polygon whose geodata has no layer in the territory's z range while the client has a floor there
            gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            px, py = x0 + gi.ravel() * 16 + 8, y0 + gj.ravel() * 16 + 8
            ins = inside(px, py, pts)
            if not ins.any():
                continue
            cand = []
            for i, j in zip(gi.ravel()[ins], gj.ravel()[ins]):
                cx, cy = (x0 - ox) // 16 + int(i), (y0 - oy) // 16 + int(j)
                k = int(g.layers[cx, cy])
                hs = g.height[cx, cy, :k]
                if not ((hs >= lo - 100) & (hs <= hi + 100)).any():
                    cand.append((int(i), int(j)))
            if not cand:
                continue
            surf = scene.client_surfaces(reg, x0, y0, nx, ny)
            seeds = []
            for i, j in cand:
                fl = [z for z in surf.walkable_bsp(i, j) if lo - 100 <= z + repair.GEO_ABOVE_FLOOR <= hi + 100]
                if fl:
                    seeds.append((i, j, repair.geo_height(min(fl, key=lambda z: abs(z - (lo + hi) / 2)))))
            if not seeds:
                continue
            seeds_total += len(seeds)
            mask = ins.reshape(nx, ny)
            allowed = mask.copy()
            for _ in range(REACH):  # grow the polygon by REACH cells
                grown = allowed.copy()
                grown[1:, :] |= allowed[:-1, :]
                grown[:-1, :] |= allowed[1:, :]
                grown[:, 1:] |= allowed[:, :-1]
                grown[:, :-1] |= allowed[:, 1:]
                allowed = grown
            flooded = repair.flood_floors(surf.walkable_bsp, seeds, (0, 0, nx, ny), (lo - 300, hi + 300), allowed)
            for (i, j), h in flooded.items():
                cells.setdefault(((x0 - ox) // 16 + i, (y0 - oy) // 16 + j), h)
        if not cells:
            continue
        stats = repair.add_layers(g, cells)
        if stats["layers_added"]:
            FIXED.mkdir(parents=True, exist_ok=True)
            l2j.write(g, FIXED / f"{reg}.l2j")
        reports.append({"spawn_file": spawn_file, "region": reg, "seeds": seeds_total, **stats})
        print(json.dumps(reports[-1]), flush=True)
    return reports


def fix_npc_points(spawn_file: str) -> list[dict]:
    """Repair the floors under the NPC points of one spawn file that have no layer near their z."""
    by_region = defaultdict(list)
    for rel, npc_id, x, y, z in npc_points():
        if rel == spawn_file:
            by_region["%d_%d" % l2j.region_of(x, y)].append((npc_id, x, y, z))
    reports = []
    for reg, points in sorted(by_region.items()):
        if not (scene.CLIENT / "maps" / f"{reg}.unr").exists() or not (STAZIS / f"{reg}.l2j").exists():
            continue
        g = load_region(reg)
        ox, oy = g.origin
        cells: dict[tuple[int, int], int] = {}
        fixed_npcs = []
        for npc_id, x, y, z in points:
            cx, cy = g.cell_of(x, y)
            k = int(g.layers[cx, cy])
            if k and np.min(np.abs(g.height[cx, cy, :k] - z)) <= NPC_DZ:
                continue
            x0 = max(((x - NPC_REACH * 16) // 16) * 16, ox)
            y0 = max(((y - NPC_REACH * 16) // 16) * 16, oy)
            n = 2 * NPC_REACH + 1
            surf = scene.client_surfaces(reg, x0, y0, n, n)
            i, j = (x - x0) // 16, (y - y0) // 16
            # The NPC stands on the geodata layer, 32 above the client's floor: a floor (BSP or a mesh top - town
            # houses are meshes) at its height is a real one, a mesh top is trusted no farther than that.
            floors = [f for f in surf.walkable(i, j) if abs(repair.geo_height(f) - z) <= NPC_SEED_DZ]
            if not floors:
                print(json.dumps({"region": reg, "npc_id": npc_id, "point": [x, y, z], "skipped": "no client floor"}))
                continue
            seed = repair.geo_height(min(floors, key=lambda f: abs(repair.geo_height(f) - z)))
            # Through mesh floors too: the doorway, steps and counters of a house are static meshes; the reach box
            # and the z band keep the flood inside the NPC's storey.
            flooded = repair.flood_floors(surf.walkable, [(i, j, seed)], (0, 0, n, n), (z - 120, z + 120))
            for (a, b), h in flooded.items():
                cells.setdefault(((x0 - ox) // 16 + a, (y0 - oy) // 16 + b), h)
            fixed_npcs.append(npc_id)
        if not cells:
            continue
        stats = repair.add_layers(g, cells)
        if stats["layers_added"]:
            FIXED.mkdir(parents=True, exist_ok=True)
            l2j.write(g, FIXED / f"{reg}.l2j")
        reports.append({"spawn_file": spawn_file, "region": reg, "npcs": fixed_npcs, **stats})
        print(json.dumps(reports[-1]), flush=True)
    return reports


def fix_interiors(spawn_file: str, margin: int = 600, band: int = 250) -> list[dict]:
    """
    Give the town's houses their floors: client floor cells whose geodata has layers only higher up, under a
    client roof (a ceiling or a mesh surface between the floor and that layer), and that border on walkable geodata at the floor's height (a doorway) seed a flood over such cells; the flooded
    floor gets layers. A house without a door to the street (no bordering cell) is left as it is.
    """
    from .townmap import PAVING, SLAB, roofed  # noqa: PLC0415

    points = [(x, y, z) for rel, _npc_id, x, y, z in npc_points() if rel == spawn_file]
    by_region = defaultdict(list)
    for p in points:
        by_region["%d_%d" % l2j.region_of(p[0], p[1])].append(p)
    reg, points = max(by_region.items(), key=lambda kv: len(kv[1]))
    g = load_region(reg)
    ox, oy = g.origin
    xs, ys, zs = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
    x0, y0 = max(((min(xs) - margin) // 16) * 16, ox), max(((min(ys) - margin) // 16) * 16, oy)
    x1, y1 = min(max(xs) + margin, ox + l2j.REGION_SIZE - 16), min(max(ys) + margin, oy + l2j.REGION_SIZE - 16)
    nx, ny = (x1 - x0) // 16, (y1 - y0) // 16
    zlo, zhi = min(zs) - band, max(zs) + band
    surf = scene.client_surfaces(reg, x0, y0, nx, ny)

    def floors(i: int, j: int) -> list[float]:
        ceil = surf.ceilings.get((i, j), [])
        return [z for z in surf.walkable(i, j) if zlo <= z <= zhi and not any(z < c <= z + SLAB for c in ceil)]

    def layers(i: int, j: int) -> np.ndarray:
        cx, cy = (x0 - ox) // 16 + i, (y0 - oy) // 16 + j
        return g.height[cx, cy, :int(g.layers[cx, cy])].astype(np.int32)

    roof_only = np.zeros((nx, ny), bool)
    hole_floor: dict[tuple[int, int], int] = {}
    for i in range(nx):
        for j in range(ny):
            hs = layers(i, j)
            if not len(hs):
                continue
            for z in floors(i, j):
                d = hs - int(z)
                if ((np.abs(d) <= 40) | ((d > 0) & (d <= PAVING))).any() or (d < -40).any():
                    continue
                if not roofed(surf, i, j, z, int(hs[d > PAVING].min())):
                    continue  # buried, not a room
                roof_only[i, j] = True
                hole_floor[(i, j)] = repair.geo_height(z)
                break
    seeds = []
    for (i, j), h in hole_floor.items():
        for _bit, dx, dy, _opp in repair.DIRECTIONS:
            a, b = i + dx, j + dy
            if 0 <= a < nx and 0 <= b < ny and not roof_only[a, b]:
                hs = layers(a, b)
                if len(hs) and (np.abs(hs - h) <= repair.MAX_STEP).any():
                    seeds.append((i, j, h))
                    break
    flooded = repair.flood_floors(floors, seeds, (0, 0, nx, ny), (zlo, zhi), roof_only) if seeds else {}
    cells = {((x0 - ox) // 16 + i, (y0 - oy) // 16 + j): h for (i, j), h in flooded.items()}
    if not cells:
        return []
    stats = repair.add_layers(g, cells)
    if stats["layers_added"]:
        FIXED.mkdir(parents=True, exist_ok=True)
        l2j.write(g, FIXED / f"{reg}.l2j")
    report = {"spawn_file": spawn_file, "region": reg, "roof_only_cells": int(roof_only.sum()),
              "doorway_seeds": len(seeds), **stats}
    print(json.dumps(report), flush=True)
    return [report]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spawn_files", nargs="+")
    ap.add_argument("--npcs", action="store_true", help="repair under the NPC points instead of the monster territories")
    ap.add_argument("--interiors", action="store_true", help="give the houses of the spawn file's town their floors")
    args = ap.parse_args()
    name = "fix_interiors.jsonl" if args.interiors else "fix_npcs.jsonl" if args.npcs else "fix_layers.jsonl"
    log = ROOT / "geodata_work" / "report" / name
    with open(log, "a") as f:
        for sf in args.spawn_files:
            fix = fix_interiors if args.interiors else fix_npc_points if args.npcs else fix_spawn_file
            for entry in fix(sf):
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
