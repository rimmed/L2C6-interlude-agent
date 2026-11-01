"""
Add the floor layers a spawn file's territories lack, from the client's BSP floors (rooms and caves: the
reliable kind; mesh tops and terrain are ambiguous outdoors) (``repair``), region by region, writing the result to ``geodata_work/fixed`` (starting from a fixed region when there is one).

Usage (from ``l2_agent/``)::

    python -m geo.fix Gludio/Wasteland.xml [more spawn files ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import l2j, repair, scene
from .scan import inside, territories

ROOT = Path(__file__).resolve().parents[2]
STAZIS = ROOT / "geodata_work" / "stazis"
FIXED = ROOT / "geodata_work" / "fixed"
MARGIN = 64
#: cells beyond a territory's polygon the added floor may reach (to join the layers around it)
REACH = 2


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spawn_files", nargs="+")
    args = ap.parse_args()
    log = ROOT / "geodata_work" / "report" / "fix_layers.jsonl"
    with open(log, "a") as f:
        for sf in args.spawn_files:
            for entry in fix_spawn_file(sf):
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
