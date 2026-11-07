"""
Map of a town's geodata against the client's floors, before and after a repair, side by side (as ``geo.compare``
draws a dungeon): one pixel per cell, north up, over the box of the town's NPC points.

Colours: black nothing in the z band, grey geodata and client agree (a layer up to ``PAVING`` over a client floor
is the street's paving over the terrain), red a client floor whose geodata has layers only higher up (a roof: a
character there is put on it), dark grey the same without a roof over it (a surface buried in a rock or under the
terrain: no defect), brown a client surface between geodata layers (a ledge, a balcony or an upper storey
without its layer), orange a client floor without a geodata layer near, blue a geodata layer without a client
floor. NPC points: white; with ``--results`` (the click test CSV) green when the
test character got the NPC's dialog, magenta when it did not.

Usage (from ``l2_agent/``)::

    python -m geo.townmap Giran/GiranNPCs.xml --before ../geodata_work/stazis OLD_FIXED --after ../geodata_work/stazis ../geodata_work/fixed
"""

from __future__ import annotations

import argparse
import csv
import struct
import zlib
from pathlib import Path

import numpy as np

from . import l2j, scene
from .compare import NEAR
from .scan import load_overlay, npc_points

#: a geodata layer at most this far above a client floor is the town's paving over it, not a roof
PAVING = 80
#: a floor with a ceiling at most this far over it lies under a slab (the town's platform), not in a room
SLAB = scene.CLEARANCE + 8
COLORS = np.array([[0, 0, 0], [90, 90, 90], [255, 40, 40], [255, 170, 0], [60, 120, 255], [150, 75, 0], [45, 45, 45]],
                  np.uint8)


def roofed(surf: scene.Surfaces, i: int, j: int, z: float, top: int) -> bool:
    """
    Whether the client has a roof over the floor ``z`` below the geodata layer ``top``: a ceiling or a mesh
    surface in between. Without one the floor is buried (a mesh top inside a rock, a plane under the terrain)
    and the layer above is the ground, not a roof.
    """
    lo, hi = z + 64, top + 40
    if any(lo < c < hi for c in surf.ceilings.get((i, j), [])):
        return True
    return any(lo < h < hi for zs, _t in (surf.spans or {}).get((i, j), []) for h in zs)
MARGIN = 600
BAND = 250
GAP = 8


def write_rgb(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def classify(region: l2j.Region, surf: scene.Surfaces, x0: int, y0: int, nx: int, ny: int, zlo: int, zhi: int):
    """
    ``geo.compare``'s classes, for a town: a client floor with a geodata layer up to ``PAVING`` above it is matched
    (the street's paving mesh over the terrain), not a hole; a hole is red only under a layer higher up (a roof).
    """
    ox, oy = region.origin
    cls = np.zeros((nx, ny), np.uint8)
    stats = {"floors": 0, "matched": 0, "under_roof": 0, "buried": 0, "upper": 0, "hole": 0, "phantom": 0}
    for i in range(nx):
        cx = (x0 - ox) // 16 + i
        for j in range(ny):
            cy = (y0 - oy) // 16 + j
            k = int(region.layers[cx, cy])
            hs = region.height[cx, cy, :k].astype(np.int32)
            ceil = surf.ceilings.get((i, j), [])
            # the terrain under a town's BSP slab (its underside right over it) is no floor; walkable() only drops
            # BSP floors under such a gap
            walk = [z for z in surf.walkable(i, j) if not any(z < c <= z + SLAB for c in ceil)]
            fl = [z for z in walk if zlo <= z <= zhi]
            worst = 1 if fl or any(zlo <= h <= zhi for h in hs) else 0
            for z in fl:
                stats["floors"] += 1
                d = hs - int(z)
                if k and ((np.abs(d) <= NEAR) | ((d > 0) & (d <= PAVING))).any():
                    stats["matched"] += 1
                elif k and (d > PAVING).any() and not (d < -NEAR).any():
                    if not roofed(surf, i, j, z, int(hs[d > PAVING].min())):
                        stats["buried"] += 1
                        worst = worst if worst != 1 else 6
                        continue
                    stats["under_roof"] += 1
                    worst = 2
                elif k and (d > PAVING).any():
                    # a ledge, balcony or upper storey over a floor the geodata has: maybe a missing upper floor
                    stats["upper"] += 1
                    worst = worst if worst in (2, 3) else 5
                else:
                    stats["hole"] += 1
                    worst = 2 if worst == 2 else 3
            for h in hs:
                if zlo <= h <= zhi and (not walk or min(abs(h - z) for z in walk) > PAVING):
                    stats["phantom"] += 1
                    if worst == 1:
                        worst = 4
            cls[i, j] = worst
    return cls, stats


def mark(img: np.ndarray, i: int, j: int, color: tuple[int, int, int], r: int = 3) -> None:
    h, w, _ = img.shape
    img[max(j - r, 0):min(j + r + 1, h), max(i - r, 0):min(i + r + 1, w)] = (0, 0, 0)
    img[max(j - r + 1, 0):min(j + r, h), max(i - r + 1, 0):min(i + r, w)] = color


def town_map(spawn_file: str, before: list[str], after: list[str], results: str | None, out: Path) -> dict:
    points = [(npc_id, x, y, z) for rel, npc_id, x, y, z in npc_points() if rel == spawn_file]
    # the region holding most of the points; the box stays inside it
    regions = {}
    for p in points:
        regions.setdefault("%d_%d" % l2j.region_of(p[1], p[2]), []).append(p)
    reg, points = max(regions.items(), key=lambda kv: len(kv[1]))
    old, new = load_overlay(before, reg), load_overlay(after, reg)
    ox, oy = new.origin
    xs, ys, zs = [p[1] for p in points], [p[2] for p in points], [p[3] for p in points]
    x0 = max(((min(xs) - MARGIN) // 16) * 16, ox)
    y0 = max(((min(ys) - MARGIN) // 16) * 16, oy)
    x1 = min(max(xs) + MARGIN, ox + l2j.REGION_SIZE - 16)
    y1 = min(max(ys) + MARGIN, oy + l2j.REGION_SIZE - 16)
    nx, ny = (x1 - x0) // 16, (y1 - y0) // 16
    zlo, zhi = min(zs) - BAND, max(zs) + BAND
    surf = scene.client_surfaces(reg, x0, y0, nx, ny)
    cls_old, stats_old = classify(old, surf, x0, y0, nx, ny, zlo, zhi)
    cls_new, stats_new = classify(new, surf, x0, y0, nx, ny, zlo, zhi)
    tested = {}
    if results:
        for row in csv.reader(open(results)):
            if row and row[0] == spawn_file:
                tested[(int(row[1]), int(row[2]), int(row[3]))] = row[5] == "dialog"
    panels = []
    for cls in (cls_old, cls_new):
        img = COLORS[cls.T].copy()
        for npc_id, x, y, _z in points:
            ok = tested.get((npc_id, x, y))
            mark(img, (x - x0) // 16, (y - y0) // 16, (255, 255, 255) if ok is None else (0, 220, 0) if ok else (255, 0, 255))
        panels.append(img)
    gap = np.full((ny, GAP, 3), 255, np.uint8)
    write_rgb(out, np.concatenate([panels[0], gap, panels[1]], axis=1))
    return {"spawn_file": spawn_file, "region": reg, "box": [x0, y0, x1, y1], "z": [zlo, zhi],
            "before": stats_old, "after": stats_new, "png": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spawn_files", nargs="+")
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--results", help="click test CSV")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "geodata_work" / "report" / "towns"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sf in args.spawn_files:
        name = Path(sf).stem
        info = town_map(sf, args.before, args.after, args.results, out_dir / f"{name}.png")
        print(info, flush=True)


if __name__ == "__main__":
    main()
