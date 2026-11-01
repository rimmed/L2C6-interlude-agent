"""
Compare the geodata of a dungeon with the client's floors: cells where the client shows a floor but the
geodata has no layer there (the server then puts a character on the nearest other layer - above it:
"thrown out of the cave"), and geodata layers with no floor in the client.

Usage (from ``l2_agent/``)::

    python -m geo.compare DevilsIsle 21_24 [--png out.png]
"""

from __future__ import annotations

import argparse
import glob
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

from . import l2j, scene

DATA = Path(__file__).resolve().parents[2] / "L2J_Mobius" / "L2J_Mobius_CT_0_Interlude" / "dist" / "game" / "data"
NEAR = 40  # a geodata layer this close to a client floor matches it (heights are stored in steps of 8)


def spawn_box(name: str, margin: int = 500) -> tuple[int, int, int, int, int, int]:
    """x0, y0, x1, y1, min z, max z of the spawn territories of ``spawns/**/<name>.xml``."""
    f = glob.glob(str(DATA / "spawns" / "**" / f"{name}.xml"), recursive=True)[0]
    xs, ys, zs = [], [], []
    for t in ET.parse(f).getroot().iter("territory"):
        for n in t.findall("node"):
            xs.append(int(n.get("x")))
            ys.append(int(n.get("y")))
        zs += [int(t.get("minZ")), int(t.get("maxZ"))]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin, min(zs), max(zs)


def compare(region: l2j.Region, surf: scene.Surfaces, x0: int, y0: int, nx: int, ny: int, zlo: int, zhi: int):
    """
    Per cell: 0 nothing in the z band, 1 geodata and client agree, 2 a client floor without a geodata layer
    while the geodata has one above it (a character there is put on that one: thrown out), 3 a client
    floor without any geodata layer near, 4 a geodata layer without a client floor near.
    """
    ox, oy = region.origin
    cls = np.zeros((nx, ny), np.uint8)
    stats = {"floors": 0, "matched": 0, "hole_above": 0, "hole": 0, "layers": 0, "phantom": 0}
    for i in range(nx):
        cx = (x0 - ox) // 16 + i
        for j in range(ny):
            cy = (y0 - oy) // 16 + j
            k = int(region.layers[cx, cy])
            hs = region.height[cx, cy, :k]
            walk = surf.walkable(i, j)
            fl = [z for z in walk if zlo <= z <= zhi]
            worst = 1 if fl or any(zlo <= h <= zhi for h in hs) else 0
            for z in fl:
                stats["floors"] += 1
                if k and np.min(np.abs(hs - z)) <= NEAR:
                    stats["matched"] += 1
                elif k and (hs > z + NEAR).any():
                    stats["hole_above"] += 1
                    worst = 2
                else:
                    stats["hole"] += 1
                    worst = 2 if worst == 2 else 3
            for h in hs:
                if zlo <= h <= zhi:
                    stats["layers"] += 1
                    if not walk or min(abs(h - z) for z in walk) > NEAR:
                        stats["phantom"] += 1
                        if worst == 1:
                            worst = 4
            cls[i, j] = worst
    return cls, stats


def write_png(path: Path, cls: np.ndarray) -> None:
    """Map of the classification: x to the right, y down (north up), one pixel per cell."""
    colors = np.array([[0, 0, 0], [90, 90, 90], [255, 40, 40], [255, 170, 0], [60, 120, 255]], np.uint8)
    img = colors[cls.T]  # rows = y
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spawns")
    ap.add_argument("region")
    ap.add_argument("--geodata", default=str(Path(__file__).resolve().parents[2] / "geodata_work" / "stazis"))
    ap.add_argument("--png")
    ap.add_argument("--zlo", type=int)
    ap.add_argument("--zhi", type=int)
    args = ap.parse_args()
    x0, y0, x1, y1, zmin, zmax = spawn_box(args.spawns)
    x0, y0 = (x0 // 16) * 16, (y0 // 16) * 16
    nx, ny = (x1 - x0) // 16, (y1 - y0) // 16
    zlo = args.zlo if args.zlo is not None else zmin - 300
    zhi = args.zhi if args.zhi is not None else zmax + 300
    region = l2j.read(Path(args.geodata) / f"{args.region}.l2j")
    surf = scene.client_surfaces(args.region, x0, y0, nx, ny)
    cls, stats = compare(region, surf, x0, y0, nx, ny, zlo, zhi)
    print(f"{args.spawns} {args.region} box x {x0}..{x0 + nx * 16} y {y0}..{y0 + ny * 16} z {zlo}..{zhi}")
    print(stats)
    print("cells: matched", int((cls == 1).sum()), "hole under a layer", int((cls == 2).sum()), "hole", int((cls == 3).sum()),
          "phantom", int((cls == 4).sum()))
    if args.png:
        write_png(Path(args.png), cls)


if __name__ == "__main__":
    main()
