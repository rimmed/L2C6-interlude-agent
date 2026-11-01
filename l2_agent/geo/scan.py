"""
Scan every monster spawn territory for points without a geodata layer inside the territory's z range:
the server would put a monster (and the agent sent next to it) on another layer - the "thrown out of a
cave" defect. Writes a CSV per spawn territory and prints the worst spawn files.

Usage (from ``l2_agent/``)::

    python -m geo.scan [--geodata ../geodata_work/stazis] [--out ../geodata_work/report/scan_layers.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import l2j
from .compare import DATA


def monster_ids() -> set[int]:
    ids = set()
    for f in glob.glob(str(DATA / "stats" / "npcs" / "*.xml")):
        for n in ET.parse(f).getroot().iter("npc"):
            if n.get("type") == "Monster":
                ids.add(int(n.get("id")))
    return ids


def inside(x: np.ndarray, y: np.ndarray, pts: list[tuple[int, int]]) -> np.ndarray:
    c = np.zeros(len(x), bool)
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[i - 1]
        if y1 == y2:
            continue
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        c ^= cross
    return c


def territories():
    """(spawn file, index, nodes, min z, max z) of the territories that spawn monsters."""
    mids = monster_ids()
    for f in sorted(glob.glob(str(DATA / "spawns" / "**" / "*.xml"), recursive=True)):
        rel = os.path.relpath(f, DATA / "spawns")
        root = ET.parse(f).getroot()
        i = 0
        for parent in root.iter():
            terrs = parent.findall("territories/territory") + parent.findall("territory")
            if not terrs:
                continue
            if not any(int(n.get("id")) in mids for n in parent.iter("npc")):
                continue
            for t in terrs:
                pts = [(int(n.get("x")), int(n.get("y"))) for n in t.findall("node")]
                if len(pts) >= 3 and t.get("minZ") is not None:
                    yield rel, i, pts, int(t.get("minZ")), int(t.get("maxZ"))
                    i += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geodata", default=str(Path(__file__).resolve().parents[2] / "geodata_work" / "stazis"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "geodata_work" / "report" / "scan_layers.csv"))
    ap.add_argument("--samples", type=int, default=300)
    args = ap.parse_args()
    regions: dict[str, l2j.Region | None] = {}
    rng = np.random.default_rng(0)
    rows = []
    per_file = defaultdict(lambda: [0, 0])
    for rel, idx, pts, lo, hi in territories():
        xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        x = rng.integers(xs.min(), xs.max() + 1, args.samples)
        y = rng.integers(ys.min(), ys.max() + 1, args.samples)
        keep = inside(x, y, pts)
        x, y = x[keep], y[keep]
        if not len(x):
            continue
        missing = 0
        for px, py in zip(x, y):
            rx, ry = l2j.region_of(int(px), int(py))
            key = f"{rx}_{ry}"
            if key not in regions:
                path = Path(args.geodata) / f"{key}.l2j"
                regions[key] = l2j.read(path) if path.exists() else None
            r = regions[key]
            if r is None:
                continue
            cx, cy = r.cell_of(int(px), int(py))
            k = int(r.layers[cx, cy])
            hs = r.height[cx, cy, :k]
            if not ((hs >= lo - 100) & (hs <= hi + 100)).any():
                missing += 1
        region = "%d_%d" % l2j.region_of(int(xs.mean()), int(ys.mean()))
        rows.append((rel, idx, region, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), lo, hi, len(x), missing))
        per_file[rel][0] += len(x)
        per_file[rel][1] += missing
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spawn_file", "territory", "region", "x0", "y0", "x1", "y1", "min_z", "max_z", "points", "no_layer_in_z"])
        w.writerows(rows)
    worst = sorted(((m / n, m, n, f) for f, (n, m) in per_file.items() if n), reverse=True)
    print(f"{len(rows)} territories in {len(per_file)} spawn files; points without a layer in z: "
          f"{sum(m for _, m, _, _ in worst)} of {sum(n for _, _, n, _ in worst)}")
    for share, m, n, f in worst[:25]:
        if m:
            print(f"  {share:6.1%} {m:6d}/{n:<6d} {f}")


if __name__ == "__main__":
    main()
