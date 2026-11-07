"""
Scan every monster spawn territory for points without a geodata layer inside the territory's z range:
the server would put a monster (and the agent sent next to it) on another layer - the "thrown out of a
cave" defect. Writes a CSV per spawn territory and prints the worst spawn files.

``--npcs`` scans the fixed spawn points of the other NPCs instead (merchants, guards, teleporters): a point
without a layer near its z (an NPC inside a building whose geodata has only the roof) cannot be walked to.

Usage (from ``l2_agent/``)::

    python -m geo.scan [--geodata ../geodata_work/stazis] [--out ../geodata_work/report/scan_layers.csv]
    python -m geo.scan --npcs --geodata ../geodata_work/stazis ../geodata_work/fixed
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


#: an NPC point counts as covered when a layer lies within this of its z
NPC_DZ = 48


def npc_points():
    """(spawn file, npc id, x, y, z) of the fixed spawn points of the NPCs that are not monsters."""
    mids = monster_ids()
    for f in sorted(glob.glob(str(DATA / "spawns" / "**" / "*.xml"), recursive=True)):
        rel = os.path.relpath(f, DATA / "spawns")
        root = ET.parse(f).getroot()
        if root.get("enabled", "true").lower() == "false":
            continue
        for n in root.iter("npc"):
            if n.get("x") is not None and int(n.get("id")) not in mids:
                yield rel, int(n.get("id")), int(n.get("x")), int(n.get("y")), int(n.get("z"))


def load_overlay(folders: list[str], key: str) -> l2j.Region | None:
    """Region ``key`` from the last of ``folders`` that has it (as the server overlays them)."""
    for folder in reversed(folders):
        path = Path(folder) / f"{key}.l2j"
        if path.exists():
            return l2j.read(path)
    return None


def scan_npcs(folders: list[str], out: str) -> None:
    regions: dict[str, l2j.Region | None] = {}
    rows, per_file = [], defaultdict(lambda: [0, 0])
    for rel, npc_id, x, y, z in npc_points():
        key = "%d_%d" % l2j.region_of(x, y)
        if key not in regions:
            regions[key] = load_overlay(folders, key)
        r = regions[key]
        if r is None:
            continue
        cx, cy = r.cell_of(x, y)
        k = int(r.layers[cx, cy])
        hs = r.height[cx, cy, :k]
        dz = int(np.min(np.abs(hs - z))) if k else 9999
        rows.append((rel, npc_id, key, x, y, z, dz, " ".join(str(int(h)) for h in hs)))
        per_file[rel][0] += 1
        per_file[rel][1] += dz > NPC_DZ
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spawn_file", "npc_id", "region", "x", "y", "z", "dz", "layers"])
        w.writerows(rows)
    bad = sum(m for _, m in per_file.values())
    print(f"{len(rows)} NPC points in {len(per_file)} spawn files; without a layer within {NPC_DZ} of their z: {bad}")
    for f, (n, m) in sorted(per_file.items(), key=lambda kv: -kv[1][1])[:25]:
        if m:
            print(f"  {m:4d}/{n:<4d} {f}")


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
    report = Path(__file__).resolve().parents[2] / "geodata_work" / "report"
    ap.add_argument("--geodata", nargs="+", default=[str(Path(__file__).resolve().parents[2] / "geodata_work" / "stazis")],
                    help="geodata folders, later ones replacing regions of earlier ones")
    ap.add_argument("--out", help="CSV (default report/scan_layers.csv, or report/scan_npcs.csv with --npcs)")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--npcs", action="store_true", help="scan the fixed NPC spawn points instead of monster territories")
    args = ap.parse_args()
    if args.npcs:
        scan_npcs(args.geodata, args.out or str(report / "scan_npcs.csv"))
        return
    args.out = args.out or str(report / "scan_layers.csv")
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
                regions[key] = load_overlay(args.geodata, key)
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
