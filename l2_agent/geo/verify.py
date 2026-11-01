"""
Check the territories the scan flagged (``scan_layers.csv``) against the client: a point without a geodata
layer in the territory's z range is a geodata defect when the client has a walkable floor there, and only
loose territory bounds when it has none.

Usage (from ``l2_agent/``)::

    python -m geo.verify [--min-missing 5]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import l2j, scene

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=str(ROOT / "geodata_work" / "report" / "scan_layers.csv"))
    ap.add_argument("--geodata", default=str(ROOT / "geodata_work" / "stazis"))
    ap.add_argument("--out", default=str(ROOT / "geodata_work" / "report" / "verify_layers.csv"))
    ap.add_argument("--min-missing", type=int, default=5)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.scan)) if int(r["no_layer_in_z"]) >= args.min_missing]
    rows.sort(key=lambda r: r["region"])
    out, per_file = [], defaultdict(lambda: [0, 0, 0])
    rng = np.random.default_rng(1)
    regions: dict[str, l2j.Region] = {}
    for r in rows:
        x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
        lo, hi = int(r["min_z"]), int(r["max_z"])
        reg = r["region"]
        if not (scene.CLIENT / "maps" / f"{reg}.unr").exists() or not (Path(args.geodata) / f"{reg}.l2j").exists():
            continue
        if reg not in regions:
            regions.clear()
            regions[reg] = l2j.read(Path(args.geodata) / f"{reg}.l2j")
        g = regions[reg]
        # the territory's nodes are not in the CSV: use its box, and points of it whose region is this one
        bx0, by0 = (x0 // 16) * 16, (y0 // 16) * 16
        nx, ny = (x1 - bx0) // 16 + 1, (y1 - by0) // 16 + 1
        if nx * ny > 400_000:
            continue
        surf = scene.client_surfaces(reg, bx0, by0, nx, ny)
        defect = loose = 0
        cells = []
        for _ in range(600):
            i, j = int(rng.integers(0, nx)), int(rng.integers(0, ny))
            x, y = bx0 + i * 16 + 8, by0 + j * 16 + 8
            if l2j.region_of(x, y) != tuple(map(int, reg.split("_"))):
                continue
            cx, cy = g.cell_of(x, y)
            k = int(g.layers[cx, cy])
            hs = g.height[cx, cy, :k]
            if ((hs >= lo - 100) & (hs <= hi + 100)).any():
                continue
            walk = [z for z in surf.walkable(i, j) if lo - 100 <= z + 32 <= hi + 100]
            if walk:
                defect += 1
                cells.append((x, y, int(walk[0])))
            else:
                loose += 1
        out.append((r["spawn_file"], r["territory"], reg, x0, y0, x1, y1, lo, hi, defect, loose,
                    ";".join(f"{x},{y},{z}" for x, y, z in cells[:5])))
        per_file[r["spawn_file"]][0] += defect
        per_file[r["spawn_file"]][1] += loose
        print(f"{r['spawn_file']} #{r['territory']} {reg}: client floor but no layer {defect}, no floor in z either {loose}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spawn_file", "territory", "region", "x0", "y0", "x1", "y1", "min_z", "max_z", "defect_points", "loose_points", "examples"])
        w.writerows(out)
    print("\nper spawn file (defect points / loose-bounds points):")
    for f, (d, loose, _) in sorted(per_file.items(), key=lambda kv: -kv[1][0]):
        print(f"  {d:5d} / {loose:5d}  {f}")


if __name__ == "__main__":
    main()
