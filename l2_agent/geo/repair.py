"""
Repair geodata from the client's geometry: add the floor layers the geodata lacks (a cave under an
island that only has the island's surface) and connect them to the existing layers.

The geodata stores a floor 32 above the client's floor polygon (measured on matching cells), in steps
of 8. Two neighbouring layers are connected when their heights differ by at most ``MAX_STEP`` (the
server refuses a step up of more than 40).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import l2j

GEO_ABOVE_FLOOR = 32
MAX_STEP = 40
# direction bit, dx, dy, opposite bit (x east, y south)
DIRECTIONS = ((l2j.EAST, 1, 0, l2j.WEST), (l2j.WEST, -1, 0, l2j.EAST), (l2j.SOUTH, 0, 1, l2j.NORTH), (l2j.NORTH, 0, -1, l2j.SOUTH))


def geo_height(floor_z: float) -> int:
    return int(round((floor_z + GEO_ABOVE_FLOOR) / 8.0)) * 8


def flood_floors(walkable, seeds: list[tuple[int, int, int]], box: tuple[int, int, int, int],
                 band: tuple[int, int], allowed=None) -> dict[tuple[int, int], int]:
    """
    Geodata heights of the client floor reachable from ``seeds`` ((i, j, geodata height) cells of the
    surfaces grid) through neighbours whose floor is at most ``MAX_STEP`` away, inside ``box``
    (i0, j0, i1, j1) and the z ``band``, and only through cells ``allowed`` (a boolean array over the box,
    ``None`` = all). ``walkable(i, j)`` lists the client floors of a cell.
    """
    i0, j0, i1, j1 = box
    out: dict[tuple[int, int], int] = {}
    queue = deque()
    for i, j, h in seeds:
        if (i, j) not in out:
            out[(i, j)] = h
            queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        h = out[(i, j)]
        for _bit, dx, dy, _opp in DIRECTIONS:
            a, b = i + dx, j + dy
            if not (i0 <= a < i1 and j0 <= b < j1) or (a, b) in out or (allowed is not None and not allowed[a, b]):
                continue
            cands = [geo_height(z) for z in walkable(a, b) if band[0] <= z <= band[1]]
            cands = [c for c in cands if abs(c - h) <= MAX_STEP]
            if cands:
                out[(a, b)] = min(cands, key=lambda c: abs(c - h))
                queue.append((a, b))
    return out


def add_layers(region: l2j.Region, cells: dict[tuple[int, int], int], skip_near: int = 48) -> dict[str, int]:
    """
    Add a layer at the given height to every region cell ``(cx, cy)`` of ``cells`` that has no layer
    within ``skip_near`` of it (an existing layer a bit off the client floor stays the only one), then set
    the passable directions of the new layers and of their neighbours' layers towards them. Returns counts.
    """
    added = {}
    for (cx, cy), h in cells.items():
        k = int(region.layers[cx, cy])
        hs = region.height[cx, cy, :k]
        if k and np.min(np.abs(hs - h)) <= skip_near:
            continue
        added[(cx, cy)] = h
    need = int(max([region.layers[c] + 1 for c in added] + [region.height.shape[2]]))
    if need > region.height.shape[2]:
        pad = need - region.height.shape[2]
        region.height = np.concatenate([region.height, np.zeros(region.height.shape[:2] + (pad,), region.height.dtype)], 2)
        region.nswe = np.concatenate([region.nswe, np.zeros(region.nswe.shape[:2] + (pad,), region.nswe.dtype)], 2)
    for (cx, cy), h in added.items():
        k = int(region.layers[cx, cy])
        hs = list(region.height[cx, cy, :k]) + [h]
        ns = list(region.nswe[cx, cy, :k]) + [0]
        order = np.argsort(hs)[::-1]  # top layer first, as the files store them
        region.height[cx, cy, :k + 1] = np.array(hs)[order]
        region.nswe[cx, cy, :k + 1] = np.array(ns)[order]
        region.layers[cx, cy] = k + 1
    opened = 0
    for (cx, cy), h in added.items():
        li = _layer(region, cx, cy, h)
        bits = 0
        for bit, dx, dy, opp in DIRECTIONS:
            a, b = cx + dx, cy + dy
            if not (0 <= a < l2j.REGION_CELLS and 0 <= b < l2j.REGION_CELLS):
                continue
            n = int(region.layers[a, b])
            if not n:
                continue
            hs = region.height[a, b, :n]
            m = int(np.argmin(np.abs(hs - h)))
            if abs(int(hs[m]) - h) <= MAX_STEP:
                bits |= bit
                if not region.nswe[a, b, m] & opp:
                    region.nswe[a, b, m] |= opp
                    opened += 1
        region.nswe[cx, cy, li] = bits
    return {"cells": len(cells), "layers_added": len(added), "neighbour_directions_opened": opened}


def _layer(region: l2j.Region, cx: int, cy: int, h: int) -> int:
    k = int(region.layers[cx, cy])
    return int(np.argmin(np.abs(region.height[cx, cy, :k] - h)))
