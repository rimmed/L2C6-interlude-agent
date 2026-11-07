"""
Path finding on the geodata (``l2jc6_geodata.parquet``, see ``l2_NN/tools/geodata2parquet.py``).

The server finds no path for some walks a player can make (into a shop whose counter shuts the merchant in, out of
a building whose cell holds only the roof, around the village fence), so the agent plans its own and sends the turns
as ordinary moves. The cells of the box around start and goal become a graph, built with numpy: a node per layer of
a cell, an edge per step the server allows - out of a cell only in its open directions (NSWE), onto the neighbour's
layer nearest in height, at most ``MAX_STEP`` up or down - and scipy's breadth-first search finds the way; straight
legs between its turns make the route.

The geodata has the sea bottom as ground, but a character in the water swims and cannot climb a steep bank, so the
paths never go into the water zones (``l2jc6_zones.parquet``); a character already in the water may only leave it.
Regions are read from the table when a path needs them, one row group each.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from .knowledge import DATA_DIR, Point

log = logging.getLogger(__name__)

CELL = 16
REGION_CELLS = 2048
TILE_ZERO_X, TILE_ZERO_Y = 20, 18
EAST, WEST, SOUTH, NORTH = 1, 2, 4, 8
#: the highest step (world units) between neighbour cells the character can climb
MAX_STEP = 48
#: cells of room around start and goal in the searched box; a failed search tries again with this many times more
MARGIN = 96
GROW = 3
#: a goal cell may be wrong (a roof): the reachable cell nearest to it within this many cells will do
GOAL_CELLS = 8
#: regions kept in memory
CACHE = 16
_MOVES = ((1, 0, EAST, WEST), (-1, 0, WEST, EAST), (0, 1, SOUTH, NORTH), (0, -1, NORTH, SOUTH))


class GeoMap:
    """The cell grid of the regions in the geodata table."""

    def __init__(self, path: Path | str = DATA_DIR / "l2jc6_geodata.parquet",
                 zones: Path | str | None = DATA_DIR / "l2jc6_zones.parquet"):
        self.file = pq.ParquetFile(path)
        names = self.file.read(columns=["rx", "ry"]).to_pylist()
        groups = [g for g in range(self.file.num_row_groups) for _ in range(self.file.metadata.row_group(g).num_rows)]
        offsets = [i - groups.index(g) for i, g in enumerate(groups)]
        #: (rx, ry) -> (row group, row in it) of the region
        self.index = {(r["rx"], r["ry"]): (groups[i], offsets[i]) for i, r in enumerate(names)}
        self._regions: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.water = _water_zones(zones) if zones is not None and Path(zones).exists() else []
        self._wet: dict[tuple[int, int, int], bool] = {}
        log.info("Geodata: %d region(s), %d water zone(s)", len(self.index), len(self.water))

    # ------------------------------------------------------------------
    # Regions and cells
    # ------------------------------------------------------------------

    def region(self, rx: int, ry: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(layers, height, nswe) arrays of region ``rx_ry``, read on first use."""
        key = (rx, ry)
        if key in self._regions:
            self._regions.move_to_end(key)
            return self._regions[key]
        where = self.index.get(key)
        if where is None:
            return None
        group, offset = where
        row = self.file.read_row_group(group).slice(offset, 1).to_pylist()[0]
        k, shape = row["max_layers"], (REGION_CELLS, REGION_CELLS)
        data = (np.frombuffer(row["layers"], np.uint8).reshape(shape),
                np.frombuffer(row["height"], "<i2").reshape(*shape, k),
                np.frombuffer(row["nswe"], np.uint8).reshape(*shape, k))
        self._regions[key] = data
        while len(self._regions) > CACHE:
            self._regions.popitem(last=False)
        return data

    @staticmethod
    def cell(x: float, y: float) -> tuple[int, int]:
        return int(x) // CELL, int(y) // CELL

    @staticmethod
    def centre(cx: int, cy: int) -> tuple[int, int]:
        return cx * CELL + CELL // 2, cy * CELL + CELL // 2

    @staticmethod
    def _region_of(cx: int, cy: int) -> tuple[int, int]:
        return (cx >> 11) + TILE_ZERO_X, (cy >> 11) + TILE_ZERO_Y

    def covers(self, x: float, y: float) -> bool:
        return self._region_of(*self.cell(x, y)) in self.index

    def layer(self, cx: int, cy: int, z: float) -> tuple[int, int] | None:
        """(height, nswe) of the layer of cell ``(cx, cy)`` nearest to ``z``; ``None`` outside the map or no layer."""
        region = self.region(*self._region_of(cx, cy))
        if region is None:
            return None
        layers, height, nswe = region
        lx, ly = cx & 2047, cy & 2047
        n = int(layers[lx, ly])
        if n == 0:
            return None
        h = height[lx, ly, :n].astype(np.int32)
        i = int(np.argmin(np.abs(h - int(z)))) if n > 1 else 0
        return int(h[i]), int(nswe[lx, ly, i])

    def _step(self, cx: int, cy: int, z: int, dx: int, dy: int) -> int | None:
        """Height after one step from cell ``(cx, cy)`` at ``z`` by ``(dx, dy)``, ``None`` when it is blocked."""
        here = self.layer(cx, cy, z)
        if here is None:
            return None
        for mx, my, out, into in _MOVES:
            if (mx, my) == (dx, dy):
                if not here[1] & out:
                    return None
                there = self.layer(cx + dx, cy + dy, here[0])
                if there is None or not there[1] & into or abs(there[0] - here[0]) > MAX_STEP:
                    return None
                if self.wet(cx + dx, cy + dy, there[0]) and not self.wet(cx, cy, here[0]):
                    return None  # no way into the water
                return there[0]
        # Diagonal: both ways round the corner must be open.
        a = self._step(cx, cy, z, dx, 0)
        b = self._step(cx, cy, z, 0, dy)
        if a is None or b is None:
            return None
        c = self._step(cx + dx, cy, a, 0, dy)
        d = self._step(cx, cy + dy, b, dx, 0)
        return c if c is not None and d is not None else None

    def wet(self, cx: int, cy: int, height: int) -> bool:
        """Whether a character on the layer at ``height`` of cell ``(cx, cy)`` is in a water zone (swims)."""
        key = (cx, cy, height >> 3)
        wet = self._wet.get(key)
        if wet is None:
            x, y = self.centre(cx, cy)
            wet = any(zone.contains(x, y, height) for zone in self.water)
            self._wet[key] = wet
        return wet

    # ------------------------------------------------------------------
    # The grid of a box
    # ------------------------------------------------------------------

    def grid(self, cx0: int, cy0: int, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(layers (nx, ny), height (nx, ny, k), nswe (nx, ny, k)) of the cells of a box, across regions."""
        parts = []
        for rx in range((cx0 >> 11), ((cx0 + nx - 1) >> 11) + 1):
            for ry in range((cy0 >> 11), ((cy0 + ny - 1) >> 11) + 1):
                region = self.region(rx + TILE_ZERO_X, ry + TILE_ZERO_Y)
                if region is not None:
                    parts.append((rx, ry, region))
        k = max((p[2][1].shape[2] for p in parts), default=1)
        layers = np.zeros((nx, ny), np.uint8)
        height = np.zeros((nx, ny, k), np.int32)
        nswe = np.zeros((nx, ny, k), np.uint8)
        for rx, ry, (rl, rh, rn) in parts:
            ox, oy = rx << 11, ry << 11
            ax, bx = max(cx0, ox), min(cx0 + nx, ox + REGION_CELLS)
            ay, by = max(cy0, oy), min(cy0 + ny, oy + REGION_CELLS)
            if ax >= bx or ay >= by:
                continue
            dst = (slice(ax - cx0, bx - cx0), slice(ay - cy0, by - cy0))
            src = (slice(ax - ox, bx - ox), slice(ay - oy, by - oy))
            layers[dst] = rl[src]
            height[dst][..., :rh.shape[2]] = rh[src]
            nswe[dst][..., :rn.shape[2]] = rn[src]
        return layers, height, nswe

    def _wet_grid(self, cx0: int, cy0: int, nx: int, ny: int, height: np.ndarray) -> np.ndarray | None:
        """Per node (nx, ny, k): in a water zone. ``None`` when no water zone touches the box."""
        x0, y0 = cx0 * CELL, cy0 * CELL
        x1, y1 = x0 + nx * CELL, y0 + ny * CELL
        zones = [z for z in self.water if z.box[0] <= x1 and z.box[1] >= x0 and z.box[2] <= y1 and z.box[3] >= y0]
        if not zones:
            return None
        xs = (cx0 + np.arange(nx)) * CELL + CELL // 2
        ys = (cy0 + np.arange(ny)) * CELL + CELL // 2
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        wet = np.zeros(height.shape, bool)
        for zone in zones:
            inside = zone.contains_many(gx, gy)
            if inside.any():
                wet |= inside[..., None] & (height >= zone.min_z) & (height <= zone.max_z)
        return wet

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def find_path(self, start: Point, goal: Point) -> list[Point] | None:
        """
        Turn points from ``start`` to ``goal`` (to the reachable cell nearest to the goal within ``GOAL_CELLS`` when
        the goal's own cell is out of reach), or ``None`` when there is no way.
        """
        if not (self.covers(start.x, start.y) and self.covers(goal.x, goal.y)):
            return None
        s, g = self.cell(start.x, start.y), self.cell(goal.x, goal.y)
        margin = MARGIN
        for _ in range(3):
            cells = self._search(s, start.z, g, goal.z, margin)
            if cells is not None:
                return self._turns(cells) if len(cells) > 1 else [Point(*self.centre(*cells[0][0]), cells[0][1])]
            margin *= GROW
        log.info("Geodata: no way from %s to %s", start, goal)
        return None

    def _search(self, s: tuple[int, int], sz: float, g: tuple[int, int], gz: float, margin: int):
        cx0, cy0 = min(s[0], g[0]) - margin, min(s[1], g[1]) - margin
        nx, ny = abs(s[0] - g[0]) + 2 * margin + 1, abs(s[1] - g[1]) + 2 * margin + 1
        layers, height, nswe = self.grid(cx0, cy0, nx, ny)
        k = height.shape[2]
        valid = np.arange(k)[None, None, :] < layers[..., None]
        index = np.full(valid.shape, -1, np.int64)
        index[valid] = np.arange(int(valid.sum()))
        wet = self._wet_grid(cx0, cy0, nx, ny, height)
        far = np.where(valid, height, 1 << 20)  # an empty layer never is the nearest
        src_all, dst_all = [], []
        for dx, dy, out, into in _MOVES:
            a = (slice(max(-dx, 0), nx - max(dx, 0)), slice(max(-dy, 0), ny - max(dy, 0)))
            b = (slice(max(dx, 0), nx - max(-dx, 0)), slice(max(dy, 0), ny - max(-dy, 0)))
            h, open_out = height[a], (nswe[a] & out) > 0
            # the neighbour's layer nearest in height to each layer here
            nearest = np.argmin(np.abs(far[b][:, :, None, :] - h[..., None]), axis=3)
            hn = np.take_along_axis(height[b], nearest, axis=2)
            nn = np.take_along_axis(nswe[b], nearest, axis=2)
            ok = valid[a] & open_out & (layers[b][..., None] > 0) & ((nn & into) > 0) & (np.abs(hn - h) <= MAX_STEP)
            if wet is not None:
                ok &= ~(np.take_along_axis(wet[b], nearest, axis=2) & ~wet[a])
            src = index[a][ok]
            dst = np.take_along_axis(index[b], nearest, axis=2)[ok]
            src_all.append(src)
            dst_all.append(dst)
        src, dst = np.concatenate(src_all), np.concatenate(dst_all)
        n_nodes = int(valid.sum())
        graph = csr_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n_nodes, n_nodes))
        si, sj = s[0] - cx0, s[1] - cy0
        if not layers[si, sj]:
            return None
        sl = int(np.argmin(np.abs(np.where(valid[si, sj], height[si, sj], 1 << 20) - int(sz))))
        start_node = int(index[si, sj, sl])
        order, pred = breadth_first_order(graph, start_node, directed=True, return_predecessors=True)
        reached = np.zeros(n_nodes, bool)
        reached[order] = True
        # the reached node nearest to the goal (and to its height) within GOAL_CELLS
        gi, gj = g[0] - cx0, g[1] - cy0
        best, best_key = -1, None
        for di in range(-GOAL_CELLS, GOAL_CELLS + 1):
            for dj in range(-GOAL_CELLS, GOAL_CELLS + 1):
                i, j = gi + di, gj + dj
                if not (0 <= i < nx and 0 <= j < ny) or di * di + dj * dj > GOAL_CELLS * GOAL_CELLS:
                    continue
                for li in range(int(layers[i, j])):
                    node = int(index[i, j, li])
                    if reached[node]:
                        key = (di * di + dj * dj, abs(int(height[i, j, li]) - int(gz)))
                        if best_key is None or key < best_key:
                            best, best_key = node, key
        if best < 0:
            return None
        coords = np.argwhere(valid)  # node -> (i, j, l), in index order
        cells = []
        node = best
        while node >= 0:
            i, j, li = coords[node]
            cells.append(((cx0 + int(i), cy0 + int(j)), int(height[i, j, li])))
            node = int(pred[node]) if node != start_node else -1
        cells.reverse()
        return cells

    def _turns(self, cells: list[tuple[tuple[int, int], int]]) -> list[Point]:
        """The path's cells cut down to the points where a straight walk would leave the free cells."""
        points: list[Point] = []
        i = 0
        while i < len(cells) - 1:
            j = min(len(cells) - 1, i + 160)
            while j > i + 1 and not self._straight(cells[i], cells[j]):
                j = max(i + 1, j - 8) if j - i > 16 else j - 1
            (cx, cy), z = cells[j]
            x, y = self.centre(cx, cy)
            points.append(Point(x, y, z))
            i = j
        return points

    def _straight(self, a: tuple[tuple[int, int], int], b: tuple[tuple[int, int], int]) -> bool:
        """Whether a straight walk from cell ``a`` to cell ``b`` stays on open cells."""
        (x0, y0), z = a
        (x1, y1), _ = b
        n = max(abs(x1 - x0), abs(y1 - y0))
        cx, cy = x0, y0
        for k in range(1, n + 1):
            nx = x0 + round((x1 - x0) * k / n)
            ny = y0 + round((y1 - y0) * k / n)
            z = self._step(cx, cy, z, nx - cx, ny - cy)
            if z is None:
                return False
            cx, cy = nx, ny
        return True


class _Zone:
    """A water zone: a box, a polygon or a cylinder between ``min_z`` and ``max_z`` (the surface)."""

    def __init__(self, shape: str, nodes: list, min_z: int, max_z: int, radius: int | None):
        self.shape, self.min_z, self.max_z, self.radius = shape, min_z, max_z, radius or 0
        self.nodes = [(int(x), int(y)) for x, y in nodes]
        xs, ys = [n[0] for n in self.nodes], [n[1] for n in self.nodes]
        r = self.radius if shape == "Cylinder" else 0
        self.box = (min(xs) - r, max(xs) + r, min(ys) - r, max(ys) + r)

    def contains(self, x: float, y: float, z: float) -> bool:
        if not self.min_z <= z <= self.max_z:
            return False
        return bool(self.contains_many(np.array([x]), np.array([y]))[0])

    def contains_many(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Whether the points (any shape of arrays) lie inside the zone's outline (its height is not checked)."""
        x0, x1, y0, y1 = self.box
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if self.shape == "Cylinder":
            cx, cy = self.nodes[0]
            return inside & (np.hypot(x - cx, y - cy) <= self.radius)
        if self.shape == "NPoly":
            poly = np.zeros(x.shape, bool)
            for (ax, ay), (bx, by) in zip(self.nodes, self.nodes[1:] + self.nodes[:1]):
                if ay == by:
                    continue
                crosses = (ay > y) != (by > y)
                poly ^= crosses & (x < ax + (y - ay) * (bx - ax) / (by - ay))
            return inside & poly
        return inside  # Cuboid: the box


def _water_zones(path: Path | str) -> list[_Zone]:
    table = pq.read_table(path, columns=["type", "shape", "nodes", "min_z", "max_z", "radius"]).to_pylist()
    return [_Zone(r["shape"], r["nodes"], r["min_z"], r["max_z"], r["radius"]) for r in table
            if r["type"] == "WaterZone" and r["nodes"] and r["min_z"] is not None and r["max_z"] is not None]
