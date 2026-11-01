"""
Read and write L2J geodata regions (``XX_YY.l2j``), as ``geoengine/geodata`` of the server loads them.

A region covers 32768 x 32768 world units: 256 x 256 blocks of 8 x 8 cells of 16 x 16 units. Blocks are
stored x-major (block ``bx * 256 + by``), cells too (``cx * 8 + cy``), all little-endian. A block is

* ``0`` flat: one ``int16`` height, every direction open;
* ``1`` complex: 64 ``int16`` cells, one layer each;
* ``2`` multilayer: per cell a ``uint8`` layer count and that many ``int16`` layers.

A layer packs the height and the passable directions: ``height = (int16)(data & 0xFFF0) >> 1``,
``nswe = data & 0x0F`` (east 1, west 2, south 4, north 8). The server picks the layer nearest to the
asked z.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BLOCKS = 256
CELLS = 8
REGION_CELLS = BLOCKS * CELLS  # 2048
CELL_SIZE = 16
REGION_SIZE = REGION_CELLS * CELL_SIZE  # 32768
TILE_ZERO_X, TILE_ZERO_Y = 20, 18
FLAT, COMPLEX, MULTILAYER = 0, 1, 2
EAST, WEST, SOUTH, NORTH = 1, 2, 4, 8
ALL = 15


def region_origin(rx: int, ry: int) -> tuple[int, int]:
    """World x, y of the north-west corner of region ``rx_ry``."""
    return (rx - TILE_ZERO_X) * REGION_SIZE, (ry - TILE_ZERO_Y) * REGION_SIZE


def region_of(x: int, y: int) -> tuple[int, int]:
    """Region ``(rx, ry)`` of a world point."""
    return (x >> 15) + TILE_ZERO_X, (y >> 15) + TILE_ZERO_Y


def decode(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Layer words -> (height, nswe)."""
    d = data.astype(np.int32) & 0xFFFF
    return ((d & 0xFFF0).astype(np.uint16).view(np.int16).astype(np.int32) >> 1), (d & 0x0F)


def encode(height: np.ndarray, nswe: np.ndarray) -> np.ndarray:
    """(height, nswe) -> layer words (heights are stored in steps of 8; a flat block keeps its exact height)."""
    return (((np.asarray(height, np.int32) << 1) & 0xFFF0) | (np.asarray(nswe, np.int32) & 0x0F)).astype(np.uint16)


@dataclass
class Region:
    """
    One region. ``layers[x, y]`` is the layer count of cell ``(x, y)`` (local cell coordinates, x east,
    y south); ``height`` / ``nswe`` are ``(2048, 2048, max_layers)`` with the cell's layers top-down as
    stored, padded with 0. ``block_type`` keeps the stored block types, so an unchanged region writes
    back byte for byte.
    """

    rx: int
    ry: int
    layers: np.ndarray
    height: np.ndarray
    nswe: np.ndarray
    block_type: np.ndarray

    @property
    def origin(self) -> tuple[int, int]:
        return region_origin(self.rx, self.ry)

    def cell_of(self, x: int, y: int) -> tuple[int, int]:
        ox, oy = self.origin
        return (x - ox) // CELL_SIZE, (y - oy) // CELL_SIZE

    def nearest_layer(self, cx: int, cy: int, z: int) -> int:
        """Index of the layer the server uses at cell ``(cx, cy)`` for a point at height ``z``."""
        n = int(self.layers[cx, cy])
        return int(np.argmin(np.abs(self.height[cx, cy, :n] - z)))


def read(path: str | Path) -> Region:
    raw = Path(path).read_bytes()
    rx, ry = map(int, Path(path).stem.split("_"))
    layers = np.zeros((REGION_CELLS, REGION_CELLS), np.uint8)
    words: list[np.ndarray] = [None] * (BLOCKS * BLOCKS)  # per block: (64, k) layer words
    block_type = np.zeros(BLOCKS * BLOCKS, np.uint8)
    flat_heights: dict[int, int] = {}
    pos = 0
    for b in range(BLOCKS * BLOCKS):
        t = raw[pos]
        pos += 1
        block_type[b] = t
        if t == FLAT:
            h = struct.unpack_from("<h", raw, pos)[0]
            pos += 2
            words[b] = np.full((64, 1), encode(h, ALL), np.uint16)
            flat_heights[b] = h  # exact: a flat height is not limited to steps of 8
            n = np.ones(64, np.uint8)
        elif t == COMPLEX:
            words[b] = np.frombuffer(raw, "<u2", 64, pos).reshape(64, 1).copy()
            pos += 128
            n = np.ones(64, np.uint8)
        elif t == MULTILAYER:
            n = np.zeros(64, np.uint8)
            cells = []
            for c in range(64):
                k = raw[pos]
                pos += 1
                cells.append(np.frombuffer(raw, "<u2", k, pos))
                pos += 2 * k
                n[c] = k
            w = np.zeros((64, int(n.max())), np.uint16)
            for c, cell in enumerate(cells):
                w[c, :len(cell)] = cell
            words[b] = w
        else:
            raise ValueError(f"{path}: block {b} has type {t}")
        bx, by = divmod(b, BLOCKS)
        layers[bx * CELLS:(bx + 1) * CELLS, by * CELLS:(by + 1) * CELLS] = n.reshape(CELLS, CELLS)
    if pos != len(raw):
        raise ValueError(f"{path}: {len(raw) - pos} bytes left after the last block")
    k = int(layers.max())
    data = np.zeros((REGION_CELLS, REGION_CELLS, k), np.uint16)
    for b, w in enumerate(words):
        bx, by = divmod(b, BLOCKS)
        cube = np.zeros((64, k), np.uint16)
        cube[:, :w.shape[1]] = w
        data[bx * CELLS:(bx + 1) * CELLS, by * CELLS:(by + 1) * CELLS] = cube.reshape(CELLS, CELLS, k)
    height, nswe = decode(data)
    pad = np.arange(k)[None, None, :] >= layers[:, :, None]
    height[pad] = 0
    nswe[pad] = 0
    for b, h in flat_heights.items():
        bx, by = divmod(b, BLOCKS)
        height[bx * CELLS:(bx + 1) * CELLS, by * CELLS:(by + 1) * CELLS, 0] = h
    return Region(rx, ry, layers, height, nswe, block_type)


def write(region: Region, path: str | Path) -> None:
    """Write a region; a block keeps its stored type unless its cells no longer fit it."""
    out = bytearray()
    words = encode(region.height, region.nswe)
    for b in range(BLOCKS * BLOCKS):
        bx, by = divmod(b, BLOCKS)
        sl = (slice(bx * CELLS, (bx + 1) * CELLS), slice(by * CELLS, (by + 1) * CELLS))
        n = region.layers[sl].reshape(64)
        w = words[sl].reshape(64, -1)
        h = region.height[sl].reshape(64, -1)[:, 0]
        ns = region.nswe[sl].reshape(64, -1)[:, 0]
        t = int(region.block_type[b])
        single = bool((n == 1).all())
        if t == FLAT and not (single and (h == h[0]).all() and (ns == ALL).all()):
            t = COMPLEX
        if t == COMPLEX and not single:
            t = MULTILAYER
        out.append(t)
        if t == FLAT:
            out += struct.pack("<h", int(h[0]))
        elif t == COMPLEX:
            out += w[:, 0].astype("<u2").tobytes()
        else:
            for c in range(64):
                out.append(int(n[c]))
                out += w[c, :n[c]].astype("<u2").tobytes()
    Path(path).write_bytes(bytes(out))
