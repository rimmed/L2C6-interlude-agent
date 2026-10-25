"""
Extract icons from a Lineage II texture package (``Icon.utx`` of the Interlude client) as PNG files.

The client files are Unreal Engine 2 packages behind a Lineage II header: ``Lineage2Ver121`` (28 bytes
of UTF-16) followed by the package XOR-ed byte by byte with 0x38. A texture object is its
properties (``Format``, ``USize``, ``VSize`` ...) followed by its mip maps; the first mip is the
full-size picture. Supported formats: DXT1, DXT3, DXT5 and RGBA8 (the icons are 32x32 DXT1).

Only numpy is needed - the PNG files are written with zlib.

Usage (from ``l2_agent/``)::

    uv run l2_client/tools/extract_icons.py ../client_textures/Icon.utx -o ../client_textures/icons --prefix skill

writes ``skill1177.png`` (Wind Strike) etc. The icons belong to the game client: keep them out of git.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HEADER = "Lineage2Ver121".encode("utf-16-le")
XOR_KEY = 0x38
UNREAL_TAG = 0x9E2A83C1
# ETextureFormat of Unreal Engine 2
TEXF_DXT1, TEXF_RGBA8, TEXF_DXT3, TEXF_DXT5 = 3, 5, 7, 8


def decrypt(raw: bytes) -> bytes:
    """The plain package of a Lineage II file (or the bytes as they are when they have no L2 header)."""
    if raw.startswith(HEADER):
        return (np.frombuffer(raw[len(HEADER):], dtype=np.uint8) ^ XOR_KEY).tobytes()
    return raw


def read_index(buf: bytes, pos: int) -> tuple[int, int]:
    """Unreal compact index: sign bit + 6 bits, then 7 bits per byte."""
    b = buf[pos]
    pos += 1
    value = b & 0x3F
    if b & 0x40:
        shift = 6
        while True:
            c = buf[pos]
            pos += 1
            value |= (c & 0x7F) << shift
            shift += 7
            if not c & 0x80:
                break
    return (-value if b & 0x80 else value), pos


class Package:
    """Names, imports and exports of an Unreal Engine 2 package."""

    def __init__(self, data: bytes):
        tag, self.version, _licensee, _flags, n_names, names_at, n_exports, exports_at, n_imports, imports_at = \
            struct.unpack_from("<IHHIIIIIII", data, 0)
        if tag != UNREAL_TAG:
            raise ValueError(f"not an Unreal package (tag {tag:#x})")
        self.data = data
        pos, self.names = names_at, []
        for _ in range(n_names):
            length, pos = read_index(data, pos)
            self.names.append(data[pos:pos + length - 1].decode("latin-1"))
            pos += length + 4  # string with its terminating zero, then the name flags
        pos, self.imports = imports_at, []
        for _ in range(n_imports):
            _class_package, pos = read_index(data, pos)
            class_name, pos = read_index(data, pos)
            pos += 4  # outer package
            object_name, pos = read_index(data, pos)
            self.imports.append((self.names[class_name], self.names[object_name]))
        pos, self.exports = exports_at, []
        for _ in range(n_exports):
            class_index, pos = read_index(data, pos)
            _super, pos = read_index(data, pos)
            package = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            name, pos = read_index(data, pos)
            pos += 4  # object flags
            size, pos = read_index(data, pos)
            offset = 0
            if size > 0:
                offset, pos = read_index(data, pos)
            cls = self.imports[-class_index - 1][1] if class_index < 0 else ""
            self.exports.append({"class": cls, "name": self.names[name], "package": package, "size": size,
                                 "offset": offset})

    def properties(self, pos: int) -> tuple[dict, int]:
        """The tagged properties at ``pos`` (only the simple types are decoded) and the position after them."""
        data, out = self.data, {}
        while True:
            name_index, pos = read_index(data, pos)
            name = self.names[name_index]
            if name == "None":
                return out, pos
            info = data[pos]
            pos += 1
            kind, size_type, is_array = info & 0x0F, (info >> 4) & 7, info & 0x80
            if kind == 10:  # struct: its name follows
                _struct, pos = read_index(data, pos)
            size = {0: 1, 1: 2, 2: 4, 3: 12, 4: 16}.get(size_type)
            if size is None:
                fmt, width = {5: ("<B", 1), 6: ("<H", 2), 7: ("<i", 4)}[size_type]
                size = struct.unpack_from(fmt, data, pos)[0]
                pos += width
            if kind == 3:  # bool: the value is the array bit
                out[name] = bool(is_array)
                continue
            if is_array:
                b = data[pos]
                pos += 1 if b < 0x80 else 2 if b < 0xC0 else 4
            value = data[pos:pos + size]
            if kind == 1:
                out[name] = value[0]
            elif kind == 2:
                out[name] = struct.unpack_from("<i", value)[0]
            elif kind == 5:
                out[name] = read_index(value, 0)[0]
            pos += size

    def texture(self, export: dict) -> np.ndarray | None:
        """The first mip of a texture as an RGBA array, ``None`` for a format this module cannot decode."""
        props, pos = self.properties(export["offset"])
        width, height, fmt = props.get("USize", 0), props.get("VSize", 0), props.get("Format", 0)
        pos += 4  # Lineage II: an int before the mip array
        mips, pos = read_index(self.data, pos)
        if mips < 1 or width <= 0 or height <= 0:
            return None
        pos += 4  # lazy array: the position after it
        size, pos = read_index(self.data, pos)
        raw = self.data[pos:pos + size]
        if fmt == TEXF_RGBA8:
            bgra = np.frombuffer(raw, dtype=np.uint8)[: width * height * 4].reshape(height, width, 4)
            return bgra[..., [2, 1, 0, 3]].copy()
        if fmt in (TEXF_DXT1, TEXF_DXT3, TEXF_DXT5):
            return decode_dxt(raw, width, height, fmt)
        return None


def _rgb565(c: np.ndarray) -> np.ndarray:
    r, g, b = (c >> 11) & 0x1F, (c >> 5) & 0x3F, c & 0x1F
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], axis=-1).astype(np.int32)


def decode_dxt(raw: bytes, width: int, height: int, fmt: int) -> np.ndarray:
    """Decode DXT1 / DXT3 / DXT5 blocks into an RGBA array of ``height x width``."""
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    block = 8 if fmt == TEXF_DXT1 else 16
    blocks = np.frombuffer(raw[: bw * bh * block], dtype=np.uint8).reshape(bh * bw, block)
    color = blocks[:, block - 8:]
    c0 = color[:, 0].astype(np.uint16) | (color[:, 1].astype(np.uint16) << 8)
    c1 = color[:, 2].astype(np.uint16) | (color[:, 3].astype(np.uint16) << 8)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1) | (fmt != TEXF_DXT1)
    p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    palette = np.stack([p0, p1, p2, p3], axis=1)  # (n, 4, 3)
    bits = color[:, 4:8].astype(np.uint32)
    codes = bits[:, 0] | (bits[:, 1] << 8) | (bits[:, 2] << 16) | (bits[:, 3] << 24)
    index = (codes[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3  # (n, 16)
    rgb = np.take_along_axis(palette, index[..., None].astype(np.int64).repeat(3, axis=2), axis=1)
    alpha = np.full((len(blocks), 16), 255, dtype=np.int32)
    if fmt == TEXF_DXT1:
        alpha[(~four[:, None]) & (index == 3)] = 0  # 3-colour blocks: code 3 is transparent black
    elif fmt == TEXF_DXT3:
        nibbles = np.unpackbits(blocks[:, :8, None], axis=2, bitorder="little").reshape(len(blocks), 16, 4)
        alpha = (nibbles @ np.array([1, 2, 4, 8])) * 17
    else:  # DXT5
        a0, a1 = blocks[:, 0].astype(np.int32), blocks[:, 1].astype(np.int32)
        steps = np.arange(8)
        table = np.where((a0 > a1)[:, None],
                         np.select([steps == 0, steps == 1], [a0[:, None], a1[:, None]],
                                   ((8 - steps) * a0[:, None] + (steps - 1) * a1[:, None]) // 7),
                         np.select([steps == 0, steps == 1, steps == 6, steps == 7],
                                   [a0[:, None], a1[:, None], 0, 255],
                                   ((6 - steps) * a0[:, None] + (steps - 1) * a1[:, None]) // 5))
        abits = np.zeros(len(blocks), dtype=np.uint64)
        for i in range(6):
            abits |= blocks[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
        aindex = ((abits[:, None] >> (3 * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
        alpha = np.take_along_axis(table, aindex, axis=1)
    pixels = np.concatenate([rgb, alpha[..., None]], axis=2).astype(np.uint8)  # (n, 16, 4)
    image = pixels.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return image[:height, :width]


def write_png(path: Path, rgba: np.ndarray) -> None:
    """A minimal RGBA PNG writer (no image library needed)."""
    height, width = rgba.shape[:2]
    rows = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract the textures of a Lineage II .utx package as PNG files.")
    p.add_argument("package", type=Path, help="e.g. client_textures/Icon.utx")
    p.add_argument("-o", "--output", type=Path, required=True, help="Directory for the PNG files")
    p.add_argument("--prefix", default="", help="Only textures whose name starts with this (e.g. 'skill')")
    args = p.parse_args(argv)

    package = Package(decrypt(args.package.read_bytes()))
    args.output.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for export in package.exports:
        if export["class"] != "Texture" or not export["name"].lower().startswith(args.prefix.lower()):
            continue
        image = package.texture(export)
        if image is None:
            skipped += 1
            continue
        write_png(args.output / f"{export['name'].lower()}.png", image)
        written += 1
    print(f"{written} textures written to {args.output} ({skipped} in an unsupported format)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
