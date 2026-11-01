"""
Unreal Engine 2 packages of the Lineage II Interlude client (``maps/*.unr``, ``staticmeshes/*.usx``).

The files start with the UTF-16 header ``Lineage2Ver111`` (28 bytes); the package after it is XOR-ed
byte by byte with ``0xAC``. Package version 123.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER_SIZE = 28
UNREAL_TAG = 0x9E2A83C1


def load(path: str | Path) -> bytes:
    """The plain package of a client file."""
    raw = Path(path).read_bytes()
    if raw[:HEADER_SIZE].decode("utf-16-le", "replace").startswith("Lineage2Ver"):
        key = raw[HEADER_SIZE] ^ (UNREAL_TAG & 0xFF)
        return (np.frombuffer(raw[HEADER_SIZE:], np.uint8) ^ key).tobytes()
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


@dataclass
class Export:
    """An export table entry: an object stored in the package."""

    index: int
    cls: str
    name: str
    outer: int
    flags: int
    size: int
    offset: int


@dataclass
class Prop:
    """A serialized property: its type, raw value bytes, struct name and array index."""

    kind: int
    raw: bytes
    struct: str = ""
    array_index: int = 0


@dataclass
class Package:
    """A decrypted package with its name, import and export tables."""

    data: bytes
    names: list[str] = field(default_factory=list)
    imports: list[tuple[str, str, int, str]] = field(default_factory=list)  # class package, class, outer, name
    exports: list[Export] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | Path) -> "Package":
        p = cls(load(path))
        p._read_tables()
        return p

    def _read_tables(self) -> None:
        d = self.data
        tag, self.version, self.licensee, _flags, n_names, names_at, n_exports, exports_at, n_imports, imports_at = \
            struct.unpack_from("<IHHIIIIIII", d, 0)
        if tag != UNREAL_TAG:
            raise ValueError(f"not an Unreal package (tag {tag:#x})")
        pos = names_at
        for _ in range(n_names):
            length, pos = read_index(d, pos)
            self.names.append(d[pos:pos + length - 1].decode("latin-1"))
            pos += length + 4
        pos = imports_at
        for _ in range(n_imports):
            cp, pos = read_index(d, pos)
            cn, pos = read_index(d, pos)
            outer = struct.unpack_from("<i", d, pos)[0]
            pos += 4
            on, pos = read_index(d, pos)
            self.imports.append((self.names[cp], self.names[cn], outer, self.names[on]))
        pos = exports_at
        for i in range(n_exports):
            ci, pos = read_index(d, pos)
            _super, pos = read_index(d, pos)
            outer = struct.unpack_from("<i", d, pos)[0]
            pos += 4
            name, pos = read_index(d, pos)
            flags = struct.unpack_from("<I", d, pos)[0]
            pos += 4
            size, pos = read_index(d, pos)
            offset = 0
            if size > 0:
                offset, pos = read_index(d, pos)
            self.exports.append(Export(i + 1, self.object_name(ci) if ci else "Class", self.names[name], outer, flags, size, offset))

    def object_name(self, ref: int) -> str:
        """Name of an object reference (export > 0, import < 0)."""
        if ref > 0:
            return self.exports[ref - 1].name
        if ref < 0:
            return self.imports[-ref - 1][3]
        return ""

    def object_path(self, ref: int) -> str:
        """``Package.Group.Name`` of an imported object (the export name for an export)."""
        if ref >= 0:
            return self.object_name(ref)
        parts, r = [], ref
        while r < 0:
            imp = self.imports[-r - 1]
            parts.append(imp[3])
            r = imp[2]
        return ".".join(reversed(parts))

    def properties(self, pos: int) -> tuple[dict[str, Prop], int]:
        """Tagged properties at ``pos``: name -> Prop (raw value bytes), and the position after them."""
        d, out = self.data, {}
        while True:
            ni, pos = read_index(d, pos)
            name = self.names[ni]
            if name == "None":
                return out, pos
            info = d[pos]
            pos += 1
            kind, size_type, is_array = info & 0x0F, (info >> 4) & 7, info & 0x80
            sname = ""
            if kind == 10:
                si, pos = read_index(d, pos)
                sname = self.names[si]
            size = {0: 1, 1: 2, 2: 4, 3: 12, 4: 16}.get(size_type)
            if size is None:
                fmt, width = {5: ("<B", 1), 6: ("<H", 2), 7: ("<i", 4)}[size_type]
                size = struct.unpack_from(fmt, d, pos)[0]
                pos += width
            if kind == 3:
                out[name] = Prop(kind, b"\x01" if is_array else b"\x00")
                continue
            idx = 0
            if is_array:
                b = d[pos]
                if b < 0x80:
                    idx, pos = b, pos + 1
                elif b < 0xC0:
                    idx, pos = struct.unpack_from(">H", d, pos)[0] & 0x3FFF, pos + 2
                else:
                    idx, pos = struct.unpack_from(">I", d, pos)[0] & 0x3FFFFFFF, pos + 4
            key = name if idx == 0 else f"{name}[{idx}]"
            out[key] = Prop(kind, d[pos:pos + size], sname, idx)
            pos += size

    def export_properties(self, e: Export) -> tuple[dict[str, Prop], int]:
        """Properties of an export; for objects with a state frame (actors) it is skipped first."""
        pos = e.offset
        if e.flags & 0x02000000:  # RF_HasStack
            node, pos = read_index(self.data, pos)
            _state, pos = read_index(self.data, pos)
            pos += 12  # probe mask (8) + latent action (4)
            if node:
                _off, pos = read_index(self.data, pos)
        return self.properties(pos)


def vector(p: Prop) -> tuple[float, float, float]:
    return struct.unpack_from("<3f", p.raw)


def rotator(p: Prop) -> tuple[int, int, int]:
    """(pitch, yaw, roll) in Unreal units (65536 = 360 degrees)."""
    return struct.unpack_from("<3i", p.raw)


def as_object(p: Prop) -> int:
    return read_index(p.raw, 0)[0]


def as_float(p: Prop) -> float:
    return struct.unpack_from("<f", p.raw)[0]


@dataclass
class Mesh:
    """Render geometry of a static mesh: vertices (n, 3) and triangles (m, 3) indexing them."""

    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    #: per triangle: its section's material has EnableCollision (leaves and decals do not)
    collides: np.ndarray = None


def material_collision(p: Package, props: dict[str, Prop]) -> list[bool]:
    """``EnableCollision`` of every entry of a static mesh's ``Materials`` array (one per section; default on)."""
    if "Materials" not in props:
        return []
    raw = props["Materials"].raw
    sub = Package(raw, p.names, p.imports, p.exports)
    n, pos = read_index(raw, 0)
    out = []
    for _ in range(n):
        item, pos = sub.properties(pos)
        out.append(item["EnableCollision"].raw != b"\x00" if "EnableCollision" in item else True)
    return out


def _tarray(d: bytes, pos: int, item: int) -> tuple[bytes, int, int]:
    n, pos = read_index(d, pos)
    return d[pos:pos + n * item], n, pos + n * item


def static_mesh(p: Package, e: Export) -> Mesh:
    """
    Geometry of a ``StaticMesh`` export (Lineage II, package version 123):
    bounding box (25) + sphere (16), sections (index count + 14 bytes each: is-strip int, first index,
    min / max vertex, triangles, primitives as uint16), bounding box (25), vertex stream (positions +
    normals, 24 bytes each) + revision, color and alpha streams (4 bytes each) + revision, UV streams
    (8 bytes per UV + coordinate index + revision), index buffer (uint16) + revision.
    """
    d = p.data
    props, pos = p.properties(e.offset)
    collision = material_collision(p, props)
    pos += 25 + 16
    n_sections, pos = read_index(d, pos)
    sections = []
    for _ in range(n_sections):
        is_strip, first, vmin, vmax, ntri, nprim = struct.unpack_from("<i5H", d, pos)
        sections.append((is_strip, first, ntri))
        pos += 14
    pos += 25
    raw, nv, pos = _tarray(d, pos, 24)
    verts = np.frombuffer(raw, "<f4").reshape(nv, 6)[:, :3].astype(np.float64)
    pos += 4
    for _ in range(2):  # color, alpha
        _raw, _n, pos = _tarray(d, pos, 4)
        pos += 4
    n_uv, pos = read_index(d, pos)
    for _ in range(n_uv):
        _raw, _n, pos = _tarray(d, pos, 8)
        pos += 8
    raw, ni, pos = _tarray(d, pos, 2)
    idx = np.frombuffer(raw, "<u2").astype(np.int64)
    tris, collides = [], []
    for s_index, (is_strip, first, ntri) in enumerate(sections):
        if is_strip:
            s = idx[first:first + ntri + 2]
            t = np.stack([s[:-2], s[1:-1], s[2:]], 1)
            t[1::2] = t[1::2][:, [1, 0, 2]]
        else:
            t = idx[first:first + 3 * ntri].reshape(-1, 3)
        tris.append(t)
        collides.append(np.full(len(t), collision[s_index] if s_index < len(collision) else True))
    tri = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    if len(tri) and tri.max() >= nv:
        raise ValueError(f"{e.name}: index {tri.max()} >= {nv} vertices")
    return Mesh(e.name, verts, tri, np.concatenate(collides) if collides else np.zeros(0, bool))


def rotation_matrix(pitch: int, yaw: int, roll: int) -> np.ndarray:
    """Unreal ``FRotationMatrix`` (rows = transformed X, Y, Z axes; a point is ``v @ M``)."""
    p, y, r = (np.array([pitch, yaw, roll], np.float64) * (2 * np.pi / 65536))
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    return np.array([
        [cp * cy, cp * sy, sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
        [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
    ])


@dataclass
class Placed:
    """A static mesh actor of a map: its mesh path and the world transform."""

    mesh: str
    location: np.ndarray
    rotation: tuple[int, int, int]
    scale: np.ndarray
    pre_pivot: np.ndarray

    def transform(self, vertices: np.ndarray) -> np.ndarray:
        return ((vertices * self.scale) - self.pre_pivot) @ rotation_matrix(*self.rotation) + self.location


def mesh_actors(p: Package) -> list[Placed]:
    """Static mesh actors of a map with their transforms."""
    out = []
    for e in p.exports:
        if e.cls not in ("StaticMeshActor", "MovableStaticMeshActor"):
            continue
        props, _ = p.export_properties(e)
        if "StaticMesh" not in props:
            continue
        scale = np.ones(3) * (as_float(props["DrawScale"]) if "DrawScale" in props else 1.0)
        if "DrawScale3D" in props:
            scale = scale * np.array(vector(props["DrawScale3D"]))
        out.append(Placed(
            p.object_path(as_object(props["StaticMesh"])),
            np.array(vector(props["Location"])) if "Location" in props else np.zeros(3),
            rotator(props["Rotation"]) if "Rotation" in props else (0, 0, 0),
            scale,
            np.array(vector(props["PrePivot"])) if "PrePivot" in props else np.zeros(3),
        ))
    return out


def struct_properties(p: Package, raw: bytes) -> dict[str, Prop]:
    """Tagged properties of a struct value (``Scale`` and the like are stored as nested properties)."""
    sub = Package(raw, p.names, p.imports, p.exports)
    props, _ = sub.properties(0)
    return props


def polys(p: Package, e: Export) -> list[tuple[np.ndarray, np.ndarray]]:
    """Polygons of a ``Polys`` export: (local vertices (k, 3), normal (3,) pointing out of the brush)."""
    d = p.data
    _props, pos = p.properties(e.offset)
    num = struct.unpack_from("<i", d, pos)[0]
    pos += 8  # count, capacity
    out = []
    for _ in range(num):
        nv, pos = read_index(d, pos)
        normal = np.array(struct.unpack_from("<3f", d, pos + 12))
        pos += 48  # base, normal, texture u, texture v
        out.append((np.frombuffer(d, "<f4", nv * 3, pos).reshape(nv, 3).astype(np.float64), normal))
        pos += 12 * nv
        pos += 4  # poly flags
        for _ in range(5):  # actor, item name, material, link, brush poly
            _v, pos = read_index(d, pos)
        pos += 8  # light map scale, a Lineage II int
    return out


def brush_polys(p: Package, model: Export) -> list[tuple[np.ndarray, np.ndarray]]:
    """The polygons of a brush's model (its Polys object)."""
    d = p.data
    _props, pos = p.properties(model.offset)
    pos += 41  # bounding box and sphere
    for _ in range(5):  # vectors, points, nodes, surfs, verts: empty for a brush
        n, pos = read_index(d, pos)
        if n:
            return []
    _shared, zones = struct.unpack_from("<ii", d, pos)
    pos += 8
    if zones:
        return []
    ref, pos = read_index(d, pos)
    if ref <= 0 or p.exports[ref - 1].cls != "Polys":
        return []
    return polys(p, p.exports[ref - 1])


def _scale(p: Package, prop: Prop | None) -> np.ndarray:
    if prop is None:
        return np.ones(3)
    sub = struct_properties(p, prop.raw)
    return np.array(vector(sub["Scale"])) if "Scale" in sub else np.ones(3)


def brushes(p: Package) -> list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]:
    """(CSG operation: 1 add / 2 subtract, [(world polygon, world normal)]) of every brush of a map."""
    out = []
    for e in p.exports:
        if e.cls != "Brush":
            continue
        props, _ = p.export_properties(e)
        if "Brush" not in props:
            continue
        ref = as_object(props["Brush"])
        if ref <= 0:
            continue
        local = brush_polys(p, p.exports[ref - 1])
        if not local:
            continue
        loc = np.array(vector(props["Location"])) if "Location" in props else np.zeros(3)
        pivot = np.array(vector(props["PrePivot"])) if "PrePivot" in props else np.zeros(3)
        rot = rotation_matrix(*rotator(props["Rotation"])) if "Rotation" in props else np.eye(3)
        main, post = _scale(p, props.get("MainScale")), _scale(p, props.get("PostScale"))
        op = props["CsgOper"].raw[0] if "CsgOper" in props else 0
        world = []
        for v, n in local:
            nw = ((n / np.where(main == 0, 1, main)) @ rot) / np.where(post == 0, 1, post)
            world.append(((((v - pivot) * main) @ rot) * post + loc, nw / (np.linalg.norm(nw) or 1)))
        out.append((op, world))
    return out


def bsp_polygons(p: Package) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Polygons of the level's compiled BSP (the largest ``Model`` of a map, what the client renders and
    collides with after CSG), as (world vertices (k, 3), plane normal facing the open side, the surface's
    poly flags).

    Lineage II layout (package 123): bounding box + sphere (41), vectors and points (12 bytes each),
    nodes (plane 16, zone mask 8, flags 1, 7 compact indices - vertex pool, surf, back, front, plane,
    collision bound, render bound -, sphere 16, 16 bytes, zones 2, vertex count 1, 5 ints), surfs
    (material index, flags int, 6 indices, plane 16, light map scale, int), verts (point, side indices).
    """
    e = max((x for x in p.exports if x.cls == "Model"), key=lambda x: x.size)
    d = p.data
    _props, pos = p.properties(e.offset)
    pos += 41
    nvec, pos = read_index(d, pos)
    pos += nvec * 12
    npt, pos = read_index(d, pos)
    points = np.frombuffer(d, "<f4", npt * 3, pos).reshape(npt, 3).astype(np.float64)
    pos += npt * 12
    nn, pos = read_index(d, pos)
    nodes = []
    for _ in range(nn):
        plane = struct.unpack_from("<4f", d, pos)
        pos += 25
        vert_pool, pos = read_index(d, pos)
        surf, pos = read_index(d, pos)
        for _ in range(5):
            _v, pos = read_index(d, pos)
        pos += 32
        n_verts = d[pos + 2]
        pos += 23
        nodes.append((np.array(plane[:3]), vert_pool, n_verts, surf))
    ns, pos = read_index(d, pos)
    surf_flags = []
    for _ in range(ns):
        _v, pos = read_index(d, pos)
        surf_flags.append(struct.unpack_from("<I", d, pos)[0])
        pos += 4
        for _ in range(6):
            _v, pos = read_index(d, pos)
        pos += 24
    nv, pos = read_index(d, pos)
    vert_point = np.zeros(nv, np.int64)
    for i in range(nv):
        vert_point[i], pos = read_index(d, pos)
        _side, pos = read_index(d, pos)
    out = []
    for normal, pool, k, surf in nodes:
        if k >= 3 and pool + k <= nv:
            out.append((points[vert_point[pool:pool + k]], normal, surf_flags[surf] if 0 <= surf < ns else 0))
    return out


def texture_mip0(p: Package, e: Export) -> tuple[int, int, int, bytes]:
    """(format, width, height, raw bytes of the first mip) of a texture export (Lineage II layout)."""
    props, pos = p.properties(e.offset)
    width = struct.unpack_from("<i", props["USize"].raw)[0] if "USize" in props else 0
    height = struct.unpack_from("<i", props["VSize"].raw)[0] if "VSize" in props else 0
    fmt = props["Format"].raw[0] if "Format" in props else 0
    pos += 4  # Lineage II: an int before the mip array
    _mips, pos = read_index(p.data, pos)
    pos += 4  # lazy array: the position after it
    size, pos = read_index(p.data, pos)
    return fmt, width, height, p.data[pos:pos + size]


@dataclass
class Terrain:
    """A map's terrain: ``heights`` (rows = y, cols = x) of the vertex grid and its placement."""

    heights: np.ndarray
    location: np.ndarray
    scale: np.ndarray

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain z at world points (bilinear over the triangles' corners; NaN outside the grid)."""
        h, w = self.heights.shape
        fx = (np.asarray(x, np.float64) - self.location[0]) / self.scale[0] + w / 2
        fy = (np.asarray(y, np.float64) - self.location[1]) / self.scale[1] + h / 2
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < w - 1) & (iy < h - 1)
        ix, iy = np.clip(ix, 0, w - 2), np.clip(iy, 0, h - 2)
        tx, ty = fx - ix, fy - iy
        H = self.heights.astype(np.float64)
        z = (H[iy, ix] * (1 - tx) * (1 - ty) + H[iy, ix + 1] * tx * (1 - ty) + H[iy + 1, ix] * (1 - tx) * ty
             + H[iy + 1, ix + 1] * tx * ty)
        z = (z - 32768) * self.scale[2] / 256 + self.location[2]
        return np.where(ok, z, np.nan)


def terrain(p: Package, textures_dir: Path) -> Terrain | None:
    """The terrain of a map, its height map read from the texture package it names."""
    t = next((e for e in p.exports if e.cls == "TerrainInfo"), None)
    if t is None:
        return None
    props, _ = p.export_properties(t)
    path = p.object_path(as_object(props["TerrainMap"]))  # Package.Group.Name
    pkg_name, name = path.split(".")[0], path.split(".")[-1]
    files = {f.stem.lower(): f for f in Path(textures_dir).glob("*.utx")}
    tp = Package.open(files[pkg_name.lower()])
    te = next(x for x in tp.exports if x.name == name and x.cls == "Texture")
    fmt, w, h, raw = texture_mip0(tp, te)
    if len(raw) < w * h * 2:
        raise ValueError(f"{path}: format {fmt}, {len(raw)} bytes for {w}x{h}")
    heights = np.frombuffer(raw, "<u2", w * h).reshape(h, w)
    return Terrain(heights, np.array(vector(props["Location"])), np.array(vector(props["TerrainScale"])))
