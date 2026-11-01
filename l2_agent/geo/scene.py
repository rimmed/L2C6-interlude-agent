"""World triangles of a client map region, for comparing the geodata with what the client shows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import unreal

CLIENT = Path(__file__).resolve().parents[2] / "client_files"
#: height above a floor up to which a static mesh blocks it (a step a character climbs is lower)
KNEE = 40
#: room a character needs over a floor (the gap under a floor slab is lower)
CLEARANCE = 128
#: BSP surface flags without collision
PF_NOT_SOLID, PF_PORTAL = 0x00000008, 0x04000000


@lru_cache(maxsize=4)
def map_package(region: str) -> unreal.Package:
    """The client map ``maps/<region>.unr`` (cached: a map is tens of megabytes)."""
    return unreal.Package.open(CLIENT / "maps" / f"{region}.unr")


@lru_cache(maxsize=1)
def _usx_files() -> dict[str, Path]:
    return {f.stem.lower(): f for f in (CLIENT / "staticmeshes").glob("*.usx")}


@lru_cache(maxsize=64)
def _package(name: str) -> unreal.Package | None:
    path = _usx_files().get(name.lower())
    return unreal.Package.open(path) if path else None


@lru_cache(maxsize=4096)
def _mesh(path: str) -> unreal.Mesh | None:
    pkg_name, _, name = path.partition(".")
    pkg = _package(pkg_name)
    if pkg is None:
        return None
    name = name.rsplit(".", 1)[-1]
    for e in pkg.exports:
        if e.cls == "StaticMesh" and e.name.lower() == name.lower():
            return unreal.static_mesh(pkg, e)
    return None


def mesh_triangles(region: str) -> tuple[np.ndarray, list[str]]:
    """World triangles (m, 3, 3) of the static meshes of map ``region`` (e.g. ``21_24``) and the meshes not found."""
    p = map_package(region)
    tris, missing = [], []
    for a in unreal.mesh_actors(p):
        m = _mesh(a.mesh)
        if m is None:
            missing.append(a.mesh)
            continue
        if len(m.triangles):
            tris.append(a.transform(m.vertices)[m.triangles])
    return (np.concatenate(tris) if tris else np.zeros((0, 3, 3))), missing


def floor_heights(tris: np.ndarray, x0: int, y0: int, nx: int, ny: int, cell: int = 16,
                  min_normal_z: float = 0.3) -> dict[tuple[int, int], list[float]]:
    """
    Heights of the upward faces (floors, normal z >= ``min_normal_z``) of ``tris`` at the centres of the
    ``nx`` x ``ny`` cells of ``cell`` units from world ``(x0, y0)``: ``{(i, j): [z, ...]}``.
    """
    out: dict[tuple[int, int], list[float]] = {}
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    nl = np.linalg.norm(n, axis=1)
    ok = nl > 1e-6
    nz = np.zeros(len(tris))
    nz[ok] = n[ok, 2] / nl[ok]
    # Unreal is left-handed: an upward face has a negative cross product z with this winding; take both
    sel = np.abs(nz) >= min_normal_z
    sel &= (tris[:, :, 0].max(1) >= x0) & (tris[:, :, 0].min(1) < x0 + nx * cell)
    sel &= (tris[:, :, 1].max(1) >= y0) & (tris[:, :, 1].min(1) < y0 + ny * cell)
    for t in tris[sel]:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
        i0 = max(int(np.floor((min(ax, bx, cx) - x0) / cell)), 0)
        i1 = min(int(np.ceil((max(ax, bx, cx) - x0) / cell)), nx)
        j0 = max(int(np.floor((min(ay, by, cy) - y0) / cell)), 0)
        j1 = min(int(np.ceil((max(ay, by, cy) - y0) / cell)), ny)
        if i0 >= i1 or j0 >= j1:
            continue
        gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        px = x0 + (gi + 0.5) * cell
        py = y0 + (gj + 0.5) * cell
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-9:
            continue
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        z = l1 * az + l2 * bz + l3 * cz
        for i, j, zz in zip(gi[inside], gj[inside], z[inside]):
            out.setdefault((int(i), int(j)), []).append(float(zz))
    return out


def brush_triangles(region: str) -> np.ndarray:
    """World triangles (m, 3, 3) of the brushes (BSP) of map ``region``, polygons fanned."""
    return np.array([t for t, _floor in _brush_faces(region)]) if _brush_faces(region) else np.zeros((0, 3, 3))


@lru_cache(maxsize=8)
def _brush_faces(region: str) -> list[tuple[np.ndarray, int]]:
    """Brush triangles with their role: +1 a floor (walkable side up), -1 a ceiling, 0 a wall."""
    p = map_package(region)
    out = []
    for op, polys in unreal.brushes(p):
        for poly, normal in polys:
            if np.ptp(poly[:, :2], axis=0).max() > 40000:
                continue  # the level's world-sized builder brush
            nz = normal[2] if op == 1 else -normal[2]  # a subtracted brush is open space: its faces look inwards
            role = 1 if nz > 0.5 else -1 if nz < -0.5 else 0
            for k in range(1, len(poly) - 1):
                out.append((np.array((poly[0], poly[k], poly[k + 1])), role))
    return out


def _rasterize(t: np.ndarray, x0: int, y0: int, nx: int, ny: int, cell: int):
    """Cells (i, j) whose centre lies in triangle ``t`` (projected on xy) and the triangle's z there."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    i0 = max(int(np.floor((min(ax, bx, cx) - x0) / cell)), 0)
    i1 = min(int(np.ceil((max(ax, bx, cx) - x0) / cell)), nx)
    j0 = max(int(np.floor((min(ay, by, cy) - y0) / cell)), 0)
    j1 = min(int(np.ceil((max(ay, by, cy) - y0) / cell)), ny)
    if i0 >= i1 or j0 >= j1:
        return None
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-9:
        return None
    gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    px, py = x0 + (gi + 0.5) * cell, y0 + (gj + 0.5) * cell
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    l3 = 1 - l1 - l2
    inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
    if not inside.any():
        return None
    return gi[inside], gj[inside], (l1 * az + l2 * bz + l3 * cz)[inside]


@lru_cache(maxsize=8)
def _bsp_faces(region: str, min_normal_z: float = 0.5) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Floor and ceiling triangles of the level BSP (plane facing up / down, the open side)."""
    p = map_package(region)
    floors, ceilings = [], []
    for poly, normal, flags in unreal.bsp_polygons(p):
        if np.ptp(poly[:, :2], axis=0).max() > 40000:
            continue  # the world's outer hull
        if flags & (PF_NOT_SOLID | PF_PORTAL):
            continue  # water surfaces and zone portals have no collision
        tris = [np.array((poly[0], poly[k], poly[k + 1])) for k in range(1, len(poly) - 1)]
        if normal[2] >= min_normal_z:
            floors += tris
        elif normal[2] <= -min_normal_z:
            ceilings += tris
    return floors, ceilings


@dataclass
class Surfaces:
    """What the client has over each cell: floor heights, ceiling heights and the terrain height (NaN = none)."""

    floors: dict
    ceilings: dict
    terrain: np.ndarray
    #: per static mesh over the cell: (its surface heights there, its top)
    spans: dict = None
    #: the floors of a cell that are mesh tops (the rest are BSP polygons)
    mesh_tops: dict = None

    def walkable(self, i: int, j: int) -> list[float]:
        """
        Floors a character can stand on: the terrain unless a static mesh covers it (a rock or a hillside
        shell over it), floors above the terrain or under a ceiling that is below it (a cave), and not under a
        static mesh standing on them (one with a surface at the floor that rises above the knee).
        """
        tz = self.terrain[i, j]
        spans = (self.spans or {}).get((i, j), [])
        # terrain under a colliding static mesh surface (a rock, a hillside or mountain shell over it, open
        # below or sunk into the ground) is not walkable; under a bridge or an arch this misses a real floor,
        # which only leaves a layer unadded (caves are BSP or meshes, not terrain)
        buried = not np.isnan(tz) and any(any(z > tz + KNEE for z in zs) for zs, _t in spans)
        out = [] if np.isnan(tz) or buried else [float(tz)]
        ceil = self.ceilings.get((i, j), [])
        for f in self.floors.get((i, j), []):
            if not (np.isnan(tz) or f >= tz - 24 or any(f + 32 < c < tz for c in ceil)):
                continue
            # the underside gap of a floor slab, or a floor under mesh tiles laid over it: a ceiling or a
            # mesh surface right over it
            if any(f < c < f + CLEARANCE for c in ceil) or any(f + 8 < z < f + CLEARANCE for zs, _t in spans for z in zs):
                continue
            # a mesh standing on this floor: it has a surface down at the floor and rises above the knee
            if any(t > f + KNEE and any(f - 24 <= z <= f + KNEE for z in zs) for zs, t in spans):
                continue
            # the top of a mesh hidden inside another mesh (surfaces below it and above the knee)
            if f in self.mesh_tops.get((i, j), ()) and any(min(zs) < f - 8 and t > f + KNEE for zs, t in spans):
                continue
            out.append(f)
        return out

    def walkable_bsp(self, i: int, j: int) -> list[float]:
        """The walkable floors of a cell that are BSP polygons (rooms, caves): the reliable ones to add layers from."""
        tops = (self.mesh_tops or {}).get((i, j), ())
        tz = self.terrain[i, j]
        return [f for f in self.walkable(i, j) if f not in tops and (np.isnan(tz) or f != float(tz))]


def client_surfaces(region: str, x0: int, y0: int, nx: int, ny: int, cell: int = 16) -> Surfaces:
    """Floors (upward BSP polygons, mesh tops), ceilings (downward BSP polygons, mesh bottoms) and terrain over the cells."""
    floors: dict[tuple[int, int], list[float]] = {}
    ceilings: dict[tuple[int, int], list[float]] = {}
    spans: dict[tuple[int, int], list[tuple[list[float], float]]] = {}
    mesh_tops: dict[tuple[int, int], set[float]] = {}
    bsp_floors, bsp_ceilings = _bsp_faces(region)
    for tris, out in ((bsp_floors, floors), (bsp_ceilings, ceilings)):
        for t in tris:
            hit = _rasterize(t, x0, y0, nx, ny, cell)
            if hit:
                for i, j, z in zip(*hit):
                    out.setdefault((int(i), int(j)), []).append(float(z))
    p = map_package(region)
    for a in unreal.mesh_actors(p):
        m = _mesh(a.mesh)
        if m is None or not len(m.triangles):
            continue
        w = a.transform(m.vertices)
        if w[:, 0].max() < x0 or w[:, 0].min() >= x0 + nx * cell or w[:, 1].max() < y0 or w[:, 1].min() >= y0 + ny * cell:
            continue
        hits: dict[tuple[int, int], list[float]] = {}
        for t in w[m.triangles[m.collides]]:  # leaves and the like have no collision
            hit = _rasterize(t, x0, y0, nx, ny, cell)
            if hit:
                for i, j, z in zip(*hit):
                    hits.setdefault((int(i), int(j)), []).append(float(z))
        for k, zs in hits.items():
            top, bottom = max(zs), min(zs)
            floors.setdefault(k, []).append(top)
            mesh_tops.setdefault(k, set()).add(top)
            spans.setdefault(k, []).append((zs, top))
            if bottom < top - 32:
                ceilings.setdefault(k, []).append(bottom)
    t = unreal.terrain(p, CLIENT / "textures")
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    tz = t.height_at(x0 + (gi + 0.5) * cell, y0 + (gj + 0.5) * cell) if t else np.full((nx, ny), np.nan)
    return Surfaces(floors, ceilings, tz, spans, mesh_tops)
