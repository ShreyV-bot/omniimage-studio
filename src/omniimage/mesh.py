"""Triangle-mesh data model, primitives, modifiers, UV tools and file I/O (OBJ, STL; glTF lives in gltf.py).

Conventions: right-handed, **Y up**, counter-clockwise faces point outward, UV origin at the **top-left** of an image
(glTF convention: v = 0 is the top row). Every mesh carries per-vertex material attributes (linear colour, roughness,
metallic, emission) so meshes with different looks can be merged into one scene, optional per-vertex UVs, and optional
texture maps (one index per face into ``textures``).
Meshes are treated as immutable by the node graph: every operation returns a new Mesh.
"""
from __future__ import annotations
import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_COLOR = (0.8, 0.8, 0.8)


def _as_rgba(a, name: str) -> np.ndarray:
    """Normalise a map to a contiguous float32 (H, W, 4) array (gray expands to RGB, missing alpha becomes 1)."""
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4) or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"texture {name} must be a non-empty (H, W, 1|3|4) array")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
    return np.ascontiguousarray(a)


@dataclass
class Texture:
    """A set of maps for one surface, all sharing one UV transform (uv * scale + offset; wrapping repeats).

    ``image`` is the base colour in scene-linear RGBA (a 1x1 white default when only other maps are given). The other maps
    are *data*, not colour, and are stored as-is: ``normal`` is a tangent-space normal map (RGB = n * 0.5 + 0.5, +Y up like
    glTF; ``normal_scale`` scales the XY tilt), ``rough_metal`` follows glTF (G = roughness, B = metallic; both multiply
    the mesh's roughness / metallic values), ``emissive`` is linear colour multiplying the mesh's emission, and
    ``occlusion`` (R channel) darkens ambient light by ``occlusion_strength``. Maps may have different resolutions."""
    image: Optional[np.ndarray] = None
    scale: Tuple[float, float] = (1.0, 1.0)
    offset: Tuple[float, float] = (0.0, 0.0)
    normal: Optional[np.ndarray] = None
    normal_scale: float = 1.0
    rough_metal: Optional[np.ndarray] = None
    emissive: Optional[np.ndarray] = None
    occlusion: Optional[np.ndarray] = None
    occlusion_strength: float = 1.0

    MAPS = ("image", "normal", "rough_metal", "emissive", "occlusion")

    def __post_init__(self):
        self.image = np.ones((1, 1, 4), np.float32) if self.image is None else _as_rgba(self.image, "image")
        for name in self.MAPS[1:]:
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, _as_rgba(v, name))

    @property
    def has_extra_maps(self) -> bool:
        return any(getattr(self, n) is not None for n in self.MAPS[1:])


@dataclass
class Mesh:
    verts: np.ndarray                       # (N, 3) float32
    faces: np.ndarray                       # (M, 3) int32, triangles
    color: Optional[np.ndarray] = None      # (N, 3) linear RGB
    rough: Optional[np.ndarray] = None      # (N,)
    metal: Optional[np.ndarray] = None      # (N,)
    emit: Optional[np.ndarray] = None       # (N, 3) linear RGB radiance
    smooth: bool = True                     # smooth (interpolated) vs flat shading
    uv: Optional[np.ndarray] = None         # (N, 2) or None
    face_tex: Optional[np.ndarray] = None   # (M,) int32 index into textures, -1 = untextured
    textures: List[Texture] = field(default_factory=list)

    def __post_init__(self):
        self.verts = np.ascontiguousarray(self.verts, np.float32).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, np.int32).reshape(-1, 3)
        n, mf = len(self.verts), len(self.faces)
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= n):
            raise ValueError("face index out of range")
        self.color = np.tile(np.array(DEFAULT_COLOR, np.float32), (n, 1)) if self.color is None else np.ascontiguousarray(self.color, np.float32).reshape(n, 3)
        self.rough = np.full(n, 0.5, np.float32) if self.rough is None else np.ascontiguousarray(self.rough, np.float32).reshape(n)
        self.metal = np.zeros(n, np.float32) if self.metal is None else np.ascontiguousarray(self.metal, np.float32).reshape(n)
        self.emit = np.zeros((n, 3), np.float32) if self.emit is None else np.ascontiguousarray(self.emit, np.float32).reshape(n, 3)
        if self.uv is not None:
            self.uv = np.ascontiguousarray(self.uv, np.float32).reshape(n, 2)
        self.face_tex = np.full(mf, -1, np.int32) if self.face_tex is None else np.ascontiguousarray(self.face_tex, np.int32).reshape(mf)
        self.textures = list(self.textures)
        if self.face_tex.size and self.face_tex.max() >= len(self.textures):
            raise ValueError("face_tex refers to a missing texture")

    # ---- basics -----------------------------------------------------------
    @property
    def nverts(self) -> int:
        return len(self.verts)

    @property
    def nfaces(self) -> int:
        return len(self.faces)

    @property
    def has_uv(self) -> bool:
        return self.uv is not None

    def copy(self) -> "Mesh":
        return Mesh(self.verts.copy(), self.faces.copy(), self.color.copy(), self.rough.copy(), self.metal.copy(),
                    self.emit.copy(), self.smooth, None if self.uv is None else self.uv.copy(), self.face_tex.copy(),
                    list(self.textures))

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.nverts:
            z = np.zeros(3, np.float32)
            return z, z
        return self.verts.min(0), self.verts.max(0)

    def face_normals(self, unit: bool = True) -> np.ndarray:
        v = self.verts
        n = np.cross(v[self.faces[:, 1]] - v[self.faces[:, 0]], v[self.faces[:, 2]] - v[self.faces[:, 0]])
        if unit:
            l = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(l, 1e-20)
        return n.astype(np.float32)

    def vertex_normals(self, crease_deg: float = 60.0) -> np.ndarray:
        """Area-weighted face normals around each vertex. Vertices that share a position (UV seams, sharp-edge
        duplicates) are blended only if their own normals differ by less than ``crease_deg``, so seams stay invisible
        while cube edges stay sharp. ``crease_deg >= 180`` blends every coincident vertex."""
        fn = self.face_normals(unit=False).astype(np.float64)
        vn = np.zeros((self.nverts, 3), np.float64)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        own = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
        if self.nverts < 2:
            return own.astype(np.float32)
        key = np.round(self.verts.astype(np.float64) / 1e-5).astype(np.int64)
        _, gid = np.unique(key, axis=0, return_inverse=True)
        gid = gid.reshape(-1)
        sizes = np.bincount(gid)
        if sizes.max() == 1:
            return own.astype(np.float32)
        order = np.argsort(gid, kind="stable")
        g, own_s, vn_s = gid[order], own[order], vn[order]
        res = vn_s.copy()
        cos_t = -2.0 if crease_deg >= 180 else np.cos(np.radians(crease_deg))
        for d in range(1, int(sizes.max())):
            i = np.nonzero(g[d:] == g[:-d])[0]
            if not len(i):
                break
            j = i + d
            ok = (own_s[i] * own_s[j]).sum(1) > cos_t
            ii, jj = i[ok], j[ok]
            np.add.at(res, ii, vn_s[jj])
            np.add.at(res, jj, vn_s[ii])
        res = res / np.maximum(np.linalg.norm(res, axis=1, keepdims=True), 1e-20)
        out = np.empty_like(res)
        out[order] = res
        return out.astype(np.float32)

    def surface_area(self) -> float:
        return float(np.linalg.norm(self.face_normals(unit=False), axis=1).sum() / 2)

    def volume(self) -> float:
        """Signed volume (positive for outward-facing closed meshes)."""
        v = self.verts.astype(np.float64)
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

    # ---- combination / transforms ----------------------------------------
    def transformed(self, m4: np.ndarray) -> "Mesh":
        m4 = np.asarray(m4, np.float64)
        v = np.c_[self.verts.astype(np.float64), np.ones(self.nverts)] @ m4.T
        out = self.copy()
        out.verts = (v[:, :3] / v[:, 3:4]).astype(np.float32)
        if np.linalg.det(m4[:3, :3]) < 0:        # mirroring flips winding; keep faces outward
            out.faces = out.faces[:, ::-1].copy()
        return out

    def merged(self, other: "Mesh") -> "Mesh":
        n = self.nverts
        uv = None
        if self.uv is not None or other.uv is not None:
            a = self.uv if self.uv is not None else np.zeros((n, 2), np.float32)
            b = other.uv if other.uv is not None else np.zeros((other.nverts, 2), np.float32)
            uv = np.r_[a, b]
        ft = np.r_[self.face_tex, np.where(other.face_tex >= 0, other.face_tex + len(self.textures), -1)]
        return Mesh(np.r_[self.verts, other.verts], np.r_[self.faces, other.faces + n], np.r_[self.color, other.color],
                    np.r_[self.rough, other.rough], np.r_[self.metal, other.metal], np.r_[self.emit, other.emit],
                    self.smooth and other.smooth, uv, ft, self.textures + other.textures)

    def with_material(self, color=None, rough=None, metal=None, emit=None) -> "Mesh":
        out = self.copy()
        if color is not None:
            out.color[:] = np.asarray(color, np.float32)[:3]
        if rough is not None:
            out.rough[:] = rough
        if metal is not None:
            out.metal[:] = metal
        if emit is not None:
            out.emit[:] = np.asarray(emit, np.float32)[:3]
        return out

    def with_texture(self, texture: Texture) -> "Mesh":
        """Use ``texture`` on every face (requires UVs)."""
        if self.uv is None:
            raise ValueError("this mesh has no UVs; use a primitive that has them or add a mesh_uv_project node first")
        out = self.copy()
        out.textures = [texture]
        out.face_tex = np.zeros(self.nfaces, np.int32)
        return out

    def unwelded(self) -> "Mesh":
        """Give every face its own three vertices (needed for per-face attributes such as box UV projection)."""
        idx = self.faces.reshape(-1)
        return Mesh(self.verts[idx], np.arange(len(idx)).reshape(-1, 3), self.color[idx], self.rough[idx], self.metal[idx],
                    self.emit[idx], self.smooth, None if self.uv is None else self.uv[idx], self.face_tex.copy(), list(self.textures))

    def tangent_frame(self):
        """Per-vertex tangent T (direction of increasing u) and bitangent B (direction of increasing image-up, i.e. -d/dv),
        unit length, area-weighted over the faces that share the vertex, so a normal map's +X / +Y match glTF. Vertices
        without usable UVs get zero vectors. Returns two (N, 3) float32 arrays."""
        T = np.zeros((self.nverts, 3), np.float64)
        B = np.zeros((self.nverts, 3), np.float64)
        if self.uv is None or not self.nfaces:
            return T.astype(np.float32), B.astype(np.float32)
        f = self.faces
        p = self.verts.astype(np.float64)[f]
        uv = self.uv.astype(np.float64)[f]
        sc = np.ones((self.nfaces, 2))
        for ti, t in enumerate(self.textures):                       # a texture's scale changes the u / v directions
            sc[self.face_tex == ti] = t.scale
        uv = uv * sc[:, None, :]
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        ok = np.abs(det) > 1e-14
        r = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)[:, None]
        dpdu = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
        dpdv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
        area = np.linalg.norm(np.cross(e1, e2), axis=1)[:, None]
        unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)
        Tf, Bf = unit(dpdu) * area, unit(-dpdv) * area
        for k in range(3):
            np.add.at(T, f[:, k], Tf)
            np.add.at(B, f[:, k], Bf)
        unit_safe = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)
        return unit_safe(T).astype(np.float32), unit_safe(B).astype(np.float32)

    def _weld_indices(self, tol: float = 1e-5):
        key = np.round(self.verts.astype(np.float64) / max(tol, 1e-12)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        return first, inv.reshape(-1)

    def weld(self, tol: float = 1e-5) -> "Mesh":
        """Merge vertices at (almost) identical positions and drop degenerate faces. Positions and material of the first
        vertex win; UVs and textures are dropped because seams are exactly what welding removes."""
        if not self.nverts:
            return self.copy()
        first, inv = self._weld_indices(tol)
        f = inv[self.faces]
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        return Mesh(self.verts[first], f[ok], self.color[first], self.rough[first], self.metal[first], self.emit[first], self.smooth)

    # ---- topology ---------------------------------------------------------
    def edges(self):
        """Unique undirected edges (E, 2) and, per face corner, the edge index (M, 3)."""
        f = self.faces
        e = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1).reshape(-1, 2)
        e = np.sort(e, axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        return uniq, inv.reshape(-1, 3)

    def is_watertight(self) -> bool:
        w = self.weld()
        if not w.nfaces:
            return False
        _, inv = w.edges()
        counts = np.bincount(inv.reshape(-1))
        return bool((counts == 2).all())

    def euler_characteristic(self) -> int:
        w = self.weld()
        used = np.unique(w.faces)
        return int(len(used) - len(w.edges()[0]) + w.nfaces)

# ============================================================================ transforms
def trs_matrix(t=(0, 0, 0), r_deg=(0, 0, 0), s=(1, 1, 1)) -> np.ndarray:
    """Translate * RotZ * RotY * RotX * Scale (rotations in degrees, applied X first)."""
    rx, ry, rz = np.radians(r_deg)
    cx, sx_, cy, sy_, cz, sz_ = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    RX = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    RY = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    RZ = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = RZ @ RY @ RX @ np.diag(s)
    m[:3, 3] = t
    return m


# ============================================================================ primitives
def _compact(m: Mesh) -> Mesh:
    """Drop zero-area triangles and unused vertices."""
    area = np.linalg.norm(m.face_normals(unit=False), axis=1)
    f = m.faces[area > 1e-12]
    ft = m.face_tex[area > 1e-12]
    used = np.unique(f)
    remap = np.full(m.nverts, -1, np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(m.verts[used], remap[f], m.color[used], m.rough[used], m.metal[used], m.emit[used], m.smooth,
                None if m.uv is None else m.uv[used], ft, list(m.textures))


def _flip(m: Mesh) -> Mesh:
    out = m.copy()
    out.faces = out.faces[:, ::-1].copy()
    return out


def _grid(nu: int, nv: int, fn, flip=False):
    """Vertex grid (nu+1)*(nv+1) with positions fn(u, v) for u, v in [0, 1]; two triangles per cell.
    Returns verts, faces and the (u, v) parameter of every vertex."""
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    verts = fn(uu, vv)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    if flip:
        faces = faces[:, ::-1]
    return verts, faces, uu, vv


def plane(size_x=2.0, size_z=2.0, sub_x=1, sub_z=1) -> Mesh:
    """Horizontal plane in XZ facing +Y. UV: u runs along +X, v along +Z (v = 0 at the far edge, so a texture is upright
    when seen from the default +Z camera)."""
    v, f, uu, ww = _grid(max(1, sub_x), max(1, sub_z), lambda u, w: np.stack([(u - .5) * size_x, 0 * u, (w - .5) * size_z], 1), flip=True)
    return Mesh(v, f, uv=np.stack([uu, ww], 1))


def cube(size=(1.0, 1.0, 1.0), sub=1) -> Mesh:
    """Cube with sharp edges (each face has its own vertices). Every face maps the full 0..1 UV square, upright."""
    half = [s / 2 for s in size]
    sub = max(1, sub)
    out = None
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        a1, a2 = [(1, 2), (2, 0), (0, 1)][axis]

        def fn(u, w, axis=axis, sign=sign, a1=a1, a2=a2):
            p = np.zeros((len(u), 3))
            p[:, axis] = sign * half[axis]
            p[:, a1] = (u * 2 - 1) * half[a1]
            p[:, a2] = (w * 2 - 1) * half[a2]
            return p
        v, f, uu, ww = _grid(sub, sub, fn, flip=(sign < 0))
        p = v.copy()
        x01 = (p[:, 0] / (2 * half[0]) + .5); y01 = (p[:, 1] / (2 * half[1]) + .5); z01 = (p[:, 2] / (2 * half[2]) + .5)
        if axis == 1:
            uv = np.stack([x01, z01], 1)
        elif axis == 0:
            uv = np.stack([1 - z01 if sign > 0 else z01, 1 - y01], 1)
        else:
            uv = np.stack([x01 if sign > 0 else 1 - x01, 1 - y01], 1)
        piece = Mesh(v, f, uv=uv)
        out = piece if out is None else out.merged(piece)
    return out


def uv_sphere(radius=1.0, segments=32, rings=16) -> Mesh:
    """UV sphere with its poles on the Y axis. Equirectangular UVs (u around, v from the north pole down); the seam
    and the poles have duplicate vertices so textures do not smear."""
    segments, rings = max(3, segments), max(2, rings)

    def fn(u, w):
        th, ph = np.pi * u, 2 * np.pi * w
        return np.stack([radius * np.sin(th) * np.cos(ph), radius * np.cos(th), radius * np.sin(th) * np.sin(ph)], 1)
    v, f, uu, ww = _grid(rings, segments, fn)
    m = _compact(Mesh(v, f, uv=np.stack([1 - ww, uu], 1)))
    return _flip(m) if m.volume() < 0 else m


def icosphere(radius=1.0, subdivisions=2) -> Mesh:
    """Geodesic sphere (evenly sized triangles). It has no UVs; add a mesh_uv_project node to texture it."""
    t = (1 + 5 ** .5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
                  [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    m = Mesh(v / np.linalg.norm(v, axis=1, keepdims=True), f)
    for _ in range(int(np.clip(subdivisions, 0, 6))):
        m = subdivide(m, smooth=False)
        m = Mesh(m.verts / np.linalg.norm(m.verts, axis=1, keepdims=True), m.faces)
    m.verts *= radius
    return m


def _outward_radial(m: Mesh) -> Mesh:
    """Flip an open surface of revolution about Y if its normals point toward the axis."""
    fn, c = m.face_normals(), m.verts[m.faces].mean(1)
    return _flip(m) if float((fn[:, [0, 2]] * c[:, [0, 2]]).sum()) < 0 else m


def cylinder(radius=0.5, height=1.0, segments=32, caps=True, radius_top=None) -> Mesh:
    """Cylinder along Y (a cone when ``radius_top`` is 0). The side wraps once in u (seam duplicated); the caps use a planar
    map. Side and caps use separate vertices so the rim stays sharp."""
    segments = max(3, segments)
    rt = radius if radius_top is None else radius_top

    def fn(u, w):
        ang, r = 2 * np.pi * w, radius + (rt - radius) * u
        return np.stack([r * np.cos(ang), -height / 2 + u * height, r * np.sin(ang)], 1)
    v, f, uu, ww = _grid(1, segments, fn)
    out = _outward_radial(_compact(Mesh(v, f, uv=np.stack([1 - ww, 1 - uu], 1))))
    if caps:
        j = np.arange(segments); k = (j + 1) % segments
        for y, rad, up in ((height / 2, rt, True), (-height / 2, radius, False)):
            if rad <= 1e-9:
                continue
            ang = 2 * np.pi * j / segments
            ring = np.stack([rad * np.cos(ang), np.full(segments, y), rad * np.sin(ang)], 1)
            pts = np.r_[[[0, y, 0]], ring]
            faces = np.stack([np.zeros(segments, int), 1 + k, 1 + j], 1) if up else np.stack([np.zeros(segments, int), 1 + j, 1 + k], 1)
            uvs = np.stack([0.5 + pts[:, 0] / (2 * rad), 0.5 + pts[:, 2] / (2 * rad)], 1)
            out = out.merged(Mesh(pts, faces, uv=uvs))
    return out


def cone(radius=0.5, height=1.0, segments=32, caps=True) -> Mesh:
    return cylinder(radius, height, segments, caps, radius_top=0.0)


def torus(major=1.0, minor=0.3, segments=48, tube_segments=16) -> Mesh:
    """Donut around Y. u wraps around the big circle, v around the tube (seams duplicated)."""
    segments, tube_segments = max(3, segments), max(3, tube_segments)

    def fn(u, w):
        a, b = 2 * np.pi * u, 2 * np.pi * w
        r = major + minor * np.cos(b)
        return np.stack([r * np.cos(a), minor * np.sin(b), r * np.sin(a)], 1)
    v, f, uu, ww = _grid(segments, tube_segments, fn)
    m = Mesh(v, f, uv=np.stack([1 - uu, ww], 1))
    return _flip(m) if m.volume() < 0 else m


# ============================================================================ modifiers
def _neighbor_sums(m: Mesh):
    e, _ = m.edges()
    nsum = np.zeros((m.nverts, 3), np.float64)
    cnt = np.zeros(m.nverts, np.int64)
    np.add.at(nsum, e[:, 0], m.verts[e[:, 1]]); np.add.at(nsum, e[:, 1], m.verts[e[:, 0]])
    np.add.at(cnt, e[:, 0], 1); np.add.at(cnt, e[:, 1], 1)
    return e, nsum, cnt


def _boundary(m: Mesh):
    e, inv = m.edges()
    counts = np.bincount(inv.reshape(-1), minlength=len(e))
    return e, inv, counts


def _welded_view(m: Mesh):
    """Position-welded topology of ``m``: (welded mesh, old->welded vertex map). Used for geometry that must not tear at
    UV seams or sharp-edge duplicates."""
    first, wid = m._weld_indices()
    wf = wid[m.faces]
    ok = (wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2]) & (wf[:, 2] != wf[:, 0])
    return Mesh(m.verts[first], wf[ok]), wid


def _loop_points(wm: Mesh):
    """Loop-subdivision positions on a welded mesh: (new positions of old vertices (N,3), edge points (E,3), edges (E,2))."""
    V, F = wm.verts.astype(np.float64), wm.faces
    e, inv, counts = _boundary(wm)
    E = len(e)
    occ_edge = inv.reshape(-1)
    occ_opp = F[:, [2, 0, 1]].reshape(-1)
    order = np.argsort(occ_edge, kind="stable")
    start = np.searchsorted(occ_edge[order], np.arange(E))
    opp1 = occ_opp[order[start]]
    opp2 = occ_opp[order[np.minimum(start + 1, len(order) - 1)]]
    a, b = V[e[:, 0]], V[e[:, 1]]
    mid = np.where((counts >= 2)[:, None], 0.375 * (a + b) + 0.125 * (V[opp1] + V[opp2]), 0.5 * (a + b))
    _, nsum, cnt = _neighbor_sums(wm)
    be = e[counts == 1]
    bsum = np.zeros_like(V); bcnt = np.zeros(wm.nverts, np.int64)
    np.add.at(bsum, be[:, 0], V[be[:, 1]]); np.add.at(bsum, be[:, 1], V[be[:, 0]])
    np.add.at(bcnt, be[:, 0], 1); np.add.at(bcnt, be[:, 1], 1)
    n = np.maximum(cnt, 1)
    beta = np.where(n > 3, 3.0 / (8.0 * n), 3.0 / 16.0)
    newV = V.copy()
    interior = (bcnt == 0) & (cnt > 0)
    newV[interior] = ((1 - n * beta)[:, None] * V + beta[:, None] * nsum)[interior]
    bv = bcnt > 0
    newV[bv] = (0.75 * V + 0.125 * bsum)[bv]
    return newV, mid, e


def subdivide(m: Mesh, levels: int = 1, smooth: bool = True) -> Mesh:
    """1-to-4 triangle subdivision. ``smooth=True`` moves the points to Loop positions (rounded surface, boundaries
    preserved); ``smooth=False`` only inserts edge midpoints (shape unchanged). UVs, materials and textures are
    interpolated, and vertices are not merged, so UV seams and sharp-edge duplicates stay intact (the smoothed positions
    are computed on the position-welded topology so nothing tears)."""
    out = m.copy()
    for _ in range(int(np.clip(levels, 0, 5))):
        if not out.nfaces:
            break
        N, F = out.nverts, out.faces
        eu, inv = out.edges()
        mid_idx = N + np.arange(len(eu))
        m01, m12, m20 = mid_idx[inv[:, 0]], mid_idx[inv[:, 1]], mid_idx[inv[:, 2]]
        a_, b_, c_ = F[:, 0], F[:, 1], F[:, 2]
        faces = np.concatenate([np.stack([a_, m01, m20], 1), np.stack([b_, m12, m01], 1),
                                np.stack([c_, m20, m12], 1), np.stack([m01, m12, m20], 1)])
        lin = lambda arr: np.r_[arr, 0.5 * (arr[eu[:, 0]] + arr[eu[:, 1]])]
        verts = lin(out.verts.astype(np.float64))
        if smooth:
            wm, wid = _welded_view(out)
            if wm.nfaces:
                newV, mid_w, ew = _loop_points(wm)
                Nw = wm.nverts
                used = np.zeros(Nw, bool); used[wm.faces.ravel()] = True
                sel = used[wid]
                verts[:N][sel] = newV[wid[sel]]
                ekey = ew[:, 0].astype(np.int64) * Nw + ew[:, 1]
                a, b = wid[eu[:, 0]], wid[eu[:, 1]]
                k = np.minimum(a, b).astype(np.int64) * Nw + np.maximum(a, b)
                pos = np.minimum(np.searchsorted(ekey, k), len(ekey) - 1)
                hit = (ekey[pos] == k) & (a != b)
                verts[N:][hit] = mid_w[pos[hit]]
        out = Mesh(verts, faces, lin(out.color), lin(out.rough), lin(out.metal), lin(out.emit), out.smooth,
                   None if out.uv is None else lin(out.uv), np.tile(out.face_tex, 4), out.textures)
    return out


def smooth_laplacian(m: Mesh, iterations: int = 1, factor: float = 0.5, keep_boundary: bool = True) -> Mesh:
    """Move every vertex toward the average of its neighbours (computed on the welded topology, so seams do not open)."""
    wm, wid = _welded_view(m)
    if not wm.nfaces:
        return m.copy()
    for _ in range(int(iterations)):
        e, inv, counts = _boundary(wm)
        _, nsum, cnt = _neighbor_sums(wm)
        avg = nsum / np.maximum(cnt, 1)[:, None]
        move = wm.verts + factor * (avg - wm.verts)
        if keep_boundary:
            bnd = np.zeros(wm.nverts, bool)
            bnd[e[counts == 1].ravel()] = True
            move[bnd] = wm.verts[bnd]
        wm.verts = move.astype(np.float32)
    out = m.copy()
    out.verts = wm.verts[wid]
    return out


def _wrap_fix(u: np.ndarray, polar: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-face (M,3) periodic coordinate: shift each corner by whole turns so a face never spans the 0/1 seam. Corners that
    sit on the axis (``polar``, where the angle is undefined) take the mean of the face's other corners."""
    m = len(u)
    polar = np.zeros_like(u, bool) if polar is None else polar
    ref = u[np.arange(m), np.argmax(~polar, axis=1)][:, None]
    u = ref + (u - ref + 0.5) % 1.0 - 0.5
    cnt = (~polar).sum(1, keepdims=True)
    mean_np = np.where(~polar, u, 0.0).sum(1, keepdims=True) / np.maximum(cnt, 1)
    return np.where(polar, mean_np, u)


def uv_project(m: Mesh, mode: str = "planar", axis: str = "y", scale: float = 1.0, offset=(0.0, 0.0)) -> Mesh:
    """Generate UVs from positions. Modes: planar (along ``axis``), cylindrical / spherical (around Y), box (per-face
    dominant axis; the mesh is unwelded). ``scale`` tiles the result, ``offset`` shifts it."""
    ax = "xyz".index(axis)
    lo, hi = m.bounds()
    ext = np.maximum(hi - lo, 1e-9)
    size = float(ext.max())
    def planar_uv(p, a):                                           # p (...,3) positions, a axis index
        q = (p - lo) / size
        if a == 0:
            return np.stack([q[..., 2], 1 - q[..., 1]], -1)
        if a == 1:
            return np.stack([q[..., 0], q[..., 2]], -1)
        return np.stack([q[..., 0], 1 - q[..., 1]], -1)
    if mode == "planar":
        out = m.copy()
        out.uv = planar_uv(m.verts.astype(np.float64), ax).astype(np.float32)
    elif mode in ("cylindrical", "spherical"):
        out = m.unwelded()
        p = out.verts.astype(np.float64) - (lo + hi) / 2
        u = (1 - np.arctan2(p[:, 2], p[:, 0]) / (2 * np.pi)).reshape(-1, 3) % 1.0
        u = _wrap_fix(u, (np.hypot(p[:, 0], p[:, 2]) < 1e-6 * size).reshape(-1, 3))
        if mode == "cylindrical":
            v = ((hi[1] - out.verts[:, 1]) / ext[1]).reshape(-1, 3)
        else:
            r = np.maximum(np.linalg.norm(p, axis=1), 1e-12)
            v = (np.arccos(np.clip(p[:, 1] / r, -1, 1)) / np.pi).reshape(-1, 3)
        out.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)
    elif mode == "box":
        out = m.unwelded()
        n = np.abs(out.face_normals())
        dom = n.argmax(1)
        tri = out.verts.reshape(-1, 3, 3).astype(np.float64)
        uv = np.zeros((len(tri), 3, 2))
        for a in range(3):
            sel = dom == a
            if sel.any():
                uv[sel] = planar_uv(tri[sel], a)
        out.uv = uv.reshape(-1, 2).astype(np.float32)
    else:
        raise ValueError(f"unknown projection {mode!r} (planar, cylindrical, spherical, box)")
    out.uv = (out.uv * scale + np.asarray(offset, np.float32)).astype(np.float32)
    return out


def value_noise3(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """Smooth deterministic 3D value noise in [0, 1]; p is (N, 3)."""
    p = np.asarray(p, np.float64)
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)

    def h(ix, iy, iz):
        n = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1274126177) & 0xFFFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
        n = n ^ (n >> 16)
        return (n & 0xFFFF) / 65535.0
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out


def displace_noise(m: Mesh, amount=0.2, scale=2.0, octaves=3, seed=0, offset=(0, 0, 0)) -> Mesh:
    out = m.copy()
    n = out.vertex_normals(crease_deg=180)      # average over coincident vertices so seams and sharp edges don't tear
    p = out.verts.astype(np.float64) * scale + np.asarray(offset, np.float64)
    val, amp, norm = 0, 1.0, 0
    for o in range(max(1, int(octaves))):
        val = val + amp * value_noise3(p * (2 ** o), seed + o)
        norm += amp
        amp *= 0.5
    out.verts = (out.verts + n * ((val / norm - 0.5) * 2 * amount)[:, None]).astype(np.float32)
    return out


def twist(m: Mesh, angle_deg=90.0, axis="y") -> Mesh:
    out = m.copy()
    ax = "xyz".index(axis)
    a1, a2 = [(1, 2), (2, 0), (0, 1)][ax]
    lo, hi = out.verts[:, ax].min(), out.verts[:, ax].max()
    t = (out.verts[:, ax] - (lo + hi) / 2) / max(hi - lo, 1e-9)
    ang = np.radians(angle_deg) * t
    c, s = np.cos(ang), np.sin(ang)
    x, y = out.verts[:, a1].copy(), out.verts[:, a2].copy()
    out.verts[:, a1] = c * x - s * y
    out.verts[:, a2] = s * x + c * y
    return out


def array(m: Mesh, count=3, offset=(1.5, 0, 0)) -> Mesh:
    out = m
    for i in range(1, max(1, int(count))):
        out = out.merged(m.transformed(trs_matrix(np.asarray(offset) * i)))
    return out.copy() if out is m else out


def mirror(m: Mesh, axis="x", merge_original=True) -> Mesh:
    s = [1, 1, 1]; s["xyz".index(axis)] = -1
    mir = m.transformed(np.diag(s + [1]))
    return m.merged(mir) if merge_original else mir




# ============================================================================ file I/O
def read_obj(path: str) -> Mesh:
    """Wavefront OBJ: v / vt / f. Polygons are fan-triangulated, negative indices work, and (v, vt) pairs become UV-carrying
    vertices. Normals and materials are ignored (use mesh_texture to add images)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"mesh not found: {path}")
    verts, uvs, corners = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                p = line.split()
                uvs.append([float(p[1]), 1.0 - float(p[2]) if len(p) > 2 else 0.0])       # OBJ v is bottom-up
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    i = int(parts[0])
                    vi = i - 1 if i > 0 else len(verts) + i
                    ti = -1
                    if len(parts) > 1 and parts[1]:
                        t = int(parts[1])
                        ti = t - 1 if t > 0 else len(uvs) + t
                    idx.append((vi, ti))
                for k in range(1, len(idx) - 1):
                    corners += [idx[0], idx[k], idx[k + 1]]
    if not verts or not corners:
        raise ValueError(f"{path}: no geometry found")
    V = np.array(verts, np.float32)
    c = np.array(corners, np.int64)
    if (c[:, 1] >= 0).any() and uvs:
        pairs, inv = np.unique(c, axis=0, return_inverse=True)
        U = np.array(uvs, np.float32)
        uv = np.where((pairs[:, 1] >= 0)[:, None], U[np.maximum(pairs[:, 1], 0)], 0.0)
        return Mesh(V[pairs[:, 0]], inv.reshape(-1, 3), uv=uv)
    return Mesh(V, c[:, 0].reshape(-1, 3))


def write_obj(m: Mesh, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# OmniImage Studio\n")
        for v in m.verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if m.uv is not None:
            for t in m.uv:
                f.write(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}\n")
            for t in m.faces + 1:
                f.write(f"f {t[0]}/{t[0]} {t[1]}/{t[1]} {t[2]}/{t[2]}\n")
        else:
            for t in m.faces + 1:
                f.write(f"f {t[0]} {t[1]} {t[2]}\n")


def read_stl(path: str) -> Mesh:
    if not os.path.exists(path):
        raise FileNotFoundError(f"mesh not found: {path}")
    raw = open(path, "rb").read()
    if len(raw) >= 84 and 84 + struct.unpack("<I", raw[80:84])[0] * 50 == len(raw):       # binary
        n = struct.unpack("<I", raw[80:84])[0]
        rec = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
        tri = rec["v"].reshape(-1, 3)
    else:
        txt = raw.decode("utf-8", errors="replace")
        tri = np.array([[float(x) for x in l.split()[1:4]] for l in txt.splitlines() if l.strip().startswith("vertex")], np.float32)
    if not len(tri):
        raise ValueError(f"{path}: no triangles found")
    m = Mesh(tri, np.arange(len(tri)).reshape(-1, 3))
    return m.weld()


def write_stl(m: Mesh, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tri = m.verts[m.faces]
    fn = m.face_normals()
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = fn, tri
    with open(path, "wb") as f:
        f.write(b"OmniImage Studio".ljust(80, b" ")); f.write(struct.pack("<I", len(tri))); f.write(rec.tobytes())




def read_mesh(path: str) -> Mesh:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return read_obj(path)
    if ext == ".stl":
        return read_stl(path)
    if ext in (".glb", ".gltf"):
        from .gltf import read_gltf
        return read_gltf(path)
    raise ValueError(f"unsupported mesh format {ext!r} (supported: .obj, .stl, .glb, .gltf)")


def write_mesh(m: Mesh, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return write_obj(m, path)
    if ext == ".stl":
        return write_stl(m, path)
    if ext == ".glb":
        from .gltf import write_glb
        return write_glb(m, path)
    raise ValueError(f"unsupported mesh format {ext!r} (supported for writing: .obj, .stl, .glb)")


PRIMITIVES = {
    "cube": lambda **k: cube(**k), "plane": lambda **k: plane(**k), "sphere": lambda **k: uv_sphere(**k),
    "icosphere": lambda **k: icosphere(**k), "cylinder": lambda **k: cylinder(**k), "cone": lambda **k: cone(**k),
    "torus": lambda **k: torus(**k),
}
