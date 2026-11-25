"""CPU software renderer for triangle meshes: z-buffer rasteriser + deferred physically-inspired shading
(GGX specular, Lambert diffuse, hemispherical environment light, key/fill/rim lights, shadow-mapped key light),
supersampling, wireframe overlay and compositing passes (depth, normal).

Everything is vectorised NumPy; the only Python loop is per-triangle rasterisation, so meshes up to ~30k
triangles render in a few seconds. Output is scene-linear RGBA."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .mesh import Mesh, plane

PI = np.pi


def _norm(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def dir_from_angles(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector from a target toward azimuth/elevation (az 0 = +Z, 90 = +X; Y up)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], np.float64)


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    fov: float = 40.0                 # vertical field of view, degrees
    aspect: float = 16 / 9
    ortho: bool = False
    ortho_height: float = 2.0
    near: float = 0.05
    far: float = 100.0
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    def basis(self):
        f = _norm(np.asarray(self.target, np.float64) - np.asarray(self.eye, np.float64))
        up = np.asarray(self.up, np.float64)
        if abs(np.dot(f, up)) > 0.999:
            up = np.array([0.0, 0.0, 1.0])
        r = _norm(np.cross(f, up))
        u = np.cross(r, f)
        return f, r, u

    def project(self, pts: np.ndarray, W: int, H: int):
        """-> sx, sy (pixels, y down), z (linear in screen space, NDC), depth (distance along view axis)."""
        f, r, u = self.basis()
        d = pts.astype(np.float64) - np.asarray(self.eye, np.float64)
        x, y, z = d @ r, d @ u, d @ f
        if self.ortho:
            hh = self.ortho_height / 2
            xn, yn = x / (hh * self.aspect), y / hh
            zn = 2 * (z - self.near) / (self.far - self.near) - 1
        else:
            th = np.tan(np.radians(self.fov) / 2)
            zs = np.maximum(z, 1e-9)
            xn, yn = x / (zs * th * self.aspect), y / (zs * th)
            zn = (self.far + self.near) / (self.far - self.near) - 2 * self.far * self.near / ((self.far - self.near) * zs)
        sx = (xn * 0.5 + 0.5) * W
        sy = (1 - (yn * 0.5 + 0.5)) * H
        return sx, sy, zn, z

    def rays(self, W: int, H: int):
        """Unit view-ray directions for every pixel centre, (H, W, 3)."""
        f, r, u = self.basis()
        xs = (np.arange(W) + 0.5) / W * 2 - 1
        ys = 1 - (np.arange(H) + 0.5) / H * 2
        if self.ortho:
            return np.broadcast_to(f, (H, W, 3))
        th = np.tan(np.radians(self.fov) / 2)
        d = f[None, None] + xs[None, :, None] * th * self.aspect * r[None, None] + ys[:, None, None] * th * u[None, None]
        return _norm(d)


def rasterize(sx, sy, sz, faces, W, H, valid=None):
    """Z-buffer. Returns (tri id (H,W) int32, screen barycentric w0, w1 (H,W), z (H,W))."""
    zbuf = np.full((H, W), np.inf, np.float32)
    tri = np.full((H, W), -1, np.int32)
    b0 = np.zeros((H, W), np.float32)
    b1 = np.zeros((H, W), np.float32)
    if not len(faces):
        return tri, b0, b1, zbuf
    X, Y, Z = sx[faces], sy[faces], sz[faces]
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    keep = np.abs(area) > 1e-9
    if valid is not None:
        keep &= valid
    keep &= (X.max(1) >= 0) & (X.min(1) <= W) & (Y.max(1) >= 0) & (Y.min(1) <= H)
    x0s, x1s, x2s = X[:, 0].tolist(), X[:, 1].tolist(), X[:, 2].tolist()
    y0s, y1s, y2s = Y[:, 0].tolist(), Y[:, 1].tolist(), Y[:, 2].tolist()
    z0s, z1s, z2s = Z[:, 0].tolist(), Z[:, 1].tolist(), Z[:, 2].tolist()
    ar = area.tolist()
    ix0 = np.maximum(np.floor(X.min(1) - 0.5), 0).astype(int).tolist()
    ix1 = np.minimum(np.ceil(X.max(1) - 0.5), W - 1).astype(int).tolist()
    iy0 = np.maximum(np.floor(Y.min(1) - 0.5), 0).astype(int).tolist()
    iy1 = np.minimum(np.ceil(Y.max(1) - 0.5), H - 1).astype(int).tolist()
    for t in np.nonzero(keep)[0].tolist():
        xa, xb = ix0[t], ix1[t]
        ya, yb = iy0[t], iy1[t]
        if xa > xb or ya > yb:
            continue
        px = np.arange(xa, xb + 1, dtype=np.float32)[None, :] + 0.5
        py = np.arange(ya, yb + 1, dtype=np.float32)[:, None] + 0.5
        a = ar[t]
        x0, x1, x2, y0, y1, y2 = x0s[t], x1s[t], x2s[t], y0s[t], y1s[t], y2s[t]
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / a
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        z = w0 * z0s[t] + w1 * z1s[t] + w2 * z2s[t]
        sl = (slice(ya, yb + 1), slice(xa, xb + 1))
        win = inside & (z < zbuf[sl])
        if win.any():
            zbuf[sl] = np.where(win, z, zbuf[sl])
            tri[sl] = np.where(win, t, tri[sl])
            b0[sl] = np.where(win, w0, b0[sl])
            b1[sl] = np.where(win, w1, b1[sl])
    return tri, b0, b1, zbuf


@dataclass
class RenderSettings:
    width: int = 960
    height: int = 540
    ssaa: int = 2
    # camera
    camera_mode: str = "auto"                 # auto: orbit around the object / manual
    azimuth: float = 30.0
    elevation: float = 20.0
    zoom: float = 1.0
    fov: float = 40.0
    ortho: bool = False
    eye: Tuple[float, float, float] = (3.0, 2.0, 4.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # lights
    key_azimuth: float = 70.0                 # relative to the camera in auto mode (absolute in manual mode)
    key_elevation: float = 42.0
    key_intensity: float = 1.6
    key_color: Tuple[float, float, float] = (1.0, 0.96, 0.9)
    fill_intensity: float = 0.25
    rim_intensity: float = 0.45
    ambient: float = 0.5
    sky_top: Tuple[float, float, float] = (0.12, 0.2, 0.42)       # linear
    sky_bottom: Tuple[float, float, float] = (0.34, 0.32, 0.3)    # linear
    background: str = "sky"                   # sky | transparent | color
    background_color: Tuple[float, float, float] = (0.02, 0.02, 0.03)
    shadows: bool = True
    shadow_res: int = 1024
    floor: str = "none"                       # none | plane | checker
    exposure: float = 0.0
    tonemap: str = "aces"                     # aces | none (none keeps unbounded scene-linear values)
    wireframe: bool = False
    wire_color: Tuple[float, float, float] = (0.02, 0.02, 0.02)
    wire_width: float = 1.0


def _env(dirs, top, bottom, strength):
    t = (dirs[..., 1] * 0.5 + 0.5)[..., None]
    return (np.asarray(bottom) * (1 - t) + np.asarray(top) * t) * strength


def _checker_floor(y, half, tiles, c0=(0.55, 0.55, 0.55), c1=(0.18, 0.18, 0.2)) -> Mesh:
    parts = None
    step = 2 * half / tiles
    for i in range(tiles):
        for j in range(tiles):
            p = plane(step, step).transformed(np.array([[1, 0, 0, -half + (i + .5) * step], [0, 1, 0, y],
                                                       [0, 0, 1, -half + (j + .5) * step], [0, 0, 0, 1]]))
            p = p.with_material(color=c0 if (i + j) % 2 == 0 else c1, rough=0.7)
            parts = p if parts is None else parts.merged(p)
    return parts


# ---------------------------------------------------------------------------- texture sampling
def _mips(tex, name: str = "image") -> list:
    """Lazily built box-filtered mip pyramid for one map of a Texture (cached on the object)."""
    cache = tex.__dict__.setdefault("_mip_cache", {})
    m = cache.get(name)
    if m is None:
        import cv2
        m = [np.ascontiguousarray(getattr(tex, name), np.float32)]
        while min(m[-1].shape[:2]) > 1:
            h, w = m[-1].shape[:2]
            nh, nw = max(1, h // 2), max(1, w // 2)
            m.append(cv2.resize(m[-1], (nw, nh), interpolation=cv2.INTER_AREA).reshape(nh, nw, 4))
        cache[name] = m
    return m


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear lookup with repeat wrapping; (u, v) in texture space, v = 0 at the top row."""
    h, w = img.shape[:2]
    x, y = u * w - 0.5, v * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    xa, xb, ya, yb = x0 % w, (x0 + 1) % w, y0 % h, (y0 + 1) % h
    top = img[ya, xa] * (1 - fx) + img[ya, xb] * fx
    bot = img[yb, xa] * (1 - fx) + img[yb, xb] * fx
    return top * (1 - fy) + bot * fy


def sample_texture(tex, u, v, lod, name: str = "image") -> np.ndarray:
    """Trilinear sample of one map of ``tex`` (a mesh.Texture) at UVs (u, v), mip level ``lod`` per sample. -> (n, 4)."""
    mips = _mips(tex, name)
    top = len(mips) - 1
    lod = np.clip(lod, 0, top)
    l0 = np.floor(lod).astype(np.int64)
    fr = (lod - l0)[:, None]
    l1 = np.minimum(l0 + 1, top)
    out = np.zeros((len(u), 4), np.float64)
    for lv in np.unique(np.r_[l0, l1]):
        for lvl_arr, wgt in ((l0, 1 - fr), (l1, fr)):
            sel = lvl_arr == lv
            if sel.any():
                out[sel] += _bilinear(mips[lv], u[sel], v[sel]) * wgt[sel]
    return out


def _uv_at(px, py, X, Y, invw, UV):
    """Perspective-correct UV at screen positions (px, py) for per-sample triangles (X, Y screen coords (K,3), invw (K,3),
    UV (K,3,2)). Works for positions outside the triangle too (analytic extrapolation), which gives exact derivatives."""
    a = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    a = np.where(np.abs(a) < 1e-12, 1e-12, a)
    w0 = ((X[:, 1] - px) * (Y[:, 2] - py) - (X[:, 2] - px) * (Y[:, 1] - py)) / a
    w1 = ((X[:, 2] - px) * (Y[:, 0] - py) - (X[:, 0] - px) * (Y[:, 2] - py)) / a
    pc = np.stack([w0, w1, 1 - w0 - w1], 1) * invw
    s = pc.sum(1, keepdims=True)
    s = np.where(np.abs(s) < 1e-20, 1e-20, s)
    return (UV * (pc / s)[..., None]).sum(1)


def texture_maps(scene: Mesh, tri_ids, ft, px, py, sx, sy, invw) -> dict:
    """Sample every texture map of the faces under the given pixels. Returns per-pixel arrays (K = number of pixels):
    ``color`` (K,3) multiplier for the albedo, ``rough`` / ``metal`` (K,) multipliers, ``emis`` (K,3) multiplier,
    ``ao`` (K,) ambient factor, ``normal`` (K,3) tangent-space unit normals and ``has_normal`` (K,) bool. Pixels whose face
    has no texture (or no such map) get neutral values (1, 1, 1, 1, (0,0,1), False)."""
    K = len(tri_ids)
    out = {"color": np.ones((K, 3)), "rough": np.ones(K), "metal": np.ones(K), "emis": np.ones((K, 3)), "ao": np.ones(K),
           "normal": np.tile([0.0, 0.0, 1.0], (K, 1)), "has_normal": np.zeros(K, bool)}
    tid = scene.face_tex[tri_ids]
    if scene.uv is None or not (tid >= 0).any():
        return out
    X, Y = sx[ft], sy[ft]
    UV = scene.uv.astype(np.float64)[ft]
    iw = invw[ft]
    for t in np.unique(tid[tid >= 0]):
        tex = scene.textures[int(t)]
        sel = np.nonzero(tid == t)[0]
        args = (X[sel], Y[sel], iw[sel], UV[sel])
        c = _uv_at(px[sel], py[sel], *args)
        cx = _uv_at(px[sel] + 1, py[sel], *args)
        cy = _uv_at(px[sel], py[sel] + 1, *args)
        sc = np.asarray(tex.scale, np.float64)
        c, dx, dy = c * sc + np.asarray(tex.offset, np.float64), (cx - c) * sc, (cy - c) * sc
        c = np.clip(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0), -1e7, 1e7)         # keeps the integer texel index sane
        dx, dy = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

        def look(name):
            h, w = getattr(tex, name).shape[:2]
            rho = np.maximum(np.hypot(dx[:, 0] * w, dx[:, 1] * h), np.hypot(dy[:, 0] * w, dy[:, 1] * h))
            return sample_texture(tex, c[:, 0], c[:, 1], np.log2(np.maximum(rho, 1e-8)), name)  # rho is finite here

        out["color"][sel] = look("image")[:, :3]
        if tex.rough_metal is not None:
            rm = look("rough_metal")
            out["rough"][sel], out["metal"][sel] = rm[:, 1], rm[:, 2]
        if tex.emissive is not None:
            out["emis"][sel] = look("emissive")[:, :3]
        if tex.occlusion is not None:
            out["ao"][sel] = 1.0 + tex.occlusion_strength * (look("occlusion")[:, 0] - 1.0)
        if tex.normal is not None:
            n = look("normal")[:, :3] * 2.0 - 1.0
            n[:, :2] *= tex.normal_scale
            out["normal"][sel] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            out["has_normal"][sel] = True
    return out


def _gather(arr, faces_t, b):
    return (arr[faces_t] * b[..., None]).sum(1) if arr.ndim == 2 else (arr[faces_t] * b).sum(1)


def _sanitized(mesh: Mesh) -> Mesh:
    """A mesh from an untrusted file (or a buggy node) may hold NaN / inf: drop faces that use a non-finite vertex and zero
    non-finite UVs, so one bad value cannot poison the camera fit or the texture lookups. No copy when all is finite."""
    bad_v = ~np.isfinite(mesh.verts).all(1)
    bad_uv = mesh.uv is not None and not np.isfinite(mesh.uv).all()
    if not bad_v.any() and not bad_uv:
        return mesh
    out = mesh.copy()
    if bad_v.any():
        keep = ~bad_v[out.faces].any(1)
        out.verts = np.where(bad_v[:, None], 0.0, out.verts).astype(np.float32)
        out.faces, out.face_tex = out.faces[keep], out.face_tex[keep]
    if bad_uv:
        out.uv = np.nan_to_num(out.uv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def render(mesh: Mesh, s: Optional[RenderSettings] = None) -> Dict[str, np.ndarray]:
    """Render ``mesh``. Returns {"image": RGBA float32, "depth": (H,W) 0..1 (near=1, background 0), "normal": RGBA}."""
    s = s or RenderSettings()
    ss = int(np.clip(s.ssaa, 1, 4))
    W, H = int(s.width) * ss, int(s.height) * ss
    mesh = _sanitized(mesh)
    if not mesh.nfaces:
        mesh = Mesh(np.zeros((3, 3)), np.zeros((0, 3), np.int32))
    lo, hi = mesh.bounds()
    center = (lo.astype(np.float64) + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
    scene = mesh
    if s.floor != "none" and mesh.nfaces:
        y = float(lo[1]) - 1e-3 * radius
        half = 6 * radius
        floor = _checker_floor(y, half, 12) if s.floor == "checker" else plane(2 * half, 2 * half).transformed(
            np.array([[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])).with_material(color=(0.45, 0.45, 0.47), rough=0.8)
        scene = mesh.merged(floor)
    scene = scene.copy()
    V = scene.verts.astype(np.float64)
    F = scene.faces

    # ---- camera
    aspect = s.width / s.height
    if s.camera_mode == "manual":
        cam = Camera(np.array(s.eye, np.float64), np.array(s.target, np.float64), s.fov, aspect, s.ortho)
        cam.ortho_height = 2 * float(np.linalg.norm(np.array(s.eye) - np.array(s.target))) * np.tan(np.radians(s.fov) / 2)
    else:
        th = np.tan(np.radians(s.fov) / 2)
        dist = radius * max(1.0, 1.0 / aspect) / np.sin(np.radians(s.fov) / 2) / max(s.zoom, 1e-3) if not s.ortho else radius * 4
        eye = center + dir_from_angles(s.azimuth, s.elevation) * dist
        cam = Camera(eye, center, s.fov, aspect, s.ortho)
        cam.ortho_height = 2 * radius / max(s.zoom, 1e-3) * max(1.0, 1.0 / aspect)
    cam.near = max(float(np.linalg.norm(cam.eye - cam.target)) * 0.01, 1e-3)
    cam.far = float(np.linalg.norm(cam.eye - cam.target)) + 40 * radius

    sx, sy, sz, depth = cam.project(V, W, H)
    valid = (depth[F] > cam.near).all(1)       # triangles crossing the near plane are dropped
    invw = 1.0 / np.maximum(depth, 1e-9) if not cam.ortho else np.ones_like(depth)
    tri, w0, w1, _ = rasterize(sx, sy, sz, F, W, H, valid)
    covered = tri >= 0
    idx = np.nonzero(covered)
    ft = F[tri[idx]]                                   # (K, 3)
    sw = np.stack([w0[idx], w1[idx], 1 - w0[idx] - w1[idx]], 1)
    pc = sw * invw[ft]
    bary = pc / np.maximum(pc.sum(1, keepdims=True), 1e-20)

    def interp(a):
        a = np.asarray(a, np.float64)
        if a.ndim == 2:
            return (a[ft] * bary[..., None]).sum(1)
        return (a[ft] * bary).sum(1)

    P = interp(V)
    fnorm = scene.face_normals().astype(np.float64)[tri[idx]]
    N = _norm(interp(scene.vertex_normals())) if scene.smooth else fnorm
    albedo = interp(scene.color)
    rough_v, metal_v, emit = interp(scene.rough), interp(scene.metal), interp(scene.emit)
    maps = None
    if scene.textures and scene.uv is not None:
        maps = texture_maps(scene, tri[idx], ft, idx[1] + 0.5, idx[0] + 0.5, sx, sy, invw)
        albedo = albedo * maps["color"]
        rough_v, metal_v, emit = rough_v * maps["rough"], metal_v * maps["metal"], emit * maps["emis"]
    rough = np.clip(rough_v, 0.04, 1.0)
    metal = np.clip(metal_v, 0, 1)
    ao = maps["ao"] if maps is not None else 1.0
    view = _norm(np.asarray(cam.eye, np.float64) - P) if not cam.ortho else np.broadcast_to(-cam.basis()[0], P.shape)
    flip = (N * view).sum(1) < 0                       # double-sided: decided from the geometric normal
    if maps is not None and maps["has_normal"].any():
        Tv, Bv = scene.tangent_frame()
        T, Bt = interp(Tv), interp(Bv)
        T = _norm(T - N * (N * T).sum(1, keepdims=True))
        Bt = _norm(Bt - N * (N * Bt).sum(1, keepdims=True))
        nm = maps["normal"]
        mapped = _norm(T * nm[:, 0:1] + Bt * nm[:, 1:2] + N * nm[:, 2:3])
        N = np.where(maps["has_normal"][:, None], mapped, N)
    N = np.where(flip[:, None], -N, N)
    NdotV = np.clip((N * view).sum(1), 1e-3, 1)

    # ---- lights
    kaz = s.key_azimuth + (s.azimuth if s.camera_mode != "manual" else 0.0)
    key_L = dir_from_angles(kaz, s.key_elevation)
    lights = [(key_L, np.array(s.key_color) * s.key_intensity),
              (dir_from_angles(kaz - 130, 15), np.array([0.75, 0.85, 1.0]) * s.fill_intensity),
              (dir_from_angles(kaz + 180, 35), np.array([1.0, 1.0, 1.0]) * s.rim_intensity)]
    F0 = 0.04 * (1 - metal)[:, None] + albedo * metal[:, None]
    diff_albedo = albedo * (1 - metal)[:, None]
    a = rough ** 2
    k = (rough + 1) ** 2 / 8
    g1 = lambda x: x / (x * (1 - k) + k)

    shadow = np.ones(len(P))
    if s.shadows and s.key_intensity > 0 and mesh.nfaces:
        shadow = _shadow(mesh, P, N, key_L, center, radius, int(s.shadow_res))

    col = np.zeros_like(P)
    for li, (L, radiance) in enumerate(lights):
        if not np.any(radiance > 0):
            continue
        NdotL = np.clip((N @ L), 0, 1)
        Hh = _norm(L[None] + view)
        NdotH = np.clip((N * Hh).sum(1), 0, 1)
        VdotH = np.clip((view * Hh).sum(1), 0, 1)
        D = a ** 2 / (PI * (NdotH ** 2 * (a ** 2 - 1) + 1) ** 2 + 1e-12)
        Fr = F0 + (1 - F0) * ((1 - VdotH) ** 5)[:, None]
        G = g1(np.maximum(NdotL, 1e-4)) * g1(NdotV)
        spec = (D * G / (4 * NdotV + 1e-6))[:, None] * Fr * PI
        vis = shadow if li == 0 else 1.0
        col += (diff_albedo * NdotL[:, None] + spec * (NdotL > 0)[:, None]) * radiance * np.reshape(vis, (-1, 1))
    top, bottom = np.array(s.sky_top), np.array(s.sky_bottom)
    amb = _env(N, top, bottom, s.ambient)
    col += diff_albedo * amb * np.reshape(ao, (-1, 1))
    refl = _norm(2 * (N * view).sum(1, keepdims=True) * N - view)
    f_amb = F0 + (np.maximum(1 - rough, F0.max(1))[:, None] - F0) * ((1 - NdotV) ** 5)[:, None]
    col += f_amb * _env(refl, top, bottom, s.ambient) * (1 - 0.65 * rough)[:, None] * np.reshape(ao, (-1, 1))
    col += emit
    col *= 2.0 ** s.exposure

    # ---- compose
    img = np.zeros((H, W, 4), np.float32)
    if s.background == "transparent":
        pass
    else:
        if s.background == "color":
            img[..., :3] = np.array(s.background_color, np.float32)
        else:
            t = np.linspace(1, 0, H)[:, None, None]
            img[..., :3] = ((np.array(bottom) * (1 - t) + np.array(top) * t) * 1.0).astype(np.float32) * np.ones((1, W, 1), np.float32)
        img[..., 3] = 1.0
        img[..., :3] *= 2.0 ** s.exposure
    if s.wireframe and len(idx[0]):
        X, Y = sx[ft], sy[ft]
        area2 = np.abs((X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0]))
        elen = np.stack([np.hypot(X[:, 2] - X[:, 1], Y[:, 2] - Y[:, 1]), np.hypot(X[:, 0] - X[:, 2], Y[:, 0] - Y[:, 2]),
                         np.hypot(X[:, 1] - X[:, 0], Y[:, 1] - Y[:, 0])], 1)
        dist = sw * (area2[:, None] / np.maximum(elen, 1e-9))
        wire = np.clip(1.5 - dist.min(1) / (0.5 * s.wire_width * ss), 0, 1)
        col = col * (1 - wire[:, None]) + np.array(s.wire_color) * wire[:, None]
    img[idx[0], idx[1], :3] = col.astype(np.float32)
    img[idx[0], idx[1], 3] = 1.0

    # ---- passes
    dpass = np.zeros((H, W), np.float32)
    if len(idx[0]):
        dd = (depth[ft] * bary).sum(1)
        dmin, dmax = float(dd.min()), float(dd.max())
        dpass[idx] = (1.0 - (dd - dmin) / max(dmax - dmin, 1e-9) * 0.95).astype(np.float32)
    npass = np.zeros((H, W, 4), np.float32)
    npass[idx[0], idx[1], :3] = (N * 0.5 + 0.5).astype(np.float32)
    npass[idx[0], idx[1], 3] = 1.0

    def down(a):
        if ss == 1:
            return a
        sh = (int(s.height), ss, int(s.width), ss) + a.shape[2:]
        return a.reshape(sh).mean((1, 3)).astype(np.float32)
    out_img = img
    if s.background == "transparent" and ss > 1:                     # average premultiplied, then un-premultiply
        pm = img.copy(); pm[..., :3] *= pm[..., 3:4]
        d = down(pm); al = d[..., 3:4]
        d[..., :3] = np.where(al > 1e-6, d[..., :3] / np.maximum(al, 1e-6), 0)
        out_img = d
    else:
        out_img = down(img)
    out_img = out_img.astype(np.float32)
    if s.tonemap == "aces":
        x = np.maximum(out_img[..., :3], 0)
        out_img[..., :3] = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    return {"image": out_img, "depth": down(dpass), "normal": down(npass)}


def _shadow(mesh: Mesh, P, N, L, center, radius, res) -> np.ndarray:
    """Soft (3x3 PCF) shadow factor in [0, 1] for world points P lit by a directional light from direction L."""
    hh = radius * 1.7
    eye = center + L * radius * 5
    cam = Camera(eye, center, ortho=True, aspect=1.0, ortho_height=2 * hh, near=radius * 0.5, far=radius * 10)
    sx, sy, sz, d = cam.project(mesh.verts.astype(np.float64), res, res)
    _, _, _, zmap = rasterize(sx, sy, sz, mesh.faces, res, res)
    px, py, pz, _ = cam.project(P, res, res)
    ndc_bias = (0.012 * radius + 0.02 * radius * (1 - np.clip(N @ L, 0, 1))) * 2 / (cam.far - cam.near)
    lit = np.zeros(len(P)); n = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            ix = np.round(px - 0.5 + dx).astype(int); iy = np.round(py - 0.5 + dy).astype(int)
            inside = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
            zm = np.where(inside, zmap[np.clip(iy, 0, res - 1), np.clip(ix, 0, res - 1)], np.inf)
            lit += (pz - ndc_bias <= zm); n += 1
    return lit / n
