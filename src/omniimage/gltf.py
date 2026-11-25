"""glTF 2.0 / GLB reader and writer for :class:`omniimage.mesh.Mesh`.

Reads: .glb and .gltf (embedded data URIs, or external .bin / image files located inside the .gltf's own folder),
node hierarchies baked to world space, triangle / strip / fan primitives, indexed or not, strided and normalised
accessors, UVs, COLOR_0, the metallic-roughness material (factors, emissive, ``KHR_materials_emissive_strength``) and its
five maps (base colour, normal, metallic-roughness, emissive, occlusion; ``KHR_texture_transform`` offset/scale). Compressed
files (Draco, meshopt) and sparse accessors are refused with a clear message rather than imported wrongly.

Writes: one binary .glb with a single mesh. Faces are grouped into primitives by material (roughness / metallic /
emission / texture set), vertex colours go to COLOR_0, maps are embedded as PNG, UV scale/offset are baked in.

The file is an untrusted input: every offset, count and index is range-checked, external files must resolve (after symlinks)
inside the model's folder, images are size-checked before decoding, and any malformed structure surfaces as ``GltfError``
(never as a stray KeyError / MemoryError).

Not covered yet: skins, morph targets, animations, cameras, lights, per-map UV sets or transforms within one material
(the first map's wins), sampler wrap modes (textures always repeat) and texture rotation.
"""
from __future__ import annotations
import base64
import json
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.media import linear_to_srgb, srgb_to_linear
from .mesh import Mesh, Texture

_GLB_MAGIC, _GLB_VERSION = 0x46546C67, 2
_CHUNK_JSON, _CHUNK_BIN = 0x4E4F534A, 0x004E4942
_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_UNSUPPORTED_REQUIRED = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_texture_basisu", "EXT_texture_webp"}
_MAX_DEPTH = 64
_MAX_ELEMENTS = 1 << 26                  # accessor count cap (67M); a file cannot make us allocate more than this
_MAX_IMAGE_PIXELS = 1 << 26              # 8192 x 8192; checked from the header, before any pixel is decoded
_MAX_FILE_BYTES = 1 << 30                # 1 GiB


class GltfError(ValueError):
    pass


# ============================================================================ reading
def _loads(raw: bytes, what: str):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:                               # bad UTF-8, bad JSON, absurd nesting (RecursionError)
        raise GltfError(f"{what}: not valid glTF JSON ({type(e).__name__})")


class _Doc:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mesh not found: {path}")
        self.base = os.path.dirname(os.path.realpath(path))
        if os.path.getsize(path) > _MAX_FILE_BYTES:
            raise GltfError(f"{path}: file is larger than the {_MAX_FILE_BYTES >> 20} MiB limit")
        with open(path, "rb") as f:
            raw = f.read()
        self.glb_bin: Optional[bytes] = None
        if raw[:4] == b"glTF":
            self.j = self._parse_glb(raw)
        else:
            self.j = _loads(raw, path)
        if not isinstance(self.j, dict) or "asset" not in self.j:
            raise GltfError(f"{path}: missing the glTF 'asset' block")
        if not str(self.j["asset"].get("version", "2")).startswith("2"):
            raise GltfError("only glTF 2.0 is supported")
        bad = _UNSUPPORTED_REQUIRED & set(self.j.get("extensionsRequired", []))
        if bad:
            raise GltfError(f"this file needs unsupported extension(s): {', '.join(sorted(bad))} "
                            "(re-export it without compression)")
        self._buffers: Dict[int, bytes] = {}
        self._images: Dict[int, Optional[np.ndarray]] = {}

    def _parse_glb(self, raw: bytes) -> dict:
        if len(raw) < 20:
            raise GltfError("truncated GLB file")
        magic, version, length = struct.unpack_from("<III", raw, 0)
        if magic != _GLB_MAGIC or version != _GLB_VERSION:
            raise GltfError("not a GLB 2.0 file")
        pos, js = 12, None
        end = min(length, len(raw))
        while pos + 8 <= end:
            clen, ctype = struct.unpack_from("<II", raw, pos)
            body = raw[pos + 8: pos + 8 + clen]
            if len(body) != clen:
                raise GltfError("truncated GLB chunk")
            if ctype == _CHUNK_JSON and js is None:
                js = body
            elif ctype == _CHUNK_BIN and self.glb_bin is None:
                self.glb_bin = bytes(body)
            pos += 8 + ((clen + 3) & ~3)
        if js is None:
            raise GltfError("GLB has no JSON chunk")
        return _loads(js, "GLB JSON chunk")

    # ---- external resources (kept inside the file's folder)
    def _uri_bytes(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            head, _, data = uri.partition(",")
            return base64.b64decode(data) if head.endswith(";base64") else data.encode("latin-1")
        from urllib.parse import unquote
        target = os.path.realpath(os.path.join(self.base, unquote(uri)))
        if os.path.commonpath([self.base, target]) != self.base:
            raise GltfError(f"external resource {uri!r} lies outside the model's folder")
        if not os.path.isfile(target):
            raise GltfError(f"external resource not found: {uri}")
        with open(target, "rb") as f:
            return f.read()

    def buffer(self, i: int) -> bytes:
        if i not in self._buffers:
            try:
                b = self.j["buffers"][i]
            except (KeyError, IndexError, TypeError):
                raise GltfError(f"buffer {i} does not exist")
            if "uri" in b:
                data = self._uri_bytes(b["uri"])
            elif i == 0 and self.glb_bin is not None:
                data = self.glb_bin
            else:
                raise GltfError(f"buffer {i} has no data")
            if len(data) < int(b.get("byteLength", 0)):
                raise GltfError(f"buffer {i} is shorter than its declared byteLength")
            self._buffers[i] = data
        return self._buffers[i]

    def view_bytes(self, vi: int) -> Tuple[bytes, int, Optional[int]]:
        try:
            v = self.j["bufferViews"][vi]
        except (KeyError, IndexError, TypeError):
            raise GltfError(f"bufferView {vi} does not exist")
        buf = self.buffer(int(v["buffer"]))
        off, ln = int(v.get("byteOffset", 0)), int(v["byteLength"])
        if off < 0 or ln < 0 or off + ln > len(buf):
            raise GltfError(f"bufferView {vi} runs past the end of its buffer")
        return buf[off:off + ln], off, v.get("byteStride")

    def accessor(self, ai: int) -> np.ndarray:
        """Accessor data as (count, ncomp) float64 for floats / normalised ints, else the raw integer dtype."""
        try:
            a = self.j["accessors"][ai]
        except (KeyError, IndexError, TypeError):
            raise GltfError(f"accessor {ai} does not exist")
        if "sparse" in a:
            raise GltfError("sparse accessors are not supported")
        ct, count, nc = int(a["componentType"]), int(a["count"]), _NCOMP[a["type"]]
        if ct not in _COMP:
            raise GltfError(f"unsupported component type {ct}")
        dt = np.dtype(_COMP[ct]).newbyteorder("<")
        if count < 0 or count > _MAX_ELEMENTS:
            raise GltfError(f"accessor {ai} has an unreasonable element count ({count})")
        if "bufferView" not in a:
            return np.zeros((count, nc), dt)
        data, _, stride = self.view_bytes(int(a["bufferView"]))
        esz = dt.itemsize * nc
        stride = int(stride) if stride else esz
        off = int(a.get("byteOffset", 0))
        if count and (off < 0 or stride < esz or off + (count - 1) * stride + esz > len(data)):
            raise GltfError(f"accessor {ai} runs past the end of its bufferView")
        if count == 0:
            return np.zeros((0, nc), dt)
        arr = np.ndarray((count, nc), dt, buffer=data, offset=off, strides=(stride, dt.itemsize)).copy()
        if a.get("normalized") and ct != 5126:
            info = np.iinfo(_COMP[ct])
            arr = np.maximum(arr.astype(np.float64) / float(info.max), -1.0) if info.min < 0 else arr.astype(np.float64) / float(info.max)
        return arr

    def image(self, ii: int) -> Optional[np.ndarray]:
        """Decoded image as (H, W, 4) float32 *sRGB-encoded* RGBA, or None if it cannot be decoded."""
        if ii in self._images:
            return self._images[ii]
        import cv2
        try:
            im = self.j["images"][ii]
            if "uri" in im:
                raw = self._uri_bytes(im["uri"])
            else:
                raw = bytes(self.view_bytes(int(im["bufferView"]))[0])
        except (KeyError, IndexError, TypeError):
            raise GltfError(f"image {ii} has no data")
        _check_image_header(raw, ii)
        dec = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        out = None
        if dec is not None:
            scale = 65535.0 if dec.dtype == np.uint16 else 255.0
            a = dec.astype(np.float32) / scale
            if a.ndim == 2:
                a = np.stack([a, a, a, np.ones_like(a)], -1)
            elif a.shape[2] == 3:
                a = np.concatenate([a[..., ::-1], np.ones(a.shape[:2] + (1,), np.float32)], -1)
            else:
                a = np.concatenate([a[..., 2::-1], a[..., 3:4]], -1)
            out = np.ascontiguousarray(a)
        self._images[ii] = out
        return out


def _check_image_header(raw: bytes, ii: int):
    """Refuse decompression bombs: read width x height from the header (PNG IHDR directly, other formats via PIL's lazy
    open) before OpenCV allocates the full image."""
    w = h = None
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        w, h = struct.unpack(">II", raw[16:24])
    else:
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
        except Exception:
            return                                       # not parseable here; imdecode will decide (and may return None)
    if w * h > _MAX_IMAGE_PIXELS:
        raise GltfError(f"image {ii} is {w} x {h}, larger than the {_MAX_IMAGE_PIXELS >> 20}-megapixel limit")


def _node_matrix(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], np.float64).reshape(4, 4).T
    t = np.array(n.get("translation", (0, 0, 0)), np.float64)
    q = np.array(n.get("rotation", (0, 0, 0, 1)), np.float64)
    s = np.array(n.get("scale", (1, 1, 1)), np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def _mesh_instances(j: dict):
    """Yield (mesh index, world matrix) for every mesh instance in the default scene."""
    nodes = j.get("nodes", [])
    meshes = j.get("meshes", [])
    if not nodes:
        for mi in range(len(meshes)):
            yield mi, np.eye(4)
        return
    if j.get("scenes"):
        sc = j["scenes"][int(j.get("scene", 0)) if int(j.get("scene", 0)) < len(j["scenes"]) else 0]
        roots = list(sc.get("nodes", []))
    else:
        child = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in child]
    stack = [(r, np.eye(4), 0) for r in reversed(roots)]
    visits = 0
    while stack:
        ni, parent, depth = stack.pop()
        visits += 1
        if depth > _MAX_DEPTH or visits > 100000 or not (0 <= ni < len(nodes)):
            continue
        n = nodes[ni]
        world = parent @ _node_matrix(n)
        if "mesh" in n:
            yield int(n["mesh"]), world
        for c in reversed(n.get("children", [])):
            stack.append((int(c), world, depth + 1))


def _texture_pixels(doc: _Doc, tex_index) -> Optional[np.ndarray]:
    try:
        src = doc.j["textures"][int(tex_index)].get("source")
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return doc.image(int(src)) if src is not None else None


def _material_texture(doc: _Doc, midx: Optional[int], mat: dict, cache: dict) -> Tuple[Optional[Texture], int]:
    """One mesh.Texture (all of the material's maps) plus the TEXCOORD set it uses, or (None, 0)."""
    if midx in cache:
        return cache[midx]
    pbr = mat.get("pbrMetallicRoughness", {})
    slots = {"image": pbr.get("baseColorTexture"), "normal": mat.get("normalTexture"),
             "rough_metal": pbr.get("metallicRoughnessTexture"), "emissive": mat.get("emissiveTexture"),
             "occlusion": mat.get("occlusionTexture")}
    maps: Dict[str, np.ndarray] = {}
    first = None                                          # (texCoord, scale, offset) shared by the whole set
    for name, info in slots.items():
        if not isinstance(info, dict) or "index" not in info:
            continue
        img = _texture_pixels(doc, info["index"])
        if img is None:
            continue
        ext = (info.get("extensions") or {}).get("KHR_texture_transform", {})
        this = (int(ext.get("texCoord", info.get("texCoord", 0))), tuple(float(x) for x in ext.get("scale", (1.0, 1.0))),
                tuple(float(x) for x in ext.get("offset", (0.0, 0.0))))
        if first is None:
            first = this
        elif this != first:
            continue                                      # a map with its own UV set / transform cannot share the set
        if name in ("image", "emissive"):                 # colour maps are sRGB-encoded; the rest are linear data
            img = np.concatenate([srgb_to_linear(img[..., :3]), img[..., 3:4]], -1)
        maps[name] = img
    if not maps:
        cache[midx] = (None, 0)
    else:
        n, o = slots["normal"], slots["occlusion"]
        cache[midx] = (Texture(maps.get("image"), first[1], first[2], normal=maps.get("normal"),
                               normal_scale=float(n.get("scale", 1.0)) if "normal" in maps else 1.0,
                               rough_metal=maps.get("rough_metal"), emissive=maps.get("emissive"),
                               occlusion=maps.get("occlusion"),
                               occlusion_strength=float(o.get("strength", 1.0)) if "occlusion" in maps else 1.0), first[0])
    return cache[midx]


def read_gltf(path: str) -> Mesh:
    """Load a .glb / .gltf as one Mesh. Raises FileNotFoundError, or GltfError (a ValueError) for anything unreadable."""
    try:
        return _read_gltf(path)
    except (GltfError, FileNotFoundError):
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError, MemoryError, struct.error) as e:
        raise GltfError(f"{path}: malformed glTF ({type(e).__name__}: {e})")


def _read_gltf(path: str) -> Mesh:
    doc = _Doc(path)
    j = doc.j
    mats = j.get("materials", [])
    tex_cache: dict = {}
    tex_list: List[Texture] = []
    tex_ids: Dict[int, int] = {}
    V, FC, COL, ROU, MET, EMI, UV, FT = [], [], [], [], [], [], [], []
    offset = 0
    any_uv = False
    for mi, world in _mesh_instances(j):
        try:
            gm = j["meshes"][mi]
        except (IndexError, KeyError):
            continue
        det = np.linalg.det(world[:3, :3])
        for prim in gm.get("primitives", []):
            mode = int(prim.get("mode", 4))
            if mode not in (4, 5, 6):
                continue                                          # points / lines carry no surface
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = np.nan_to_num(doc.accessor(int(attrs["POSITION"])).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if pos.shape[1] != 3:
                raise GltfError("POSITION must be a VEC3 accessor")
            n = len(pos)
            if "indices" in prim:
                idx = doc.accessor(int(prim["indices"])).reshape(-1).astype(np.int64)
            else:
                idx = np.arange(n, dtype=np.int64)
            if len(idx) and (idx.min() < 0 or idx.max() >= n):
                raise GltfError("index out of range")
            if mode == 4:
                idx = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            elif mode == 5:
                k = np.arange(max(len(idx) - 2, 0))
                tri = np.stack([idx[k], np.where(k % 2 == 0, idx[k + 1], idx[k + 2]), np.where(k % 2 == 0, idx[k + 2], idx[k + 1])], 1) if len(k) else np.zeros((0, 3), np.int64)
                idx = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
            else:
                k = np.arange(1, max(len(idx) - 1, 1))
                idx = np.stack([np.full(len(k), idx[0]), idx[k], idx[k + 1]], 1) if len(idx) >= 3 else np.zeros((0, 3), np.int64)
            if not len(idx):
                continue
            # material
            midx = int(prim["material"]) if "material" in prim and 0 <= int(prim["material"]) < len(mats) else None
            mat = mats[midx] if midx is not None else {}
            pbr = mat.get("pbrMetallicRoughness", {})
            base = np.array(pbr.get("baseColorFactor", (1, 1, 1, 1)), np.float64)[:3]
            rough = float(pbr.get("roughnessFactor", 1.0))
            metal = float(pbr.get("metallicFactor", 1.0))
            emis = np.array(mat.get("emissiveFactor", (0, 0, 0)), np.float64)
            strength = (mat.get("extensions") or {}).get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)
            emis = emis * float(strength)
            col = np.tile(base, (n, 1))
            if "COLOR_0" in attrs:
                c = doc.accessor(int(attrs["COLOR_0"])).astype(np.float64)
                if len(c) == n and c.shape[1] in (3, 4):
                    col = col * np.clip(c[:, :3], 0, None)
            uv = None
            tex, tc = _material_texture(doc, midx, mat, tex_cache)
            uv_name = f"TEXCOORD_{tc}"
            if uv_name in attrs:
                u = np.nan_to_num(doc.accessor(int(attrs[uv_name])).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                if len(u) == n and u.shape[1] == 2:
                    uv = u
            elif tc != 0 and "TEXCOORD_0" in attrs:
                uv = np.nan_to_num(doc.accessor(int(attrs["TEXCOORD_0"])).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if uv is not None:
                any_uv = True
            tid = -1
            if tex is not None and uv is not None:
                tid = tex_ids.setdefault(id(tex), len(tex_list))
                if tid == len(tex_list):
                    tex_list.append(tex)
            v4 = np.c_[pos, np.ones(n)] @ world.T
            verts = v4[:, :3] / np.where(np.abs(v4[:, 3:4]) < 1e-12, 1, v4[:, 3:4])
            f = idx[:, ::-1] if det < 0 else idx
            V.append(verts); FC.append(f + offset); COL.append(col)
            ROU.append(np.full(n, rough)); MET.append(np.full(n, metal)); EMI.append(np.tile(emis, (n, 1)))
            UV.append(uv if uv is not None else np.zeros((n, 2))); FT.append(np.full(len(f), tid, np.int32))
            offset += n
    if not V:
        raise GltfError(f"{path}: no triangle geometry found")
    return Mesh(np.concatenate(V), np.concatenate(FC), np.concatenate(COL), np.concatenate(ROU), np.concatenate(MET),
                np.concatenate(EMI), True, np.concatenate(UV) if any_uv else None, np.concatenate(FT), tex_list)


# ============================================================================ writing
def _pad4(b: bytes, fill: bytes = b"\0") -> bytes:
    return b + fill * (-len(b) % 4)


def _encode_png(img: np.ndarray, colour: bool = True) -> bytes:
    """PNG bytes for a linear (H, W, 4) map. ``colour`` maps are sRGB-encoded with alpha; data maps are stored as-is (RGB)."""
    import cv2
    if colour:
        rgb = np.clip(linear_to_srgb(np.clip(img[..., :3], 0, 1)), 0, 1)
        out = np.concatenate([rgb, np.clip(img[..., 3:4], 0, 1)], -1)[..., [2, 1, 0, 3]]
    else:
        out = np.clip(img[..., :3], 0, 1)[..., ::-1]
    ok, buf = cv2.imencode(".png", (out * 255 + 0.5).astype(np.uint8))
    if not ok:
        raise GltfError("could not encode texture as PNG")
    return buf.tobytes()


def write_glb(m: Mesh, path: str):
    if not m.nfaces:
        raise GltfError("cannot write an empty mesh")
    src = m if m.smooth else m.unwelded()
    normals = src.vertex_normals(crease_deg=60.0 if m.smooth else 0.0)
    has_uv = src.uv is not None

    # faces grouped by material
    fv = src.faces
    r = np.round(src.rough[fv].mean(1), 3)
    mt = np.round(src.metal[fv].mean(1), 3)
    em = np.round(src.emit[fv].mean(1), 3)
    key = np.column_stack([src.face_tex, r, mt, em])
    groups, ginv = np.unique(key, axis=0, return_inverse=True)
    ginv = ginv.reshape(-1)

    bin_parts: List[bytes] = []
    views: List[dict] = []
    accessors: List[dict] = []

    def add_view(data: bytes, target: Optional[int] = None) -> int:
        off = sum(len(p) for p in bin_parts)
        bin_parts.append(_pad4(data))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_acc(arr: np.ndarray, ctype: int, typ: str, target: Optional[int], minmax: bool = False) -> int:
        a = {"bufferView": add_view(np.ascontiguousarray(arr).tobytes(), target), "componentType": ctype,
             "count": int(len(arr)), "type": typ}
        if minmax:
            a["min"] = [float(x) for x in arr.min(0)]
            a["max"] = [float(x) for x in arr.max(0)]
        accessors.append(a)
        return len(accessors) - 1

    images, textures, materials, primitives = [], [], [], []
    tex_map: Dict[int, dict] = {}
    ext_used = set()
    for gi, grp in enumerate(groups):
        tid, rough, metal = grp[0], grp[1], grp[2]
        fsel = fv[ginv == gi]
        used, inv = np.unique(fsel, return_inverse=True)
        idx = inv.reshape(-1, 3).astype(np.uint32)
        attrs = {"POSITION": add_acc(src.verts[used].astype(np.float32), 5126, "VEC3", 34962, True),
                 "NORMAL": add_acc(normals[used].astype(np.float32), 5126, "VEC3", 34962),
                 "COLOR_0": add_acc(np.clip(src.color[used], 0, None).astype(np.float32), 5126, "VEC3", 34962)}
        tid = int(tid)
        mat = {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": float(metal),
                                        "roughnessFactor": float(rough)}}
        if has_uv:
            uv = src.uv[used].astype(np.float64)
            if tid >= 0:
                t = src.textures[tid]
                uv = uv * np.asarray(t.scale, np.float64) + np.asarray(t.offset, np.float64)
            attrs["TEXCOORD_0"] = add_acc(uv.astype(np.float32), 5126, "VEC2", 34962)
        if tid >= 0 and has_uv:
            if tid not in tex_map:
                t = src.textures[tid]
                info: Dict[str, dict] = {}

                def add_tex(png: bytes) -> int:
                    images.append({"bufferView": add_view(png), "mimeType": "image/png"})
                    textures.append({"source": len(images) - 1})
                    return len(textures) - 1
                if not (t.image.shape[:2] == (1, 1) and np.allclose(t.image, 1.0)):        # skip the "white" placeholder
                    info["base"] = {"index": add_tex(_encode_png(t.image))}
                if t.normal is not None:
                    info["normal"] = {"index": add_tex(_encode_png(t.normal, colour=False)), "scale": float(t.normal_scale)}
                if t.rough_metal is not None:
                    info["rm"] = {"index": add_tex(_encode_png(t.rough_metal, colour=False))}
                if t.emissive is not None:
                    info["emissive"] = {"index": add_tex(_encode_png(t.emissive))}
                if t.occlusion is not None:
                    info["occlusion"] = {"index": add_tex(_encode_png(t.occlusion, colour=False)),
                                         "strength": float(t.occlusion_strength)}
                tex_map[tid] = info
            info = tex_map[tid]
            if "base" in info:
                mat["pbrMetallicRoughness"]["baseColorTexture"] = dict(info["base"])
            if "rm" in info:
                mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = dict(info["rm"])
            if "normal" in info:
                mat["normalTexture"] = dict(info["normal"])
            if "emissive" in info:
                mat["emissiveTexture"] = dict(info["emissive"])
            if "occlusion" in info:
                mat["occlusionTexture"] = dict(info["occlusion"])
        e = src.emit[used].mean(0).astype(np.float64)
        if e.max() > 1e-6:
            s = float(e.max())
            if s > 1.0:
                mat["emissiveFactor"] = [float(x) for x in e / s]
                mat["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": s}}
                ext_used.add("KHR_materials_emissive_strength")
            else:
                mat["emissiveFactor"] = [float(x) for x in e]
        materials.append(mat)
        primitives.append({"attributes": attrs, "indices": add_acc(idx.reshape(-1, 1), 5125, "SCALAR", 34963),
                           "material": len(materials) - 1, "mode": 4})

    doc = {"asset": {"version": "2.0", "generator": "OmniImage Studio"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0, "name": "mesh"}], "meshes": [{"primitives": primitives}], "materials": materials,
           "accessors": accessors, "bufferViews": views}
    binblob = b"".join(bin_parts)
    doc["buffers"] = [{"byteLength": len(binblob)}]
    if images:
        doc["images"], doc["textures"] = images, textures
    if ext_used:
        doc["extensionsUsed"] = sorted(ext_used)
    js = _pad4(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(binblob)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total))
        f.write(struct.pack("<II", len(js), _CHUNK_JSON)); f.write(js)
        f.write(struct.pack("<II", len(binblob), _CHUNK_BIN)); f.write(binblob)
