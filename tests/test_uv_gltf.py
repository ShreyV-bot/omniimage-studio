"""UVs, textures, texture sampling in the renderer, OBJ vt round trip and glTF / GLB import-export."""
import base64
import json
import struct

import cv2
import numpy as np
import pytest

import omniimage as o
from omniimage import mesh as M
from omniimage.core.params import ParamError
from omniimage.core.media import srgb_to_linear
from omniimage.gltf import GltfError, read_gltf, write_glb
from omniimage.raster import RenderSettings, render, sample_texture


def checker(n=64, cells=8, a=(0.9, 0.1, 0.1), b=(0.1, 0.1, 0.9)):
    yy, xx = np.mgrid[0:n, 0:n]
    c = (((xx // (n // cells)) + (yy // (n // cells))) % 2).astype(np.float32)[..., None]
    return np.concatenate([c * np.array(a) + (1 - c) * np.array(b), np.ones((n, n, 1))], -1).astype(np.float32)


# ------------------------------------------------------------------ UV data model
@pytest.mark.parametrize("make", [M.plane, M.cube, M.uv_sphere, M.cylinder, M.cone, M.torus])
def test_primitives_have_uvs_in_unit_square_and_stay_closed(make):
    m = make()
    assert m.has_uv and m.uv.shape == (m.nverts, 2)
    assert m.uv.min() >= -1e-6 and m.uv.max() <= 1 + 1e-6
    if make is not M.plane:
        assert m.is_watertight() and m.volume() > 0


def test_cube_uv_is_upright_on_front_face():
    c = M.cube()
    front = np.nonzero(c.face_normals()[:, 2] > 0.9)[0]
    for vi in np.unique(c.faces[front]):
        x, y, _ = c.verts[vi]
        assert c.uv[vi] == pytest.approx([x + 0.5, 0.5 - y], abs=1e-6)          # u grows with +x, v = 0 at the top


def test_sphere_uv_no_degenerate_uv_triangles_and_seam_is_duplicated():
    s = M.uv_sphere(1, 24, 12)
    tri = s.uv[s.faces]
    area = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    assert (area > 0).all()
    assert s.nverts > s.weld().nverts                                      # seam vertices are separate


def test_vertex_normals_cube_sharp_sphere_smooth_across_seam():
    c = M.cube()
    assert np.allclose(np.abs(c.vertex_normals()).max(1), 1, atol=1e-4)
    s = M.uv_sphere(1, 24, 12)
    n = s.vertex_normals()
    key = np.round(s.verts, 4)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    for g in np.unique(inv[np.bincount(inv)[inv] > 1])[:20]:                    # coincident seam vertices agree
        pts = n[inv == g]
        assert np.allclose(pts, pts[0], atol=1e-3)


def test_merge_offsets_texture_ids_and_pads_uvs():
    t1, t2 = M.Texture(checker()), M.Texture(checker(a=(0, 1, 0)))
    a = M.cube().with_texture(t1)
    b = M.uv_sphere(1, 8, 4).with_texture(t2)
    j = a.merged(b)
    assert len(j.textures) == 2 and set(np.unique(j.face_tex)) == {0, 1}
    j2 = a.merged(M.icosphere(1, 0))                                       # one side without UVs
    assert j2.uv.shape == (j2.nverts, 2) and (j2.face_tex[a.nfaces:] == -1).all()


def test_with_texture_requires_uv():
    with pytest.raises(ValueError):
        M.icosphere(1, 1).with_texture(M.Texture(checker()))


def test_subdivide_keeps_uv_seams_materials_and_shape():
    s = M.uv_sphere(1, 16, 8).with_material(color=(1, 0, 0), rough=0.2)
    t = M.subdivide(s, 2)
    assert t.has_uv and t.is_watertight() and t.nfaces == s.nfaces * 16
    assert np.allclose(t.color, [1, 0, 0]) and np.allclose(t.rough, 0.2)
    assert t.uv.min() >= -1e-6 and t.uv.max() <= 1 + 1e-6
    assert 0.85 * s.volume() < t.volume() < 1.02 * s.volume()              # Loop rounds slightly inward but does not collapse
    assert M.subdivide(M.cube().with_texture(M.Texture(checker())), 1).face_tex.shape == (12 * 4,)


def test_subdivide_unsmoothed_cube_keeps_exact_shape():
    t = M.subdivide(M.cube(), 2, smooth=False)
    assert t.volume() == pytest.approx(1.0, rel=1e-5) and t.surface_area() == pytest.approx(6.0, rel=1e-5)


def test_smooth_laplacian_does_not_open_seams():
    s = M.smooth_laplacian(M.uv_sphere(1, 16, 8), 5)
    assert s.is_watertight() and s.has_uv


@pytest.mark.parametrize("mode", ["planar", "cylindrical", "spherical", "box"])
def test_uv_project_modes(mode):
    m = M.uv_project(M.icosphere(1, 2), mode, scale=2.0, offset=(0.25, 0.0))
    assert m.has_uv and m.uv.shape == (m.nverts, 2)
    assert m.faces.shape == M.icosphere(1, 2).faces.shape
    if mode == "planar":
        assert m.nverts == M.icosphere(1, 2).nverts and m.uv[:, 0].min() == pytest.approx(0.25, abs=1e-5)
    with pytest.raises(ValueError):
        M.uv_project(M.cube(), "nope")


def test_uv_project_wrap_has_no_face_spanning_the_whole_texture():
    m = M.uv_project(M.uv_sphere(1, 32, 16), "cylindrical")
    span = np.ptp(m.uv[m.faces][..., 0], axis=1)
    assert span.max() < 0.5


# ------------------------------------------------------------------ OBJ with UVs
def test_obj_roundtrip_keeps_uvs_faces_and_order(tmp_path):
    t = M.torus(1, 0.3, 24, 8)
    M.write_obj(t, str(tmp_path / "t.obj"))
    r = M.read_obj(str(tmp_path / "t.obj"))
    assert r.nverts == t.nverts and np.array_equal(r.faces, t.faces)
    assert np.allclose(r.uv, t.uv, atol=1e-5) and np.allclose(r.verts, t.verts, atol=1e-5)


def test_obj_reader_handles_quads_negative_indices_and_missing_uv(tmp_path):
    p = tmp_path / "q.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf -4/-4 -3/-3 -2/-2 -1/-1\n")
    q = M.read_obj(str(p))
    assert q.nfaces == 2 and q.has_uv and q.uv[:, 1].tolist() == [1.0, 1.0, 0.0, 0.0]        # OBJ v is flipped to top-down
    p2 = tmp_path / "n.obj"
    p2.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert not M.read_obj(str(p2)).has_uv


# ------------------------------------------------------------------ texture sampling and rendering
def test_sample_texture_bilinear_wrap_and_mips():
    img = checker(64, 8)
    tex = M.Texture(img)
    u = np.array([0.5 / 64, 0.5 / 64 + 1.0, 0.5 / 64 - 3.0])
    v = np.array([0.5 / 64] * 3)
    out = sample_texture(tex, u, v, np.zeros(3))
    assert np.allclose(out[0], out[1], atol=1e-5) and np.allclose(out[0], out[2], atol=1e-5)        # repeat wrapping
    assert np.allclose(out[0, :3], img[0, 0, :3], atol=1e-5)
    hi = sample_texture(tex, np.array([0.3]), np.array([0.3]), np.array([8.0]))                    # coarsest mip = average
    assert np.allclose(hi[0, :3], img[..., :3].mean((0, 1)), atol=0.02)


def test_textured_render_shows_texture_colours_and_not_untextured():
    cube = M.cube()
    s = RenderSettings(width=160, height=120, ssaa=1, shadows=False, ambient=0.8)
    plain = render(cube, s)["image"]
    tex = render(cube.with_texture(M.Texture(checker(a=(1, 0, 0), b=(0, 0, 1)))), s)["image"]
    assert not np.allclose(plain, tex)
    a = tex[..., 3] > 0.5
    r, b = tex[..., 0][a], tex[..., 2][a]
    assert (r > 2 * b).any() and (b > 2 * r).any()                         # both red and blue checks appear


def test_texture_orientation_marker_top_left_on_front_face():
    img = np.full((64, 64, 4), 0.2, np.float32); img[..., 3] = 1
    img[:16, :16] = (1, 1, 1, 1)                                           # bright block at the image's top-left
    r = render(M.cube().with_texture(M.Texture(img)),
               RenderSettings(width=200, height=200, ssaa=1, camera_mode="manual", eye=(0, 0, 3.2), target=(0, 0, 0), fov=30,
                              shadows=False, ambient=0.8, key_azimuth=90, key_elevation=10, tonemap="none"))["image"]
    lum = r[..., :3].mean(-1)
    corner = lambda ys, xs: float(lum[ys, xs].mean())
    tl, tr = corner(slice(36, 60), slice(36, 60)), corner(slice(36, 60), slice(140, 164))
    bl, br = corner(slice(140, 164), slice(36, 60)), corner(slice(140, 164), slice(140, 164))
    assert tl > 2 * max(tr, bl, br)                                        # marker upper left: not mirrored, not flipped


def test_texture_scale_offset_and_alpha_handling():
    t = M.Texture(checker(32, 4), scale=(2, 2), offset=(0.5, 0))
    r = render(M.cube().with_texture(t), RenderSettings(width=96, height=96, ssaa=1, shadows=False))["image"]
    assert np.isfinite(r).all()
    t3 = M.Texture(np.ones((8, 8, 3), np.float32) * 0.5)                   # RGB gets an opaque alpha
    assert t3.image.shape == (8, 8, 4) and t3.image[..., 3].min() == 1


def test_minified_texture_is_filtered_not_noisy():
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.repeat((((xx + yy) % 2).astype(np.float32))[..., None], 3, -1)
    img = np.concatenate([img, np.ones((n, n, 1), np.float32)], -1)           # 1-pixel checker: pure aliasing bait
    m = M.plane(4, 4).transformed(M.trs_matrix((0, 0, 0), (0, 0, 0), (1, 1, 1))).with_texture(M.Texture(img, scale=(8, 8)))
    r = render(m, RenderSettings(width=200, height=200, ssaa=1, camera_mode="manual", eye=(0, 6, 0.01), target=(0, 0, 0), fov=40,
                                 shadows=False, tonemap="none", ambient=1.0, key_intensity=0, fill_intensity=0, rim_intensity=0))["image"]
    inner = r[80:120, 80:120, 0]
    assert inner.std() < 0.1                                                # trilinear mips average the checker to grey


# ------------------------------------------------------------------ glTF: hand-built, spec-conformant files
def build_gltf(extra_nodes=None, strided=True, mode=4, uri_buf=None):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    uv = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], np.float32)
    if strided:
        inter = np.zeros((4, 5), np.float32); inter[:, :3] = pos; inter[:, 3:] = uv
        vb = inter.tobytes(); stride = 20
    else:
        vb = pos.tobytes() + uv.tobytes(); stride = None
    ib = np.array([0, 1, 2, 2, 1, 3] if mode == 4 else [0, 1, 2, 3], np.uint16).tobytes()
    ib += b"\0\0" if len(ib) % 4 else b""
    blob = vb + ib
    uri = uri_buf or "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
         "nodes": [{"mesh": 0, "translation": [1, 2, 3], "scale": [2, 2, 2]}],
         "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0, "mode": mode}]}],
         "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.25, 0.125, 1], "metallicFactor": 0.0, "roughnessFactor": 0.4},
                        "emissiveFactor": [0.1, 0, 0]}],
         "buffers": [{"uri": uri, "byteLength": len(blob)}],
         "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(vb), **({"byteStride": stride} if stride else {})},
                         {"buffer": 0, "byteOffset": len(vb), "byteLength": len(ib)}],
         "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                        "min": [0, 0, 0], "max": [1, 1, 0]},
                       {"bufferView": 0 if strided else 0, "byteOffset": 12 if strided else 48, "componentType": 5126, "count": 4, "type": "VEC2"},
                       {"bufferView": 1, "componentType": 5123, "count": 6 if mode == 4 else 4, "type": "SCALAR"}]}
    if not strided:                                                           # separate tightly packed views
        j["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 32},
                            {"buffer": 0, "byteOffset": len(vb), "byteLength": len(ib)}]
        j["accessors"][0]["bufferView"] = 0
        j["accessors"][1] = {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"}
        j["accessors"][2]["bufferView"] = 2
    if extra_nodes:
        j.update(extra_nodes)
    return j


def test_read_handmade_gltf_strided_transform_and_material(tmp_path):
    p = tmp_path / "quad.gltf"
    p.write_text(json.dumps(build_gltf()))
    m = read_gltf(str(p))
    assert m.nverts == 4 and m.nfaces == 2
    assert m.verts[:, 0].min() == pytest.approx(1) and m.verts[:, 0].max() == pytest.approx(3)         # scale 2 then translate
    assert m.verts[:, 2] == pytest.approx(np.full(4, 3.0))
    assert np.allclose(m.color, [0.5, 0.25, 0.125]) and np.allclose(m.rough, 0.4) and np.allclose(m.metal, 0)
    assert np.allclose(m.emit[0], [0.1, 0, 0]) and np.allclose(m.uv[3], [1, 0])
    assert (m.face_normals()[:, 2] > 0.99).all()                                                      # CCW seen from +Z


def test_read_handmade_gltf_tightly_packed_views_and_strip(tmp_path):
    (tmp_path / "a.gltf").write_text(json.dumps(build_gltf(strided=False)))
    assert read_gltf(str(tmp_path / "a.gltf")).nfaces == 2
    (tmp_path / "s.gltf").write_text(json.dumps(build_gltf(mode=5)))
    s = read_gltf(str(tmp_path / "s.gltf"))
    assert s.nfaces == 2 and (s.face_normals()[:, 2] > 0.99).all()                                     # strip winding alternates


def test_read_node_hierarchy_and_mirrored_node_keeps_faces_outward(tmp_path):
    cube = M.cube()
    pos = cube.verts.astype(np.float32).tobytes()
    ib = cube.faces.astype(np.uint16).tobytes()
    blob = pos + ib
    j = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}],
         "nodes": [{"children": [1, 2], "translation": [0, 5, 0]},
                   {"mesh": 0, "scale": [-1, 1, 1], "translation": [1, 0, 0]},
                   {"mesh": 0, "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1]}],
         "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
         "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode(), "byteLength": len(blob)}],
         "bufferViews": [{"buffer": 0, "byteLength": len(pos)}, {"buffer": 0, "byteOffset": len(pos), "byteLength": len(ib)}],
         "accessors": [{"bufferView": 0, "componentType": 5126, "count": cube.nverts, "type": "VEC3"},
                       {"bufferView": 1, "componentType": 5123, "count": cube.nfaces * 3, "type": "SCALAR"}]}
    (tmp_path / "h.gltf").write_text(json.dumps(j))
    m = read_gltf(str(tmp_path / "h.gltf"))
    assert m.nfaces == 24 and m.volume() == pytest.approx(2.0, rel=1e-5)                              # both cubes outward (+1 each)
    lo, hi = m.bounds()
    assert hi[1] == pytest.approx(5.5) and lo[0] == pytest.approx(0.5)


def test_read_refuses_escape_sparse_and_compressed(tmp_path):
    sub = tmp_path / "models"; sub.mkdir()
    (tmp_path / "secret.bin").write_bytes(b"\0" * 80)
    bad = build_gltf(uri_buf="../secret.bin")
    (sub / "evil.gltf").write_text(json.dumps(bad))
    with pytest.raises(GltfError, match="outside"):
        read_gltf(str(sub / "evil.gltf"))
    j = build_gltf(); j["accessors"][0]["sparse"] = {"count": 1}
    (sub / "sparse.gltf").write_text(json.dumps(j))
    with pytest.raises(GltfError, match="sparse"):
        read_gltf(str(sub / "sparse.gltf"))
    j = build_gltf(); j["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    (sub / "draco.gltf").write_text(json.dumps(j))
    with pytest.raises(GltfError, match="draco|Draco|unsupported"):
        read_gltf(str(sub / "draco.gltf"))
    (sub / "junk.glb").write_bytes(b"glTF" + b"\0" * 8)
    with pytest.raises(GltfError):
        read_gltf(str(sub / "junk.glb"))
    with pytest.raises(FileNotFoundError):
        read_gltf(str(sub / "missing.glb"))


def test_read_truncated_buffer_is_an_error_not_a_crash(tmp_path):
    j = build_gltf(); j["buffers"][0]["byteLength"] = 10**6
    (tmp_path / "t.gltf").write_text(json.dumps(j))
    with pytest.raises(GltfError):
        read_gltf(str(tmp_path / "t.gltf"))
    j = build_gltf(); j["accessors"][2]["count"] = 10**6
    (tmp_path / "t2.gltf").write_text(json.dumps(j))
    with pytest.raises(GltfError):
        read_gltf(str(tmp_path / "t2.gltf"))


def test_read_embedded_texture_converts_srgb_to_linear_and_applies_transform(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8); img[...] = (0, 128, 255)                      # BGR in OpenCV -> RGB (255, 128, 0)
    ok, png = cv2.imencode(".png", img)
    j = build_gltf()
    j["images"] = [{"uri": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}]
    j["textures"] = [{"source": 0}]
    j["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {
        "index": 0, "extensions": {"KHR_texture_transform": {"scale": [2, 3], "offset": [0.1, 0.2]}}}
    j["extensionsUsed"] = ["KHR_texture_transform"]
    (tmp_path / "tex.gltf").write_text(json.dumps(j))
    m = read_gltf(str(tmp_path / "tex.gltf"))
    assert len(m.textures) == 1 and (m.face_tex == 0).all()
    px = m.textures[0].image[0, 0]
    assert px[0] == pytest.approx(1.0, abs=1e-4) and px[1] == pytest.approx(float(srgb_to_linear(np.array(128 / 255))), abs=1e-3) and px[2] == pytest.approx(0, abs=1e-4)
    assert m.textures[0].scale == (2.0, 3.0) and m.textures[0].offset == pytest.approx((0.1, 0.2))


def test_read_normalized_integer_uv_and_color(tmp_path):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    uv = np.array([[0, 0], [65535, 0], [0, 65535]], np.uint16)
    col = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], np.uint8)
    parts = [pos.tobytes(), uv.tobytes(), col.tobytes()]
    blob = b"".join(parts)
    offs = [0, 36, 48]
    j = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
         "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1, "COLOR_0": 2}}]}],
         "buffers": [{"uri": "data:;base64," + base64.b64encode(blob).decode(), "byteLength": len(blob)}],
         "bufferViews": [{"buffer": 0, "byteOffset": o_, "byteLength": len(p_)} for o_, p_ in zip(offs, parts)],
         "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                       {"bufferView": 1, "componentType": 5123, "normalized": True, "count": 3, "type": "VEC2"},
                       {"bufferView": 2, "componentType": 5121, "normalized": True, "count": 3, "type": "VEC4"}]}
    (tmp_path / "n.gltf").write_text(json.dumps(j))
    m = read_gltf(str(tmp_path / "n.gltf"))
    assert m.nfaces == 1 and np.allclose(m.uv[1], [1, 0]) and np.allclose(m.color[2], [0, 0, 1])           # default material factors are 1


# ------------------------------------------------------------------ GLB writer
def test_glb_roundtrip_geometry_uv_colour_material(tmp_path):
    m = M.uv_sphere(1.0, 20, 10).with_material(color=(0.2, 0.5, 0.9), rough=0.3, metal=0.7)
    write_glb(m, str(tmp_path / "s.glb"))
    r = read_gltf(str(tmp_path / "s.glb"))
    assert r.nfaces == m.nfaces and r.volume() == pytest.approx(m.volume(), rel=1e-4) and r.is_watertight()
    assert np.allclose(r.color[0], [0.2, 0.5, 0.9], atol=1e-6) and np.allclose(r.rough, 0.3, atol=1e-3) and np.allclose(r.metal, 0.7, atol=1e-3)
    assert r.has_uv and np.allclose(np.sort(r.uv, axis=0), np.sort(m.uv, axis=0), atol=1e-6)


def test_glb_roundtrip_preserves_textured_render(tmp_path):
    t = M.Texture(checker(64, 8), scale=(2, 1), offset=(0.25, 0))
    m = M.cube().with_texture(t)
    write_glb(m, str(tmp_path / "c.glb"))
    r = read_gltf(str(tmp_path / "c.glb"))
    assert len(r.textures) == 1 and (r.face_tex == 0).all()
    s = RenderSettings(width=120, height=90, ssaa=1, shadows=False)
    a, b = render(m, s)["image"], render(r, s)["image"]
    assert np.abs(a - b).mean() < 0.01                                  # 8-bit PNG quantisation only


def test_glb_two_materials_two_primitives_and_emission(tmp_path):
    a = M.cube().with_material(color=(1, 0, 0), rough=0.9)
    b = M.uv_sphere(0.5, 12, 6).transformed(M.trs_matrix((3, 0, 0))).with_material(color=(0, 1, 0), emit=(4, 2, 0), metal=1.0, rough=0.1)
    m = a.merged(b)
    write_glb(m, str(tmp_path / "two.glb"))
    raw = open(tmp_path / "two.glb", "rb").read()
    jlen = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + jlen])
    assert len(doc["meshes"][0]["primitives"]) == 2 and len(doc["materials"]) == 2
    assert "KHR_materials_emissive_strength" in doc.get("extensionsUsed", [])
    r = read_gltf(str(tmp_path / "two.glb"))
    em = r.emit.max(1)
    assert em.max() == pytest.approx(4.0, rel=1e-3) and (r.rough.min() == pytest.approx(0.1, abs=1e-3))
    assert r.nfaces == m.nfaces


def test_glb_flat_shading_exports_face_normals_and_empty_is_refused(tmp_path):
    flat = M.cube()
    flat.smooth = False
    write_glb(flat, str(tmp_path / "f.glb"))
    assert read_gltf(str(tmp_path / "f.glb")).nfaces == 12
    with pytest.raises(GltfError):
        write_glb(M.Mesh(np.zeros((3, 3)), np.zeros((0, 3), np.int32)), str(tmp_path / "e.glb"))


def test_glb_chunks_are_aligned_and_header_length_matches(tmp_path):
    write_glb(M.torus(1, .3, 12, 6), str(tmp_path / "a.glb"))
    raw = open(tmp_path / "a.glb", "rb").read()
    magic, ver, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67 and ver == 2 and total == len(raw) and len(raw) % 4 == 0
    jl, jt = struct.unpack_from("<II", raw, 12)
    assert jt == 0x4E4F534A and jl % 4 == 0


def test_read_write_mesh_dispatch_and_unknown_extension(tmp_path):
    m = M.cube()
    M.write_mesh(m, str(tmp_path / "x.glb"))
    assert M.read_mesh(str(tmp_path / "x.glb")).nfaces == 12
    with pytest.raises(ValueError):
        M.write_mesh(m, str(tmp_path / "x.fbx"))
    with pytest.raises(ValueError):
        M.read_mesh(str(tmp_path / "x.fbx"))


# ------------------------------------------------------------------ graph nodes
def test_mesh_texture_and_uv_project_nodes_in_a_graph():
    g = o.Graph()
    chk = g.add("checker")
    sph = g.add("mesh_icosphere", subdivisions=2)
    uvp = g.add("mesh_uv_project", mode="spherical")
    tx = g.add("mesh_texture", scale_u=2, scale_v=2)
    rn = g.add("render_mesh", width=160, height=120, quality=1)
    g.connect(sph, uvp); g.connect(uvp, tx, dst_port="mesh"); g.connect(chk, tx, dst_port="image"); g.connect(tx, rn)
    textured = g.evaluate(rn)
    g2 = o.Graph()
    s2 = g2.add("mesh_icosphere", subdivisions=2); r2 = g2.add("render_mesh", width=160, height=120, quality=1); g2.connect(s2, r2)
    assert textured.shape[:2] == (120, 160) and not np.allclose(textured, g2.evaluate(r2))


def test_mesh_texture_node_requires_uvs():
    spec = o.get_spec("mesh_texture")
    with pytest.raises(ParamError, match="UV"):
        spec.fn({"mesh": M.icosphere(1, 1), "image": checker()}, {"scale_u": 1.0, "scale_v": 1.0, "offset_u": 0.0, "offset_v": 0.0}, None)


def test_mesh_load_and_write_nodes_handle_glb(tmp_path):
    p = str(tmp_path / "n.glb")
    spec_w, spec_l = o.get_spec("mesh_write"), o.get_spec("mesh_load")
    spec_w.fn({"mesh": M.torus()}, {"path": p}, None)
    out = spec_l.fn({}, {"path": p}, None)
    assert out.nfaces == M.torus().nfaces


# ------------------------------------------------------------------ Copilot
@pytest.mark.parametrize("shape", ["cube", "sphere", "icosphere", "torus"])
def test_copilot_builds_textured_3d_scene(tmp_path, shape):
    from omniimage.ai_assistant import Copilot, Toolbox
    c = Copilot(workdir=str(tmp_path))
    r = c.ask(f"make a checker textured {shape} 3d render")
    assert r.files and (tmp_path / "scene3d.png").exists()
    types = {n.type for n in c.project.graph.nodes.values()}
    assert "mesh_texture" in types and "checker" in types
    assert ("mesh_uv_project" in types) == (shape == "icosphere")
    assert not o.Graph.validate(c.project.graph)
    tb = Toolbox(c.project, str(tmp_path))
    assert not tb.call("create_3d_scene", {"shape": "cube", "texture": "wood"})["ok"]
    assert tb.call("create_3d_scene", {"shape": "plane", "texture": "noise"})["ok"]


# ------------------------------------------------------------------ GUI preview of a textured mesh
def test_gui_previews_textured_mesh_and_sandbox_blocks_glb_write_outside_workdir(tmp_path):
    from omniimage.gui.server import Studio
    from omniimage.ai_assistant import Toolbox
    studio = Studio(workdir=str(tmp_path / "work"))
    c = studio.copilot
    c.ask("make a checker textured cube 3d render")
    tx = next(i for i, n in c.project.graph.nodes.items() if n.type == "mesh_texture")
    png, kind = studio.render_png(tx, maxw=320)
    assert kind == "mesh" and png[:8] == b"\x89PNG\r\n\x1a\n"
    tb = Toolbox(c.project, str(tmp_path / "work"))
    r = tb.call("add_node", {"type": "mesh_write", "params": {"path": str(tmp_path / "outside.glb")}})
    node = c.project.graph.nodes[r["id"]] if r.get("ok") else None
    assert node is None or str(tmp_path / "work") in str(node.params.get("path"))
