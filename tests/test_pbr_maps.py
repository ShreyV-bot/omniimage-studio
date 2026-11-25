"""PBR texture maps (normal, roughness/metallic, emissive, occlusion): data model, tangent frames, renderer, nodes, glTF I/O,
and the glTF parser's behaviour on hostile or broken files."""
import base64
import json
import os
import struct
import zlib

import cv2
import numpy as np
import pytest

import omniimage as o
from omniimage import mesh as M
from omniimage.gltf import GltfError, read_gltf, write_glb
from omniimage.raster import RenderSettings, render, texture_maps

RNG = np.random.default_rng(7)


def flat_normal(n=16):
    return np.tile(np.array([0.5, 0.5, 1.0], np.float32), (n, n, 1))


def tilted_normal(nx, ny, n=16):
    v = np.array([nx, ny, np.sqrt(max(1 - nx * nx - ny * ny, 0))])
    return np.tile((v * 0.5 + 0.5).astype(np.float32), (n, n, 1))


def top_view(**kw):
    """Camera straight above, key light off, lots of ambient: a clean look at one flat plane."""
    base = dict(width=64, height=64, ssaa=1, camera_mode="manual", eye=(0, 4, 0.001), target=(0, 0, 0), fov=30, shadows=False,
                floor="none", tonemap="none", background="color", background_color=(0, 0, 0))
    base.update(kw)
    return RenderSettings(**base)


# ------------------------------------------------------------------ data model
def test_texture_extra_maps_are_normalised_and_validated():
    t = M.Texture(None, normal=np.zeros((4, 4, 3)), occlusion=np.ones((2, 2)))
    assert t.image.shape == (1, 1, 4) and t.normal.shape == (4, 4, 4) and t.occlusion.shape == (2, 2, 4)
    assert t.has_extra_maps and not M.Texture(np.ones((2, 2, 4))).has_extra_maps
    with pytest.raises(ValueError):
        M.Texture(None, normal=np.zeros((4, 4, 2)))


def test_tangent_frame_matches_gltf_convention_and_is_orthogonal_to_normals():
    pl = M.plane(2, 2)
    T, B = pl.tangent_frame()
    assert np.allclose(T, [1, 0, 0], atol=1e-5) and np.allclose(B, [0, 0, -1], atol=1e-5)      # u -> +X, image-up -> -Z
    for m in (M.cube(), M.uv_sphere(1, 24, 12), M.torus(1, .3, 24, 8), M.cylinder()):
        T, B = m.tangent_frame()
        N = m.vertex_normals()
        assert np.linalg.norm(T, axis=1).min() > 0.99 and np.linalg.norm(B, axis=1).min() > 0.99
        assert np.abs((T * N).sum(1)).max() < 0.35                      # near-tangent even where smoothing bends the normal
    s = M.uv_sphere(1, 32, 16)
    eq = np.argmin(np.abs(s.verts[:, 1]) + np.abs(s.verts[:, 2]) + (s.verts[:, 0] < 0))                  # a point on the equator, +X side
    assert s.tangent_frame()[1][eq][1] > 0.9                                                          # image-up points to the north pole
    assert not np.any(M.icosphere(1, 1).tangent_frame()[0])                                         # no UVs -> zero frame


# ------------------------------------------------------------------ renderer: normal mapping
def test_normal_map_tilts_world_normal_along_tangent_and_bitangent():
    pl = M.plane(4, 4)
    s = top_view()
    flat = render(pl.with_texture(M.Texture(None, normal=flat_normal())), s)["normal"][32, 32, :3]
    assert np.allclose(flat, [0.5, 1.0, 0.5], atol=0.02)                                             # unchanged: +Y
    tx = render(pl.with_texture(M.Texture(None, normal=tilted_normal(0.6, 0.0))), s)["normal"][32, 32, :3]
    assert tx[0] > 0.75 and tx[1] < 0.95 and abs(tx[2] - 0.5) < 0.03                                 # leans toward +X (tangent)
    ty = render(pl.with_texture(M.Texture(None, normal=tilted_normal(0.0, 0.6))), s)["normal"][32, 32, :3]
    assert ty[2] < 0.25 and abs(ty[0] - 0.5) < 0.03                                                  # leans toward -Z (bitangent)


def test_normal_strength_zero_equals_flat_and_flat_map_equals_no_map():
    sph = M.uv_sphere(1, 32, 16)
    s = RenderSettings(width=96, height=72, ssaa=1, shadows=False, floor="none")
    plain = render(sph, s)["image"]
    flat = render(sph.with_texture(M.Texture(None, normal=flat_normal())), s)["image"]
    off = render(sph.with_texture(M.Texture(None, normal=tilted_normal(0.7, 0.2), normal_scale=0.0)), s)["image"]
    assert np.abs(plain - flat).mean() < 0.01 and np.abs(plain - off).mean() < 0.01
    bumpy = render(sph.with_texture(M.Texture(None, normal=tilted_normal(0.7, 0.2))), s)["image"]
    assert np.abs(plain - bumpy).mean() > 0.01


def test_normal_mapping_works_on_flat_shaded_and_back_faces():
    c = M.cube(); c.smooth = False
    r = render(c.with_texture(M.Texture(None, normal=tilted_normal(0.5, 0.5))), RenderSettings(width=64, height=48, ssaa=1, shadows=False))
    assert np.isfinite(r["image"]).all() and np.isfinite(r["normal"]).all()
    pl = M.plane(2, 2).with_texture(M.Texture(None, normal=tilted_normal(0.5, 0)))
    under = render(pl, RenderSettings(width=48, height=48, ssaa=1, camera_mode="manual", eye=(0, -4, 0.001), target=(0, 0, 0),
                                       shadows=False, floor="none", fov=30))
    n = under["normal"][24, 24, :3]
    assert n[1] < 0.2                                                                              # viewed from below: normal flipped toward the camera


# ------------------------------------------------------------------ renderer: other maps
def test_texture_maps_return_neutral_values_without_maps_and_sampled_values_with_them():
    pl = M.plane(2, 2)
    rm = np.dstack([np.ones((8, 8)), np.full((8, 8), .25), np.full((8, 8), .5), np.ones((8, 8))]).astype(np.float32)
    tex = M.Texture(None, rough_metal=rm, emissive=np.full((8, 8, 3), .3, np.float32), occlusion=np.full((8, 8), .4, np.float32), occlusion_strength=0.5)
    m = pl.with_texture(tex)
    tri = np.array([0, 0])
    ft = m.faces[tri]                                                   # (K, 3) vertex indices per pixel, as render() passes them
    sx, sy, iw = np.array([0., 10., 0., 10.]), np.array([0., 0., 10., 10.]), np.ones(4)
    out = texture_maps(m, tri, ft, np.array([2.0, 3.0]), np.array([2.0, 3.0]), sx, sy, iw)
    assert np.allclose(out["rough"], .25, atol=1e-4) and np.allclose(out["metal"], .5, atol=1e-4)
    assert np.allclose(out["emis"], .3, atol=1e-4) and np.allclose(out["ao"], 1 + .5 * (.4 - 1), atol=1e-4)
    none = texture_maps(pl, tri[:1], ft[:1], np.array([2.0]), np.array([2.0]), sx, sy, iw)
    assert none["rough"][0] == 1 and none["ao"][0] == 1 and not none["has_normal"][0]


def test_roughness_map_changes_highlights():
    sph = M.uv_sphere(1, 48, 24).with_material(color=(0.8, 0.8, 0.8), metal=0.0)
    s = RenderSettings(width=128, height=96, ssaa=1, shadows=False, floor="none", tonemap="none")
    def peak(r):
        rm = np.dstack([np.ones((4, 4)), np.full((4, 4), r), np.zeros((4, 4)), np.ones((4, 4))]).astype(np.float32)
        t = M.Texture(None, rough_metal=rm)
        m = sph.with_texture(t)
        m.rough[:] = 1.0
        return render(m, s)["image"][..., :3].max()
    assert peak(0.05) > peak(1.0) * 1.15


def test_emissive_map_glows_in_the_dark_and_occlusion_darkens_ambient_only():
    pl = M.plane(4, 4)
    dark = dict(key_intensity=0.0, fill_intensity=0.0, rim_intensity=0.0, ambient=0.0)
    red = np.zeros((8, 8, 3), np.float32); red[..., 0] = 1.0
    lit = render(pl.with_material(emit=(1, 1, 1)).with_texture(M.Texture(None, emissive=red)), top_view(**dark))["image"][32, 32]
    assert lit[0] > 0.5 and lit[1] < 0.05 and lit[2] < 0.05                                          # emits only the map's red
    off = render(pl.with_texture(M.Texture(None, emissive=red)), top_view(**dark))["image"][32, 32]
    assert off[:3].max() < 0.05                                                                     # factor 0 = glTF default: no glow
    amb = dict(key_intensity=0.0, fill_intensity=0.0, rim_intensity=0.0, ambient=1.0)
    base = render(pl, top_view(**amb))["image"][32, 32, :3].mean()
    black = np.zeros((8, 8, 3), np.float32)
    ao_full = render(pl.with_texture(M.Texture(None, occlusion=black)), top_view(**amb))["image"][32, 32, :3].mean()
    ao_none = render(pl.with_texture(M.Texture(None, occlusion=black, occlusion_strength=0.0)), top_view(**amb))["image"][32, 32, :3].mean()
    assert base > 0.05 and ao_full < base * 0.2 and abs(ao_none - base) < 0.01


# ------------------------------------------------------------------ nodes
def test_normal_from_height_directions_and_resolution_independence():
    spec = o.get_spec("normal_from_height")
    run = lambda img, **k: spec.fn({"image": img}, {"strength": 4.0, "wrap": False, "flip_y": False, **k}, None)
    def ramp(n, axis):
        g = np.linspace(0, 1, n, dtype=np.float32)
        h = np.tile(g, (n, 1)) if axis == "x" else np.tile(g[:, None], (1, n))
        return np.dstack([h, h, h, np.ones_like(h)])
    rx = run(ramp(64, "x"))[32, 32]
    assert rx[0] < 0.45 and abs(rx[1] - 0.5) < 0.01 and rx[2] > 0.5                                  # rising to +x -> normal leans to -x
    ry = run(ramp(64, "y"))[32, 32]
    assert ry[1] > 0.55 and abs(ry[0] - 0.5) < 0.01                                                 # rising downward -> leans to +Y (image-up)
    assert run(ramp(64, "y"), flip_y=True)[32, 32][1] < 0.45
    assert abs(run(ramp(64, "x"))[32, 32][0] - run(ramp(256, "x"))[32, 32][0]) < 0.02              # same look at any resolution
    out = run(ramp(32, "x"))
    assert out.shape == (32, 32, 4) and out[..., 3].min() == 1
    v = out[..., :3] * 2 - 1
    assert np.allclose(np.linalg.norm(v, axis=-1), 1, atol=1e-4)


def test_normal_from_height_wrap_is_shift_equivariant():
    spec = o.get_spec("normal_from_height")
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    h = 0.5 + 0.5 * np.sin(xx / 32 * 2 * np.pi) * np.cos(yy / 32 * 2 * np.pi)
    img = np.dstack([h, h, h, np.ones_like(h)])
    f = lambda a: spec.fn({"image": a}, {"strength": 4.0, "wrap": True, "flip_y": False}, None)
    assert np.allclose(np.roll(f(img), 5, axis=1), f(np.roll(img, 5, axis=1)), atol=1e-5)


def test_mesh_texture_node_packs_maps_and_replaces_factors():
    spec = o.get_spec("mesh_texture")
    prm = dict(scale_u=1.0, scale_v=1.0, offset_u=0.0, offset_v=0.0, normal_strength=0.8, roughness_scale=0.9, metallic_scale=1.0,
               emissive_strength=3.0, occlusion_strength=0.5)
    rough = np.full((8, 8, 4), 0.3, np.float32); rough[..., 3] = 1
    metal = np.full((16, 16, 4), 0.7, np.float32)                                                  # different size: gets resized
    nrm, emi, occ = (np.full((4, 4, 4), 0.5, np.float32), np.full((4, 4, 4), 1.0, np.float32), np.full((4, 4, 4), 0.2, np.float32))
    out = spec.fn({"mesh": M.cube(), "normal": nrm, "roughness": rough, "metallic": metal, "emissive": emi, "occlusion": occ}, prm, None)
    t = out.textures[0]
    assert t.rough_metal.shape[:2] == (8, 8) or t.rough_metal.shape[:2] == (16, 16)
    assert np.allclose(t.rough_metal[..., 1].mean(), 0.3, atol=1e-3) and np.allclose(t.rough_metal[..., 2].mean(), 0.7, atol=1e-3)
    assert np.allclose(out.rough, 0.9) and out.metal.min() == 1.0 and np.allclose(out.emit, 3.0)
    assert t.normal_scale == 0.8 and t.occlusion_strength == 0.5 and np.allclose(t.occlusion[..., 0], 0.2)
    with pytest.raises(o.core.params.ParamError, match="at least one"):
        spec.fn({"mesh": M.cube()}, prm, None)
    with pytest.raises(o.core.params.ParamError, match="UV"):
        spec.fn({"mesh": M.icosphere(1, 1), "normal": nrm}, prm, None)
    colour_only = spec.fn({"mesh": M.cube(), "image": np.ones((2, 2, 4), np.float32)}, prm, None)
    assert np.allclose(colour_only.rough, 0.5) and colour_only.emit.max() == 0                      # untouched when no map is connected


def test_bump_mapped_sphere_in_a_graph_differs_from_plain():
    g = o.Graph()
    sph = g.add("mesh_sphere", segments=48, rings=24)
    nz = g.add("noise", width=256, height=256, scale=24.0)
    nm = g.add("normal_from_height", strength=12.0)
    tx = g.add("mesh_texture", scale_u=3.0, scale_v=2.0)
    rn = g.add("render_mesh", width=160, height=120, quality=1)
    g.connect(nz, nm); g.connect(sph, tx, dst_port="mesh"); g.connect(nm, tx, dst_port="normal"); g.connect(tx, rn)
    bumpy = g.evaluate(rn)
    g2 = o.Graph()
    s2 = g2.add("mesh_sphere", segments=48, rings=24); r2 = g2.add("render_mesh", width=160, height=120, quality=1); g2.connect(s2, r2)
    assert np.abs(bumpy - g2.evaluate(r2)).mean() > 0.003


def test_copilot_bump_texture(tmp_path):
    from omniimage.ai_assistant import Copilot
    c = Copilot(workdir=str(tmp_path))
    r = c.ask("make a bumpy rock sphere 3d render")
    types = {n.type for n in c.project.graph.nodes.values()}
    assert r.files and "normal_from_height" in types and "mesh_texture" in types


# ------------------------------------------------------------------ glTF: all maps
def png_b64(arr_u8_rgb):
    ok, buf = cv2.imencode(".png", arr_u8_rgb[..., ::-1])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def quad_gltf(material, images):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    uv = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], np.float32)
    ib = np.array([0, 1, 2, 2, 1, 3], np.uint16).tobytes()
    blob = pos.tobytes() + uv.tobytes() + ib
    return {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
            "materials": [material], "images": [{"uri": u} for u in images], "textures": [{"source": i} for i in range(len(images))],
            "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode(), "byteLength": len(blob)}],
            "bufferViews": [{"buffer": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 32},
                            {"buffer": 0, "byteOffset": 80, "byteLength": len(ib)}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
                          {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
                          {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}]}


def test_read_all_five_maps_with_correct_colour_handling(tmp_path):
    g128 = np.full((4, 4, 3), 128, np.uint8)
    mat = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 1}},
           "normalTexture": {"index": 2, "scale": 0.5}, "emissiveTexture": {"index": 3}, "emissiveFactor": [1, 1, 1],
           "occlusionTexture": {"index": 4, "strength": 0.25}}
    j = quad_gltf(mat, [png_b64(g128)] * 5)
    (tmp_path / "all.gltf").write_text(json.dumps(j))
    m = read_gltf(str(tmp_path / "all.gltf"))
    t = m.textures[0]
    srgb128 = ((128 / 255 + 0.055) / 1.055) ** 2.4
    assert t.image[0, 0, 0] == pytest.approx(srgb128, abs=2e-3)                    # colour maps are decoded from sRGB
    assert t.emissive[0, 0, 0] == pytest.approx(srgb128, abs=2e-3)
    for data in (t.normal, t.rough_metal, t.occlusion):
        assert data[0, 0, 0] == pytest.approx(128 / 255, abs=2e-3)                 # data maps are NOT colour-converted
    assert t.normal_scale == 0.5 and t.occlusion_strength == 0.25 and np.allclose(m.emit, 1.0)


def test_maps_with_a_different_uv_transform_than_the_first_are_dropped_not_misapplied(tmp_path):
    g = np.full((4, 4, 3), 200, np.uint8)
    mat = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
           "normalTexture": {"index": 1, "extensions": {"KHR_texture_transform": {"scale": [4, 4]}}}}
    (tmp_path / "m.gltf").write_text(json.dumps(quad_gltf(mat, [png_b64(g)] * 2)))
    t = read_gltf(str(tmp_path / "m.gltf")).textures[0]
    assert t.normal is None and t.image[0, 0, 0] > 0.5


def test_glb_roundtrip_of_all_maps_renders_the_same(tmp_path):
    n = 32
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    h = 0.5 + 0.5 * np.sin(xx / n * 6.28) * np.sin(yy / n * 6.28)
    gx, gy = np.gradient(h, axis=1), np.gradient(h, axis=0)
    nmap = np.dstack([-gx * 6, gy * 6, np.ones_like(h)]); nmap = nmap / np.linalg.norm(nmap, axis=-1, keepdims=True) * 0.5 + 0.5
    rm = np.dstack([np.ones((n, n)), 0.2 + 0.6 * h, 0.5 * h, np.ones((n, n))]).astype(np.float32)
    tex = M.Texture(checker_img(), scale=(2, 2), normal=nmap.astype(np.float32), normal_scale=0.8, rough_metal=rm,
                    occlusion=np.repeat(h[..., None], 3, -1), occlusion_strength=0.7)
    m = M.uv_sphere(1, 32, 16).with_texture(tex)
    write_glb(m, str(tmp_path / "all.glb"))
    r = read_gltf(str(tmp_path / "all.glb"))
    t = r.textures[0]
    assert t.normal is not None and t.rough_metal is not None and t.occlusion is not None and t.emissive is None
    s = RenderSettings(width=120, height=90, ssaa=1, shadows=False)
    assert np.abs(render(m, s)["image"] - render(r, s)["image"]).mean() < 0.012                       # 8-bit quantisation only


def checker_img(n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    c = (((xx // 8) + (yy // 8)) % 2).astype(np.float32)[..., None]
    return np.concatenate([c * np.array([.9, .6, .1]) + (1 - c) * np.array([.1, .2, .7]), np.ones((n, n, 1))], -1).astype(np.float32)


def test_glb_skips_placeholder_base_colour_when_only_data_maps_exist(tmp_path):
    m = M.cube().with_texture(M.Texture(None, normal=flat_normal(8)))
    write_glb(m, str(tmp_path / "n.glb"))
    raw = open(tmp_path / "n.glb", "rb").read()
    doc = json.loads(raw[20:20 + struct.unpack_from("<I", raw, 12)[0]])
    mat = doc["materials"][0]
    assert "normalTexture" in mat and "baseColorTexture" not in mat["pbrMetallicRoughness"] and len(doc["images"]) == 1
    assert read_gltf(str(tmp_path / "n.glb")).textures[0].normal is not None


# ------------------------------------------------------------------ glTF parser vs hostile files (negative AND positive)
def tri_blob():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    return pos.tobytes() + uv.tobytes()                                                           # 36 + 24 bytes


def tri_gltf(buffer_uri, image_uri=None):
    blob = tri_blob()
    prim = {"attributes": {"POSITION": 0}}
    j = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}], "meshes": [{"primitives": [prim]}],
         "buffers": [{"uri": buffer_uri, "byteLength": len(blob)}],
         "bufferViews": [{"buffer": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 24}],
         "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                       {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}]}
    if image_uri:
        prim["attributes"]["TEXCOORD_0"] = 1
        prim["material"] = 0
        j["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        j["images"] = [{"uri": image_uri}]; j["textures"] = [{"source": 0}]
    return j, blob


def b64_uri(blob):
    return "data:;base64," + base64.b64encode(blob).decode()


def test_external_files_inside_the_folder_load_but_escapes_are_refused(tmp_path):
    root = tmp_path / "model"; root.mkdir()
    j, blob = tri_gltf("tri.bin")
    (root / "tri.bin").write_bytes(blob)
    (root / "ok.gltf").write_text(json.dumps(j))
    assert read_gltf(str(root / "ok.gltf")).nfaces == 1                                           # positive: normal relative file works
    (root / "sub").mkdir(); (root / "sub" / "tri.bin").write_bytes(blob)
    j["buffers"][0]["uri"] = "sub/tri.bin"
    (root / "sub.gltf").write_text(json.dumps(j))
    assert read_gltf(str(root / "sub.gltf")).nfaces == 1                                          # positive: sub-folder works
    secret = tmp_path / "secret.bin"; secret.write_bytes(blob)
    (root / "link.bin").symlink_to(secret)
    for bad in ("../secret.bin", "%2e%2e/secret.bin", str(secret), "sub/../../secret.bin", "link.bin", "file://" + str(secret)):
        j["buffers"][0]["uri"] = bad
        (root / "bad.gltf").write_text(json.dumps(j))
        with pytest.raises(GltfError):
            read_gltf(str(root / "bad.gltf"))                                                      # negative: every route out is refused


def test_images_get_the_same_folder_confinement(tmp_path):
    root = tmp_path / "m"; root.mkdir()
    ok, png = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    (tmp_path / "outside.png").write_bytes(png.tobytes())
    (root / "inside.png").write_bytes(png.tobytes())
    j, blob = tri_gltf(b64_uri(tri_blob()), "inside.png")
    (root / "a.gltf").write_text(json.dumps(j))
    assert read_gltf(str(root / "a.gltf")).nfaces == 1
    j["images"][0]["uri"] = "../outside.png"
    (root / "b.gltf").write_text(json.dumps(j))
    with pytest.raises(GltfError, match="outside"):
        read_gltf(str(root / "b.gltf"))


def fake_png_header(w, h):
    """A syntactically valid PNG whose header claims w x h pixels but carries almost no data (a decompression bomb)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 16)) + chunk(b"IEND", b"")


def test_image_bomb_is_refused_before_decoding_but_a_normal_image_passes(tmp_path):
    for dims, should_pass in (((64, 64), True), ((60000, 60000), False)):
        data = fake_png_header(*dims) if not should_pass else cv2.imencode(".png", np.zeros(dims + (3,), np.uint8))[1].tobytes()
        j, _ = tri_gltf(b64_uri(tri_blob()), "data:image/png;base64," + base64.b64encode(data).decode())
        p = tmp_path / f"i{dims[0]}.gltf"
        p.write_text(json.dumps(j))
        if should_pass:
            assert read_gltf(str(p)).nfaces == 1
        else:
            with pytest.raises(GltfError, match="megapixel"):
                read_gltf(str(p))


@pytest.mark.parametrize("mutate", [
    lambda j: j["accessors"][0].pop("componentType"),
    lambda j: j["accessors"][0].update(count=-5),
    lambda j: j["accessors"][0].update(count=10 ** 12),
    lambda j: j["accessors"][0].pop("bufferView"),                                       # zero-filled accessor, count stays sane -> fine
    lambda j: j["accessors"][0].update(count=10 ** 9),
    lambda j: j["accessors"][0].update(type="VEC9"),
    lambda j: j["bufferViews"][0].update(byteOffset=-4),
    lambda j: j["bufferViews"][0].update(byteLength="lots"),
    lambda j: j["meshes"][0]["primitives"][0].update(indices=99),
    lambda j: j["meshes"][0]["primitives"][0]["attributes"].update(POSITION="x"),
    lambda j: j.update(scenes=[{"nodes": ["a"]}]),
    lambda j: j.update(nodes=[{"mesh": 0, "matrix": [1, 2, 3]}]),
    lambda j: j.update(buffers=[{"byteLength": 36}]),
])
def test_malformed_files_raise_gltf_error_never_a_stray_exception(tmp_path, mutate):
    j, _ = tri_gltf(b64_uri(tri_blob()))
    mutate(j)
    p = tmp_path / "m.gltf"
    p.write_text(json.dumps(j))
    try:
        read_gltf(str(p))
    except GltfError:
        pass                                                                              # the only acceptable failure type


def test_accessor_without_data_cannot_force_a_huge_allocation(tmp_path):
    j, _ = tri_gltf(b64_uri(tri_blob()))
    j["accessors"][0].pop("bufferView")                                  # spec: such an accessor reads as zeros -> count must be capped
    j["accessors"][0]["count"] = 3
    (tmp_path / "ok.gltf").write_text(json.dumps(j))
    assert read_gltf(str(tmp_path / "ok.gltf")).nverts == 3                # positive: a small zero accessor is legal
    j["accessors"][0]["count"] = 4 * 10 ** 9
    (tmp_path / "huge.gltf").write_text(json.dumps(j))
    with pytest.raises(GltfError, match="element count"):
        read_gltf(str(tmp_path / "huge.gltf"))


def test_hostile_structures_terminate_quickly(tmp_path):
    j, _ = tri_gltf(b64_uri(tri_blob()))
    j["nodes"] = [{"children": [0], "mesh": 0}]                                           # a node that is its own child
    (tmp_path / "cycle.gltf").write_text(json.dumps(j))
    assert read_gltf(str(tmp_path / "cycle.gltf")).nfaces >= 1                            # depth-capped, does not hang
    (tmp_path / "deep.gltf").write_text("[" * 200000 + "]" * 200000)
    with pytest.raises(GltfError):
        read_gltf(str(tmp_path / "deep.gltf"))
    (tmp_path / "bad.gltf").write_bytes(b"\xff\xfe not json")
    with pytest.raises(GltfError):
        read_gltf(str(tmp_path / "bad.gltf"))
    glb = b"glTF" + struct.pack("<II", 2, 1000) + struct.pack("<II", 900, 0x4E4F534A) + b"{}"
    (tmp_path / "trunc.glb").write_bytes(glb)
    with pytest.raises(GltfError):
        read_gltf(str(tmp_path / "trunc.glb"))


# ------------------------------------------------------------------ robustness against non-finite and degenerate data
@pytest.mark.parametrize("where,value", [("uv", np.nan), ("uv", np.inf), ("verts", np.nan), ("verts", np.inf)])
def test_render_survives_non_finite_geometry_and_uvs(where, value):
    nm = RNG.random((8, 8, 3)).astype(np.float32)
    m = M.cube().with_texture(M.Texture(RNG.random((8, 8, 4)).astype(np.float32), normal=nm))
    getattr(m, where)[:3] = value
    with np.errstate(all="raise"):                                                               # no silent invalid casts either
        r = render(m, RenderSettings(width=40, height=30, ssaa=1, shadows=False, floor="none"))
    assert np.isfinite(r["image"]).all() and np.isfinite(r["normal"]).all()
    healthy = render(M.cube().with_texture(m.textures[0]), RenderSettings(width=40, height=30, ssaa=1, shadows=False, floor="none"))["image"]
    assert np.abs(r["image"] - healthy).mean() < 0.2                                               # the rest of the cube still draws


def test_empty_textures_are_refused_and_odd_shapes_work():
    with pytest.raises(ValueError, match="non-empty"):
        M.Texture(np.ones((0, 4, 4), np.float32))
    for shape in ((1, 64), (64, 1), (1, 1)):
        t = M.Texture(np.ones(shape + (4,), np.float32), normal=np.full(shape + (3,), 0.5, np.float32))
        assert np.isfinite(render(M.cube().with_texture(t), RenderSettings(width=32, height=24, ssaa=1))["image"]).all()


def test_degenerate_uvs_and_mixed_faces_do_not_break_normal_mapping():
    d = M.cube(); d.uv[:] = 0.3                                                                  # every triangle has zero UV area
    t = M.Texture(None, normal=tilted_normal(0.5, 0.5), scale=(-2, 3))                            # also a mirrored scale
    mixed = d.with_texture(t).merged(M.uv_sphere(0.5, 12, 6)).merged(M.cube().with_texture(t))
    r = render(mixed, RenderSettings(width=48, height=36, ssaa=1, shadows=False))
    assert np.isfinite(r["image"]).all()
    T, B = M.plane().with_texture(t).tangent_frame()
    assert T[0][0] < -0.99                                                                       # a negative u-scale flips the tangent


def test_gltf_with_nan_positions_and_uvs_loads_as_finite(tmp_path):
    j, blob = tri_gltf(b64_uri(np.full(15, np.nan, np.float32).tobytes()[:0] + tri_blob()), "data:image/png;base64," +
                       base64.b64encode(cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()).decode())
    nan_blob = np.concatenate([np.array([[np.nan, 0, 0], [np.inf, 0, 0], [0, 1, 0]], np.float32).reshape(-1),
                               np.array([[np.nan, 0], [0, np.inf], [0, 1]], np.float32).reshape(-1)]).tobytes()
    j["buffers"][0]["uri"] = b64_uri(nan_blob)
    (tmp_path / "nan.gltf").write_text(json.dumps(j))
    m = read_gltf(str(tmp_path / "nan.gltf"))
    assert np.isfinite(m.verts).all() and np.isfinite(m.uv).all()
