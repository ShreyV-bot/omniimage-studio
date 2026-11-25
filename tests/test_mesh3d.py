import numpy as np
import pytest
import omniimage as o
from omniimage import mesh as M
from omniimage.core.params import ParamError
from omniimage.raster import RenderSettings, render, Camera, rasterize


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("make,euler", [(M.cube, 2), (M.uv_sphere, 2), (M.icosphere, 2), (M.cylinder, 2), (M.cone, 2), (M.torus, 0)])
def test_primitives_are_closed_manifolds_with_outward_faces(make, euler):
    m = make()
    assert m.is_watertight() and m.euler_characteristic() == euler
    assert m.volume() > 0                                    # positive signed volume = faces point outward
    fn, c = m.face_normals(), m.verts[m.faces].mean(1)
    if make in (M.cube, M.uv_sphere, M.icosphere):           # convex around the origin: normals agree with position
        assert (np.einsum("ij,ij->i", fn, c) > 0).all()


def test_analytic_volumes_and_areas():
    assert M.cube((2, 3, 4)).volume() == pytest.approx(24, rel=1e-5)
    assert M.cube((2, 3, 4)).surface_area() == pytest.approx(2 * (6 + 8 + 12), rel=1e-5)
    s = M.uv_sphere(1.0, 96, 48)
    assert s.volume() == pytest.approx(4 / 3 * np.pi, rel=0.01) and s.surface_area() == pytest.approx(4 * np.pi, rel=0.01)
    assert M.cylinder(0.5, 2.0, 128).volume() == pytest.approx(np.pi * 0.25 * 2, rel=0.01)
    assert M.cone(0.5, 1.0, 128).volume() == pytest.approx(np.pi * 0.25 / 3, rel=0.01)
    assert M.torus(1.0, 0.3, 96, 48).volume() == pytest.approx(2 * np.pi ** 2 * 1.0 * 0.09, rel=0.01)


def test_vertex_normals_are_unit_and_point_out_on_sphere():
    m = M.icosphere(2.0, 2)
    n = m.vertex_normals()
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-5)
    assert (np.einsum("ij,ij->i", n, m.verts / np.linalg.norm(m.verts, axis=1, keepdims=True)) > 0.99).all()


def test_transform_order_and_mirror_keeps_faces_outward():
    m = M.cube((1, 1, 1)).transformed(M.trs_matrix((5, 0, 0), (0, 90, 0), (2, 1, 1)))
    lo, hi = m.bounds()
    assert np.allclose((lo + hi) / 2, [5, 0, 0], atol=1e-5) and np.allclose(hi - lo, [1, 1, 2], atol=1e-5)  # scaled in x then rotated
    mir = M.cube().transformed(np.diag([-1, 1, 1, 1.0]))
    assert mir.volume() > 0


def test_weld_and_join_keep_materials():
    a = M.cube().with_material(color=(1, 0, 0), metal=1.0)
    b = M.cube().transformed(M.trs_matrix((3, 0, 0))).with_material(color=(0, 0, 1))
    j = a.merged(b)
    assert j.nverts == 48 and j.nfaces == 24
    assert np.allclose(j.color[0], [1, 0, 0]) and np.allclose(j.color[-1], [0, 0, 1]) and j.metal[0] == 1 and j.metal[-1] == 0
    assert a.weld().nverts == 8


def test_subdivision_counts_and_loop_smoothing():
    ico = M.icosphere(1.0, 0)
    flat = M.subdivide(ico, 1, smooth=False)
    assert flat.nfaces == 80 and flat.is_watertight()
    assert flat.volume() == pytest.approx(ico.volume(), rel=1e-5)          # midpoints only: same shape
    sm = M.subdivide(ico, 2, smooth=True)
    assert sm.nfaces == 320 and sm.is_watertight() and sm.euler_characteristic() == 2
    r = np.linalg.norm(sm.verts, axis=1)
    assert r.std() < 0.03 and 0.7 < r.mean() < 1.0                          # Loop converges toward a (slightly shrunk) sphere
    plane = M.subdivide(M.plane(2, 2), 2)
    assert plane.nfaces == 32 and np.allclose(plane.verts[:, 1], 0, atol=1e-6)   # flat stays flat
    lo, hi = plane.bounds()                                                   # Loop rounds the corners slightly but never grows the sheet
    assert -1.0 <= lo[0] < -0.9 and 0.9 < hi[0] <= 1.0 and -1.0 <= lo[2] < -0.9


def test_laplacian_smoothing_reduces_roughness():
    rough = M.displace_noise(M.icosphere(1, 3), amount=0.15, scale=6)
    smooth = M.smooth_laplacian(rough, 10, 0.5)
    r0, r1 = np.linalg.norm(rough.verts, axis=1).std(), np.linalg.norm(smooth.verts, axis=1).std()
    assert r1 < r0 * 0.5


def test_noise_displacement_is_deterministic_and_bounded():
    a = M.displace_noise(M.icosphere(1, 2), 0.2, 3, 3, seed=7)
    b = M.displace_noise(M.icosphere(1, 2), 0.2, 3, 3, seed=7)
    c = M.displace_noise(M.icosphere(1, 2), 0.2, 3, 3, seed=8)
    assert np.array_equal(a.verts, b.verts) and not np.allclose(a.verts, c.verts)
    r = np.linalg.norm(a.verts, axis=1)
    assert r.min() >= 0.79 and r.max() <= 1.21 and r.std() > 0.01
    n = M.value_noise3(np.random.default_rng(0).random((5000, 3)) * 10)
    assert 0 <= n.min() and n.max() <= 1 and 0.3 < n.mean() < 0.7


def test_twist_array_mirror():
    t = M.twist(M.cube((1, 2, 1), 4), 90, "y")
    assert t.verts[:, 1].min() == pytest.approx(-1) and not np.allclose(t.verts[:, 0], M.cube((1, 2, 1), 4).verts[:, 0])
    ar = M.array(M.cube(), 4, (2, 0, 0))
    assert ar.nfaces == 48 and ar.bounds()[1][0] == pytest.approx(6.5)
    mi = M.mirror(M.cube().transformed(M.trs_matrix((1, 0, 0))), "x")
    assert mi.bounds()[0][0] == pytest.approx(-1.5) and mi.volume() == pytest.approx(2.0, rel=1e-5)


# ------------------------------------------------------------------ file I/O
def test_obj_roundtrip_and_quads_negative_indices(tmp_path):
    m = M.torus(1, .3, 12, 6)
    M.write_obj(m, str(tmp_path / "t.obj"))
    b = M.read_obj(str(tmp_path / "t.obj"))
    assert b.nverts == m.nverts and np.array_equal(b.faces, m.faces) and np.allclose(b.verts, m.verts, atol=1e-5)
    (tmp_path / "q.obj").write_text("# quad with texture/normal indices\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n"
                                    "f 1/1/1 2/1/1 3/1/1 4/1/1\nf -4 -3 -2\n")
    q = M.read_obj(str(tmp_path / "q.obj"))
    assert q.nfaces == 3 and q.surface_area() == pytest.approx(1.0 + 0.5)
    (tmp_path / "bad.obj").write_text("# nothing\n")
    with pytest.raises(ValueError):
        M.read_obj(str(tmp_path / "bad.obj"))


def test_stl_roundtrip_binary_and_ascii(tmp_path):
    m = M.uv_sphere(1, 12, 6)
    M.write_stl(m, str(tmp_path / "s.stl"))
    b = M.read_stl(str(tmp_path / "s.stl"))
    assert b.nfaces == m.nfaces and b.nverts == m.weld().nverts and b.volume() == pytest.approx(m.volume(), rel=1e-4) and b.is_watertight()
    (tmp_path / "a.stl").write_text("solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\nendsolid\n")
    assert M.read_stl(str(tmp_path / "a.stl")).nfaces == 1
    with pytest.raises(ValueError):
        M.read_mesh(str(tmp_path / "x.fbx"))


# ------------------------------------------------------------------ rasteriser / renderer
def _sphere_scene():
    return M.uv_sphere(1.0, 32, 16)


def test_rasterizer_depth_ordering():
    near = np.array([[0, 0, 0.2], [10, 0, 0.2], [0, 10, 0.2]], float)
    far = np.array([[0, 0, 0.8], [10, 0, 0.8], [0, 10, 0.8]], float)
    verts = np.r_[far, near]
    sx, sy, sz = verts[:, 0], verts[:, 1], verts[:, 2]
    tri, w0, w1, z = rasterize(sx, sy, sz, np.array([[0, 1, 2], [3, 4, 5]]), 10, 10)
    assert tri[1, 1] == 1 and z[1, 1] == pytest.approx(0.2, abs=1e-5)       # nearer triangle wins regardless of draw order
    tri2, *_ = rasterize(sx, sy, sz, np.array([[3, 4, 5], [0, 1, 2]]), 10, 10)
    assert tri2[1, 1] == 0
    assert tri[9, 9] == -1                                                    # outside the triangle


def test_sphere_coverage_matches_projection():
    s = RenderSettings(width=200, height=200, ssaa=2, background="transparent", shadows=False, fov=40)
    r = render(_sphere_scene(), s)
    img = r["image"]
    assert img.shape == (200, 200, 4) and img.dtype == np.float32 and np.isfinite(img).all()
    area = img[..., 3].sum()
    assert 0.5 * 200 * 200 * 0.3 < area < 200 * 200 * 0.8                     # sphere fills a sensible fraction
    assert img[100, 100, 3] > 0.99 and img[2, 2, 3] == 0                      # centre covered, corner empty
    ys, xs = np.nonzero(img[..., 3] > 0.5)
    assert abs(xs.mean() - 100) < 2 and abs(ys.mean() - 100) < 2              # centred (auto camera orbits the centre)


def test_lighting_is_brighter_toward_the_light_and_specular_changes_with_roughness():
    base = dict(width=160, height=160, ssaa=1, background="transparent", shadows=False, fill_intensity=0, rim_intensity=0, ambient=0,
                camera_mode="manual", eye=(0, 0, 5), target=(0, 0, 0), tonemap="none", key_azimuth=90, key_elevation=0)
    lit = render(_sphere_scene(), RenderSettings(**base))["image"]
    right, left = lit[80, 120:140, :3].mean(), lit[80, 20:40, :3].mean()      # light comes from +X
    assert right > 5 * max(left, 1e-4)
    glossy = render(_sphere_scene().with_material(rough=0.15), RenderSettings(**{**base, "key_azimuth": 20, "key_elevation": 20}))["image"]
    matte = render(_sphere_scene().with_material(rough=1.0), RenderSettings(**{**base, "key_azimuth": 20, "key_elevation": 20}))["image"]
    assert glossy[..., :3].max() > matte[..., :3].max() * 1.5                 # tight highlight is brighter


def test_metal_tints_reflection_with_base_colour():
    s = RenderSettings(width=96, height=96, ssaa=1, shadows=False, tonemap="none", background="transparent")
    gold = render(_sphere_scene().with_material(color=(1.0, 0.6, 0.1), metal=1.0, rough=0.3), s)["image"]
    c = gold[..., :3][gold[..., 3] > 0.5].mean(0)
    assert c[0] > c[1] > c[2]                                                 # R > G > B for gold-ish metal


def test_shadow_darkens_floor_under_object():
    s = RenderSettings(width=200, height=120, ssaa=1, floor="plane", camera_mode="manual", eye=(0, 6, 0.01), target=(0, 0, 0),
                       key_azimuth=0, key_elevation=60, tonemap="none", fov=50, background="color")
    sphere = M.uv_sphere(0.6, 24, 12).transformed(M.trs_matrix((0, 0.6, 0)))
    on = render(sphere, s)["image"][..., :3].mean(2)
    off = render(sphere, RenderSettings(**{**s.__dict__, "shadows": False}))["image"][..., :3].mean(2)
    assert (on < off * 0.8).sum() > 100                                       # a shadow patch exists only with shadows on
    assert (on <= off + 1e-4).all()                                           # shadows never add light


def test_passes_depth_normal_and_wireframe():
    r = render(_sphere_scene(), RenderSettings(width=100, height=100, ssaa=1, shadows=False))
    d, n = r["depth"], r["normal"]
    assert d.shape == (100, 100) and d[0, 0] == 0 and d[50, 50] > 0.9         # near surface at the centre is brightest
    cols = np.nonzero(d[50] > 0)[0]
    assert len(cols) > 20 and d[50, 50] > d[50, cols[0]] > 0                  # the sphere's rim is farther than its centre
    from omniimage.raster import dir_from_angles
    assert np.allclose(n[50, 50, :3], dir_from_angles(30, 20) * 0.5 + 0.5, atol=0.03)   # centre pixel faces the orbiting camera
    w = render(M.icosphere(1, 1), RenderSettings(width=120, height=120, ssaa=1, shadows=False, wireframe=True,
                                                    wire_color=(0, 0, 0), tonemap="none"))["image"]
    p = render(M.icosphere(1, 1), RenderSettings(width=120, height=120, ssaa=1, shadows=False, tonemap="none"))["image"]
    assert (w[..., :3].sum(2) < p[..., :3].sum(2) * 0.5).sum() > 100          # wire lines darken some pixels


def test_antialiasing_smooths_edges():
    a = render(_sphere_scene(), RenderSettings(width=80, height=80, ssaa=1, background="transparent", shadows=False))["image"][..., 3]
    b = render(_sphere_scene(), RenderSettings(width=80, height=80, ssaa=3, background="transparent", shadows=False))["image"][..., 3]
    partial = lambda x: ((x > 0.05) & (x < 0.95)).sum()
    assert partial(b) > 3 * max(partial(a), 1)


def test_orbit_turntable_and_determinism():
    m = M.cube((2.0, 1.0, 0.4))                              # a torus would look identical from every azimuth
    s = lambda az: RenderSettings(width=96, height=54, ssaa=1, azimuth=az, shadows=False)
    a, a2, b = render(m, s(0))["image"], render(m, s(0))["image"], render(m, s(90))["image"]
    assert np.array_equal(a, a2) and np.abs(a - b).mean() > 0.005
    ortho = render(m, RenderSettings(width=96, height=54, ssaa=1, ortho=True, shadows=False))["image"]
    assert np.isfinite(ortho).all()


def test_empty_mesh_and_crossing_near_plane_do_not_crash():
    r = render(M.Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32)), RenderSettings(width=32, height=18, ssaa=1))
    assert r["image"].shape == (18, 32, 4)
    inside = render(M.cube((10, 10, 10)), RenderSettings(width=32, height=18, ssaa=1, camera_mode="manual", eye=(0, 0, 0), target=(0, 0, 1)))
    assert np.isfinite(inside["image"]).all()


# ------------------------------------------------------------------ nodes / pipeline integration
def test_mesh_nodes_pipeline_and_3d_feeds_compositing(tmp_path):
    g = o.Graph()
    t = g.add("mesh_torus", segments=24, tube_segments=8)
    mat = g.add("mesh_material", color="#ff8800", metallic=0.0, roughness=0.4, emission_color="#ff8800", emission_strength=0.0)
    tw = g.add("mesh_twist", angle=45)
    r = g.add("render_mesh", width=128, height=72, quality=1, azimuth="=frame*10", floor="checker")
    g.chain(t, mat, tw, r)
    a, b = g.evaluate(r, ctx=o.Ctx(0)), g.evaluate(r, ctx=o.Ctx(9))
    assert a.shape == (72, 128, 4) and np.abs(a - b).mean() > 0.003            # turntable animates
    # depth pass drives a 2D node: mask -> blur -> image
    depth = g.evaluate(r, "depth")
    assert depth.shape == (72, 128) and 0 <= depth.min() and depth.max() <= 1
    # the beauty pass goes through 2D colour nodes and a scope
    sat = g.add("saturation", amount=0.0); sc = g.add("scope_waveform", width=64, height=32)
    g.connect(r, sat); g.connect(sat, sc)
    assert g.evaluate(sc).shape == (32, 64, 4)
    # file round trip through nodes
    w = g.add("mesh_write", path=str(tmp_path / "out.obj")); g.connect(tw, w); g.evaluate(w)
    ld = g.add("mesh_load", path=str(tmp_path / "out.obj")); rr = g.add("render_mesh", width=64, height=36, quality=1); g.connect(ld, rr)
    assert g.evaluate(rr).shape == (36, 64, 4)
    with pytest.raises(o.GraphError):
        g.connect(r, g.add("mesh_twist"))                                      # image -> mesh is a type error


def test_mesh_load_requires_path_and_type_checking():
    g = o.Graph()
    n = g.add("mesh_load")
    with pytest.raises(ParamError, match="path"):
        g.evaluate(n)


def test_showcase_preset_and_cli(tmp_path, capsys):
    from omniimage.presets import showcase_3d
    from omniimage.cli import main
    g, out = showcase_3d(160, 90, quality=1)
    assert g.evaluate(out).shape == (90, 160, 4) and not g.validate()
    assert main(["mesh", "make", "cone", "-o", str(tmp_path / "c.stl"), "--subdivide", "1"]) == 0
    assert main(["mesh", "info", str(tmp_path / "c.stl")]) == 0
    assert "watertight=True" in capsys.readouterr().out
    assert main(["mesh", "render", str(tmp_path / "c.stl"), "-o", str(tmp_path / "c.png"), "--width", "96", "--height", "54",
                 "--quality", "1", "--floor", "--wireframe"]) == 0
    assert (tmp_path / "c.png").exists()
    assert main(["mesh", "make", "teapot", "-o", str(tmp_path / "t.obj")]) == 1


def test_copilot_builds_3d_scene_offline_and_sandboxes_mesh_writes(tmp_path):
    from omniimage.ai_assistant import Copilot, Toolbox
    c = Copilot(workdir=str(tmp_path))
    r = c.ask("render a gold 3D torus")
    assert (tmp_path / "scene3d.png").exists() and any(n.type == "render_mesh" for n in c.project.graph.nodes.values())
    assert any(n.type == "mesh_material" and n.params["metallic"] == 1.0 for n in c.project.graph.nodes.values())
    tb = Toolbox(o.Project(), str(tmp_path / "w"))
    assert not tb.call("add_node", {"type": "mesh_write", "params": {"path": "../../evil.obj"}})["ok"]
    assert not tb.call("add_node", {"type": "write_image", "params": {"path": "/etc/evil.png"}})["ok"]
    assert tb.call("create_3d_scene", {"shape": "cube", "turntable": True})["ok"]
    assert not tb.call("create_3d_scene", {"shape": "teapot"})["ok"]


def test_gui_previews_mesh_nodes(tmp_path):
    from omniimage.gui.server import Studio
    st = Studio(workdir=str(tmp_path))
    st.project.apply({"op": "add_node", "type": "mesh_torus", "id": "t"})
    png, kind = st.render_png("t", maxw=160)
    assert kind == "mesh" and png[:4] == b"\x89PNG"
