import numpy as np
import pytest
import omniimage as o
from omniimage.lut import LUT3D, bake_lut


def test_identity_is_identity():
    lut = LUT3D.identity(9)
    x = np.random.default_rng(0).random((50, 3)).astype(np.float32)
    assert np.allclose(lut.apply(x), x, atol=1e-5)


def test_trilinear_exact_for_linear_function():
    lut = LUT3D.from_function(lambda v: v * 0.5 + 0.25, size=5)
    x = np.random.default_rng(1).random((200, 3)).astype(np.float32)
    assert np.allclose(lut.apply(x), x * 0.5 + 0.25, atol=1e-5)


def test_cube_roundtrip(tmp_path):
    lut = LUT3D.from_function(lambda v: v[:, ::-1], size=5, title="swap")
    path = str(tmp_path / "t.cube")
    lut.write_cube(path)
    back = LUT3D.read_cube(path)
    assert back.size == 5 and back.title == "swap"
    assert np.allclose(back.table, lut.table, atol=1e-5)
    x = np.array([[0.2, 0.5, 0.9]], np.float32)
    assert np.allclose(back.apply(x), [[0.9, 0.5, 0.2]], atol=1e-5)


def test_cube_red_fastest_order(tmp_path):
    p = tmp_path / "o.cube"
    rows = ["LUT_3D_SIZE 2"] + [f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1)]
    p.write_text("\n".join(rows))
    lut = LUT3D.read_cube(str(p))
    assert np.allclose(lut.apply(np.array([[1.0, 0.0, 0.0]], np.float32)), [[1, 0, 0]])
    assert np.allclose(lut.apply(np.array([[0.0, 0.0, 1.0]], np.float32)), [[0, 0, 1]])


def test_1d_cube_is_expanded(tmp_path):
    p = tmp_path / "one.cube"
    p.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
    lut = LUT3D.read_cube(str(p))
    x = np.array([[0.3, 0.6, 0.9]], np.float32)
    assert np.allclose(lut.apply(x), x, atol=1e-4)


def test_bad_cube_row_count(tmp_path):
    p = tmp_path / "bad.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n")
    with pytest.raises(ValueError):
        LUT3D.read_cube(str(p))


def grade_graph():
    g = o.Graph()
    src = g.add("lut_input", id="src")
    lin = g.add("srgb_to_linear")
    ex = g.add("exposure", stops=0.4)
    sat = g.add("saturation", amount=1.3)
    out = g.add("linear_to_srgb")
    g.chain(src, lin, ex, sat, out)
    return g, src, out


def test_bake_matches_direct_evaluation():
    g, src, out = grade_graph()
    lut = bake_lut(g, out, src, size=33)
    # evaluate the same graph on random pixels by overriding the input
    rng = np.random.default_rng(3)
    px = rng.random((40, 30, 3)).astype(np.float32)
    img = np.concatenate([px, np.ones((40, 30, 1), np.float32)], -1)
    direct = g.evaluate(out, overrides={src: {"image": img}}, override_token="t")[..., :3]
    # inputs near the top of the range clip in the direct path; compare in-gamut pixels only
    ok = (direct < 0.98).all(-1)
    baked = lut.apply(px)
    err = np.abs(baked[ok] - direct[ok])
    # 33-point LUTs cannot follow the steep sRGB toe near black exactly: bound mean and worst case separately
    assert err.mean() < 0.003
    assert err.max() < 0.08


def test_bake_rejects_spatial_nodes():
    g = o.Graph()
    src = g.add("lut_input")
    lin = g.add("srgb_to_linear")
    bl = g.add("blur_gaussian")
    out = g.add("linear_to_srgb")
    g.chain(src, lin, bl, out)
    with pytest.raises(o.GraphError, match="cannot bake"):
        bake_lut(g, out, src)


def test_bake_requires_lut_input_node():
    g = o.Graph()
    s = g.add("solid", width=4, height=4)
    e = g.add("exposure")
    g.connect(s, e)
    with pytest.raises(o.GraphError, match="lut_input"):
        bake_lut(g, e, s)


def test_all_looks_bake_to_finite_valid_luts(tmp_path):
    from omniimage.presets import LOOKS, make_look_lut
    assert len(LOOKS) >= 14
    for name in LOOKS:
        lut = make_look_lut(name, 17)
        assert np.isfinite(lut.table).all(), name
        assert np.abs(lut.table - LUT3D.identity(17).table).max() > 0.01, f"{name} did nothing"
        p = str(tmp_path / f"{name}.cube")
        lut.write_cube(p)
        assert LUT3D.read_cube(p).size == 17


def test_lut_apply_node_intensity(tmp_path):
    from omniimage.presets import make_look_lut
    make_look_lut("noir", 17).write_cube(str(tmp_path / "n.cube"))
    g = o.Graph()
    s = g.add("gradient", width=16, height=8, kind="radial", color_a="#ff0000", color_b="#00ff00")
    full = g.add("lut_apply", path=str(tmp_path / "n.cube"), intensity=1.0)
    half = g.add("lut_apply", path=str(tmp_path / "n.cube"), intensity=0.5)
    g.connect(s, full); g.connect(s, half)
    a, b, c = g.evaluate(s), g.evaluate(full), g.evaluate(half)
    assert np.allclose(c[..., :3], (a[..., :3] + b[..., :3]) / 2, atol=1e-4)
    assert abs(float(b[..., 0].mean() - b[..., 1].mean())) < 0.05      # noir is (nearly) monochrome
