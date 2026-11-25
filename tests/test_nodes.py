"""Smoke test every registered node: it must run on default parameters and return finite, correctly typed data."""
import numpy as np
import pytest
import omniimage as o
from omniimage.core.ports import PortType as T

SKIP = {"load_image", "load_audio", "read_video_frame", "lut_load", "lut_apply", "write_image", "write_audio",
        "lut_input", "test_counter", "mesh_load", "mesh_write"}


def make_input(g, port_type):
    if port_type == T.IMAGE:
        return g.add("gradient", width=48, height=32, kind="radial", color_a="#203050", color_b="#f0c080")
    if port_type == T.MASK:
        return g.add("shape_mask", width=48, height=32)
    if port_type == T.AUDIO:
        return g.add("osc", duration=0.25, freq_start=300, freq_end=600)
    if port_type == T.MESH:
        return g.add("mesh_torus", segments=12, tube_segments=6)
    raise AssertionError(port_type)


NODE_TYPES = sorted(t for t in o.REGISTRY if t not in SKIP and o.REGISTRY[t].source == "builtin")


def test_registry_is_big_enough():
    assert len(o.REGISTRY) >= 90


@pytest.mark.parametrize("type_", NODE_TYPES)
def test_node_runs_with_defaults(type_):
    spec = o.REGISTRY[type_]
    g = o.Graph()
    params = {}
    for k in ("width", "height"):
        if k in spec.params and (not spec.inputs or type_ == "render_mesh"):
            params[k] = 48 if k == "width" else 32
    if "duration" in spec.params and not spec.inputs:
        params["duration"] = 0.25
    n = g.add(type_, **params)
    for port, pt in spec.inputs.items():
        src = make_input(g, pt)
        if type_ == "stereo_width":
            src = g.chain(src, g.add("pan"))
        g.connect(src, n, dst_port=port)
    res = g.evaluate(n, ctx=o.Ctx(5))
    assert not g.validate()
    for port, pt in spec.outputs.items():
        val = g.evaluate(n, port, ctx=o.Ctx(5))
        if pt == T.IMAGE:
            assert val.dtype == np.float32 and val.ndim == 3 and val.shape[2] == 4, port
            assert np.isfinite(val).all(), f"{type_}.{port} has NaN/inf"
        elif pt == T.MASK:
            assert val.ndim == 2 and np.isfinite(val).all()
        elif pt == T.MESH:
            from omniimage.mesh import Mesh
            assert isinstance(val, Mesh) and val.nfaces > 0 and np.isfinite(val.verts).all()
        elif pt == T.AUDIO:
            assert isinstance(val, o.AudioBuf) and np.isfinite(val.samples).all()


def test_every_node_has_documentation():
    for s in o.REGISTRY.values():
        assert len(s.doc) > 10, s.type
        for k, p in s.params.items():
            assert p.type in ("float", "int", "bool", "str", "choice", "color", "list", "any"), (s.type, k)


def test_pointwise_nodes_are_really_pointwise():
    """A LUT-safe node must give the same result for a pixel regardless of its neighbours / position."""
    rng = np.random.default_rng(0)
    img = np.concatenate([rng.random((16, 16, 3)).astype(np.float32), np.ones((16, 16, 1), np.float32)], -1)
    flipped = np.ascontiguousarray(img[::-1, ::-1])
    for spec in o.REGISTRY.values():
        if not spec.pointwise or spec.type in ("lut_apply",) or set(spec.inputs) != {"image"}:
            continue
        g = o.Graph()
        n = g.add(spec.type)
        out1 = g.evaluate(n, overrides=None, ctx=o.Ctx(0)) if False else None
        g2 = o.Graph()
        src = g2.add("solid", width=16, height=16)
        n2 = g2.add(spec.type)
        g2.connect(src, n2)
        a = g2.evaluate(n2, overrides={src: {"image": img}}, override_token="a")
        b = g2.evaluate(n2, overrides={src: {"image": flipped}}, override_token="b")
        assert np.allclose(a[::-1, ::-1], b, atol=1e-5), f"{spec.type} is not pointwise"


def test_blend_modes_basic_math():
    g = o.Graph()
    a = g.add("solid", width=4, height=4, color="0.5,0.5,0.5")
    b = g.add("solid", width=4, height=4, color="0.5,0.5,0.5")
    from omniimage.core.media import srgb_to_linear
    ga = srgb_to_linear(np.float32(0.5))
    for mode, expect in (("multiply", ga * ga), ("add", ga + ga), ("over", ga), ("difference", 0.0)):
        m = g.add("merge", mode=mode)
        g.connect(a, m, dst_port="a"); g.connect(b, m, dst_port="b")
        assert g.evaluate(m)[0, 0, 0] == pytest.approx(expect, abs=1e-5), mode


def test_over_respects_alpha():
    g = o.Graph()
    fg = g.add("solid", width=4, height=4, color="1,0,0,0.5")
    bg = g.add("solid", width=4, height=4, color="0,0,1,1")
    m = g.add("merge", mode="over")
    g.connect(fg, m, dst_port="a"); g.connect(bg, m, dst_port="b")
    px = g.evaluate(m)[0, 0]
    assert px[3] == pytest.approx(1.0) and px[0] == pytest.approx(0.5, abs=1e-5) and px[2] == pytest.approx(0.5, abs=1e-5)


def test_chroma_key_removes_green_keeps_subject():
    g = o.Graph()
    img = np.zeros((20, 20, 4), np.float32); img[..., 3] = 1
    from omniimage.core.media import srgb_to_linear
    img[:, :10, :3] = srgb_to_linear(np.array([0.0, 1.0, 0.0], np.float32))     # green screen
    img[:, 10:, :3] = srgb_to_linear(np.array([0.8, 0.3, 0.2], np.float32))     # subject
    src = g.add("solid", width=20, height=20)
    k = g.add("chroma_key")
    g.connect(src, k)
    out = g.evaluate(k, "image", overrides={src: {"image": img}}, override_token="k")
    assert out[5, 2, 3] < 0.05 and out[5, 17, 3] > 0.95


def test_transform_identity_and_flip():
    g = o.Graph()
    s = g.add("checker", width=32, height=32, size=4)
    t = g.add("transform")
    g.connect(s, t)
    assert np.allclose(g.evaluate(t), g.evaluate(s), atol=1e-5)
    f = g.add("flip", horizontal=True, vertical=False)
    g.connect(s, f)
    assert np.allclose(g.evaluate(f), g.evaluate(s)[:, ::-1])


def test_film_grain_changes_every_frame_but_is_deterministic():
    g = o.Graph()
    s = g.add("solid", width=32, height=32, color="0.5,0.5,0.5")
    n = g.add("film_grain", amount=0.2)
    g.connect(s, n)
    a, b, a2 = g.evaluate(n, ctx=o.Ctx(1)), g.evaluate(n, ctx=o.Ctx(2)), g.evaluate(n, ctx=o.Ctx(1))
    assert not np.allclose(a, b) and np.allclose(a, a2)


def test_image_io_roundtrip(tmp_path):
    g = o.Graph()
    s = g.add("gradient", width=32, height=16, color_a="#102030", color_b="#f0e0d0")
    w = g.add("write_image", path=str(tmp_path / "a.png"), bits="16")
    g.connect(s, w)
    g.evaluate(w)
    r = g.add("load_image", path=str(tmp_path / "a.png"))
    assert np.abs(g.evaluate(r) - g.evaluate(s)).max() < 2e-3
    # EXR keeps linear float values exactly
    w2 = g.add("write_image", path=str(tmp_path / "a.exr"))
    g.connect(s, w2); g.evaluate(w2)
    r2 = g.add("load_image", path=str(tmp_path / "a.exr"))
    assert np.abs(g.evaluate(r2) - g.evaluate(s)).max() < 2e-3
