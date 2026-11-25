import numpy as np
import pytest
import omniimage as o
from omniimage import colorspaces as cs


# ---------- colour management ----------
def test_known_reference_values():
    # sRGB red -> ACEScg is a published reference value
    out = cs.convert(np.array([[1.0, 0.0, 0.0]], np.float32), "linear_rec709", "acescg")[0]
    assert np.allclose(out, [0.6131, 0.0702, 0.0206], atol=2e-3)
    # 18% grey lands on the published code values of each log curve
    g = np.full((1, 3), 0.18, np.float32)
    assert cs.convert(g, "linear_rec709", "logc3_awg3")[0, 0] == pytest.approx(0.3910, abs=2e-3)
    assert cs.convert(g, "linear_rec709", "slog3_sgamut3")[0, 0] == pytest.approx(0.4106, abs=2e-3)
    assert cs.convert(g, "linear_rec709", "acescct")[0, 0] == pytest.approx(0.4135, abs=2e-3)


@pytest.mark.parametrize("space", sorted(cs.SPACES))
def test_roundtrip_every_space(space):
    rng = np.random.default_rng(0)
    x = rng.random((200, 3)).astype(np.float32) * 0.9 + 0.01
    back = cs.convert(cs.convert(x, "linear_rec709", space), space, "linear_rec709")
    assert np.allclose(back, x, atol=2e-3), space


@pytest.mark.parametrize("space", ["linear_rec2020", "linear_p3d65", "acescg", "logc3_awg3"])
def test_white_stays_neutral(space):
    w = cs.convert(np.ones((1, 3), np.float32), "linear_rec709", space)[0]
    assert np.allclose(w, w[0], atol=1e-3)          # no colour cast on white (Bradford adaptation works)


def test_colorspace_node_and_lut_bake():
    g = o.Graph()
    i = g.add("lut_input", id="i")
    c = g.add("colorspace", from_space="srgb", to_space="acescct")
    g.connect(i, c)
    lut = o.bake_lut(g, c, i, size=9)                # colorspace is pointwise, so it bakes into a LUT
    assert np.isfinite(lut.table).all()


# ---------- scopes ----------
def solid(g, color, w=64, h=48):
    return g.add("solid", width=w, height=h, color=color)


def test_waveform_shows_brightness_level():
    g = o.Graph()
    s = solid(g, "0.8,0.8,0.8")
    w = g.add("scope_waveform", width=64, height=100)
    g.connect(s, w)
    img = g.evaluate(w)
    from omniimage.core.media import linear_to_srgb
    disp = linear_to_srgb(img[..., 1])
    row = int(np.argmax(disp.mean(1)))
    assert abs(row - (1 - 0.8) * 99) <= 2             # bright trace sits at 80% height


def test_parade_separates_channels():
    g = o.Graph()
    s = solid(g, "1,0,0")
    pa = g.add("scope_parade", width=90, height=60)
    g.connect(s, pa)
    img = g.evaluate(pa)
    third = img.shape[1] // 3
    assert img[:, :third, 0].max() > 0.9 and img[:, third:, 0].max() < 0.5      # red only in the left panel
    top_red = int(np.argmax(img[:, :third, 0].mean(1)))
    top_green = int(np.argmax(img[:, third:2 * third, 1].mean(1)))
    assert top_red < 5 and top_green > 55                                  # red trace at the top, green (=0) at the bottom


def test_vectorscope_places_red_above_blue():
    g = o.Graph()
    for colour, expect in (("1,0,0", "up"), ("0,0,1", "down")):
        s = solid(g, colour)
        vs = g.add("scope_vectorscope", size=64)
        g.connect(s, vs)
        img = g.evaluate(vs)[..., 1]
        ys, xs = np.nonzero(img > 0.5)
        assert len(ys)
        # Cb/Cr: blue is right/low, red is right/up-ish. Compare vertical positions
        if expect == "up":
            assert ys.mean() < 32
        else:
            assert xs.mean() > 32


def test_histogram_peaks_at_value():
    g = o.Graph()
    s = solid(g, "0.5,0.5,0.5")
    h = g.add("scope_histogram", width=100, height=40)
    g.connect(s, h)
    img = g.evaluate(h)
    col = np.argmax(img[..., 0].sum(0))
    assert abs(col - 0.5 * 99) <= 2


# ---------- secondary grading ----------
def test_qualifier_selects_only_target_hue():
    g = o.Graph()
    from omniimage.core.media import srgb_to_linear
    img = np.ones((10, 20, 4), np.float32)
    img[:, :10, :3] = srgb_to_linear(np.array([0.9, 0.1, 0.1], np.float32))     # red half
    img[:, 10:, :3] = srgb_to_linear(np.array([0.1, 0.2, 0.9], np.float32))     # blue half
    src = g.add("solid", width=20, height=10)
    q = g.add("hsl_qualifier", hue=0, hue_width=40)
    g.connect(src, q)
    m = g.evaluate(q, overrides={src: {"image": img}}, override_token="q")
    assert m[:, :10].min() > 0.95 and m[:, 10:].max() < 0.05
    q2 = g.add("hsl_qualifier", hue=0, hue_width=40, invert=True)
    g.connect(src, q2)
    m2 = g.evaluate(q2, overrides={src: {"image": img}}, override_token="q")
    assert np.allclose(m2, 1 - m, atol=1e-5)


def test_qualifier_wraps_around_zero_hue():
    from omniimage.nodes.secondary import hsv_of
    h, s, v = hsv_of(np.array([[[1.0, 0.0, 0.1]]], np.float32))
    assert h[0, 0] > 340 or h[0, 0] < 20


def test_power_window_shape_and_invert():
    g = o.Graph()
    w = g.add("power_window", width=100, height=100, kind="rect", w=0.5, h=0.5, softness=0.0)
    m = g.evaluate(w)
    assert m[50, 50] == 1 and m[5, 5] == 0 and m[50, 5] == 0
    assert m.sum() == pytest.approx(2500, rel=0.1)
    wi = g.add("power_window", width=100, height=100, kind="rect", w=0.5, h=0.5, softness=0.0, invert=True)
    assert np.allclose(g.evaluate(wi), 1 - m)
    e = g.add("power_window", width=100, height=100, kind="ellipse", w=0.5, h=0.5, softness=0.0)
    assert g.evaluate(e).sum() == pytest.approx(np.pi * 25 * 25, rel=0.06)


def test_secondary_grade_recipe_end_to_end():
    """Grade only the sky: qualifier mask -> merge(mask)."""
    g = o.Graph()
    from omniimage.core.media import srgb_to_linear
    img = np.ones((10, 20, 4), np.float32)
    img[:, :10, :3] = srgb_to_linear(np.array([0.9, 0.1, 0.1], np.float32))
    img[:, 10:, :3] = srgb_to_linear(np.array([0.1, 0.2, 0.9], np.float32))
    src = g.add("solid", width=20, height=10)
    q = g.add("hsl_qualifier", hue=225, hue_width=60)
    dark = g.add("exposure", stops=-2)
    mg = g.add("merge", mode="over", mix=1.0)
    g.connect(src, dark)
    g.connect(dark, mg, dst_port="a"); g.connect(src, mg, dst_port="b"); g.connect(q, mg, dst_port="mask")
    g.connect(src, q)
    out = g.evaluate(mg, overrides={src: {"image": img}}, override_token="r")
    assert np.allclose(out[:, :10], img[:, :10], atol=1e-4)                  # red untouched
    assert (out[:, 10:, :3] < img[:, 10:, :3] * 0.5).all()                    # blue darkened


# ---------- design (Figma-like) ----------
def test_rect_pixel_exact_area_and_alpha():
    g = o.Graph()
    s = g.add("vector_shape", width=100, height=100, kind="rect", x=50, y=50, w=40, h=20)
    img = g.evaluate(s)
    assert img[..., 3].sum() == pytest.approx(800, abs=1.0)
    assert img[50, 50, 3] == 1 and img[10, 10, 3] == 0


def test_fractional_edge_is_antialiased():
    g = o.Graph()
    s = g.add("vector_shape", width=40, height=40, kind="rect", x=20, y=20, w=10.5, h=10.5)
    a = g.evaluate(s)[..., 3]
    assert a.sum() == pytest.approx(10.5 * 10.5, abs=0.5)
    assert a[14, 20] == pytest.approx(0.25, abs=0.02) and a[20, 14] == pytest.approx(0.25, abs=0.02)


@pytest.mark.parametrize("kind,expected", [("ellipse", np.pi * 30 * 20), ("rounded_rect", 60 * 40 - (4 - np.pi) * 10 * 10),
                                           ("line", 60 * 10)])
def test_shape_areas(kind, expected):
    g = o.Graph()
    kw = dict(w=60, h=40, radius=10) if kind != "line" else dict(w=70, h=10)
    kw = {**kw, **({"w": 60, "h": 40} if kind == "ellipse" else {})}
    s = g.add("vector_shape", width=120, height=120, kind=kind, x=60, y=60, **kw)
    if kind == "line":
        expected = 60 * 10 + np.pi * 25
    assert g.evaluate(s)[..., 3].sum() == pytest.approx(expected, rel=0.03)


def test_polygon_and_star():
    g = o.Graph()
    sq = g.add("vector_shape", width=100, height=100, kind="polygon", sides=4, x=50, y=50, w=60, h=60)
    assert g.evaluate(sq)[..., 3].sum() == pytest.approx(2 * 30 * 30, rel=0.03)     # square inscribed in r=30 circle
    st = g.add("vector_shape", width=100, height=100, kind="star", sides=5, x=50, y=50, w=80, h=80)
    a = g.evaluate(st)[..., 3]
    assert 0 < a.sum() < np.pi * 40 * 40 and a[50, 50] == 1


def test_stroke_alignment_changes_size():
    g = o.Graph()
    areas = {}
    for al in ("inside", "center", "outside"):
        s = g.add("vector_shape", width=100, height=100, kind="rect", x=50, y=50, w=40, h=40, stroke_width=6,
                  stroke_align=al, fill="0,0,0,0", stroke="#ffffff")
        a = g.evaluate(s)[..., 3]
        ys, xs = np.nonzero(a > 0.5)
        areas[al] = xs.max() - xs.min() + 1
    assert areas["inside"] == 40 and areas["center"] == 46 and areas["outside"] == 52


def test_gradient_fill_runs_from_color_one_to_two():
    g = o.Graph()
    s = g.add("vector_shape", width=100, height=100, kind="rect", x=50, y=50, w=80, h=80, gradient=True,
              gradient_angle=0, fill="#ff0000", fill2="#0000ff")
    img = g.evaluate(s)
    assert img[50, 12, 0] > img[50, 12, 2] * 5 and img[50, 88, 2] > img[50, 88, 0] * 5


def test_drop_shadow_adds_alpha_around_but_keeps_original():
    g = o.Graph()
    s = g.add("vector_shape", width=100, height=100, kind="rect", x=50, y=50, w=30, h=30)
    d = g.add("drop_shadow", dx=0, dy=10, blur=6, opacity=0.5)
    g.connect(s, d)
    a, b = g.evaluate(s), g.evaluate(d)
    assert np.allclose(b[45:55, 45:55], a[45:55, 45:55], atol=1e-5)                 # shape itself unchanged
    assert b[68, 50, 3] > 0.2 and a[68, 50, 3] == 0                                  # shadow below it
    assert b[20, 50, 3] < 0.05                                                       # none above


def test_auto_layout_row_hug_and_gap():
    g = o.Graph()
    a = g.add("vector_shape", width=40, height=30, kind="rect", x=20, y=15, w=40, h=30, fill="#ff0000")
    b = g.add("vector_shape", width=60, height=50, kind="rect", x=30, y=25, w=60, h=50, fill="#00ff00")
    l = g.add("auto_layout", direction="row", gap=10, padding=5, align="center")
    g.connect(a, l, dst_port="a"); g.connect(b, l, dst_port="b")
    img = g.evaluate(l)
    assert img.shape[:2] == (50 + 10, 40 + 10 + 60 + 10)                              # hug: padding + content + gap
    assert img[30, 25, 0] > 0.9 and img[30, 25, 1] < 0.1                              # A at left, vertically centred
    assert img[30, 80, 1] > 0.9                                                       # B to the right
    assert img[30, 47, 3] < 0.01                                                      # gap is transparent


def test_auto_layout_space_between_fixed_width():
    g = o.Graph()
    a = g.add("vector_shape", width=20, height=20, kind="rect", x=10, y=10, w=20, h=20)
    b = g.add("vector_shape", width=20, height=20, kind="rect", x=10, y=10, w=20, h=20)
    l = g.add("auto_layout", direction="row", padding=0, distribute="space_between", fixed_width=200, fixed_height=20)
    g.connect(a, l, dst_port="a"); g.connect(b, l, dst_port="b")
    img = g.evaluate(l)
    assert img[10, 5, 3] == 1 and img[10, 195, 3] == 1 and img[10, 100, 3] == 0


def test_ui_card_composition():
    """A Figma-style card: rounded rect + shadow + text composed with merge."""
    g = o.Graph()
    card = g.add("vector_shape", width=400, height=240, kind="rounded_rect", x=200, y=120, w=320, h=160, radius=20,
                 fill="#ffffff")
    sh = g.add("drop_shadow", dy=10, blur=24, opacity=0.3)
    bg = g.add("solid", width=400, height=240, color="#e9edf5")
    mg = g.add("merge", mode="over")
    g.connect(card, sh); g.connect(sh, mg, dst_port="a"); g.connect(bg, mg, dst_port="b")
    img = g.evaluate(mg)
    assert img.shape == (240, 400, 4) and np.isfinite(img).all()
    assert img[120, 200, 0] > img[120, 5, 0]                                          # white card brighter than bg
    assert img[215, 200, 0] < img[235, 5, 0]                                          # shadow darkens below the card
