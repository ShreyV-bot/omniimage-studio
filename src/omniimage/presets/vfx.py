"""Ready-made image effect chains. Each function takes a graph and the id of a source image node and returns the id
of the last node, so effects can be stacked: ``apply("vhs", g, src)``."""
from __future__ import annotations
from typing import Callable, Dict, Tuple

from ..core.graph import Graph


def _add(g, last, node_type, /, **params):
    n = g.add(node_type, **params)
    g.connect(last, n)
    return n


def vhs(g, src, amount=1.0):
    n = _add(g, src, "blur_gaussian", radius=0.8 * amount)
    n = _add(g, n, "rgb_split", dx=3.0 * amount)
    n = _add(g, n, "scanlines", strength=0.3 * amount, period=3.0)
    n = _add(g, n, "film_grain", amount=0.07 * amount, size=1.6)
    n = _add(g, n, "saturation", amount=0.85)
    return _add(g, n, "glitch", amount=0.25 * amount, slices=6, hold=5)


def film(g, src, amount=1.0):
    n = _add(g, src, "tone_map", operator="aces", exposure=0.3)
    n = _add(g, n, "glow", threshold=0.55, radius=18, intensity=0.35 * amount)
    n = _add(g, n, "film_grain", amount=0.05 * amount, size=1.4)
    return _add(g, n, "vignette", amount=0.35 * amount, softness=0.6)


def dreamy(g, src, amount=1.0):
    n = _add(g, src, "glow", threshold=0.3, radius=30, intensity=0.9 * amount)
    n = _add(g, n, "saturation", amount=1.1)
    return _add(g, n, "contrast", amount=0.92)


def glitch_fx(g, src, amount=1.0):
    n = _add(g, src, "glitch", amount=0.7 * amount, slices=20, hold=2)
    return _add(g, n, "chromatic_aberration", amount=1.2 * amount)


def toon(g, src, amount=1.0):
    smooth = _add(g, src, "bilateral", sigma_color=0.15, sigma_space=6.0)
    flat = _add(g, smooth, "posterize", levels=5)
    edges = _add(g, src, "edge_detect", strength=6.0 * amount)
    inv = _add(g, edges, "invert")
    out = g.add("merge", mode="multiply")
    g.connect(flat, out, dst_port="b")
    g.connect(inv, out, dst_port="a")
    return out


def hologram(g, src, amount=1.0):
    n = _add(g, src, "grayscale")
    n = _add(g, n, "gradient_map", colors=["#001018", "#00e5ff", "#e8ffff"])
    n = _add(g, n, "scanlines", strength=0.5, period=4.0)
    n = _add(g, n, "glitch", amount=0.3 * amount, slices=8, hold=4)
    return _add(g, n, "glow", threshold=0.4, radius=14, intensity=1.0 * amount)


def heat_haze(g, src, amount=1.0):
    w, h = _size(g, src)
    nz = g.add("noise", width=w, height=h, scale=48, octaves=3)
    mv = _add(g, nz, "transform", ty="=t*60")
    out = g.add("displace", amount=6.0 * amount)
    g.connect(src, out, dst_port="image")
    g.connect(mv, out, dst_port="map")
    return out


def _size(g, src):
    n = g.nodes[src]
    return n.params.get("width", 960), n.params.get("height", 540)


EFFECTS: Dict[str, Tuple[str, Callable]] = {
    "vhs": ("Analog tape: blur, channel split, scanlines, grain, glitch", vhs),
    "film": ("Filmic tone map, halation glow, grain, vignette", film),
    "dreamy": ("Soft bloom and lowered contrast", dreamy),
    "glitch": ("Digital slice glitch + chromatic aberration", glitch_fx),
    "toon": ("Cartoon: flat colors with ink outlines", toon),
    "hologram": ("Cyan hologram with scanlines and glow", hologram),
    "heat_haze": ("Animated heat shimmer distortion (needs source with width/height params)", heat_haze),
}


def apply_effect(name: str, g: Graph, src: str, amount: float = 1.0) -> str:
    if name not in EFFECTS:
        raise KeyError(f"unknown effect {name!r}. Available: {', '.join(sorted(EFFECTS))}")
    return EFFECTS[name][1](g, src, amount)


def neon_sign(text="OMNI", color="#ff2fd0", width=960, height=540, size=200, flicker=True) -> Tuple[Graph, str]:
    """Self-contained neon sign title: dark wall + glowing text with optional flicker. Returns (graph, output_id)."""
    g = Graph()
    bg = g.add("gradient", id="wall", width=width, height=height, kind="radial", color_a="#1a1020", color_b="#050308")
    tx = g.add("text", id="tube", width=width, height=height, text=text, size=size, color=color)
    core = g.add("exposure", id="hot", stops=0.6)
    g.connect(tx, core)
    inten = "=1.4+0.5*pulse(t,9)*sin(t*50)" if flicker else 1.4
    glow = g.add("glow", id="glow", threshold=0.0, radius=26, intensity=inten)
    g.connect(core, glow)
    out = g.add("merge", id="out", mode="over")
    g.connect(glow, out, dst_port="a")
    g.connect(bg, out, dst_port="b")
    return g, out


def green_screen_composite(fg_path: str, bg_path: str, key="#00ff00") -> Tuple[Graph, str]:
    g = Graph()
    fg = g.add("load_image", id="fg", path=fg_path)
    bg = g.add("load_image", id="bg", path=bg_path)
    k = g.add("chroma_key", id="key", key_color=key, tolerance=0.25, softness=0.15, despill=0.9)
    g.connect(fg, k)
    out = g.add("merge", id="out", mode="over")
    g.connect(k, out, src_port="image", dst_port="a")
    g.connect(bg, out, dst_port="b")
    return g, out


def demo_scene(width=960, height=540) -> Tuple[Graph, str]:
    """A colourful synthetic test image (no input files needed). Returns (graph, output_id)."""
    g = Graph()
    nz = g.add("noise", id="out_noise", width=width, height=height, scale=90, octaves=5, seed=4)
    lin = g.add("srgb_to_linear", id="out_lin")          # noise values are display-style gray
    g.connect(nz, lin)
    gm = g.add("gradient_map", id="out_map", colors=["#04101f", "#1b2a5c", "#7a2f6b", "#d9663f", "#f5c98a"])
    g.connect(lin, gm)
    ring = g.add("shape_mask", id="ring", width=width, height=height, kind="ellipse", w=0.32, h=0.58, feather=0.04)
    sun = g.add("solid", id="sun", width=width, height=height, color="#ffe2a8")
    sunkey = g.add("set_alpha", id="sun_key")
    g.connect(sun, sunkey, dst_port="image")
    g.connect(ring, sunkey, dst_port="mask")
    out = g.add("merge", id="out", mode="over")
    g.connect(sunkey, out, dst_port="a")
    g.connect(gm, out, dst_port="b")
    return g, out
