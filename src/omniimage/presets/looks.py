"""Procedural color looks. Original recipes (no third-party film data): each is a node graph of color-only
nodes, so it can be baked to a standard .cube LUT usable in Resolve, Premiere, OBS, FFmpeg, Blender and more."""
from __future__ import annotations
from typing import Callable, Dict, Tuple

from ..core.graph import Graph
from ..lut import LUT3D, bake_lut


def _start(g: Graph):
    src = g.add("lut_input", id="lut_in")
    lin = g.add("srgb_to_linear", id="to_linear")
    g.connect(src, lin)
    return src, lin


def _end(g: Graph, last: str):
    out = g.add("linear_to_srgb", id="to_srgb")
    g.connect(last, out)
    return out


def _add(g, last, node_type, /, **params):
    n = g.add(node_type, **params)
    g.connect(last, n)
    return n


def _display_curve(g, last, points, channel="master"):
    """Apply a tone curve on display-referred (sRGB) values, returning to linear afterwards."""
    a = _add(g, last, "linear_to_srgb")
    b = _add(g, a, "curves", points=points, channel=channel)
    return _add(g, b, "srgb_to_linear")


def teal_orange(g, last):
    last = _add(g, last, "contrast", amount=1.1)
    last = _add(g, last, "color_balance", shadows=[-0.6, 0.1, 0.6], midtones=[0.1, 0, -0.1],
                highlights=[0.7, 0.2, -0.6], strength=0.4)
    return _add(g, last, "saturation", amount=1.1)


def bleach_bypass(g, last):
    last = _add(g, last, "saturation", amount=0.5)
    return _add(g, last, "contrast", amount=1.25)


def noir(g, last):
    last = _add(g, last, "grayscale")
    last = _add(g, last, "contrast", amount=1.2)
    return _display_curve(g, last, [[0, 0], [0.25, 0.18], [0.75, 0.85], [1, 1]])


def vintage_fade(g, last):
    last = _display_curve(g, last, [[0, 0.09], [0.5, 0.52], [1, 0.94]])
    last = _add(g, last, "temperature_tint", temperature=0.22, tint=0.05)
    return _add(g, last, "saturation", amount=0.8)


def cinematic_warm(g, last):
    last = _add(g, last, "temperature_tint", temperature=0.25)
    last = _add(g, last, "contrast", amount=1.12)
    return _add(g, last, "saturation", amount=1.05)


def cool_moody(g, last):
    last = _add(g, last, "temperature_tint", temperature=-0.3)
    last = _add(g, last, "exposure", stops=-0.3)
    last = _add(g, last, "contrast", amount=1.2)
    return _add(g, last, "saturation", amount=0.85)


def cross_process(g, last):
    last = _display_curve(g, last, [[0, 0], [0.5, 0.62], [1, 1]], "r")
    last = _display_curve(g, last, [[0, 0], [0.5, 0.5], [1, 1]], "g")
    last = _display_curve(g, last, [[0, 0.1], [0.5, 0.45], [1, 0.85]], "b")
    return _add(g, last, "saturation", amount=1.25)


def sepia_tone(g, last):
    last = _add(g, last, "sepia", amount=1.0)
    return _add(g, last, "contrast", amount=1.1)


def matrix_green(g, last):
    last = _add(g, last, "grayscale")
    return _add(g, last, "gradient_map", colors=["#000a02", "#00ff41"])


def vaporwave(g, last):
    return _add(g, last, "gradient_map", colors=["#1b0b3a", "#ff2fa3", "#2ff3ff"], amount=0.85)


def golden_hour(g, last):
    last = _add(g, last, "exposure", stops=0.2)
    last = _add(g, last, "temperature_tint", temperature=0.45, tint=0.05)
    last = _add(g, last, "lift_gamma_gain", lift=[0.02, 0.0, -0.01], gamma=[1.05, 1.0, 0.95], gain=[1.05, 1.0, 0.9])
    return _add(g, last, "saturation", amount=1.2)


def day_for_night(g, last):
    last = _add(g, last, "exposure", stops=-1.5)
    last = _add(g, last, "temperature_tint", temperature=-0.5)
    last = _add(g, last, "saturation", amount=0.6)
    return _add(g, last, "color_balance", shadows=[-0.2, 0.0, 0.6], strength=0.4)


def high_contrast_bw(g, last):
    last = _add(g, last, "grayscale")
    return _add(g, last, "contrast", amount=1.4)


def faded_film(g, last):
    last = _display_curve(g, last, [[0, 0.07], [0.3, 0.3], [0.7, 0.72], [1, 0.93]])
    return _add(g, last, "saturation", amount=0.85)


LOOKS: Dict[str, Tuple[str, Callable]] = {
    "teal_orange": ("Blockbuster teal shadows / orange highlights", teal_orange),
    "bleach_bypass": ("Desaturated, gritty, high contrast", bleach_bypass),
    "noir": ("Black and white with deep contrast", noir),
    "vintage_fade": ("Lifted blacks, warm, muted color", vintage_fade),
    "cinematic_warm": ("Warm, slightly contrasty grade", cinematic_warm),
    "cool_moody": ("Cool, darker, desaturated mood", cool_moody),
    "cross_process": ("Colour-shifted cross-processed look", cross_process),
    "sepia_tone": ("Classic sepia", sepia_tone),
    "matrix_green": ("Green phosphor terminal look", matrix_green),
    "vaporwave": ("Purple/pink/cyan gradient look", vaporwave),
    "golden_hour": ("Warm low-sun glow", golden_hour),
    "day_for_night": ("Fake night from daylight footage", day_for_night),
    "high_contrast_bw": ("Punchy monochrome", high_contrast_bw),
    "faded_film": ("Soft faded print look", faded_film),
}


def look_graph(name: str) -> Tuple[Graph, str, str]:
    """Return (graph, input_node, output_node) for a named look."""
    if name not in LOOKS:
        raise KeyError(f"unknown look {name!r}. Available: {', '.join(sorted(LOOKS))}")
    g = Graph()
    src, lin = _start(g)
    last = LOOKS[name][1](g, lin)
    out = _end(g, last)
    return g, src, out


def make_look_lut(name: str, size: int = 33, strength: float = 1.0) -> LUT3D:
    g, src, out = look_graph(name)
    lut = bake_lut(g, out, src, size=size, title=f"OmniImage {name}")
    return lut.blend(strength) if strength != 1.0 else lut
