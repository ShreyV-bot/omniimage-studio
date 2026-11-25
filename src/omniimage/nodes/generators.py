"""Generators: nodes with no image input."""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from ..core.registry import node
from ..core.params import P, parse_color
from ..core.ports import PortType as T
from ..core.media import srgb_to_linear, ensure_rgba

IMG = {"image": T.IMAGE}
_WH = {"width": P("int", 960, 1, 16384, doc="pixels"), "height": P("int", 540, 1, 16384, doc="pixels")}


def _lin_color(c):
    c = list(c)
    rgb = srgb_to_linear(np.array(c[:3], np.float32))
    return np.array([*rgb, c[3] if len(c) > 3 else 1.0], np.float32)


@node("solid", "generator", "A flat color. The color is given sRGB-encoded (like a hex code) and stored linear.",
      outputs=IMG, params={**_WH, "color": P("color", [0.5, 0.5, 0.5, 1.0], doc="#rrggbb or r,g,b")})
def solid(inp, p, ctx):
    img = np.empty((p["height"], p["width"], 4), np.float32)
    img[:] = _lin_color(p["color"])
    return img


@node("gradient", "generator", "Linear or radial gradient between two colors (sRGB-encoded inputs).",
      outputs=IMG, params={**_WH, "kind": P("choice", "linear", choices=("linear", "radial")),
                           "color_a": P("color", [0, 0, 0, 1]), "color_b": P("color", [1, 1, 1, 1]),
                           "angle": P("float", 0.0, -360, 360, doc="degrees, linear only")})
def gradient(inp, p, ctx):
    h, w = p["height"], p["width"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx / max(w - 1, 1) - 0.5
    yy = yy / max(h - 1, 1) - 0.5
    if p["kind"] == "linear":
        a = np.deg2rad(p["angle"])
        t = xx * np.cos(a) + yy * np.sin(a) + 0.5
    else:
        t = np.sqrt(xx ** 2 + yy ** 2) * 2.0
    t = np.clip(t, 0, 1)[..., None]
    a_, b_ = _lin_color(p["color_a"]), _lin_color(p["color_b"])
    return (a_ * (1 - t) + b_ * t).astype(np.float32)


def _fbm(h, w, scale, octaves, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(scale / (2 ** o), 1.0)
        gh, gw = int(h / s) + 3, int(w / s) + 3
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = cv2.resize(grid, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        out += amp * layer
        total += amp
        amp *= 0.5
    out /= total
    out -= out.min()
    mx = out.max()
    return out / mx if mx > 0 else out


@node("noise", "generator", "Fractal (fBm) noise as a gray image in 0..1. Seeded and deterministic.",
      outputs=IMG, params={**_WH, "scale": P("float", 64.0, 2, 4096, doc="feature size in pixels"),
                           "octaves": P("int", 4, 1, 8), "seed": P("int", 1, 0, 2 ** 31 - 1)})
def noise(inp, p, ctx):
    n = _fbm(p["height"], p["width"], p["scale"], p["octaves"], p["seed"])
    return ensure_rgba(n)


@node("checker", "generator", "Checkerboard test pattern.", outputs=IMG,
      params={**_WH, "size": P("int", 32, 1, 4096), "color_a": P("color", [0.2, 0.2, 0.2, 1]),
              "color_b": P("color", [0.8, 0.8, 0.8, 1])})
def checker(inp, p, ctx):
    h, w, s = p["height"], p["width"], p["size"]
    yy, xx = np.mgrid[0:h, 0:w]
    m = (((xx // s) + (yy // s)) % 2 == 0)[..., None]
    return np.where(m, _lin_color(p["color_a"]), _lin_color(p["color_b"])).astype(np.float32)


@node("lut_input", "generator",
      "Entry point for LUT baking. Normally a horizontal gray ramp (preview); when baking a LUT it is replaced by "
      "the identity color lattice. Values are display-encoded (sRGB) code values, NOT linearised.",
      outputs=IMG, params={"width": P("int", 960, 2, 16384), "height": P("int", 540, 2, 16384)})
def lut_input(inp, p, ctx):
    ramp = np.linspace(0, 1, p["width"], dtype=np.float32)[None, :].repeat(p["height"], 0)
    return ensure_rgba(ramp)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


@node("text", "generator", "Anti-aliased text on a transparent background (built-in open font).",
      outputs=IMG, params={**_WH, "text": P("str", "OMNI", animatable=False), "size": P("int", 120, 4, 2000),
                           "color": P("color", [1, 1, 1, 1]), "x": P("float", 0.5, -2, 3, doc="center, 0..1"),
                           "y": P("float", 0.5, -2, 3, doc="center, 0..1")})
def text(inp, p, ctx):
    w, h = p["width"], p["height"]
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    f = _font(p["size"])
    d.text((p["x"] * w, p["y"] * h), p["text"], fill=255, font=f, anchor="mm")
    a = np.asarray(mask, np.float32) / 255.0
    c = _lin_color(p["color"])
    img = np.empty((h, w, 4), np.float32)
    img[..., :3] = c[:3]
    img[..., 3] = a * c[3]
    return img


@node("shape_mask", "generator", "Rectangle or ellipse mask with feathered edge.", outputs={"mask": T.MASK},
      params={**_WH, "kind": P("choice", "ellipse", choices=("ellipse", "rect")),
              "cx": P("float", 0.5, -1, 2), "cy": P("float", 0.5, -1, 2),
              "w": P("float", 0.5, 0, 4, doc="fraction of width"), "h": P("float", 0.5, 0, 4, doc="fraction of height"),
              "feather": P("float", 0.05, 0, 2)})
def shape_mask(inp, p, ctx):
    h, w = p["height"], p["width"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx / max(w - 1, 1) - p["cx"]) / max(p["w"] / 2, 1e-6)
    y = (yy / max(h - 1, 1) - p["cy"]) / max(p["h"] / 2, 1e-6)
    d = np.sqrt(x * x + y * y) if p["kind"] == "ellipse" else np.maximum(np.abs(x), np.abs(y))
    f = max(p["feather"], 1e-4)
    return np.clip((1.0 - d) / f, 0, 1).astype(np.float32)


@node("value", "generator", "A constant number (for scripts and the Copilot).", outputs={"value": T.VALUE},
      params={"value": P("float", 0.0, doc="the number")})
def value(inp, p, ctx):
    return p["value"]
