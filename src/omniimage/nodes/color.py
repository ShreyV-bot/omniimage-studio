"""Color nodes. All are per-pixel (pointwise) so any chain of them can be baked into a 3D LUT.

Working space is scene-linear RGB unless stated. Use ``linear_to_srgb`` / ``srgb_to_linear`` to bracket
display-referred operations (curves, gradient maps, LUT application)."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator

from ..core.registry import node
from ..core.params import P, parse_color, ParamError
from ..core.ports import PortType as T
from ..core.media import srgb_to_linear, linear_to_srgb, luminance, with_rgb

IO = dict(inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE}, pointwise=True)


def _vec3(v, name, default):
    if v is None:
        return np.array(default, np.float32)
    v = list(v)
    if len(v) == 1:
        v = v * 3
    if len(v) != 3:
        raise ParamError(f"{name} needs 3 numbers (r, g, b)")
    return np.array([float(x) for x in v], np.float32)


@node("srgb_to_linear", "color", "Decode sRGB code values to scene-linear light.", **IO)
def n_s2l(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, srgb_to_linear(im[..., :3]))


@node("linear_to_srgb", "color", "Encode scene-linear light to sRGB code values.", **IO)
def n_l2s(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, linear_to_srgb(im[..., :3]))


@node("exposure", "color", "Brighten or darken in photographic stops (linear light).",
      params={"stops": P("float", 0.0, -12, 12)}, **IO)
def exposure(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, im[..., :3] * np.float32(2.0 ** p["stops"]))


@node("contrast", "color", "Contrast around a pivot (default 18% gray in linear light) with a soft highlight "
      "shoulder so bright areas roll off instead of clipping hard.",
      params={"amount": P("float", 1.0, 0, 5, doc="1 = unchanged"), "pivot": P("float", 0.18, 0.001, 1),
              "shoulder": P("bool", True, doc="soft-compress values above 0.8")}, **IO)
def contrast(inp, p, ctx):
    im = inp["image"]
    x = np.maximum(im[..., :3], 0) / p["pivot"]
    y = np.power(x, p["amount"]) * p["pivot"]
    if p["shoulder"]:
        y = np.where(y > 0.8, 0.8 + 0.2 * np.tanh((y - 0.8) / 0.2), y)
    return with_rgb(im, y)


@node("saturation", "color", "Scale color saturation around luma. 0 = grayscale.",
      params={"amount": P("float", 1.0, 0, 5)}, **IO)
def saturation(inp, p, ctx):
    im = inp["image"]
    y = luminance(im)[..., None]
    return with_rgb(im, y + (im[..., :3] - y) * p["amount"])


@node("vibrance", "color", "Saturation boost that protects already-saturated colors and skin.",
      params={"amount": P("float", 0.5, -1, 2)}, **IO)
def vibrance(inp, p, ctx):
    im = inp["image"]
    rgb = im[..., :3]
    mx, mn = rgb.max(-1, keepdims=True), rgb.min(-1, keepdims=True)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    k = 1.0 + p["amount"] * (1.0 - np.clip(sat, 0, 1))
    y = luminance(im)[..., None]
    return with_rgb(im, y + (rgb - y) * k)


@node("hue_shift", "color", "Rotate hues around the gray axis.", params={"degrees": P("float", 0.0, -360, 360)}, **IO)
def hue_shift(inp, p, ctx):
    a = np.deg2rad(p["degrees"])
    c, s = np.cos(a), np.sin(a)
    k = 1.0 / 3.0
    sq = np.sqrt(1.0 / 3.0)
    m = np.array([
        [c + (1 - c) * k, k * (1 - c) - sq * s, k * (1 - c) + sq * s],
        [k * (1 - c) + sq * s, c + k * (1 - c), k * (1 - c) - sq * s],
        [k * (1 - c) - sq * s, k * (1 - c) + sq * s, c + k * (1 - c)]], np.float32)
    im = inp["image"]
    return with_rgb(im, im[..., :3] @ m.T)


@node("temperature_tint", "color", "Simple white-balance style warm/cool and green/magenta shift.",
      params={"temperature": P("float", 0.0, -1, 1, doc="+ warm, - cool"),
              "tint": P("float", 0.0, -1, 1, doc="+ magenta, - green")}, **IO)
def temperature_tint(inp, p, ctx):
    im = inp["image"]
    gain = np.array([1 + 0.30 * p["temperature"], 1 - 0.15 * p["tint"], 1 - 0.30 * p["temperature"]], np.float32)
    return with_rgb(im, im[..., :3] * gain)


@node("lift_gamma_gain", "color", "Classic three-way grade. Lists are [r, g, b].",
      params={"lift": P("list", [0.0, 0.0, 0.0], doc="shadows offset"),
              "gamma": P("list", [1.0, 1.0, 1.0], doc="midtone power (>1 brightens)"),
              "gain": P("list", [1.0, 1.0, 1.0], doc="highlight multiplier")}, **IO)
def lgg(inp, p, ctx):
    lift, gam, gain = _vec3(p["lift"], "lift", 0), _vec3(p["gamma"], "gamma", 1), _vec3(p["gain"], "gain", 1)
    im = inp["image"]
    x = im[..., :3]
    x = gain * (x + lift * (1.0 - np.clip(x, 0, 1)))
    x = np.power(np.maximum(x, 0), 1.0 / np.maximum(gam, 1e-3))
    return with_rgb(im, x)


@node("cdl", "color", "ASC CDL: out = (in * slope + offset) ^ power, then saturation.",
      params={"slope": P("list", [1, 1, 1]), "offset": P("list", [0, 0, 0]), "power": P("list", [1, 1, 1]),
              "saturation": P("float", 1.0, 0, 4)}, **IO)
def cdl(inp, p, ctx):
    s, o, pw = _vec3(p["slope"], "slope", 1), _vec3(p["offset"], "offset", 0), _vec3(p["power"], "power", 1)
    im = inp["image"]
    x = np.maximum(im[..., :3] * s + o, 0) ** pw
    y = (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722)[..., None]
    return with_rgb(im, y + (x - y) * p["saturation"])


@node("color_balance", "color", "Tint shadows, midtones and highlights separately. Each list is [r, g, b] in -1..1.",
      params={"shadows": P("list", [0, 0, 0]), "midtones": P("list", [0, 0, 0]), "highlights": P("list", [0, 0, 0]),
              "strength": P("float", 0.5, 0, 1)}, **IO)
def color_balance(inp, p, ctx):
    im = inp["image"]
    y = np.clip(luminance(im), 0, 1)[..., None]
    sh = (1 - y) ** 2
    hi = y ** 2
    mid = 1 - sh - hi
    mid = np.clip(mid, 0, 1)
    tint = (_vec3(p["shadows"], "shadows", 0) * sh + _vec3(p["midtones"], "midtones", 0) * mid
            + _vec3(p["highlights"], "highlights", 0) * hi) * p["strength"]
    return with_rgb(im, np.maximum(im[..., :3] * (1.0 + tint), 0))   # multiplicative: keeps blacks black


def _curve_lut(points, n=2048):
    pts = sorted((float(a), float(b)) for a, b in points)
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    if len(pts) < 2 or any(b - a <= 0 for a, b in zip(xs, xs[1:])):
        raise ParamError("curve points must be >= 2 distinct x values, as [[x, y], ...] in 0..1")
    f = PchipInterpolator(xs, ys, extrapolate=True)
    grid = np.linspace(0, 1, n)
    return np.clip(f(grid), 0, 1).astype(np.float32)


@node("curves", "color", "Smooth tone curve through control points [[x, y], ...] (0..1). Operates on the values "
      "it receives: put it between srgb encode/decode for a display-referred curve.",
      params={"points": P("list", [[0, 0], [1, 1]]),
              "channel": P("choice", "master", choices=("master", "r", "g", "b", "luma"))}, **IO)
def curves(inp, p, ctx):
    lut = _curve_lut(p["points"])
    grid = np.linspace(0, 1, lut.size, dtype=np.float32)
    im = inp["image"]
    rgb = im[..., :3]

    def f(v):
        v = np.asarray(v, np.float32)
        return np.interp(np.clip(v, 0, 1), grid, lut) + np.maximum(v - 1, 0)

    ch = p["channel"]
    if ch == "master":
        out = f(rgb)
    elif ch == "luma":
        y = luminance(im)[..., None]
        out = rgb * (f(y) / np.maximum(y, 1e-5))
    else:
        out = rgb.copy()
        i = "rgb".index(ch)
        out[..., i] = f(rgb[..., i])
    return with_rgb(im, out)


@node("channel_mixer", "color", "3x3 color matrix [rr, rg, rb, gr, gg, gb, br, bg, bb]: each output channel is a "
      "weighted sum of input channels.",
      params={"matrix": P("list", [1, 0, 0, 0, 1, 0, 0, 0, 1])}, **IO)
def channel_mixer(inp, p, ctx):
    m = np.array(p["matrix"], np.float32)
    if m.size != 9:
        raise ParamError("matrix needs exactly 9 numbers")
    im = inp["image"]
    return with_rgb(im, im[..., :3] @ m.reshape(3, 3).T)


@node("gradient_map", "color", "Map luminance onto a color ramp. ``colors`` are hex strings, dark to bright "
      "(sRGB-encoded as written; output is linear).",
      params={"colors": P("list", ["#000000", "#ffffff"]), "amount": P("float", 1.0, 0, 1)}, **IO)
def gradient_map(inp, p, ctx):
    cols = [srgb_to_linear(np.array(parse_color(c)[:3], np.float32) if isinstance(c, str)
                           else np.array(c[:3], np.float32)) for c in p["colors"]]
    if len(cols) < 2:
        raise ParamError("gradient_map needs at least 2 colors")
    cols = np.stack(cols)
    im = inp["image"]
    y = np.clip(linear_to_srgb(luminance(im)), 0, 1)
    pos = np.linspace(0, 1, len(cols))
    out = np.stack([np.interp(y, pos, cols[:, c]) for c in range(3)], -1).astype(np.float32)
    return with_rgb(im, im[..., :3] * (1 - p["amount"]) + out * p["amount"])


@node("invert", "color", "Invert colors.", **IO)
def invert(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, 1.0 - im[..., :3])


@node("levels", "color", "Black/white point remap with gamma.",
      params={"in_black": P("float", 0.0, -1, 2), "in_white": P("float", 1.0, -1, 4), "gamma": P("float", 1.0, 0.05, 10),
              "out_black": P("float", 0.0, -1, 2), "out_white": P("float", 1.0, -1, 4)}, **IO)
def levels(inp, p, ctx):
    im = inp["image"]
    span = max(p["in_white"] - p["in_black"], 1e-6)
    x = np.clip((im[..., :3] - p["in_black"]) / span, 0, None) ** (1.0 / p["gamma"])
    return with_rgb(im, p["out_black"] + x * (p["out_white"] - p["out_black"]))


@node("threshold", "color", "Black/white by luminance.", params={"level": P("float", 0.5, 0, 1)}, **IO)
def threshold(inp, p, ctx):
    im = inp["image"]
    v = (luminance(im) >= p["level"]).astype(np.float32)[..., None]
    return with_rgb(im, np.repeat(v, 3, -1))


@node("posterize", "color", "Reduce each channel to a number of levels.", params={"levels": P("int", 4, 2, 256)}, **IO)
def posterize(inp, p, ctx):
    im = inp["image"]
    n = p["levels"] - 1
    return with_rgb(im, np.round(np.clip(im[..., :3], 0, 1) * n) / n)


@node("grayscale", "color", "Convert to gray using Rec.709 luma.", **IO)
def grayscale(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, np.repeat(luminance(im)[..., None], 3, -1))


@node("sepia", "color", "Classic sepia tone matrix.", params={"amount": P("float", 1.0, 0, 1)}, **IO)
def sepia(inp, p, ctx):
    im = inp["image"]
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    s = im[..., :3] @ m.T
    return with_rgb(im, im[..., :3] * (1 - p["amount"]) + s * p["amount"])


@node("clamp", "color", "Clamp values to a range (e.g. broadcast-safe 0..1).",
      params={"low": P("float", 0.0, -10, 10), "high": P("float", 1.0, -10, 100)}, **IO)
def clamp(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, np.clip(im[..., :3], p["low"], p["high"]))


@node("tone_map", "color", "Compress HDR (linear) to 0..1 with a selectable operator.",
      params={"operator": P("choice", "aces", choices=("reinhard", "aces", "filmic")),
              "exposure": P("float", 0.0, -8, 8, doc="stops applied before mapping")}, **IO)
def tone_map(inp, p, ctx):
    im = inp["image"]
    x = np.maximum(im[..., :3], 0) * np.float32(2.0 ** p["exposure"])
    op = p["operator"]
    if op == "reinhard":
        y = x / (1.0 + x)
    elif op == "aces":  # Narkowicz ACES fit
        y = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    else:  # Hable/Uncharted2 filmic
        def h(v):
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F
        y = h(x * 2.0) / h(np.float32(11.2))
    return with_rgb(im, np.clip(y, 0, 1))


@node("split_tone", "color", "Tint shadows and highlights with two colors (sRGB hex).",
      params={"shadow_color": P("color", [0.1, 0.3, 0.45, 1]), "highlight_color": P("color", [1.0, 0.75, 0.5, 1]),
              "balance": P("float", 0.0, -1, 1), "amount": P("float", 0.3, 0, 1)}, **IO)
def split_tone(inp, p, ctx):
    im = inp["image"]
    sc = srgb_to_linear(np.array(p["shadow_color"][:3], np.float32))
    hc = srgb_to_linear(np.array(p["highlight_color"][:3], np.float32))
    y = np.clip(linear_to_srgb(luminance(im)), 0, 1)[..., None]
    t = np.clip((y - 0.5 - p["balance"] * 0.5) * 2 + 0.5, 0, 1)
    tint = sc * (1 - t) + hc * t
    rgb = im[..., :3]
    # soft-light style: tint modulates, preserving luminance roughly
    out = rgb * (1 - p["amount"]) + rgb * (tint * 2.0) * p["amount"]
    return with_rgb(im, out)
