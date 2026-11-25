"""Spatial and time-dependent image effects (blur, glow, grain, lens, glitch ...).

These depend on neighbouring pixels, position or frame number, so they are NOT LUT-bakeable."""
from __future__ import annotations
import numpy as np
import cv2

from ..core.registry import node
from ..core.params import P
from ..core.ports import PortType as T
from ..core.media import luminance, with_rgb, ensure_rgba

IO = dict(inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE})


def _blur(a, sigma):
    if sigma <= 0.01:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REPLICATE)


@node("blur_gaussian", "filter", "Gaussian blur. Radius is the standard deviation in pixels.",
      params={"radius": P("float", 4.0, 0, 300)}, **IO)
def blur_gaussian(inp, p, ctx):
    return _blur(inp["image"], p["radius"]).astype(np.float32)


@node("blur_directional", "filter", "Motion blur along a direction.",
      params={"length": P("float", 20.0, 1, 500), "angle": P("float", 0.0, -360, 360)}, **IO)
def blur_directional(inp, p, ctx):
    n = int(max(1, round(p["length"])))
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), p["angle"], 1.0)
    k = cv2.warpAffine(k, m, (n, n))
    s = k.sum()
    k = k / s if s > 0 else k
    return cv2.filter2D(inp["image"], -1, k, borderType=cv2.BORDER_REPLICATE)


@node("sharpen", "filter", "Unsharp mask.", params={"amount": P("float", 1.0, 0, 10), "radius": P("float", 1.5, 0.1, 50)}, **IO)
def sharpen(inp, p, ctx):
    im = inp["image"]
    blur = _blur(im, p["radius"])
    out = im + (im - blur) * p["amount"]
    out[..., 3] = im[..., 3]
    return out.astype(np.float32)


@node("median", "filter", "Median filter (size 3 or 5) for impulse noise.", params={"size": P("int", 3, 3, 5)}, **IO)
def median(inp, p, ctx):
    k = 5 if p["size"] >= 5 else 3
    return cv2.medianBlur(inp["image"], k)


@node("bilateral", "filter", "Edge-preserving smoothing.",
      params={"sigma_color": P("float", 0.1, 0.001, 5), "sigma_space": P("float", 5.0, 0.1, 100)}, **IO)
def bilateral(inp, p, ctx):
    im = inp["image"]
    rgb = cv2.bilateralFilter(np.ascontiguousarray(im[..., :3]), -1, p["sigma_color"], p["sigma_space"])
    return with_rgb(im, rgb)


@node("glow", "filter", "Bloom: blur the bright parts and add them back.",
      params={"threshold": P("float", 0.8, 0, 10), "radius": P("float", 20.0, 0, 300), "intensity": P("float", 1.0, 0, 10)},
      **IO)
def glow(inp, p, ctx):
    im = inp["image"]
    a = im[..., 3:4]
    pm = im[..., :3] * a                                   # premultiplied, so halos on transparent areas work
    bright = np.maximum(pm - p["threshold"], 0)
    g = _blur(bright, p["radius"]) + 0.5 * _blur(bright, p["radius"] * 2.5) + 0.25 * _blur(bright, p["radius"] * 6)
    pm_out = pm + g * p["intensity"]
    a_out = np.maximum(a, np.clip(g.max(-1, keepdims=True) * p["intensity"], 0, 1))
    out = np.concatenate([pm_out / np.maximum(a_out, 1e-6), a_out], -1)
    return out.astype(np.float32)


@node("vignette", "filter", "Darken (or lighten) the corners.",
      params={"amount": P("float", 0.5, -1, 1), "softness": P("float", 0.5, 0.01, 2), "roundness": P("float", 1.0, 0, 1)}, **IO)
def vignette(inp, p, ctx):
    im = inp["image"]
    h, w = im.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx / max(w - 1, 1) - 0.5) * 2
    y = (yy / max(h - 1, 1) - 0.5) * 2
    x = x * (1 + (w / h - 1) * p["roundness"]) if w > h else x
    d = np.sqrt(x * x + y * y) / 1.414
    k = np.clip((d - (1 - p["softness"])) / max(p["softness"], 1e-3), 0, 1)
    k = (k * k * (3 - 2 * k))[..., None]
    return with_rgb(im, im[..., :3] * (1 - p["amount"] * k))


@node("film_grain", "filter", "Animated film grain, stronger in midtones. Different each frame.",
      params={"amount": P("float", 0.08, 0, 1), "size": P("float", 1.2, 0.3, 10), "seed": P("int", 7, 0, 2 ** 31 - 1)},
      uses_time=True, **IO)
def film_grain(inp, p, ctx):
    im = inp["image"]
    h, w = im.shape[:2]
    rng = np.random.default_rng([p["seed"], int(ctx.frame)])
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = _blur(n, p["size"] - 0.3) * (1.0 + p["size"])
    y = np.clip(luminance(im), 0, 1)
    resp = (4 * y * (1 - y) + 0.15)[..., None]
    return with_rgb(im, im[..., :3] + n[..., None] * p["amount"] * resp)


def _scale_about_center(ch, s):
    h, w = ch.shape[:2]
    m = np.array([[s, 0, (1 - s) * (w - 1) / 2], [0, s, (1 - s) * (h - 1) / 2]], np.float32)
    return cv2.warpAffine(ch, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@node("chromatic_aberration", "filter", "Lens-style color fringing: red and blue scale differently.",
      params={"amount": P("float", 0.5, -10, 10, doc="percent scale difference")}, **IO)
def chromatic_aberration(inp, p, ctx):
    im = inp["image"].copy()
    s = p["amount"] * 0.01
    im[..., 0] = _scale_about_center(im[..., 0], 1 + s)
    im[..., 2] = _scale_about_center(im[..., 2], 1 - s)
    return im


@node("rgb_split", "filter", "Shift red and blue channels in opposite directions (glitch/anaglyph look).",
      params={"dx": P("float", 6.0, -500, 500), "dy": P("float", 0.0, -500, 500)}, **IO)
def rgb_split(inp, p, ctx):
    im = inp["image"].copy()
    h, w = im.shape[:2]
    for ch, sgn in ((0, 1), (2, -1)):
        m = np.float32([[1, 0, sgn * p["dx"]], [0, 1, sgn * p["dy"]]])
        im[..., ch] = cv2.warpAffine(im[..., ch], m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return im


@node("pixelate", "filter", "Mosaic / pixel look.", params={"size": P("int", 8, 1, 512)}, **IO)
def pixelate(inp, p, ctx):
    im = inp["image"]
    h, w = im.shape[:2]
    s = p["size"]
    small = cv2.resize(im, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


@node("edge_detect", "filter", "Sobel edge magnitude as a gray image.", params={"strength": P("float", 1.0, 0, 20)}, **IO)
def edge_detect(inp, p, ctx):
    im = inp["image"]
    y = luminance(im)
    gx, gy = cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)
    m = np.sqrt(gx * gx + gy * gy) * p["strength"]
    out = ensure_rgba(m)
    out[..., 3] = im[..., 3]
    return out


@node("normal_from_height", "filter", "Turn a height map (bright = high) into a tangent-space normal map for mesh_texture's "
      "normal port. The output is DATA (n * 0.5 + 0.5), so keep it away from colour nodes. wrap=true makes tiling textures "
      "seamless; flip_y gives DirectX-style maps.",
      params={"strength": P("float", 4.0, 0, 200, doc="bump height relative to the image width"), "wrap": P("bool", True),
              "flip_y": P("bool", False)}, **IO)
def normal_from_height(inp, p, ctx):
    im = inp["image"]
    h = im[..., :3].mean(-1).astype(np.float32)
    if p["wrap"]:
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5            # d h / d row (rows run downward)
    else:
        gy, gx = np.gradient(h)
    k = float(p["strength"]) * h.shape[1] / 16.0                     # strength is independent of resolution
    nx, ny, nz = -gx * k, gy * k, np.ones_like(h)                    # image-up convention: n_y = d h / d row
    if p["flip_y"]:
        ny = -ny
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.empty(h.shape + (4,), np.float32)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = nx / ln * 0.5 + 0.5, ny / ln * 0.5 + 0.5, nz / ln * 0.5 + 0.5, 1.0
    return out


@node("scanlines", "filter", "CRT/VHS scanlines.",
      params={"strength": P("float", 0.35, 0, 1), "period": P("float", 3.0, 1, 50)}, **IO)
def scanlines(inp, p, ctx):
    im = inp["image"]
    h = im.shape[0]
    y = np.arange(h, dtype=np.float32)
    line = 1.0 - p["strength"] * (0.5 + 0.5 * np.cos(2 * np.pi * y / p["period"]))
    return with_rgb(im, im[..., :3] * line[:, None, None])


@node("glitch", "filter", "Digital glitch: horizontal slice displacement + channel split, re-rolled every few frames.",
      params={"amount": P("float", 0.5, 0, 1), "slices": P("int", 12, 1, 200), "seed": P("int", 3, 0, 2 ** 31 - 1),
              "hold": P("int", 3, 1, 120, doc="frames each glitch pattern lasts")}, uses_time=True, **IO)
def glitch(inp, p, ctx):
    im = inp["image"].copy()
    h, w = im.shape[:2]
    rng = np.random.default_rng([p["seed"], int(ctx.frame) // p["hold"]])
    out = im.copy()
    n = p["slices"]
    ys = np.sort(rng.integers(0, h, n * 2).reshape(-1, 2), axis=1)
    for y0, y1 in ys:
        if y1 - y0 < 1 or rng.random() > p["amount"] + 0.2:
            continue
        sh = int(rng.normal(0, w * 0.06 * p["amount"]))
        out[y0:y1] = np.roll(im[y0:y1], sh, axis=1)
    dx = rng.integers(2, 2 + int(12 * p["amount"]) + 1)
    out[..., 0] = np.roll(out[..., 0], dx, axis=1)
    out[..., 2] = np.roll(out[..., 2], -dx, axis=1)
    return out


@node("lens_distortion", "filter", "Barrel (positive) or pincushion (negative) distortion.",
      params={"k1": P("float", 0.1, -1, 1), "k2": P("float", 0.0, -1, 1)}, **IO)
def lens_distortion(inp, p, ctx):
    im = inp["image"]
    h, w = im.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - w / 2) / (w / 2)
    y = (yy - h / 2) / (h / 2)
    r2 = x * x + y * y
    f = 1 + p["k1"] * r2 + p["k2"] * r2 * r2
    mx, my = x * f * (w / 2) + w / 2, y * f * (h / 2) + h / 2
    return cv2.remap(im, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


@node("transform", "filter", "Move, rotate and scale about the image center (alpha 0 outside the frame).",
      params={"tx": P("float", 0.0, -20000, 20000, doc="pixels"), "ty": P("float", 0.0, -20000, 20000, doc="pixels"),
              "rotate": P("float", 0.0, -3600, 3600, doc="degrees"), "scale": P("float", 1.0, 0.001, 100)}, **IO)
def transform(inp, p, ctx):
    im = inp["image"]
    h, w = im.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -p["rotate"], p["scale"])
    m[0, 2] += p["tx"]
    m[1, 2] += p["ty"]
    return cv2.warpAffine(im, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


@node("crop", "filter", "Crop to a pixel rectangle.",
      params={"x": P("int", 0, 0, 65535), "y": P("int", 0, 0, 65535), "width": P("int", 256, 1, 65535),
              "height": P("int", 256, 1, 65535)}, **IO)
def crop(inp, p, ctx):
    im = inp["image"]
    h, w = im.shape[:2]
    x0, y0 = min(p["x"], w - 1), min(p["y"], h - 1)
    return np.ascontiguousarray(im[y0:y0 + p["height"], x0:x0 + p["width"]])


@node("resize", "filter", "Resize to an exact size.",
      params={"width": P("int", 960, 1, 16384), "height": P("int", 540, 1, 16384),
              "filter": P("choice", "cubic", choices=("nearest", "linear", "cubic", "area", "lanczos"))}, **IO)
def resize(inp, p, ctx):
    m = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
         "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}[p["filter"]]
    return cv2.resize(inp["image"], (p["width"], p["height"]), interpolation=m)


@node("flip", "filter", "Mirror horizontally and/or vertically.",
      params={"horizontal": P("bool", True), "vertical": P("bool", False)}, **IO)
def flip(inp, p, ctx):
    im = inp["image"]
    if p["horizontal"]:
        im = im[:, ::-1]
    if p["vertical"]:
        im = im[::-1]
    return np.ascontiguousarray(im)


@node("displace", "filter", "Warp the image using a map: red moves x, green moves y (0.5 = no movement).",
      inputs={"image": T.IMAGE, "map": T.IMAGE}, outputs={"image": T.IMAGE},
      params={"amount": P("float", 20.0, -2000, 2000, doc="max pixels")})
def displace(inp, p, ctx):
    im, mp = inp["image"], inp["map"]
    h, w = im.shape[:2]
    if mp.shape[:2] != (h, w):
        mp = cv2.resize(mp, (w, h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = xx + (mp[..., 0] - 0.5) * 2 * p["amount"]
    my = yy + (mp[..., 1] - 0.5) * 2 * p["amount"]
    return cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
