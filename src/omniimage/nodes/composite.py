"""Compositing, keying and mask nodes."""
from __future__ import annotations
import numpy as np
import cv2

from ..core.registry import node
from ..core.params import P, parse_color, ParamError
from ..core.ports import PortType as T
from ..core.media import luminance, with_rgb, srgb_to_linear

BLEND_MODES = ("over", "add", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference",
               "subtract", "divide", "darken", "lighten", "color_dodge", "color_burn", "exclusion", "linear_light")


def _blend(mode, cb, cs):
    """Separable blend functions on (..., 3) arrays (straight, non-premultiplied)."""
    if mode == "over":
        return cs
    if mode == "add":
        return cb + cs
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "overlay":
        return _blend("hard_light", cs, cb)
    if mode == "hard_light":
        return np.where(cs <= 0.5, cb * 2 * cs, cb + (2 * cs - 1) - cb * (2 * cs - 1))
    if mode == "soft_light":
        d = np.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(np.maximum(cb, 0)))
        return np.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "subtract":
        return np.maximum(cb - cs, 0)
    if mode == "divide":
        return cb / np.maximum(cs, 1e-4)
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "color_dodge":
        return np.where(cs >= 1, 1.0, np.minimum(1.0, cb / np.maximum(1 - cs, 1e-4)))
    if mode == "color_burn":
        return np.where(cs <= 0, 0.0, 1.0 - np.minimum(1.0, (1 - cb) / np.maximum(cs, 1e-4)))
    if mode == "exclusion":
        return cb + cs - 2 * cb * cs
    if mode == "linear_light":
        return cb + 2 * cs - 1
    raise ParamError(f"unknown blend mode {mode}")


def _fit(img, shape):
    if img.shape[:2] != shape[:2]:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return img


@node("merge", "composite",
      "Composite A (foreground) over B (background) with a blend mode, W3C-style alpha math. Optional mask limits "
      "where A applies.",
      inputs={"a": T.IMAGE, "b": T.IMAGE, "mask": T.MASK}, outputs={"image": T.IMAGE}, optional=("mask",),
      params={"mode": P("choice", "over", choices=BLEND_MODES), "mix": P("float", 1.0, 0, 1, doc="opacity of A")})
def merge(inp, p, ctx):
    b = inp["b"]
    a = _fit(inp["a"], b.shape)
    cs, cb = a[..., :3], b[..., :3]
    a_s = a[..., 3:4] * p["mix"]
    if "mask" in inp:
        a_s = a_s * _fit(inp["mask"], b.shape)[..., None]
    a_b = b[..., 3:4]
    blended = _blend(p["mode"], cb, cs)
    a_r = a_s + a_b * (1 - a_s)
    num = (1 - a_s) * a_b * cb + (1 - a_b) * a_s * cs + a_s * a_b * blended
    cr = num / np.maximum(a_r, 1e-6)
    return np.concatenate([cr, a_r], axis=-1).astype(np.float32)


@node("mix", "composite", "Cross-fade between two images.", inputs={"a": T.IMAGE, "b": T.IMAGE}, outputs={"image": T.IMAGE},
      params={"amount": P("float", 0.5, 0, 1, doc="0 = A, 1 = B")})
def mix(inp, p, ctx):
    a, b = inp["a"], _fit(inp["b"], inp["a"].shape)
    return (a * (1 - p["amount"]) + b * p["amount"]).astype(np.float32)


@node("set_alpha", "composite", "Use a mask as the image's alpha channel.", inputs={"image": T.IMAGE, "mask": T.MASK},
      outputs={"image": T.IMAGE})
def set_alpha(inp, p, ctx):
    im = inp["image"].copy()
    im[..., 3] = _fit(inp["mask"], im.shape)
    return im


@node("premultiply", "composite", "Multiply color by alpha.", inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE})
def premultiply(inp, p, ctx):
    im = inp["image"].copy()
    im[..., :3] *= im[..., 3:4]
    return im


@node("unpremultiply", "composite", "Divide color by alpha.", inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE})
def unpremultiply(inp, p, ctx):
    im = inp["image"].copy()
    im[..., :3] /= np.maximum(im[..., 3:4], 1e-6)
    return im


@node("image_to_mask", "mask", "Extract a mask from an image channel.", inputs={"image": T.IMAGE}, outputs={"mask": T.MASK},
      params={"channel": P("choice", "luma", choices=("luma", "alpha", "r", "g", "b"))})
def image_to_mask(inp, p, ctx):
    im = inp["image"]
    c = p["channel"]
    if c == "luma":
        return luminance(im)
    return np.ascontiguousarray(im[..., {"alpha": 3, "r": 0, "g": 1, "b": 2}[c]])


@node("mask_to_image", "mask", "Show a mask as a gray image.", inputs={"mask": T.MASK}, outputs={"image": T.IMAGE})
def mask_to_image(inp, p, ctx):
    m = inp["mask"]
    out = np.repeat(m[..., None], 3, -1)
    return np.concatenate([out, np.ones_like(m)[..., None]], -1).astype(np.float32)


@node("mask_blur", "mask", "Feather a mask.", inputs={"mask": T.MASK}, outputs={"mask": T.MASK},
      params={"radius": P("float", 5.0, 0, 300)})
def mask_blur(inp, p, ctx):
    return cv2.GaussianBlur(inp["mask"], (0, 0), max(p["radius"], 0.01)) if p["radius"] > 0.01 else inp["mask"]


@node("mask_invert", "mask", "Invert a mask.", inputs={"mask": T.MASK}, outputs={"mask": T.MASK})
def mask_invert(inp, p, ctx):
    return 1.0 - inp["mask"]


@node("mask_grow", "mask", "Grow (positive) or shrink (negative) a mask by pixels.", inputs={"mask": T.MASK},
      outputs={"mask": T.MASK}, params={"pixels": P("int", 3, -200, 200)})
def mask_grow(inp, p, ctx):
    n = abs(p["pixels"])
    if n == 0:
        return inp["mask"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    return (cv2.dilate if p["pixels"] > 0 else cv2.erode)(inp["mask"], k)


@node("mask_combine", "mask", "Combine two masks.", inputs={"a": T.MASK, "b": T.MASK}, outputs={"mask": T.MASK},
      params={"op": P("choice", "multiply", choices=("multiply", "add", "max", "min", "subtract"))})
def mask_combine(inp, p, ctx):
    a, b = inp["a"], _fit(inp["b"], inp["a"].shape)
    op = p["op"]
    r = {"multiply": a * b, "add": a + b, "max": np.maximum(a, b), "min": np.minimum(a, b), "subtract": a - b}[op]
    return np.clip(r, 0, 1).astype(np.float32)


@node("chroma_key", "keying",
      "Green/blue-screen keyer. Outputs the image with alpha set, plus the matte. The key color is sRGB hex.",
      inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE, "matte": T.MASK},
      params={"key_color": P("color", [0.0, 1.0, 0.0, 1.0]), "tolerance": P("float", 0.25, 0, 1),
              "softness": P("float", 0.15, 0, 1), "despill": P("float", 0.8, 0, 1, doc="spill suppression")})
def chroma_key(inp, p, ctx):
    im = inp["image"]
    key = srgb_to_linear(np.array(p["key_color"][:3], np.float32))
    rgb = im[..., :3]

    def chroma(c):  # normalised chromaticity (removes brightness)
        s = np.maximum(c.sum(-1, keepdims=True), 1e-5)
        return c / s

    d = np.linalg.norm(chroma(rgb) - chroma(key[None, None, :]), axis=-1)
    lo, hi = p["tolerance"] * 0.5, p["tolerance"] * 0.5 + p["softness"] * 0.5 + 1e-4
    matte = np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)
    out = im.copy()
    out[..., :3] = _despill(rgb, key, p["despill"])
    out[..., 3] = im[..., 3] * matte
    return {"image": out, "matte": matte}


def _despill(rgb, key, amount):
    """Limit the dominant key channel to the average of the other two."""
    if amount <= 0:
        return rgb
    k = int(np.argmax(key))
    others = [i for i in range(3) if i != k]
    limit = rgb[..., others].mean(-1)
    out = rgb.copy()
    out[..., k] = np.minimum(rgb[..., k], limit) * amount + rgb[..., k] * (1 - amount)
    return out


@node("despill", "keying", "Remove green/blue spill from edges.", inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE},
      params={"key_color": P("color", [0.0, 1.0, 0.0, 1.0]), "amount": P("float", 1.0, 0, 1)})
def despill(inp, p, ctx):
    im = inp["image"].copy()
    im[..., :3] = _despill(im[..., :3], np.array(p["key_color"][:3], np.float32), p["amount"])
    return im


@node("luma_key", "keying", "Key by brightness: pixels between low and high become opaque.",
      inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE, "matte": T.MASK},
      params={"low": P("float", 0.1, 0, 10), "high": P("float", 0.9, 0, 10), "invert": P("bool", False)})
def luma_key(inp, p, ctx):
    im = inp["image"]
    y = luminance(im)
    m = np.clip((y - p["low"]) / max(p["high"] - p["low"], 1e-5), 0, 1).astype(np.float32)
    if p["invert"]:
        m = 1 - m
    out = im.copy()
    out[..., 3] = im[..., 3] * m
    return {"image": out, "matte": m}
