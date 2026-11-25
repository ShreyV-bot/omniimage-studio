"""Video scopes as nodes: waveform, RGB parade, vectorscope, histogram. Each turns an image into a scope image
(display-referred analysis of the sRGB-encoded picture) you can view, save, or composite."""
from __future__ import annotations
import numpy as np

from ..core.registry import node
from ..core.params import P
from ..core.ports import PortType as T
from ..core.media import linear_to_srgb, srgb_to_linear

_WH = {"width": P("int", 512, 16, 4096), "height": P("int", 256, 16, 4096), "gain": P("float", 1.0, 0.05, 20)}
IO = dict(inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE})
BG = 0.06


def _display(im):
    return np.clip(linear_to_srgb(np.clip(im[..., :3], 0, None)), 0, 1)


def _finish(rgb_disp):
    out = np.ones(rgb_disp.shape[:2] + (4,), np.float32)
    out[..., :3] = srgb_to_linear(np.clip(rgb_disp, 0, 1))
    return out


def _grid(h, w):
    g = np.zeros((h, w), np.float32)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        g[int(round((1 - v) * (h - 1))), :] = 1
    return g


def _wave_counts(values, ww, wh, img_w, gain):
    """values: (H, W) in 0..1 -> (wh, ww) intensity."""
    H, W = values.shape
    xi = np.broadcast_to((np.arange(W) * ww // W)[None, :], (H, W))
    yi = ((1 - np.clip(values, 0, 1)) * (wh - 1)).astype(np.int64)
    c = np.bincount((yi * ww + xi).ravel(), minlength=wh * ww).reshape(wh, ww).astype(np.float32)
    per_col = H * max(W / ww, 1.0)
    return 1 - np.exp(-c / per_col * gain * 40)


@node("scope_waveform", "scope", "Luma waveform: horizontal = image column, vertical = brightness (0 bottom, 1 top).",
      params=_WH, **IO)
def scope_waveform(inp, p, ctx):
    d = _display(inp["image"])
    luma = d @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    w = _wave_counts(luma, p["width"], p["height"], d.shape[1], p["gain"])
    out = np.full((p["height"], p["width"], 3), BG, np.float32) + _grid(p["height"], p["width"])[..., None] * 0.12
    out = out + w[..., None] * np.array([0.3, 1.0, 0.4], np.float32)
    return _finish(out)


@node("scope_parade", "scope", "RGB parade: three side-by-side waveforms for the red, green and blue channels.",
      params=_WH, **IO)
def scope_parade(inp, p, ctx):
    d = _display(inp["image"])
    W, H = p["width"], p["height"]
    pw = W // 3
    out = np.full((H, pw * 3, 3), BG, np.float32) + np.tile(_grid(H, pw * 3)[..., None], (1, 1, 3)) * 0.12
    for c in range(3):
        w = _wave_counts(d[..., c], pw, H, d.shape[1], p["gain"])
        tint = np.full(3, 0.15, np.float32); tint[c] = 1.0
        out[:, c * pw:(c + 1) * pw] += w[..., None] * tint
    return _finish(out)


@node("scope_vectorscope", "scope", "Vectorscope: plots chroma (Cb/Cr, Rec.709) of every pixel. Distance from the centre "
      "is saturation, angle is hue. Square output of 'size' pixels.",
      params={"size": P("int", 384, 32, 2048), "gain": P("float", 1.0, 0.05, 20)}, **IO)
def scope_vectorscope(inp, p, ctx):
    d = _display(inp["image"])
    S = p["size"]
    y = d @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    cb, cr = (d[..., 2] - y) / 1.8556, (d[..., 0] - y) / 1.5748
    xi = np.clip(((cb + 0.5) * (S - 1)).astype(np.int64), 0, S - 1)
    yi = np.clip(((0.5 - cr) * (S - 1)).astype(np.int64), 0, S - 1)
    c = np.bincount((yi * S + xi).ravel(), minlength=S * S).reshape(S, S).astype(np.float32)
    inten = 1 - np.exp(-c / max(d.shape[0] * d.shape[1] / (S * S), 1.0) * p["gain"] * 3)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.hypot(xx - (S - 1) / 2, yy - (S - 1) / 2) / (S / 2)
    ring = (np.abs(r - 0.75) < 0.5 / S * 2).astype(np.float32) + (np.abs(r - 0.37) < 0.5 / S * 2)
    cross = ((np.abs(xx - (S - 1) / 2) < 0.6) | (np.abs(yy - (S - 1) / 2) < 0.6)).astype(np.float32)
    out = np.full((S, S, 3), BG, np.float32) + (np.clip(ring, 0, 1) + cross * 0.6)[..., None] * 0.12
    out = out + inten[..., None] * np.array([0.4, 1.0, 0.5], np.float32)
    return _finish(out)


@node("scope_histogram", "scope", "RGB histogram of the picture (display-referred), additive colours.",
      params={"width": P("int", 512, 16, 4096), "height": P("int", 256, 16, 4096), "log": P("bool", False)}, **IO)
def scope_histogram(inp, p, ctx):
    d = _display(inp["image"])
    W, H = p["width"], p["height"]
    out = np.full((H, W, 3), BG, np.float32)
    rows = np.arange(H)[:, None]
    for c in range(3):
        hist = np.bincount(np.clip((d[..., c] * (W - 1)).astype(np.int64), 0, W - 1).ravel(), minlength=W).astype(np.float32)
        if p["log"]:
            hist = np.log1p(hist)
        hist = hist / max(hist.max(), 1e-9)
        bar = rows >= (H - 1 - hist[None, :] * (H - 1))
        tint = np.zeros(3, np.float32); tint[c] = 0.7
        out += bar[..., None].astype(np.float32) * tint
    return _finish(out)
