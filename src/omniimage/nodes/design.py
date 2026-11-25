"""Vector/UI design nodes (Figma-style): resolution-independent shapes with fills, gradients and strokes,
drop shadows, and auto-layout stacks. Colours are given like hex codes (sRGB) and stored linear."""
from __future__ import annotations
import cv2
import numpy as np

from ..core.registry import node
from ..core.params import P, ParamError
from ..core.ports import PortType as T
from .generators import _lin_color

_CANVAS = {"width": P("int", 960, 1, 16384), "height": P("int", 540, 1, 16384)}


def _poly_sdf(px, py, verts):
    """Signed distance (negative inside) from points to a closed polygon."""
    n = len(verts)
    d2 = np.full(px.shape, np.inf, np.float32)
    inside = np.zeros(px.shape, bool)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        t = np.clip((wx * ex + wy * ey) / max(ex * ex + ey * ey, 1e-9), 0, 1)
        dx, dy = wx - ex * t, wy - ey * t
        d2 = np.minimum(d2, dx * dx + dy * dy)
        cond = (ay > py) != (by > py)
        xint = ax + (py - ay) / np.where(by == ay, 1e-9, by - ay) * ex
        inside ^= cond & (px < xint)
    d = np.sqrt(d2)
    return np.where(inside, -d, d)


def _over(dst, src, x, y):
    """Straight-alpha 'over' of src onto dst at integer offset (clipped)."""
    h, w = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + sw, w), min(y + sh, h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    sa, da = s[..., 3:4], d[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-9)
    dst[y0:y1, x0:x1, :3] = rgb
    dst[y0:y1, x0:x1, 3:4] = oa


@node("vector_shape", "design", "Anti-aliased vector shape on a transparent canvas: rect, rounded_rect, ellipse, polygon, "
      "star or line (capsule). Fill can be solid or a two-colour linear gradient; stroke can sit inside, centred or "
      "outside the edge. Positions and sizes are in pixels; (x, y) is the shape's centre.", outputs={"image": T.IMAGE},
      params={**_CANVAS, "kind": P("choice", "rounded_rect", choices=("rect", "rounded_rect", "ellipse", "polygon", "star", "line")),
              "x": P("float", 480, -1e5, 1e5), "y": P("float", 270, -1e5, 1e5),
              "w": P("float", 300, 0, 1e5), "h": P("float", 200, 0, 1e5), "radius": P("float", 24, 0, 1e5),
              "rotation": P("float", 0, -360, 360), "sides": P("int", 5, 3, 32, doc="polygon sides / star points"),
              "inner": P("float", 0.45, 0.05, 0.95, doc="star inner radius ratio"),
              "fill": P("color", "#4f8cff"), "gradient": P("bool", False), "fill2": P("color", "#9b5cff"),
              "gradient_angle": P("float", 90, -360, 360), "stroke": P("color", "#ffffff"),
              "stroke_width": P("float", 0, 0, 1000), "stroke_align": P("choice", "inside", choices=("inside", "center", "outside")),
              "opacity": P("float", 1, 0, 1)})
def vector_shape(inp, p, ctx):
    W, H = p["width"], p["height"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = xx + 0.5 - p["x"], yy + 0.5 - p["y"]
    a = np.deg2rad(p["rotation"])
    u, v = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    bx, by = p["w"] / 2, p["h"] / 2
    k = p["kind"]
    if k in ("rect", "rounded_rect"):
        r = min(p["radius"], bx, by) if k == "rounded_rect" else 0.0
        qx, qy = np.abs(u) - (bx - r), np.abs(v) - (by - r)
        sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    elif k == "ellipse":
        bx, by = max(bx, 1e-3), max(by, 1e-3)
        k0, k1 = np.hypot(u / bx, v / by), np.hypot(u / bx ** 2, v / by ** 2)
        sdf = np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -min(bx, by))
    elif k == "line":
        r = by
        ax = max(bx - r, 0)
        sdf = np.hypot(u - np.clip(u, -ax, ax), v) - r
    else:
        n, R = p["sides"], min(bx, by)
        if k == "polygon":
            verts = [(R * np.cos(-np.pi / 2 + 2 * np.pi * i / n), R * np.sin(-np.pi / 2 + 2 * np.pi * i / n)) for i in range(n)]
        else:
            verts = [((R if i % 2 == 0 else R * p["inner"]) * np.cos(-np.pi / 2 + np.pi * i / n),
                      (R if i % 2 == 0 else R * p["inner"]) * np.sin(-np.pi / 2 + np.pi * i / n)) for i in range(2 * n)]
        sdf = _poly_sdf(u, v, verts)
    cov = np.clip(0.5 - sdf, 0, 1)
    f1, f2 = _lin_color(p["fill"]), _lin_color(p["fill2"])
    if p["gradient"]:
        g = np.deg2rad(p["gradient_angle"])
        ext = max(abs(p["w"] * np.cos(g - a)) + abs(p["h"] * np.sin(g - a)), 1e-6)
        t = np.clip(0.5 + (x * np.cos(g) + y * np.sin(g)) / ext, 0, 1)[..., None]
        fc = f1[:3] * (1 - t) + f2[:3] * t
        fa = (f1[3] * (1 - t[..., 0]) + f2[3] * t[..., 0])
    else:
        fc, fa = np.broadcast_to(f1[:3], (H, W, 3)), np.full((H, W), f1[3], np.float32)
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3], out[..., 3] = fc, fa * cov
    if p["stroke_width"] > 0:
        sw = p["stroke_width"]
        c = {"inside": -sw / 2, "center": 0.0, "outside": sw / 2}[p["stroke_align"]]
        scov = np.clip(0.5 - (np.abs(sdf - c) - sw / 2), 0, 1)
        sc = _lin_color(p["stroke"])
        layer = np.empty((H, W, 4), np.float32)
        layer[..., :3], layer[..., 3] = sc[:3], scov * sc[3]
        _over(out, layer, 0, 0)
    out[..., 3] *= p["opacity"]
    return out


@node("drop_shadow", "design", "Add a soft shadow under an image using its alpha (Figma/CSS drop shadow).",
      inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE},
      params={"dx": P("float", 0, -2000, 2000), "dy": P("float", 8, -2000, 2000), "blur": P("float", 12, 0, 500),
              "spread": P("float", 0, -200, 200, doc="grow (+) or shrink (-) the shadow, px"),
              "color": P("color", "#000000"), "opacity": P("float", 0.35, 0, 1)})
def drop_shadow(inp, p, ctx):
    im = inp["image"]
    H, W = im.shape[:2]
    a = im[..., 3].copy()
    if p["spread"]:
        k = int(abs(p["spread"])) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        a = cv2.dilate(a, ker) if p["spread"] > 0 else cv2.erode(a, ker)
    a = cv2.warpAffine(a, np.float32([[1, 0, p["dx"]], [0, 1, p["dy"]]]), (W, H), borderValue=0)
    if p["blur"] > 0:
        a = cv2.GaussianBlur(a, (0, 0), p["blur"] / 2)
    c = _lin_color(p["color"])
    sh = np.empty_like(im)
    sh[..., :3], sh[..., 3] = c[:3], np.clip(a, 0, 1) * p["opacity"] * c[3]
    out = sh.copy()
    _over(out, im, 0, 0)
    return out


@node("auto_layout", "design", "Figma-style auto layout: stack up to four images in a row or column with gap, padding "
      "and alignment on a canvas that hugs the content (or a fixed size).",
      inputs={"a": T.IMAGE, "b": T.IMAGE, "c": T.IMAGE, "d": T.IMAGE}, outputs={"image": T.IMAGE},
      optional=("b", "c", "d"),
      params={"direction": P("choice", "row", choices=("row", "column")), "gap": P("int", 16, 0, 2000),
              "padding": P("int", 16, 0, 2000), "align": P("choice", "center", choices=("start", "center", "end")),
              "distribute": P("choice", "start", choices=("start", "center", "end", "space_between")),
              "fixed_width": P("int", 0, 0, 16384, doc="0 = hug content"), "fixed_height": P("int", 0, 0, 16384),
              "background": P("color", [0, 0, 0, 0])})
def auto_layout(inp, p, ctx):
    items = [inp[k] for k in ("a", "b", "c", "d") if k in inp]
    row = p["direction"] == "row"
    gap, pad = p["gap"], p["padding"]
    main = [im.shape[1] if row else im.shape[0] for im in items]
    cross = [im.shape[0] if row else im.shape[1] for im in items]
    content_main = sum(main) + gap * (len(items) - 1)
    W = p["fixed_width"] or ((content_main if row else max(cross)) + 2 * pad)
    H = p["fixed_height"] or ((max(cross) if row else content_main) + 2 * pad)
    inner_main, inner_cross = (W if row else H) - 2 * pad, (H if row else W) - 2 * pad
    free = inner_main - content_main
    spacing = gap
    start = pad
    d = p["distribute"]
    if d == "center":
        start += free // 2
    elif d == "end":
        start += free
    elif d == "space_between" and len(items) > 1:
        spacing = gap + free / (len(items) - 1)
    bg = np.empty((H, W, 4), np.float32)
    c = _lin_color(p["background"])
    bg[:] = c
    pos = float(start)
    for im, m, cr in zip(items, main, cross):
        off = {"start": 0, "center": (inner_cross - cr) // 2, "end": inner_cross - cr}[p["align"]] + pad
        x, y = (int(round(pos)), off) if row else (off, int(round(pos)))
        _over(bg, im, x, y)
        pos += m + spacing
    return bg
