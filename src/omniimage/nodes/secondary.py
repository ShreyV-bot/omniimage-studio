"""Secondary grading: HSL qualifier (pick colours) and power windows (shapes). Both output MASKs; feed them to
``merge`` (mask input) or ``set_alpha`` to grade only part of the picture."""
from __future__ import annotations
import cv2
import numpy as np

from ..core.registry import node
from ..core.params import P
from ..core.ports import PortType as T
from ..core.media import linear_to_srgb

_WH = {"width": P("int", 960, 1, 16384), "height": P("int", 540, 1, 16384)}


def _ramp(x, lo, hi, soft):
    soft = max(soft, 1e-4)
    return np.clip((x - (lo - soft)) / soft, 0, 1) * np.clip(((hi + soft) - x) / soft, 0, 1)


def hsv_of(rgb_disp):
    r, g, b = rgb_disp[..., 0], rgb_disp[..., 1], rgb_disp[..., 2]
    mx, mn = rgb_disp.max(-1), rgb_disp.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rc = nz & (mx == r); gc = nz & (mx == g) & ~rc; bc = nz & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6
    h[gc] = (b - r)[gc] / d[gc] + 2
    h[bc] = (r - g)[bc] / d[bc] + 4
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h * 60.0, s, mx


@node("hsl_qualifier", "secondary", "Select pixels by hue / saturation / brightness with soft edges (like Resolve's "
      "qualifier). Output is a mask. Hue is in degrees (0=red, 120=green, 240=blue).",
      inputs={"image": T.IMAGE}, outputs={"mask": T.MASK},
      params={"hue": P("float", 0, 0, 360), "hue_width": P("float", 40, 0, 360), "hue_soft": P("float", 15, 0.01, 180),
              "sat_min": P("float", 0.15, 0, 1), "sat_max": P("float", 1, 0, 1), "sat_soft": P("float", 0.1, 0.001, 1),
              "lum_min": P("float", 0, 0, 1), "lum_max": P("float", 1, 0, 1), "lum_soft": P("float", 0.1, 0.001, 1),
              "blur": P("float", 0, 0, 100, doc="smooth the matte (pixels)"), "invert": P("bool", False)})
def hsl_qualifier(inp, p, ctx):
    d = np.clip(linear_to_srgb(np.clip(inp["image"][..., :3], 0, None)), 0, 1)
    h, s, v = hsv_of(d)
    dist = np.abs(((h - p["hue"] + 180.0) % 360.0) - 180.0)
    m = _ramp(dist, -1e9, p["hue_width"] / 2, p["hue_soft"]) * _ramp(s, p["sat_min"], p["sat_max"], p["sat_soft"])
    m = m * _ramp(v, p["lum_min"], p["lum_max"], p["lum_soft"])
    if p["blur"] > 0:
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), p["blur"])
    m = np.clip(m, 0, 1)
    return (1 - m if p["invert"] else m).astype(np.float32)


@node("power_window", "secondary", "Shape mask (ellipse / rectangle / rounded rectangle) with rotation and soft edge, "
      "like Resolve's power windows. Sizes are fractions of the image width/height.", outputs={"mask": T.MASK},
      params={**_WH, "kind": P("choice", "ellipse", choices=("ellipse", "rect", "rounded_rect")),
              "cx": P("float", 0.5, -1, 2), "cy": P("float", 0.5, -1, 2), "w": P("float", 0.5, 0, 4),
              "h": P("float", 0.5, 0, 4), "rotation": P("float", 0, -360, 360), "radius": P("float", 0.1, 0, 1),
              "softness": P("float", 0.05, 0, 1, doc="edge falloff, fraction of the smaller side"),
              "invert": P("bool", False)})
def power_window(inp, p, ctx):
    W, H = p["width"], p["height"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = xx - p["cx"] * (W - 1), yy - p["cy"] * (H - 1)
    a = np.deg2rad(p["rotation"])
    xr, yr = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    bx, by = max(p["w"] * W / 2, 1e-3), max(p["h"] * H / 2, 1e-3)
    if p["kind"] == "ellipse":
        k0 = np.hypot(xr / bx, yr / by)
        k1 = np.hypot(xr / bx ** 2, yr / by ** 2)
        sdf = np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -min(bx, by))
    else:
        r = p["radius"] * min(bx, by) if p["kind"] == "rounded_rect" else 0.0
        qx, qy = np.abs(xr) - (bx - r), np.abs(yr) - (by - r)
        sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    soft = max(p["softness"] * min(W, H), 1.0)
    m = np.clip(0.5 - sdf / soft, 0, 1).astype(np.float32)
    return 1 - m if p["invert"] else m


@node("gradient_window", "secondary", "Linear gradient mask (graduated filter): 1 on one side, 0 on the other.",
      outputs={"mask": T.MASK},
      params={**_WH, "angle": P("float", 90, -360, 360, doc="direction the mask fades toward, degrees (90 = downward)"),
              "position": P("float", 0.5, -1, 2), "softness": P("float", 0.3, 0.001, 2), "invert": P("bool", False)})
def gradient_window(inp, p, ctx):
    W, H = p["width"], p["height"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.deg2rad(p["angle"])
    d = ((xx / max(W - 1, 1) - 0.5) * np.cos(a) + (yy / max(H - 1, 1) - 0.5) * np.sin(a)) + 0.5 - p["position"]
    m = np.clip(0.5 - d / p["softness"], 0, 1).astype(np.float32)
    return 1 - m if p["invert"] else m
