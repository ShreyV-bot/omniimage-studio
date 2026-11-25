"""Colour management: primaries, transfer curves and conversions between common working/camera/display spaces.

The engine's working space is **scene-linear Rec.709/sRGB primaries** (``linear_rec709``). Every space here is
described by RGB primaries + white point + a transfer curve; conversion goes decode -> XYZ -> Bradford adaptation ->
target primaries -> encode. This is a compact built-in alternative to OpenColorIO for the common cases (an OCIO
config loader is on the roadmap).
"""
from __future__ import annotations
import numpy as np

D65 = (0.3127, 0.3290)
PRIMARIES = {   # (R, G, B) xy chromaticities and white point
    "rec709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), D65),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), D65),
    "p3d65": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),
    "ap1": ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), (0.32168, 0.33767)),          # ACEScg
    "awg3": ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020), D65),                  # ARRI Wide Gamut 3
    "sgamut3": ((0.730, 0.280), (0.140, 0.855), (0.100, -0.050), D65),                     # Sony S-Gamut3
}
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _xy_to_XYZ(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz_matrix(name: str) -> np.ndarray:
    r, g, b, w = PRIMARIES[name]
    m = np.stack([_xy_to_XYZ(r), _xy_to_XYZ(g), _xy_to_XYZ(b)], 1)
    s = np.linalg.solve(m, _xy_to_XYZ(w))
    return m * s


def adapt(src_white, dst_white) -> np.ndarray:
    a, b = _BRADFORD @ _xy_to_XYZ(src_white), _BRADFORD @ _xy_to_XYZ(dst_white)
    return np.linalg.inv(_BRADFORD) @ np.diag(b / a) @ _BRADFORD


def primaries_matrix(src: str, dst: str) -> np.ndarray:
    """3x3 matrix taking linear ``src``-primaries RGB to linear ``dst``-primaries RGB."""
    m = np.linalg.inv(rgb_to_xyz_matrix(dst)) @ adapt(PRIMARIES[src][3], PRIMARIES[dst][3]) @ rgb_to_xyz_matrix(src)
    return m.astype(np.float32)


# ---- transfer curves: each maps (encode, decode) over float arrays ----
def _pow(x, g):
    return np.sign(x) * np.abs(x) ** g


def _srgb_dec(v):
    return np.where(v <= 0.04045, v / 12.92, _pow((np.abs(v) + 0.055) / 1.055, 2.4) * np.sign(v))


def _srgb_enc(l):
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * _pow(np.maximum(l, 0), 1 / 2.4) - 0.055)


def _r709_enc(l):
    return np.where(l < 0.018, 4.5 * l, 1.099 * _pow(np.maximum(l, 0), 0.45) - 0.099)


def _r709_dec(v):
    return np.where(v < 0.081, v / 4.5, _pow((np.maximum(v, 0) + 0.099) / 1.099, 1 / 0.45))


_A, _B = 10.5402377416545, 0.0729055341958355


def _cct_enc(l):
    return np.where(l <= 0.0078125, _A * l + _B, (np.log2(np.maximum(l, 1e-10)) + 9.72) / 17.52)


def _cct_dec(v):
    return np.where(v <= 0.155251141552511, (v - _B) / _A, 2.0 ** (v * 17.52 - 9.72))


_LC = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809)   # LogC3 EI800


def _logc_enc(l):
    k = _LC
    return np.where(l > k["cut"], k["c"] * np.log10(np.maximum(k["a"] * l + k["b"], 1e-10)) + k["d"], k["e"] * l + k["f"])


def _logc_dec(t):
    k = _LC
    return np.where(t > k["e"] * k["cut"] + k["f"], (10.0 ** ((t - k["d"]) / k["c"]) - k["b"]) / k["a"], (t - k["f"]) / k["e"])


def _slog3_enc(l):
    return np.where(l >= 0.01125, (420.0 + np.log10(np.maximum((l + 0.01) / 0.19, 1e-10)) * 261.5) / 1023.0,
                    (l * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0)


def _slog3_dec(t):
    return np.where(t >= 171.2102946929 / 1023.0, 10.0 ** ((t * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01,
                    (t * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0))


_LIN = (lambda l: l, lambda v: v)
CURVES = {
    "linear": _LIN, "srgb": (_srgb_enc, _srgb_dec), "rec709": (_r709_enc, _r709_dec),
    "gamma22": (lambda l: _pow(np.maximum(l, 0), 1 / 2.2), lambda v: _pow(np.maximum(v, 0), 2.2)),
    "gamma24": (lambda l: _pow(np.maximum(l, 0), 1 / 2.4), lambda v: _pow(np.maximum(v, 0), 2.4)),
    "acescct": (_cct_enc, _cct_dec), "logc3": (_logc_enc, _logc_dec), "slog3": (_slog3_enc, _slog3_dec),
}

SPACES = {   # name -> (primaries, curve)
    "linear_rec709": ("rec709", "linear"), "srgb": ("rec709", "srgb"), "rec709": ("rec709", "rec709"),
    "rec709_gamma24": ("rec709", "gamma24"), "gamma22_rec709": ("rec709", "gamma22"),
    "linear_rec2020": ("rec2020", "linear"), "rec2020_gamma24": ("rec2020", "gamma24"),
    "linear_p3d65": ("p3d65", "linear"), "p3d65_gamma26": ("p3d65", "gamma24"),
    "acescg": ("ap1", "linear"), "acescct": ("ap1", "acescct"),
    "logc3_awg3": ("awg3", "logc3"), "slog3_sgamut3": ("sgamut3", "slog3"),
}


def convert(rgb: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert (..., 3) float RGB between named spaces in :data:`SPACES`."""
    if src not in SPACES or dst not in SPACES:
        raise KeyError(f"unknown colour space; choose from {sorted(SPACES)}")
    (sp, sc), (dp, dc) = SPACES[src], SPACES[dst]
    x = np.asarray(rgb, np.float32)
    lin = CURVES[sc][1](x)
    if sp != dp:
        lin = lin @ primaries_matrix(sp, dp).T
    return np.asarray(CURVES[dc][0](lin), np.float32)
