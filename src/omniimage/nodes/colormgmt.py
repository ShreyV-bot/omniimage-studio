"""Colour-management node (works with the engine's scene-linear Rec.709 working space)."""
from __future__ import annotations
from ..core.registry import node
from ..core.params import P
from ..core.ports import PortType as T
from ..core.media import with_rgb
from .. import colorspaces as cs

_CH = tuple(sorted(cs.SPACES))


@node("colorspace", "color", "Convert pixel values between colour spaces (e.g. ACEScg, ACEScct, LogC3, S-Log3, Rec.2020, "
      "P3-D65, sRGB, Rec.709). Set 'from' to how the incoming values are encoded and 'to' to what you want out. "
      "The engine's working space is linear_rec709, so a camera-log clip goes: colorspace(from=logc3_awg3, to=linear_rec709).",
      inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE}, pointwise=True,
      params={"from_space": P("choice", "linear_rec709", choices=_CH), "to_space": P("choice", "srgb", choices=_CH)})
def colorspace(inp, p, ctx):
    im = inp["image"]
    return with_rgb(im, cs.convert(im[..., :3], p["from_space"], p["to_space"]))
