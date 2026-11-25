"""Typed ports: the contract that keeps invalid node connections impossible."""
from __future__ import annotations
from enum import Enum


class PortType(str, Enum):
    IMAGE = "image"   # float32 RGBA array (H, W, 4), scene-linear unless a node says otherwise
    MASK = "mask"     # float32 array (H, W)
    AUDIO = "audio"   # AudioBuf (samples (n, channels) float32 + sample rate)
    MESH = "mesh"     # omniimage.mesh.Mesh (triangle mesh with per-vertex material attributes)
    LUT = "lut"       # LUT3D
    VALUE = "value"   # float / int / str / list
    ANY = "any"


def compatible(src: PortType, dst: PortType) -> bool:
    """A source port may feed a destination port only when types match (or one side is ANY)."""
    return src == dst or src == PortType.ANY or dst == PortType.ANY
