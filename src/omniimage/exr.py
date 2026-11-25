"""Minimal OpenEXR fallback (scanline, uncompressed, FLOAT RGBA) used when OpenCV was built without EXR support.

Preference order for EXR I/O: OpenCV -> the optional ``OpenEXR`` package -> this module.
The reader here only understands uncompressed scanline files (what :func:`write_exr` produces); for ZIP/PIZ/DWA
files from other tools install ``pip install OpenEXR``.
"""
from __future__ import annotations
import struct
import numpy as np

_MAGIC = 20000630
_FLOAT = 2


def _attr(name: str, typ: str, payload: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(payload)) + payload


def write_exr(path: str, img: np.ndarray) -> None:
    """Write an RGBA float image (H, W, 4) as uncompressed 32-bit float EXR."""
    h, w, c = img.shape
    chans = ["A", "B", "G", "R"] if c == 4 else ["B", "G", "R"]
    idx = {"R": 0, "G": 1, "B": 2, "A": 3}
    clist = b"".join(n.encode() + b"\0" + struct.pack("<iiii", _FLOAT, 0, 1, 1) for n in chans) + b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = (_attr("channels", "chlist", clist) + _attr("compression", "compression", b"\0") +
              _attr("dataWindow", "box2i", box) + _attr("displayWindow", "box2i", box) +
              _attr("lineOrder", "lineOrder", b"\0") + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)) +
              _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0)) +
              _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)) + b"\0")
    base = 8 + len(header) + 8 * h
    line_bytes = w * 4 * len(chans)
    offsets = [base + y * (8 + line_bytes) for y in range(h)]
    data = np.ascontiguousarray(img, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", _MAGIC, 2) + header)
        f.write(struct.pack(f"<{h}Q", *offsets))
        for y in range(h):
            f.write(struct.pack("<ii", y, line_bytes))
            for n in chans:
                f.write(data[y, :, idx[n]].tobytes())


def read_exr(path: str) -> np.ndarray:
    """Read an uncompressed scanline EXR into float32 RGBA."""
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 8 or struct.unpack("<i", buf[:4])[0] != _MAGIC:
        raise IOError("not an OpenEXR file")
    pos, attrs = 8, {}
    while buf[pos] != 0:
        e = buf.index(b"\0", pos); name = buf[pos:e].decode(); pos = e + 1
        e = buf.index(b"\0", pos); typ = buf[pos:e].decode(); pos = e + 1
        size = struct.unpack("<i", buf[pos:pos + 4])[0]; pos += 4
        attrs[name] = (typ, buf[pos:pos + size]); pos += size
    pos += 1
    if attrs["compression"][1][0] != 0:
        raise IOError("compressed EXR is not supported by the built-in reader; install OpenEXR (pip install OpenEXR)")
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    chans, p, cl = [], 0, attrs["channels"][1]
    while cl[p] != 0:
        e = cl.index(b"\0", p); name = cl[p:e].decode(); p = e + 1
        ptype = struct.unpack("<i", cl[p:p + 4])[0]; p += 16
        chans.append((name, ptype))
    sizes = {0: 4, 1: 2, 2: 4}
    out = np.zeros((h, w, 4), np.float32); out[..., 3] = 1.0
    table = struct.unpack(f"<{h}Q", buf[pos:pos + 8 * h])
    for off in table:
        y, n = struct.unpack("<ii", buf[off:off + 8]); q = off + 8
        for name, ptype in chans:
            nb = w * sizes[ptype]
            dt = {0: "<u4", 1: "<f2", 2: "<f4"}[ptype]
            row = np.frombuffer(buf[q:q + nb], dtype=dt).astype(np.float32); q += nb
            if name in "RGBA" and len(name) == 1:
                out[y - y0, :, "RGBA".index(name)] = row
    if not any(n == "G" for n, _ in chans) and any(n == "Y" for n, _ in chans):
        pass
    return out
