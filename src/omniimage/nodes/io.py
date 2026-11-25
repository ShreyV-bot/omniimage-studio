"""File input/output nodes (images, video frames, audio) and LUT nodes."""
from __future__ import annotations
import os
import shutil
import subprocess
import wave

import numpy as np
import cv2
from scipy.io import wavfile

from ..core.registry import node
from ..core.params import P, ParamError
from ..core.ports import PortType as T
from ..core.media import AudioBuf, srgb_to_linear, linear_to_srgb, ensure_rgba, luminance, with_rgb
from ..lut import LUT3D


def _mtime(p):
    path = p.get("path")
    try:
        return os.path.getmtime(path) if path else None
    except OSError:
        return None


def _write_exr(a: np.ndarray, path: str) -> bool:
    try:
        if cv2.imwrite(path, a[..., [2, 1, 0, 3]]):
            return True
    except cv2.error:
        pass
    try:
        import OpenEXR
        import Imath  # noqa: F401  (older API)
    except ImportError:
        OpenEXR = None
    if OpenEXR is not None and hasattr(OpenEXR, "File"):
        try:
            OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION},
                         {"RGBA": np.ascontiguousarray(a)}).write(path)
            return True
        except Exception:
            pass
    from ..exr import write_exr
    write_exr(path, a)
    return True


def _read_exr(path: str) -> np.ndarray:
    try:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except cv2.error:
        raw = None
    if raw is not None:
        a = raw.astype(np.float32)
        if a.ndim == 2:
            a = a[..., None].repeat(3, -1)
        if a.shape[2] == 3:
            a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
        return np.ascontiguousarray(a[..., [2, 1, 0, 3]])
    try:
        import OpenEXR
        if hasattr(OpenEXR, "File"):
            ch = OpenEXR.File(path).channels()
            a = np.asarray(ch["RGBA"].pixels if "RGBA" in ch else
                           np.dstack([ch[c].pixels for c in "RGB"] + ([ch["A"].pixels] if "A" in ch else [])), np.float32)
            if a.shape[2] == 3:
                a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
            return np.ascontiguousarray(a)
    except ImportError:
        pass
    from ..exr import read_exr
    return read_exr(path)


def read_image_file(path: str, decode: str = "auto") -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"image not found: {path}")
    if path.lower().endswith(".exr"):
        return _read_exr(path)
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        from PIL import Image
        raw = np.asarray(Image.open(path).convert("RGBA"))[..., [2, 1, 0, 3]]
    is_int = raw.dtype in (np.uint8, np.uint16)
    scale = 255.0 if raw.dtype == np.uint8 else 65535.0 if raw.dtype == np.uint16 else 1.0
    a = raw.astype(np.float32) / scale
    if a.ndim == 2:
        a = a[..., None].repeat(3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
    a = a[..., [2, 1, 0, 3]]  # BGRA -> RGBA
    if decode == "srgb" or (decode == "auto" and is_int):
        a[..., :3] = srgb_to_linear(a[..., :3])
    return np.ascontiguousarray(a)


@node("load_image", "io", "Load PNG/JPEG/TIFF/WebP/BMP/EXR. 8/16-bit files are decoded from sRGB to linear.",
      outputs={"image": T.IMAGE}, fingerprint=_mtime,
      params={"path": P("str", "", animatable=False), "decode": P("choice", "auto", choices=("auto", "srgb", "linear"))})
def load_image(inp, p, ctx):
    if not p["path"]:
        raise ParamError("load_image needs a path")
    return read_image_file(p["path"], p["decode"])


def write_image_file(img: np.ndarray, path: str, encode: bool = True, bits: int = 8, quality: int = 95):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    a = np.asarray(img, np.float32)
    if ext == ".exr":
        ok = _write_exr(a, path)
    else:
        rgb = a[..., :3]
        al = a[..., 3:4]
        if ext in (".jpg", ".jpeg", ".bmp"):
            rgb = rgb * al
        rgb = linear_to_srgb(rgb) if encode else np.clip(rgb, 0, 1)
        rgb = np.clip(rgb, 0, 1)
        if ext in (".jpg", ".jpeg", ".bmp"):
            out = rgb
        else:
            out = np.concatenate([rgb, np.clip(al, 0, 1)], -1)
        out = out[..., [2, 1, 0] + ([3] if out.shape[2] == 4 else [])]
        if ext in (".jpg", ".jpeg", ".bmp") or bits == 8:
            out = (out * 255.0 + 0.5).astype(np.uint8)
        else:
            out = (out * 65535.0 + 0.5).astype(np.uint16)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
        ok = cv2.imwrite(path, out, params)
    if not ok:
        raise IOError(f"could not write {path} (unsupported extension?)")
    return path


@node("write_image", "io", "Write the image to disk (PNG/JPEG/TIFF/WebP/BMP/EXR). Linear light is encoded to sRGB "
      "unless disabled. EXR is written as linear float.", inputs={"image": T.IMAGE}, sink=True,
      params={"path": P("str", "out.png", animatable=False), "encode": P("bool", True), "bits": P("choice", "8", choices=("8", "16")),
              "quality": P("int", 95, 1, 100)})
def write_image(inp, p, ctx):
    path = p["path"].replace("{frame}", f"{ctx.frame:05d}")
    write_image_file(inp["image"], path, p["encode"], int(p["bits"]), p["quality"])


_caps = {}


@node("read_video_frame", "io", "Read the frame matching the current timeline frame from a video file.",
      outputs={"image": T.IMAGE}, uses_time=True, fingerprint=_mtime,
      params={"path": P("str", "", animatable=False), "offset": P("int", 0, -100000, 100000, doc="timeline frame where the clip starts"),
              "loop": P("bool", False)})
def read_video_frame(inp, p, ctx):
    path = p["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"video not found: {path}")
    st = _caps.get(path)
    if st is None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open video {path}")
        st = _caps[path] = {"cap": cap, "pos": 0, "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    idx = ctx.frame - p["offset"]
    n = st["n"]
    if n > 0:
        idx = idx % n if p["loop"] else min(max(idx, 0), n - 1)
    cap = st["cap"]
    if idx != st["pos"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:
        raise IOError(f"could not decode frame {idx} of {path}")
    st["pos"] = idx + 1
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    return ensure_rgba(srgb_to_linear(rgb))


# ------------------------------------------------------------------------------ audio files
def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is required for this audio format but was not found on PATH")
    return exe


def read_audio_file(path: str, sr: int = 0) -> AudioBuf:
    if not os.path.exists(path):
        raise FileNotFoundError(f"audio not found: {path}")
    if path.lower().endswith(".wav") and sr == 0:
        try:
            rate, data = wavfile.read(path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
            return AudioBuf(data, int(rate))
        except Exception:
            pass
    sr = sr or 44100
    cmd = [_ffmpeg(), "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return AudioBuf(np.frombuffer(out, np.float32).reshape(-1, 2).copy(), sr)


def write_audio_file(a: AudioBuf, path: str, bits: str = "16"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    s = np.clip(a.samples, -1.0, 1.0)
    if path.lower().endswith(".wav"):
        if bits == "16":
            wavfile.write(path, a.sr, (s * 32767.0).astype(np.int16))
        elif bits == "32f":
            wavfile.write(path, a.sr, s.astype(np.float32))
        else:  # 24-bit PCM packed by hand
            v = (s * 8388607.0).astype(np.int32)
            b = np.stack([(v >> 0) & 255, (v >> 8) & 255, (v >> 16) & 255], -1).astype(np.uint8).tobytes()
            with wave.open(path, "wb") as w:
                w.setnchannels(a.channels); w.setsampwidth(3); w.setframerate(a.sr); w.writeframes(b)
        return path
    raw = s.astype(np.float32).tobytes()
    cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "f32le", "-ar", str(a.sr), "-ac", str(a.channels), "-i", "-", path]
    subprocess.run(cmd, input=raw, check=True)
    return path


@node("load_audio", "io", "Load audio (WAV natively; MP3/FLAC/OGG/AAC/... through ffmpeg).", outputs={"audio": T.AUDIO},
      fingerprint=_mtime, params={"path": P("str", "", animatable=False), "sr": P("int", 0, 0, 192000, doc="0 = keep (WAV) / 44100")})
def load_audio(inp, p, ctx):
    if not p["path"]:
        raise ParamError("load_audio needs a path")
    return read_audio_file(p["path"], p["sr"])


@node("write_audio", "io", "Write audio to disk. WAV 16/24-bit/32-bit float natively; other formats via ffmpeg.",
      inputs={"audio": T.AUDIO}, sink=True,
      params={"path": P("str", "out.wav", animatable=False), "bits": P("choice", "16", choices=("16", "24", "32f"))})
def write_audio(inp, p, ctx):
    write_audio_file(inp["audio"], p["path"], p["bits"])


@node("waveform_image", "io", "Draw an audio waveform as an image (audio -> visual bridge).",
      inputs={"audio": T.AUDIO}, outputs={"image": T.IMAGE},
      params={"width": P("int", 960, 8, 8192), "height": P("int", 270, 8, 4096), "color": P("color", [0.2, 0.9, 1.0, 1.0]),
              "thickness": P("int", 1, 1, 20)})
def waveform_image(inp, p, ctx):
    a = inp["audio"]
    mono = a.samples.mean(1)
    w, h = p["width"], p["height"]
    n = len(mono)
    edges = np.linspace(0, n, w + 1).astype(int)
    img = np.zeros((h, w, 4), np.float32)
    col = srgb_to_linear(np.array(p["color"][:3], np.float32))
    for x in range(w):
        seg = mono[edges[x]:max(edges[x + 1], edges[x] + 1)]
        if seg.size == 0:
            continue
        lo, hi = float(np.clip(seg.min(), -1, 1)), float(np.clip(seg.max(), -1, 1))
        y0, y1 = int((1 - hi) * 0.5 * (h - 1)), int((1 - lo) * 0.5 * (h - 1))
        y1 = max(y1, y0 + p["thickness"] - 1)
        img[y0:y1 + 1, x, :3] = col
        img[y0:y1 + 1, x, 3] = 1.0
    return img


# ------------------------------------------------------------------------------ LUT nodes
@node("lut_load", "lut", "Load a .cube LUT file (3D, or 1D which is expanded).", outputs={"lut": T.LUT}, fingerprint=_mtime,
      params={"path": P("str", "", animatable=False)})
def lut_load(inp, p, ctx):
    if not p["path"]:
        raise ParamError("lut_load needs a path")
    return LUT3D.read_cube(p["path"])


@node("lut_apply", "lut", "Apply a 3D LUT. 'srgb' space: encode linear to sRGB, apply, decode back (typical creative "
      "LUT). 'linear': apply directly. Provide a lut input or a .cube path.",
      inputs={"image": T.IMAGE, "lut": T.LUT}, outputs={"image": T.IMAGE}, optional=("lut",), pointwise=True,
      fingerprint=_mtime,
      params={"path": P("str", "", animatable=False), "intensity": P("float", 1.0, 0, 1),
              "space": P("choice", "srgb", choices=("srgb", "linear"))})
def lut_apply(inp, p, ctx):
    lut = inp.get("lut")
    if lut is None:
        if not p["path"]:
            raise ParamError("lut_apply needs a lut input or a path")
        lut = LUT3D.read_cube(p["path"])
    im = inp["image"]
    rgb = im[..., :3]
    enc = linear_to_srgb(rgb) if p["space"] == "srgb" else rgb
    res = lut.apply(np.clip(enc, 0, 1))
    if p["space"] == "srgb":
        res = srgb_to_linear(res)
    out = rgb * (1 - p["intensity"]) + res * p["intensity"] if p["intensity"] < 1 else res
    return with_rgb(im, out)
