"""Rendering helpers: still frames, image sequences and video (ffmpeg) with optional audio."""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

import numpy as np

from .core.graph import Ctx, Graph, GraphError
from .core.media import AudioBuf, linear_to_srgb
from .nodes.io import write_image_file, write_audio_file

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def render_still(graph: Graph, target: str, path: str, frame: int = 0, fps: float = 24.0, port: Optional[str] = None,
                 encode: bool = True, bits: int = 8) -> str:
    img = graph.evaluate(target, port, Ctx(frame, fps))
    return write_image_file(img, path, encode=encode, bits=bits)


def to_u8(img: np.ndarray) -> np.ndarray:
    rgb = linear_to_srgb(img[..., :3] * img[..., 3:4])   # composite over black for video
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def render_frames(graph: Graph, target: str, out: str, start: int = 0, end: int = 47, fps: float = 24.0,
                  audio: Optional[AudioBuf] = None, crf: int = 18, port: Optional[str] = None,
                  progress: Optional[Callable[[int, int], None]] = None) -> str:
    """Render frames ``start..end`` (inclusive). ``out`` is a video file (.mp4/.mov/...) or an image pattern such as
    ``frames/f_{frame}.png``."""
    if end < start:
        raise GraphError("end frame must be >= start frame")
    total = end - start + 1
    if out.lower().endswith(VIDEO_EXT):
        return _render_video(graph, target, out, start, end, fps, audio, crf, port, progress)
    if "{frame}" not in out:
        base, ext = os.path.splitext(out)
        out = f"{base}_{{frame}}{ext}"
    for i, f in enumerate(range(start, end + 1)):
        img = graph.evaluate(target, port, Ctx(f, fps))
        write_image_file(img, out.replace("{frame}", f"{f:05d}"))
        if progress:
            progress(i + 1, total)
    return out


def _render_video(graph, target, out, start, end, fps, audio, crf, port, progress):
    ffmpeg = shutil.which("ffmpeg")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    total = end - start + 1
    first = graph.evaluate(target, port, Ctx(start, fps))
    h, w = first.shape[:2]
    if not ffmpeg:
        import cv2
        if audio is not None:
            raise RuntimeError("ffmpeg is required to render video with audio")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i, f in enumerate(range(start, end + 1)):
            img = first if f == start else graph.evaluate(target, port, Ctx(f, fps))
            vw.write(to_u8(img)[..., ::-1])
            if progress:
                progress(i + 1, total)
        vw.release()
        return out
    tmpdir = tempfile.mkdtemp(prefix="omni_")
    try:
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if audio is not None:
            wav = os.path.join(tmpdir, "a.wav")
            write_audio_file(audio, wav, "16")
            cmd += ["-i", wav]
        ext = out.lower()
        vcodec = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"] if ext.endswith(".webm") else \
                 ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] + vcodec
        if audio is not None:
            # pad short audio with silence so the (shorter) audio never truncates the video
            cmd += ["-af", "apad", "-c:a", "libopus" if ext.endswith(".webm") else "aac", "-shortest"]
        cmd += [out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i, f in enumerate(range(start, end + 1)):
                img = first if f == start else graph.evaluate(target, port, Ctx(f, fps))
                if img.shape[:2] != (h, w):
                    raise GraphError(f"frame {f} has size {img.shape[1]}x{img.shape[0]}, expected {w}x{h}")
                proc.stdin.write(to_u8(img).tobytes())
                if progress:
                    progress(i + 1, total)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[-600:]}")
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
