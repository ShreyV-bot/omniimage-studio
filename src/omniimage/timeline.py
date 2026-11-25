"""Timeline: video/audio tracks of clips with transitions, speed, opacity and a multitrack mixer
(track/bus gain, pan, automation, ducking, loudness normalisation, limiter). Frames are the unit of time for
placement; audio fades and in-points are in seconds; automation keyframes are in seconds.

Every clip's picture/sound comes from a node graph, so any effect, generator or synthesised sound in the engine is
usable on the timeline, and file clips are just graphs with a loader node."""
from __future__ import annotations
import json
import os
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from math import gcd
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

from .core.graph import Graph, Ctx, GraphError
from .core.media import AudioBuf
from .core.params import resolve, eval_keys, parse_color
from .core.media import srgb_to_linear
from .loudness import measure_lufs

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
TRANSITIONS = ("dissolve", "dip_to_black", "wipe_left", "wipe_right")
FITS = ("contain", "cover", "stretch", "none")
BLENDS = ("over", "add", "multiply", "screen")


@dataclass
class Transition:
    type: str = "dissolve"
    duration: int = 12            # frames; the clip must overlap the previous clip by this much

    def __post_init__(self):
        if self.type not in TRANSITIONS:
            raise GraphError(f"unknown transition {self.type!r}; choose from {TRANSITIONS}")
        if self.duration < 1:
            raise GraphError("transition duration must be >= 1 frame")


def _chain(g: Graph, node: str, effects) -> str:
    for typ, params in effects or []:
        n = g.add(typ, **params)
        g.connect(node, n)
        node = n
    return node


@dataclass
class Clip:
    """A video clip. ``in_point`` is a source frame offset and ``speed`` multiplies source time (2 = twice as fast)."""
    graph: Graph
    node: str
    start: int
    duration: int
    name: str = "clip"
    port: Optional[str] = None
    in_point: float = 0.0
    speed: float = 1.0
    opacity: object = 1.0          # number, keyframes or expression (time = frames since clip start)
    fit: str = "contain"
    transition_in: Optional[Transition] = None
    media: Optional[str] = None    # original file path, used for EDL/OTIO reel names

    def __post_init__(self):
        if self.duration < 1:
            raise GraphError("clip duration must be >= 1 frame")
        if self.fit not in FITS:
            raise GraphError(f"fit must be one of {FITS}")

    @property
    def end(self) -> int:
        return self.start + self.duration

    @classmethod
    def from_graph(cls, graph: Graph, node: str, start: int, duration: int, **kw) -> "Clip":
        return cls(graph, node, start, duration, **kw)

    @classmethod
    def from_file(cls, path: str, start: int, duration: int, effects=None, **kw) -> "Clip":
        g = Graph()
        if path.lower().endswith(VIDEO_EXT):
            n = g.add("read_video_frame", path=path, loop=False)
        else:
            n = g.add("load_image", path=path)
        n = _chain(g, n, effects)
        kw.setdefault("name", os.path.basename(path))
        return cls(g, n, start, duration, media=path, **kw)

    def local_frame(self, t: int) -> int:
        return int(round(self.in_point + (t - self.start) * self.speed))


@dataclass
class AudioClip:
    graph: Optional[Graph] = None
    node: Optional[str] = None
    path: Optional[str] = None
    start: int = 0
    duration: int = 0               # frames on the timeline
    name: str = "audio"
    in_point: float = 0.0           # seconds into the source
    gain_db: float = 0.0
    fade_in: float = 0.0            # seconds
    fade_out: float = 0.0
    media: Optional[str] = None

    def __post_init__(self):
        if self.duration < 1:
            raise GraphError("audio clip duration must be >= 1 frame")
        if (self.graph is None) == (self.path is None):
            raise GraphError("an audio clip needs exactly one of: a file path, or a graph+node")

    @property
    def end(self) -> int:
        return self.start + self.duration

    @classmethod
    def from_file(cls, path: str, start: int, duration: int, **kw) -> "AudioClip":
        kw.setdefault("name", os.path.basename(path))
        return cls(path=path, start=start, duration=duration, media=path, **kw)

    @classmethod
    def from_graph(cls, graph: Graph, node: str, start: int, duration: int, **kw) -> "AudioClip":
        return cls(graph=graph, node=node, start=start, duration=duration, **kw)


@dataclass
class Track:
    kind: str                       # "video" | "audio"
    name: str
    clips: list = field(default_factory=list)
    mute: bool = False
    solo: bool = False
    gain_db: object = 0.0           # audio: number or {"keys": [[seconds, dB], ...]}
    pan: float = 0.0                # audio: -1 left .. +1 right
    bus: str = "master"             # audio routing
    blend: str = "over"             # video blend mode against the tracks below
    opacity: float = 1.0

    def add(self, clip):
        want = Clip if self.kind == "video" else AudioClip
        if not isinstance(clip, want):
            raise GraphError(f"{self.kind} track {self.name!r} needs a {want.__name__}")
        self.clips.append(clip)
        return clip


@dataclass
class Duck:
    """Sidechain ducking: lower this bus by ``amount_db`` while the ``key`` bus (e.g. dialogue) is louder than ``threshold_db``."""
    key: str
    amount_db: float = -12.0
    threshold_db: float = -40.0
    smooth: float = 0.15            # seconds


@dataclass
class Bus:
    gain_db: float = 0.0
    duck: Optional[Duck] = None


@dataclass
class Master:
    gain_db: float = 0.0
    target_lufs: Optional[float] = None   # e.g. -14 (streaming), -23 (broadcast R128)
    ceiling_db: float = -1.0              # soft limiter ceiling; None disables


# ---------------------------------------------------------------- helpers
def _fit(img: np.ndarray, W: int, H: int, mode: str) -> np.ndarray:
    h, w = img.shape[:2]
    if (w, h) == (W, H) and mode != "none":
        return img
    out = np.zeros((H, W, 4), np.float32)
    if mode == "stretch":
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    if mode == "none":
        s = 1.0
    elif mode == "contain":
        s = min(W / w, H / h)
    else:
        s = max(W / w, H / h)
    nw, nh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
    r = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    dx0, dy0 = max(x0, 0), max(y0, 0)
    cw, ch = min(nw - sx0, W - dx0), min(nh - sy0, H - dy0)
    out[dy0:dy0 + ch, dx0:dx0 + cw] = r[sy0:sy0 + ch, sx0:sx0 + cw]
    return out


def _over(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """straight-alpha a over b"""
    sa, da = a[..., 3:4], b[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (a[..., :3] * sa + b[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-9)
    return np.concatenate([rgb, oa], -1).astype(np.float32)


def _transition(a: np.ndarray, b: np.ndarray, kind: str, p: float) -> np.ndarray:
    p = float(np.clip(p, 0, 1))
    if kind == "dissolve":
        pa, pb = a[..., :3] * a[..., 3:4], b[..., :3] * b[..., 3:4]
        oa = a[..., 3:4] * (1 - p) + b[..., 3:4] * p
        return np.concatenate([(pa * (1 - p) + pb * p) / np.maximum(oa, 1e-9), oa], -1).astype(np.float32)
    if kind == "dip_to_black":
        src, k = (a, 1 - 2 * p) if p < 0.5 else (b, 2 * p - 1)
        out = src.copy()
        out[..., :3] *= max(k, 0.0)
        return out
    W = a.shape[1]
    edge = int(round(p * W))
    out = a.copy()
    if kind == "wipe_left":
        out[:, :edge] = b[:, :edge]
    elif kind == "wipe_right":
        out[:, W - edge:] = b[:, W - edge:]
    else:
        raise GraphError(f"unknown transition {kind!r}")
    return out


def _stereo(buf: AudioBuf, sr: int) -> np.ndarray:
    x = buf.samples
    if buf.sr != sr:
        g = gcd(buf.sr, sr)
        x = resample_poly(x, sr // g, buf.sr // g, axis=0)
    x = x.astype(np.float32)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    return x[:, :2]


def read_audio_any(path: str) -> AudioBuf:
    """Read a WAV directly, or extract the audio track of any media file through ffmpeg."""
    from .nodes.io import read_audio_file
    if path.lower().endswith(".wav"):
        return read_audio_file(path)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError(f"ffmpeg is required to read audio from {path}")
    d = tempfile.mkdtemp(prefix="omni_a_")
    try:
        wav = os.path.join(d, "a.wav")
        r = subprocess.run([ff, "-v", "error", "-y", "-i", path, "-vn", "-acodec", "pcm_f32le", wav],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"no readable audio in {path}: {r.stderr.decode(errors='replace')[-200:]}")
        return read_audio_file(wav)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _db_curve(value, n: int, sr: int) -> np.ndarray:
    """Gain in dB per sample from a constant or seconds-based keyframes."""
    if isinstance(value, dict) and "keys" in value:
        ks = sorted((float(a), float(b)) for a, b in value["keys"])
        t = np.arange(n) / sr
        return np.interp(t, [k[0] for k in ks], [k[1] for k in ks]).astype(np.float32)
    return np.full(n, float(value), np.float32)


def _soft_limit(x: np.ndarray, ceiling_db: float) -> np.ndarray:
    c = 10 ** (ceiling_db / 20)
    knee = 0.8 * c
    ax = np.abs(x)
    over = ax > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (c - knee) * np.tanh((ax[over] - knee) / (c - knee)))
    return y


# ---------------------------------------------------------------- the timeline
class Timeline:
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 24.0, sample_rate: int = 48000,
                 name: str = "Timeline", background="#000000"):
        self.width, self.height, self.fps, self.sample_rate = int(width), int(height), float(fps), int(sample_rate)
        self.name = name
        self.background = background
        self.tracks: List[Track] = []
        self.buses: Dict[str, Bus] = {}
        self.master = Master()
        self.markers: List[dict] = []

    # --- building
    def add_track(self, kind: str, name: Optional[str] = None, **kw) -> Track:
        if kind not in ("video", "audio"):
            raise GraphError("track kind must be 'video' or 'audio'")
        n = sum(t.kind == kind for t in self.tracks) + 1
        t = Track(kind, name or f"{'V' if kind == 'video' else 'A'}{n}", **kw)
        self.tracks.append(t)
        return t

    def add_marker(self, frame: int, name: str = "", color: str = "blue"):
        self.markers.append({"frame": int(frame), "name": name, "color": color})

    def video_tracks(self):
        return [t for t in self.tracks if t.kind == "video"]

    def audio_tracks(self):
        return [t for t in self.tracks if t.kind == "audio"]

    @property
    def duration(self) -> int:
        ends = [c.end for t in self.tracks for c in t.clips]
        return max(ends) if ends else 0

    def seconds(self, frames: float) -> float:
        return frames / self.fps

    # --- video
    def _clip_image(self, c: Clip, t: int) -> np.ndarray:
        img = c.graph.evaluate(c.node, c.port, Ctx(c.local_frame(t), self.fps))
        img = _fit(np.asarray(img, np.float32), self.width, self.height, c.fit)
        op = float(resolve(c.opacity, Ctx(t - c.start, self.fps)))
        if op != 1.0:
            img = img.copy()
            img[..., 3] *= np.clip(op, 0, 1)
        return img

    def _track_image(self, tr: Track, t: int) -> Optional[np.ndarray]:
        active = sorted((c for c in tr.clips if c.start <= t < c.end), key=lambda c: c.start)
        res = None
        for c in active:
            img = self._clip_image(c, t)
            if res is None:
                res = img
            elif c.transition_in and t < c.start + c.transition_in.duration:
                res = _transition(res, img, c.transition_in.type, (t - c.start) / c.transition_in.duration)
            else:
                res = _over(img, res)
        return res

    def render_frame(self, t: int) -> np.ndarray:
        """Composite every video track at timeline frame ``t`` onto the background (returns opaque linear RGBA)."""
        bg = np.empty((self.height, self.width, 4), np.float32)
        c = parse_color(self.background) if isinstance(self.background, str) else list(self.background)
        bg[:] = [*srgb_to_linear(np.array(c[:3], np.float32)), c[3] if len(c) > 3 else 1.0]
        canvas = bg
        solo = any(tr.solo for tr in self.video_tracks())
        for tr in self.video_tracks():
            if tr.mute or (solo and not tr.solo):
                continue
            img = self._track_image(tr, t)
            if img is None:
                continue
            a = img[..., 3:4] * tr.opacity
            cb, cs = canvas[..., :3], img[..., :3]
            mixed = {"over": cs, "add": cb + cs, "multiply": cb * cs, "screen": 1 - (1 - cb) * (1 - cs)}[tr.blend]
            canvas = np.concatenate([cb * (1 - a) + mixed * a, np.maximum(canvas[..., 3:4], a)], -1).astype(np.float32)
        return canvas

    # make a Timeline usable wherever render code expects "a graph with evaluate()"
    def evaluate(self, target=None, port=None, ctx: Optional[Ctx] = None, **_):
        return self.render_frame(ctx.frame if ctx else 0)

    # --- audio
    def _source_audio(self, c: AudioClip, cache: dict) -> AudioBuf:
        key = c.path if c.path else (id(c.graph), c.node)
        if key not in cache:
            cache[key] = read_audio_any(c.path) if c.path else c.graph.evaluate(c.node, None, Ctx(0, self.fps))
        return cache[key]

    def _render_clip(self, c: AudioClip, sr: int, cache: dict) -> np.ndarray:
        src = _stereo(self._source_audio(c, cache), sr)
        n = int(round(self.seconds(c.duration) * sr))
        i0 = int(round(c.in_point * sr))
        seg = np.zeros((n, 2), np.float32)
        avail = src[i0:i0 + n]
        seg[:len(avail)] = avail
        env = np.full(n, 10 ** (c.gain_db / 20), np.float32)
        fi, fo = int(c.fade_in * sr), int(c.fade_out * sr)
        if fi > 0:
            env[:fi] *= np.linspace(0, 1, min(fi, n), endpoint=False, dtype=np.float32)[:n]
        if fo > 0:
            k = min(fo, n)
            env[n - k:] *= np.linspace(1, 0, k, endpoint=True, dtype=np.float32)
        return seg * env[:, None]

    def render_audio(self, sr: Optional[int] = None) -> AudioBuf:
        """Mix every audio track through its bus to the master. Returns stereo float32."""
        sr = sr or self.sample_rate
        total = int(round(self.seconds(self.duration) * sr))
        cache: dict = {}
        solo = any(t.solo for t in self.audio_tracks())
        bus_buf: Dict[str, np.ndarray] = {}
        for tr in self.audio_tracks():
            if tr.mute or (solo and not tr.solo) or not tr.clips:
                continue
            buf = np.zeros((total, 2), np.float32)
            for c in tr.clips:
                seg = self._render_clip(c, sr, cache)
                s0 = int(round(self.seconds(c.start) * sr))
                e0 = min(s0 + len(seg), total)
                if e0 > s0:
                    buf[s0:e0] += seg[:e0 - s0]
            buf *= (10 ** (_db_curve(tr.gain_db, total, sr) / 20))[:, None]
            ang = (np.clip(tr.pan, -1, 1) + 1) * np.pi / 4
            buf *= np.array([np.cos(ang), np.sin(ang)], np.float32) * np.sqrt(2)
            bus_buf[tr.bus] = bus_buf.get(tr.bus, 0) + buf
        # ducking uses the *pre-duck* level of the key bus
        raw = {k: v.copy() for k, v in bus_buf.items()}
        for name, bus in self.buses.items():
            if name not in bus_buf:
                continue
            bus_buf[name] = bus_buf[name] * 10 ** (bus.gain_db / 20)
            d = bus.duck
            if d and d.key in raw:
                lvl = np.sqrt(uniform_filter1d((raw[d.key] ** 2).mean(1), max(int(0.05 * sr), 1)) + 1e-12)
                on = np.clip((20 * np.log10(lvl + 1e-12) - d.threshold_db) / 6.0, 0, 1)
                on = uniform_filter1d(on, max(int(d.smooth * sr), 1)).astype(np.float32)
                bus_buf[name] = bus_buf[name] * (10 ** (d.amount_db * on / 20))[:, None]
        mix = np.zeros((total, 2), np.float32)
        for name, b in bus_buf.items():
            mix += b
        mix *= 10 ** (self.master.gain_db / 20)
        out = AudioBuf(mix.astype(np.float32), sr)
        if self.master.target_lufs is not None:
            cur = measure_lufs(out)
            if np.isfinite(cur):
                out = AudioBuf((out.samples * 10 ** ((self.master.target_lufs - cur) / 20)).astype(np.float32), sr)
        if self.master.ceiling_db is not None:
            out = AudioBuf(_soft_limit(out.samples, self.master.ceiling_db), sr)
        return out

    # --- persistence
    def to_dict(self) -> dict:
        def tr_d(t: Track):
            base = {k: getattr(t, k) for k in ("kind", "name", "mute", "solo", "gain_db", "pan", "bus", "blend", "opacity")}
            cl = []
            for c in t.clips:
                if isinstance(c, Clip):
                    cl.append({"graph": c.graph.to_dict(), "node": c.node, "port": c.port, "start": c.start,
                               "duration": c.duration, "name": c.name, "in_point": c.in_point, "speed": c.speed,
                               "opacity": c.opacity, "fit": c.fit, "media": c.media,
                               "transition_in": vars(c.transition_in) if c.transition_in else None})
                else:
                    cl.append({"graph": c.graph.to_dict() if c.graph else None, "node": c.node, "path": c.path,
                               "start": c.start, "duration": c.duration, "name": c.name, "in_point": c.in_point,
                               "gain_db": c.gain_db, "fade_in": c.fade_in, "fade_out": c.fade_out, "media": c.media})
            base["clips"] = cl
            return base
        return {"version": 1, "name": self.name, "width": self.width, "height": self.height, "fps": self.fps,
                "sample_rate": self.sample_rate, "background": self.background, "markers": self.markers,
                "tracks": [tr_d(t) for t in self.tracks],
                "buses": {k: {"gain_db": b.gain_db, "duck": vars(b.duck) if b.duck else None} for k, b in self.buses.items()},
                "master": vars(self.master)}

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        tl = cls(d["width"], d["height"], d["fps"], d["sample_rate"], d.get("name", "Timeline"), d.get("background", "#000000"))
        tl.markers = d.get("markers", [])
        for td in d["tracks"]:
            clips = td["clips"]
            t = tl.add_track(td["kind"], td["name"], mute=td["mute"], solo=td["solo"], gain_db=td["gain_db"],
                             pan=td["pan"], bus=td["bus"], blend=td["blend"], opacity=td["opacity"])
            for c in clips:
                g = Graph.from_dict(c["graph"]) if c.get("graph") else None
                if td["kind"] == "video":
                    tr = Transition(**c["transition_in"]) if c.get("transition_in") else None
                    t.add(Clip(g, c["node"], c["start"], c["duration"], c["name"], c.get("port"), c["in_point"], c["speed"],
                               c["opacity"], c["fit"], tr, c.get("media")))
                else:
                    t.add(AudioClip(g, c.get("node"), c.get("path"), c["start"], c["duration"], c["name"], c["in_point"],
                                    c["gain_db"], c["fade_in"], c["fade_out"], c.get("media")))
        for k, b in d.get("buses", {}).items():
            tl.buses[k] = Bus(b["gain_db"], Duck(**b["duck"]) if b.get("duck") else None)
        tl.master = Master(**d.get("master", {}))
        return tl

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Timeline":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def render_timeline(tl: Timeline, out: str, start: int = 0, end: Optional[int] = None, crf: int = 18,
                    with_audio: bool = True, progress=None) -> str:
    """Render timeline frames ``start..end`` (inclusive; default whole timeline) to a video file or image sequence."""
    from .render import render_frames
    if tl.duration < 1:
        raise GraphError("timeline is empty")
    end = tl.duration - 1 if end is None else end
    audio = None
    if with_audio and tl.audio_tracks() and any(t.clips for t in tl.audio_tracks()):
        full = tl.render_audio()
        a0, a1 = int(tl.seconds(start) * full.sr), int(round(tl.seconds(end + 1) * full.sr))
        audio = AudioBuf(full.samples[a0:a1], full.sr)
    return render_frames(tl, "timeline", out, start, end, tl.fps, audio=audio, crf=crf, progress=progress)
