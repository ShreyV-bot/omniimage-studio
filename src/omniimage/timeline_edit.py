"""Editing layer for :class:`Timeline`: validated edit operations with undo/redo, used by the GUI and by scripts.

Every edit is a plain dict ``{"op": ..., ...}``. A batch of edits is atomic: if one fails, nothing changes. Frames are
the unit of placement, as in the engine. Clips on a track are always kept sorted by start frame, so a clip is addressed
by ``{"track": i, "clip": j}`` against the latest view; ops that can reorder clips return the clip's new position.

Ops: set_timeline, add_track, remove_track, set_track, add_clip, move_clip, trim_clip, split_clip, delete_clip,
ripple_delete, set_clip, set_transition, add_effect, remove_effect, set_clip_param, add_marker, remove_marker,
set_master.

All limits below exist because timelines can come from untrusted JSON and the GUI is a network service on localhost:
no op can ask for an unbounded amount of memory or time.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import wavfile

from .core.graph import Ctx, Graph, GraphError
from .core.params import resolve
from .core.ports import PortType as T, compatible
from .core.registry import get_spec
from .timeline import (AudioClip, BLENDS, Clip, FITS, Timeline, TRANSITIONS, Transition, VIDEO_EXT, _fit,
                       read_audio_any)

MAX_FRAME = 1_000_000          # about 11 hours at 24 fps
MAX_TRACKS = 64
MAX_CLIPS = 4000
MAX_DIM = 8192
MIN_DIM = 16
MAX_UNDO = 100
MAX_AUDIO_PREVIEW_SECONDS = 300
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac") + VIDEO_EXT
STILL_FRAMES = 48              # default length of a still image or generated clip


class EditError(GraphError):
    """A rejected edit (the GUI shows the message; nothing was changed)."""


# ---------------------------------------------------------------- argument validation
def _int(op: dict, key: str, lo: int = -MAX_FRAME, hi: int = MAX_FRAME, default=None) -> int:
    v = op.get(key, default)
    if v is None:
        raise EditError(f"{op.get('op')}: {key!r} is required")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != int(v):
        raise EditError(f"{op.get('op')}: {key!r} must be a whole number")
    v = int(v)
    if not lo <= v <= hi:
        raise EditError(f"{op.get('op')}: {key!r} must be between {lo} and {hi}")
    return v


def _num(op: dict, key: str, lo: float, hi: float, default=None) -> float:
    v = op.get(key, default)
    if v is None:
        raise EditError(f"{op.get('op')}: {key!r} is required")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise EditError(f"{op.get('op')}: {key!r} must be a number")
    if not lo <= v <= hi:
        raise EditError(f"{op.get('op')}: {key!r} must be between {lo} and {hi}")
    return float(v)


def _str(op: dict, key: str, maxlen: int = 120, default=None, allow_empty=False) -> str:
    v = op.get(key, default)
    if not isinstance(v, str) or (not v and not allow_empty):
        raise EditError(f"{op.get('op')}: {key!r} must be a non-empty string")
    if len(v) > maxlen or "\x00" in v:
        raise EditError(f"{op.get('op')}: {key!r} is too long or invalid")
    return v


def _bool(op: dict, key: str) -> bool:
    v = op.get(key)
    if not isinstance(v, bool):
        raise EditError(f"{op.get('op')}: {key!r} must be true or false")
    return v


def _opacity(v):
    """A number 0..1, or {"keys": [[frame, value], ...], "interp": "linear"|"smooth"|"hold"} (frames since clip start)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isfinite(v) or not 0 <= v <= 1:
            raise EditError("opacity must be between 0 and 1")
        return float(v)
    if isinstance(v, dict) and isinstance(v.get("keys"), list) and 1 <= len(v["keys"]) <= 200:
        keys = []
        for k in v["keys"]:
            if (not isinstance(k, (list, tuple)) or len(k) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in k)):
                raise EditError("opacity keys must be [frame, value] pairs of numbers")
            if not 0 <= k[1] <= 1 or abs(k[0]) > MAX_FRAME:
                raise EditError("opacity keys must have value 0..1 and a sane frame")
            keys.append([float(k[0]), float(k[1])])
        interp = v.get("interp", "linear")
        if interp not in ("linear", "smooth", "hold"):
            raise EditError("opacity interp must be linear, smooth or hold")
        return {"keys": sorted(keys), "interp": interp}
    raise EditError("opacity must be a number 0..1 or {keys: [[frame, value], ...]}")


def _sig(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def check_timeline(tl: Timeline):
    """Reject a loaded timeline whose numbers could exhaust memory or time."""
    if not (MIN_DIM <= tl.width <= MAX_DIM and MIN_DIM <= tl.height <= MAX_DIM):
        raise EditError(f"timeline size must be {MIN_DIM}..{MAX_DIM} pixels")
    if not (1 <= tl.fps <= 240):
        raise EditError("timeline fps must be 1..240")
    if not (8000 <= tl.sample_rate <= 192000):
        raise EditError("sample rate must be 8000..192000")
    if len(tl.tracks) > MAX_TRACKS or sum(len(t.clips) for t in tl.tracks) > MAX_CLIPS:
        raise EditError("too many tracks or clips")
    for t in tl.tracks:
        for c in t.clips:
            if not (0 <= c.start <= MAX_FRAME and 1 <= c.duration <= MAX_FRAME):
                raise EditError(f"clip {c.name!r} has an out-of-range start or duration")


class TimelineEditor:
    """A Timeline plus validated edits, undo/redo and the preview helpers the GUI needs.

    ``resolve_media(name) -> absolute path`` must confine names to the work folder (the GUI passes ``Studio.safe_path``);
    ``project_graph() -> Graph`` supplies the node editor's graph so a node can be placed on the timeline as a clip."""

    def __init__(self, tl: Optional[Timeline] = None, resolve_media: Optional[Callable[[str], str]] = None,
                 project_graph: Optional[Callable[[], Graph]] = None):
        self.tl = tl or Timeline()
        self.resolve_media = resolve_media or (lambda name: (_ for _ in ()).throw(EditError("no media folder configured")))
        self.project_graph = project_graph
        self._undo: List[str] = []
        self._redo: List[str] = []
        self.rev = 0
        self._peaks: Dict[str, list] = {}
        self._src: Dict[str, object] = {}
        self._wav: Optional[Tuple[int, bytes]] = None

    # ------------------------------------------------------------ history
    def _snap(self) -> str:
        return json.dumps(self.tl.to_dict())

    def _restore(self, snap: str):
        self.tl = Timeline.from_dict(json.loads(snap))

    def apply(self, ops) -> list:
        if not isinstance(ops, list) or not 1 <= len(ops) <= 200 or not all(isinstance(o, dict) for o in ops):
            raise EditError("send 1..200 edit operations, each an object with an 'op'")
        before = self._snap()
        try:
            results = [self._do(o) for o in ops]
            check_timeline(self.tl)
        except Exception:
            self._restore(before)
            raise
        if self._snap() != before:
            self._undo.append(before)
            del self._undo[:-MAX_UNDO]
            self._redo.clear()
            self.rev += 1
        return results

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snap())
        self._restore(self._undo.pop())
        self.rev += 1
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snap())
        self._restore(self._redo.pop())
        self.rev += 1
        return True

    def replace(self, tl: Timeline):
        """Swap in a whole timeline (open / new / demo). Undoable."""
        check_timeline(tl)
        self._undo.append(self._snap())
        del self._undo[:-MAX_UNDO]
        self._redo.clear()
        self.tl = tl
        self.rev += 1

    # ------------------------------------------------------------ lookup
    def _track(self, op, kind: Optional[str] = None):
        i = _int(op, "track", 0, max(len(self.tl.tracks) - 1, 0))
        if i >= len(self.tl.tracks):
            raise EditError(f"there is no track {i}")
        t = self.tl.tracks[i]
        if kind and t.kind != kind:
            raise EditError(f"track {t.name!r} is a {t.kind} track; this needs a {kind} track")
        return i, t

    def _clip(self, op, kind: Optional[str] = None):
        ti, t = self._track(op, kind)
        n = len(t.clips)
        ci = _int(op, "clip", 0, max(n - 1, 0))
        if ci >= n:
            raise EditError(f"track {t.name!r} has no clip {ci}")
        return ti, t, ci, t.clips[ci]

    @staticmethod
    def _index(t, clip) -> int:
        return next(i for i, c in enumerate(t.clips) if c is clip)

    @staticmethod
    def _sort(t):
        t.clips.sort(key=lambda c: c.start)

    def _clip_count(self) -> int:
        return sum(len(t.clips) for t in self.tl.tracks)

    # ------------------------------------------------------------ the ops
    def _do(self, op: dict):
        name = op.get("op")
        fn = getattr(self, "_op_" + name, None) if isinstance(name, str) and name.isidentifier() else None
        if fn is None:
            raise EditError(f"unknown edit op {name!r}")
        return fn(op) or {}

    def _op_set_timeline(self, op):
        tl = self.tl
        if "name" in op:
            tl.name = _str(op, "name", 80)
        if "width" in op:
            tl.width = _int(op, "width", MIN_DIM, MAX_DIM)
        if "height" in op:
            tl.height = _int(op, "height", MIN_DIM, MAX_DIM)
        if "fps" in op:
            tl.fps = _num(op, "fps", 1, 240)

    def _op_add_track(self, op):
        kind = op.get("kind")
        if kind not in ("video", "audio"):
            raise EditError("add_track: kind must be 'video' or 'audio'")
        if len(self.tl.tracks) >= MAX_TRACKS:
            raise EditError(f"at most {MAX_TRACKS} tracks")
        t = self.tl.add_track(kind, _str(op, "name", 40) if "name" in op else None)
        return {"track": len(self.tl.tracks) - 1, "name": t.name}

    def _op_remove_track(self, op):
        i, _ = self._track(op)
        del self.tl.tracks[i]

    def _op_set_track(self, op):
        _, t = self._track(op)
        if "name" in op:
            t.name = _str(op, "name", 40)
        for k in ("mute", "solo"):
            if k in op:
                setattr(t, k, _bool(op, k))
        if "opacity" in op or "blend" in op:
            if t.kind != "video":
                raise EditError("opacity and blend belong to video tracks")
            if "opacity" in op:
                t.opacity = _num(op, "opacity", 0, 1)
            if "blend" in op:
                if op["blend"] not in BLENDS:
                    raise EditError(f"blend must be one of {BLENDS}")
                t.blend = op["blend"]
        if any(k in op for k in ("gain_db", "pan", "bus")):
            if t.kind != "audio":
                raise EditError("gain, pan and bus belong to audio tracks")
            if "gain_db" in op:
                t.gain_db = _num(op, "gain_db", -100, 24)
            if "pan" in op:
                t.pan = _num(op, "pan", -1, 1)
            if "bus" in op:
                t.bus = _str(op, "bus", 40)

    # --- clips
    def _op_add_clip(self, op):
        ti, t = self._track(op)
        if self._clip_count() >= MAX_CLIPS:
            raise EditError(f"at most {MAX_CLIPS} clips")
        src = op.get("source")
        if not isinstance(src, dict) or len(src) < 1 or not ({"file", "node"} & set(src)) or {"file", "node"} <= set(src):
            raise EditError("add_clip: source must be {file: name} or {node: id}")
        fps = self.tl.fps
        start = _int(op, "start", 0, MAX_FRAME) if "start" in op else max((c.end for c in t.clips), default=0)
        want = _int(op, "duration", 1, MAX_FRAME) if "duration" in op else None
        label = _str(op, "name", 80) if "name" in op else None
        if "file" in src:
            path = self.resolve_media(_str(src, "file", 400))
            if not os.path.isfile(path):
                raise EditError(f"no such file: {src['file']}")
            low = path.lower()
            if t.kind == "video":
                if low.endswith(VIDEO_EXT):
                    n = _probe_frames(path)
                elif low.endswith(IMAGE_EXT):
                    n = STILL_FRAMES
                else:
                    raise EditError("a video track takes video or image files")
                clip = Clip.from_file(path, start, min(want or n, MAX_FRAME))
            else:
                if not low.endswith(AUDIO_EXT):
                    raise EditError("an audio track takes audio files (or a video file's sound)")
                buf = read_audio_any(path)
                n = max(1, int(math.ceil(len(buf.samples) / buf.sr * fps)))
                self._src[path] = buf
                clip = AudioClip.from_file(path, start, min(want or n, MAX_FRAME))
            if label:
                clip.name = label
        else:
            if self.project_graph is None:
                raise EditError("no node graph is available")
            g = self.project_graph()
            nid = _str(src, "node", 120)
            if nid not in g.nodes:
                raise EditError(f"the node graph has no node {nid!r}")
            spec = get_spec(g.nodes[nid].type)
            port = src.get("port") or next(iter(spec.outputs), None)
            if port not in spec.outputs:
                raise EditError(f"node {nid!r} has no output {port!r}")
            wanted = T.IMAGE if t.kind == "video" else T.AUDIO
            if spec.outputs[port] != wanted:
                raise EditError(f"{nid}.{port} is {spec.outputs[port].value}; a {t.kind} track needs {wanted.value}")
            keep = g._upstream(nid) | {nid}
            sub = Graph.from_dict({"version": 1, "nodes": [d for d in g.to_dict()["nodes"] if d["id"] in keep]})
            probe = sub.evaluate(nid, port, Ctx(0, fps))          # fail now, not on every preview frame
            if t.kind == "video":
                clip = Clip.from_graph(sub, nid, start, want or STILL_FRAMES, name=label or nid, port=port)
            else:
                n = max(1, int(math.ceil(len(probe.samples) / probe.sr * fps)))
                clip = AudioClip.from_graph(sub, nid, start, want or n, name=label or nid)
        t.add(clip)
        self._sort(t)
        return {"track": ti, "clip": self._index(t, clip)}

    def _op_move_clip(self, op):
        ti, t, ci, c = self._clip(op)
        start = _int(op, "start", 0, MAX_FRAME)
        dest = t
        dti = ti
        if "to_track" in op and op["to_track"] != ti:
            dti, dest = self._track({"op": "move_clip", "track": op["to_track"]}, t.kind)
        c.start = start
        if dest is not t:
            t.clips.pop(ci)
            dest.clips.append(c)
        self._sort(dest)
        return {"track": dti, "clip": self._index(dest, c)}

    def _op_trim_clip(self, op):
        ti, t, ci, c = self._clip(op)
        edge = op.get("edge")
        if edge not in ("left", "right"):
            raise EditError("trim_clip: edge must be 'left' or 'right'")
        frame = _int(op, "frame", 0, MAX_FRAME)
        if edge == "right":
            if frame <= c.start:
                raise EditError("a clip must keep at least one frame")
            c.duration = frame - c.start
        else:
            if frame >= c.end:
                raise EditError("a clip must keep at least one frame")
            delta = frame - c.start
            if isinstance(c, Clip):
                new_in = c.in_point + delta * c.speed
            else:
                new_in = c.in_point + delta / self.tl.fps
            if new_in < -1e-9:
                raise EditError("cannot trim before the start of the source")
            c.in_point = max(new_in, 0.0)
            c.duration -= delta
            c.start = frame
        self._sort(t)
        return {"track": ti, "clip": self._index(t, c)}

    def _op_split_clip(self, op):
        ti, t, ci, c = self._clip(op)
        frame = _int(op, "frame", 0, MAX_FRAME)
        if not c.start < frame < c.end:
            raise EditError("the split point must be inside the clip")
        if self._clip_count() >= MAX_CLIPS:
            raise EditError(f"at most {MAX_CLIPS} clips")
        cut = frame - c.start
        graph = Graph.from_dict(c.graph.to_dict()) if c.graph is not None else None
        if isinstance(c, Clip):
            op_ = c.opacity
            if isinstance(op_, str):
                raise EditError("cannot split a clip whose opacity is an expression; set a number or keyframes first")
            if isinstance(op_, dict):
                op_ = {**op_, "keys": [[k[0] - cut, k[1]] for k in op_["keys"]]}
            b = Clip(graph, c.node, frame, c.end - frame, c.name, c.port, c.in_point + cut * c.speed, c.speed, op_,
                     c.fit, None, c.media)
        else:
            b = AudioClip(graph, c.node, c.path, frame, c.end - frame, c.name, c.in_point + cut / self.tl.fps,
                          c.gain_db, 0.0, c.fade_out, c.media)
            c.fade_out = 0.0
        c.duration = cut
        t.clips.append(b)
        self._sort(t)
        return {"track": ti, "clip": self._index(t, b), "first": self._index(t, c)}

    def _op_delete_clip(self, op):
        _, t, ci, _ = self._clip(op)
        t.clips.pop(ci)

    def _op_ripple_delete(self, op):
        _, t, ci, c = self._clip(op)
        t.clips.pop(ci)
        for o in t.clips:
            if o.start >= c.end:
                o.start -= c.duration

    def _op_set_clip(self, op):
        ti, t, ci, c = self._clip(op)
        if "name" in op:
            c.name = _str(op, "name", 80)
        if isinstance(c, Clip):
            if "opacity" in op:
                c.opacity = _opacity(op["opacity"])
            if "speed" in op:
                c.speed = _num(op, "speed", 0.05, 50)
            if "fit" in op:
                if op["fit"] not in FITS:
                    raise EditError(f"fit must be one of {FITS}")
                c.fit = op["fit"]
            bad = [k for k in ("gain_db", "fade_in", "fade_out") if k in op]
        else:
            if "gain_db" in op:
                c.gain_db = _num(op, "gain_db", -100, 24)
            for k in ("fade_in", "fade_out"):
                if k in op:
                    setattr(c, k, _num(op, k, 0, 600))
            bad = [k for k in ("opacity", "speed", "fit") if k in op]
        if bad:
            raise EditError(f"{', '.join(bad)} do not apply to a {t.kind} clip")

    def _op_set_transition(self, op):
        ti, t, ci, c = self._clip(op, "video")
        if op.get("type") is None:
            c.transition_in = None
            return {"track": ti, "clip": ci}
        kind = op["type"]
        if kind not in TRANSITIONS:
            raise EditError(f"transition must be one of {TRANSITIONS}")
        dur = _int(op, "duration", 1, MAX_FRAME, default=12)
        if dur > c.duration:
            raise EditError("a transition cannot be longer than its clip")
        prev = [o for o in t.clips if o is not c and o.start < c.start]
        if not prev:
            raise EditError("a transition needs a clip before this one on the same track")
        prev_end = max(o.end for o in prev)
        if prev_end < c.start + dur:
            if not op.get("fix"):
                raise EditError(f"the previous clip must overlap this one by {dur} frames "
                                f"(it overlaps by {max(prev_end - c.start, 0)}); drag this clip earlier or use fix")
            new_start = prev_end - dur
            if new_start < 0 or new_start <= max(o.start for o in prev):
                raise EditError(f"the previous clip is too short for a {dur}-frame transition")
            c.start = new_start
        c.transition_in = Transition(kind, dur)
        self._sort(t)
        return {"track": ti, "clip": self._index(t, c)}

    # --- clip effects (nodes inside the clip's own graph)
    def _op_add_effect(self, op):
        ti, t, ci, c = self._clip(op, "video")
        typ = _str(op, "type", 80)
        spec = get_spec(typ)
        req = [p for p in spec.inputs if p not in spec.optional]
        if len(req) != 1 or not compatible(T.IMAGE, spec.inputs[req[0]]):
            raise EditError(f"{typ} needs more than one required input or no picture input, so it cannot be a one-click effect")
        if not spec.outputs or spec.outputs[next(iter(spec.outputs))] != T.IMAGE:
            raise EditError(f"{typ} does not output a picture")
        params = op.get("params", {})
        if not isinstance(params, dict):
            raise EditError("params must be an object")
        g = c.graph
        nid = g.add(typ, **params)
        g.connect(c.node, nid, c.port, req[0])
        c.node, c.port = nid, None
        g.evaluate(c.node, None, Ctx(c.local_frame(c.start), self.tl.fps))     # fail now (rolls the batch back)
        return {"track": ti, "clip": ci, "node": nid}

    def _op_remove_effect(self, op):
        ti, t, ci, c = self._clip(op, "video")
        g = c.graph
        head = g.nodes[c.node]
        if not head.inputs:
            raise EditError("that node is the clip's source; delete the clip instead")
        want = _str(op, "node", 120)
        if want != c.node:
            raise EditError("only the last effect of the chain can be removed (remove effects from the end)")
        first_port = next(iter(get_spec(head.type).inputs))
        main = head.inputs.get(first_port) or next(iter(head.inputs.values()))
        src, sp = main
        g.remove(c.node)
        c.node = src
        c.port = sp if len(get_spec(g.nodes[src].type).outputs) > 1 else None
        keep = g._upstream(c.node) | {c.node}
        for k in [k for k in g.nodes if k not in keep]:
            g.remove(k)

    def _op_set_clip_param(self, op):
        _, t, ci, c = self._clip(op)
        if c.graph is None:
            raise EditError("this clip plays a file directly; it has no node parameters")
        nid = _str(op, "node", 120)
        name = _str(op, "name", 80)
        value = op.get("value")
        if name == "path" and isinstance(value, str):
            value = self.resolve_media(value)            # a clip can only be pointed at files in the work folder
            if not os.path.isfile(value):
                raise EditError(f"no such file: {op.get('value')}")
        c.graph.set_param(nid, name, value)

    # --- markers, master
    def _op_add_marker(self, op):
        if len(self.tl.markers) >= 500:
            raise EditError("at most 500 markers")
        self.tl.add_marker(_int(op, "frame", 0, MAX_FRAME), _str(op, "name", 60, default="", allow_empty=True),
                           _str(op, "color", 20, default="blue"))

    def _op_remove_marker(self, op):
        i = _int(op, "index", 0, max(len(self.tl.markers) - 1, 0))
        if i >= len(self.tl.markers):
            raise EditError("no such marker")
        self.tl.markers.pop(i)

    def _op_set_master(self, op):
        m = self.tl.master
        if "gain_db" in op:
            m.gain_db = _num(op, "gain_db", -60, 24)
        if "target_lufs" in op:
            m.target_lufs = None if op["target_lufs"] is None else _num(op, "target_lufs", -70, 0)
        if "ceiling_db" in op:
            m.ceiling_db = None if op["ceiling_db"] is None else _num(op, "ceiling_db", -30, 0)

    # ------------------------------------------------------------ views for the UI
    def view(self) -> dict:
        tl = self.tl
        tracks = []
        for t in tl.tracks:
            clips = []
            for c in t.clips:
                d = {"name": c.name, "start": c.start, "duration": c.duration, "in_point": c.in_point,
                     "media": os.path.basename(c.media) if c.media else None}
                if isinstance(c, Clip):
                    d.update(opacity=c.opacity, speed=c.speed, fit=c.fit,
                             transition=vars(c.transition_in) if c.transition_in else None,
                             sig=_sig(c.graph.to_dict(), c.node, c.port, c.in_point, c.fit, tl.width, tl.height),
                             has_graph=True)
                else:
                    d.update(gain_db=c.gain_db, fade_in=c.fade_in, fade_out=c.fade_out, has_graph=c.graph is not None,
                             sig=_sig(c.path, c.graph.to_dict() if c.graph else None, c.node, c.in_point, c.duration,
                                      c.gain_db, c.fade_in, c.fade_out, tl.fps))
                clips.append(d)
            tracks.append({"kind": t.kind, "name": t.name, "mute": t.mute, "solo": t.solo, "gain_db": t.gain_db,
                           "pan": t.pan, "bus": t.bus, "blend": t.blend, "opacity": t.opacity, "clips": clips})
        return {"name": tl.name, "width": tl.width, "height": tl.height, "fps": tl.fps, "duration": tl.duration,
                "markers": tl.markers, "master": vars(tl.master), "tracks": tracks, "rev": self.rev,
                "can_undo": bool(self._undo), "can_redo": bool(self._redo)}

    def clip_graph(self, op) -> dict:
        _, t, ci, c = self._clip(op)
        if c.graph is None:
            return {"graph": None, "node": None, "order": []}
        return {"graph": c.graph.to_dict(), "node": c.node, "order": c.graph.topological_order()}

    def frame(self, t: int, maxw: int) -> np.ndarray:
        """Linear RGBA of timeline frame ``t``; ``maxw`` below the timeline width renders a proxy at that width."""
        tl = self.tl
        t = max(0, min(int(t), MAX_FRAME))
        w = max(MIN_DIM, min(int(maxw), tl.width))
        if w < tl.width:
            tl = copy.copy(tl)
            tl.width, tl.height = w, max(MIN_DIM, int(round(self.tl.height * w / self.tl.width)))
        return tl.render_frame(t)

    def thumb(self, op) -> np.ndarray:
        _, t, ci, c = self._clip(op, "video")
        img = c.graph.evaluate(c.node, c.port, Ctx(c.local_frame(c.start), self.tl.fps))
        img = _fit(np.asarray(img, np.float32), self.tl.width, self.tl.height, c.fit).copy()
        img[..., :3] *= img[..., 3:4]                     # over black, so a transparent title does not show its hidden colour
        img[..., 3] = 1.0
        return img

    def peaks(self, op, n: int = 200) -> list:
        """Loudness envelope (0..1) of an audio clip as it will sound (gain and fades included)."""
        _, t, ci, c = self._clip(op, "audio")
        n = max(16, min(int(n), 2000))
        key = _sig(c.path, c.graph.to_dict() if c.graph else None, c.node, c.in_point, c.duration, c.gain_db,
                   c.fade_in, c.fade_out, self.tl.fps, n)
        if key not in self._peaks:
            if c.path and c.path not in self._src:
                self._src[c.path] = read_audio_any(c.path)
                while len(self._src) > 8:
                    self._src.pop(next(iter(self._src)))
            cache = {c.path: self._src[c.path]} if c.path else {}
            seg = self.tl._render_clip(c, 4000, cache)
            m = np.abs(seg).max(1) if len(seg) else np.zeros(1, np.float32)
            idx = np.linspace(0, len(m), n + 1).astype(int)
            out = [round(float(m[a:max(b, a + 1)].max()), 3) for a, b in zip(idx[:-1], idx[1:])]
            self._peaks[key] = out
            while len(self._peaks) > 256:
                self._peaks.pop(next(iter(self._peaks)))
        return self._peaks[key]

    def mix_wav(self) -> bytes:
        """The whole timeline's audio mix as a 16-bit WAV (cached until the next edit)."""
        if self._wav and self._wav[0] == self.rev:
            return self._wav[1]
        tl = self.tl
        if not any(t.clips for t in tl.audio_tracks()):
            raise EditError("the timeline has no audio clips")
        if tl.seconds(tl.duration) > MAX_AUDIO_PREVIEW_SECONDS:
            raise EditError(f"audio preview is limited to {MAX_AUDIO_PREVIEW_SECONDS} s; render to a file instead")
        mix = tl.render_audio()
        pcm = (np.clip(mix.samples, -1, 1) * 32767).astype(np.int16)
        bio = io.BytesIO()
        wavfile.write(bio, mix.sr, pcm)
        self._wav = (self.rev, bio.getvalue())
        return self._wav[1]


def _probe_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise EditError(f"cannot open video {os.path.basename(path)}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n < 1:
        raise EditError("could not tell how many frames the video has")
    return n
