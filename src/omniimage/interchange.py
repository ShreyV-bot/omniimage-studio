"""Interchange with other editors: CMX3600 EDL and OpenTimelineIO (.otio JSON) export.

EDL carries cuts and dissolves of ONE video track (plus audio events if requested). OTIO carries all tracks, gaps,
dissolves, speed changes and markers; Resolve, Premiere, Final Cut (via adapters) and Blender's VSE can read it.
Clips made from node graphs (not files) are exported with a missing media reference so the editorial structure survives.
"""
from __future__ import annotations
import json
import os
from typing import List

from .timeline import Timeline, Clip, AudioClip


def timecode(frame: int, fps: float, base_hours: int = 0) -> str:
    n = int(round(fps))
    f = int(round(frame))
    s, ff = divmod(f, n)
    m, ss = divmod(s, 60)
    h, mm = divmod(m, 60)
    return f"{(h + base_hours) % 24:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _reel(c) -> str:
    return "AX"


def export_edl(tl: Timeline, path: str, video_track: int = 0, include_audio: bool = False) -> str:
    """Write a CMX3600 EDL (non-drop-frame, record timecode starting 01:00:00:00)."""
    vt = tl.video_tracks()
    lines: List[str] = [f"TITLE: {tl.name}", "FCM: NON-DROP FRAME", ""]
    n = 0

    def ev(ch, kind, sin, sout, rin, rout, dur=None, name=""):
        nonlocal n
        n += 1
        trans = f"D    {dur:03d}" if kind == "D" else "C       "
        lines.append(f"{n:03d}  {'AX':<8} {ch:<4} {trans:<4} {timecode(sin, tl.fps)} {timecode(sout, tl.fps)} "
                     f"{timecode(rin, tl.fps, 1)} {timecode(rout, tl.fps, 1)}")
        if name:
            lines.append(f"* FROM CLIP NAME: {name}")

    if vt:
        clips = sorted(vt[video_track].clips, key=lambda c: c.start)
        for i, c in enumerate(clips):
            tr = c.transition_in if i > 0 else None
            src_in = c.in_point
            src_out = c.in_point + c.duration * c.speed
            if tr:
                prev = clips[i - 1]
                at = prev.in_point + (c.start - prev.start) * prev.speed
                ev("V", "C", at, at, c.start, c.start, name=prev.name)
                ev("V", "D", src_in, c.in_point + tr.duration * c.speed, c.start, c.start + tr.duration, tr.duration, c.name)
                if c.duration > tr.duration:
                    ev("V", "C", c.in_point + tr.duration * c.speed, src_out, c.start + tr.duration, c.end, name=c.name)
            else:
                ev("V", "C", src_in, src_out, c.start, c.end, name=c.name)
            if c.speed != 1.0:
                lines.append(f"M2   AX       {c.speed * tl.fps:.1f}   {timecode(c.in_point, tl.fps)}")
    if include_audio:
        for at in tl.audio_tracks():
            for c in sorted(at.clips, key=lambda c: c.start):
                s_in = int(round(c.in_point * tl.fps))
                ev("AA", "C", s_in, s_in + c.duration, c.start, c.end, name=c.name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ OpenTimelineIO
def _rt(v, rate):
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": float(v)}


def _tr(start, dur, rate):
    return {"OTIO_SCHEMA": "TimeRange.1", "duration": _rt(dur, rate), "start_time": _rt(start, rate)}


def _otio_clip(c, rate, src_start, dur, kind):
    ref = ({"OTIO_SCHEMA": "ExternalReference.1", "target_url": os.path.abspath(c.media), "available_range": None,
            "metadata": {}} if c.media else {"OTIO_SCHEMA": "MissingReference.1", "name": "", "available_range": None,
                                              "metadata": {}})
    effects = []
    speed = getattr(c, "speed", 1.0)
    if speed != 1.0:
        effects.append({"OTIO_SCHEMA": "LinearTimeWarp.1", "name": "", "effect_name": "LinearTimeWarp",
                        "time_scalar": speed, "metadata": {}})
    return {"OTIO_SCHEMA": "Clip.1", "name": c.name, "source_range": _tr(src_start, dur * speed, rate),
            "media_reference": ref, "effects": effects, "markers": [], "metadata": {}, "enabled": True}


def _otio_track(track, tl, kind):
    rate = tl.fps
    children, cursor = [], 0
    clips = sorted(track.clips, key=lambda c: c.start)
    for i, c in enumerate(clips):
        start, dur, src = c.start, c.duration, c.in_point if kind == "Video" else int(round(c.in_point * rate))
        pre_tr = None
        if kind == "Video" and i > 0 and c.transition_in and children:
            tr = c.transition_in
            cut = c.start + tr.duration // 2
            prev = children[-1]
            if prev["OTIO_SCHEMA"] == "Clip.1":
                cut_in = cut - c.start
                # trim the outgoing clip to the cut point
                over = cursor - cut
                if over > 0:
                    prev["source_range"]["duration"]["value"] -= over * clips[i - 1].speed
                    cursor = cut
                children.append({"OTIO_SCHEMA": "Transition.1", "name": "", "transition_type": "SMPTE_Dissolve" if tr.type == "dissolve" else "Custom_Transition",
                                 "in_offset": _rt(cut_in, rate), "out_offset": _rt(tr.duration - cut_in, rate), "metadata": {}})
                src += cut_in * c.speed
                start, dur = cut, c.end - cut
        if start > cursor:
            children.append({"OTIO_SCHEMA": "Gap.1", "name": "", "source_range": _tr(0, start - cursor, rate),
                             "effects": [], "markers": [], "metadata": {}, "enabled": True})
        children.append(_otio_clip(c, rate, src, dur, kind))
        cursor = start + dur
    return {"OTIO_SCHEMA": "Track.1", "name": track.name, "kind": kind, "children": children, "source_range": None,
            "effects": [], "markers": [], "metadata": {}, "enabled": True}


def export_otio(tl: Timeline, path: str) -> str:
    """Write an OpenTimelineIO JSON file (all video and audio tracks, gaps, dissolves, speed, markers)."""
    tracks = [_otio_track(t, tl, "Video" if t.kind == "video" else "Audio") for t in tl.tracks]
    markers = [{"OTIO_SCHEMA": "Marker.2", "name": m["name"], "color": m["color"].upper(), "comment": "",
                "marked_range": _tr(m["frame"], 0, tl.fps), "metadata": {}} for m in tl.markers]
    doc = {"OTIO_SCHEMA": "Timeline.1", "name": tl.name, "global_start_time": None, "metadata": {},
           "tracks": {"OTIO_SCHEMA": "Stack.1", "name": "tracks", "children": tracks, "source_range": None,
                      "effects": [], "markers": markers, "metadata": {}, "enabled": True}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path
