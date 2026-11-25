"""A ready-made sample timeline (two scenes with a dissolve, a title, SFX and ducked music) for trying the editor."""
from __future__ import annotations
from ..core.graph import Graph
from ..timeline import Timeline, Clip, AudioClip, Transition, Bus, Duck
from .vfx import demo_scene, neon_sign
from .sfx import build_sfx


def demo_timeline(width: int = 640, height: int = 360, fps: float = 24.0) -> Timeline:
    tl = Timeline(width, height, fps, 48000, "OmniImage demo")
    v1, v2 = tl.add_track("video"), tl.add_track("video")
    g1, out1 = demo_scene(width, height)
    n = g1.add("film_grain", amount=0.08); g1.connect(out1, n)
    v1.add(Clip.from_graph(g1, n, 0, 36, name="Scene A"))
    g2, out2 = neon_sign("OMNI", width=width, height=height, size=int(height * 0.35))
    v1.add(Clip.from_graph(g2, out2, 30, 48, name="Neon", transition_in=Transition("dissolve", 6)))
    g3 = Graph()
    t = g3.add("text", width=width, height=height, text="OmniImage Studio", size=int(height * 0.1), y=0.88)
    v2.add(Clip.from_graph(g3, t, 6, 60, name="Title", opacity={"keys": [[0, 0], [8, 1], [52, 1], [60, 0]]}))
    a1, a2, a3 = tl.add_track("audio", bus="sfx"), tl.add_track("audio", bus="music"), tl.add_track("audio", bus="sfx")
    sg, so = build_sfx("riser")
    a1.add(AudioClip.from_graph(sg, so, 0, 36, name="Riser", fade_out=0.2))
    sg2, so2 = build_sfx("impact")
    a3.add(AudioClip.from_graph(sg2, so2, 36, 24, name="Impact"))
    pg = Graph()
    pad = pg.add("osc", waveform="saw", freq_start=110, freq_end=110, duration=3.5, gain=0.25)
    lp = pg.add("filter", type="lowpass", cutoff=900, order=2)
    pg.connect(pad, lp)
    a2.add(AudioClip.from_graph(pg, lp, 0, 84, name="Pad", gain_db=-6, fade_in=0.5, fade_out=0.8))
    tl.buses["music"] = Bus(-4.0, Duck(key="sfx", amount_db=-9, threshold_db=-45))
    tl.master.target_lufs = -16.0
    tl.add_marker(30, "dissolve")
    return tl
