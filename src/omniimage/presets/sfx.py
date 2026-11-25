"""Procedural sound-effect recipes built from audio nodes. Deterministic: same recipe, same samples."""
from __future__ import annotations
from typing import Callable, Dict, Tuple

from ..core.graph import Graph


def _add(g, last, node_type, /, **params):
    n = g.add(node_type, **params)
    g.connect(last, n)
    return n


def _mix(g, a, b, **params):
    m = g.add("mix_audio", **params)
    g.connect(a, m, dst_port="a")
    g.connect(b, m, dst_port="b")
    return m


def whoosh(g, duration=0.9):
    n = g.add("noise_audio", color="pink", duration=duration, seed=11)
    n = _add(g, n, "filter_sweep", type="bandpass", cutoff_start=250, cutoff_end=5200, q=1.6)
    n = _add(g, n, "envelope", attack=duration * 0.45, decay=duration * 0.1, sustain=0.7, release=duration * 0.45, curve="exp")
    n = _add(g, n, "reverb", rt60=0.8, mix=0.15)
    return _add(g, n, "normalize", peak_db=-3)


def laser(g, duration=0.35):
    n = g.add("osc", waveform="saw", freq_start=2400, freq_end=160, glide="exp", duration=duration)
    n = _add(g, n, "envelope", attack=0.002, decay=0.05, sustain=0.6, release=0.15)
    n = _add(g, n, "filter", type="lowpass", cutoff=6500)
    n = _add(g, n, "delay", time_ms=55, feedback=0.3, mix=0.3)
    return _add(g, n, "normalize", peak_db=-3)


def explosion(g, duration=2.2):
    nz = g.add("noise_audio", color="brown", duration=duration, seed=5, gain=0.9)
    nz = _add(g, nz, "filter_sweep", type="lowpass", cutoff_start=7000, cutoff_end=180, q=0.8)
    nz = _add(g, nz, "envelope", attack=0.004, decay=0.5, sustain=0.3, release=duration * 0.5)
    sub = g.add("osc", waveform="sine", freq_start=70, freq_end=28, glide="exp", duration=1.6)
    sub = _add(g, sub, "envelope", attack=0.003, decay=0.4, sustain=0.2, release=0.8)
    m = _mix(g, nz, sub, gain_b_db=3)
    m = _add(g, m, "distortion", drive=2.5, shape="tanh", mix=0.35)
    m = _add(g, m, "compressor", threshold_db=-20, ratio=4, makeup_db=4)
    m = _add(g, m, "reverb", rt60=1.6, mix=0.25)
    return _add(g, m, "normalize", peak_db=-1)


def coin(g):
    a = g.add("osc", waveform="square", freq_start=988, freq_end=988, duration=0.08, gain=0.5)
    b = g.add("osc", waveform="square", freq_start=1319, freq_end=1319, duration=0.45, gain=0.5)
    m = _mix(g, a, b, offset_b=0.08)
    m = _add(g, m, "envelope", attack=0.002, decay=0.15, sustain=0.5, release=0.3)
    return _add(g, m, "normalize", peak_db=-4)


def riser(g, duration=3.0):
    nz = g.add("noise_audio", color="white", duration=duration, seed=2, gain=0.6)
    nz = _add(g, nz, "filter_sweep", type="bandpass", cutoff_start=200, cutoff_end=9000, q=3.0)
    saw = g.add("osc", waveform="saw", freq_start=80, freq_end=900, glide="exp", duration=duration, gain=0.35)
    saw = _add(g, saw, "filter", type="lowpass", cutoff=3500)
    m = _mix(g, nz, saw)
    m = _add(g, m, "envelope", attack=duration * 0.95, decay=0.01, sustain=1.0, release=0.05, curve="exp")
    m = _add(g, m, "reverb", rt60=1.2, mix=0.2)
    return _add(g, m, "normalize", peak_db=-3)


def impact(g):
    sub = g.add("osc", waveform="sine", freq_start=130, freq_end=34, glide="exp", duration=1.2)
    sub = _add(g, sub, "envelope", attack=0.001, decay=0.35, sustain=0.0, release=0.4)
    click = g.add("noise_audio", color="white", duration=0.12, seed=9, gain=0.8)
    click = _add(g, click, "filter", type="lowpass", cutoff=3200)
    click = _add(g, click, "envelope", attack=0.001, decay=0.05, sustain=0.0, release=0.05)
    m = _mix(g, sub, click, gain_b_db=-6)
    m = _add(g, m, "distortion", drive=1.8, shape="tanh", mix=0.3)
    m = _add(g, m, "reverb", rt60=2.0, mix=0.3)
    return _add(g, m, "normalize", peak_db=-1)


def wind(g, duration=4.0):
    n = g.add("noise_audio", color="pink", duration=duration, seed=21)
    n = _add(g, n, "filter_sweep", type="bandpass", cutoff_start=300, cutoff_end=900, q=0.8)
    n = _add(g, n, "tremolo", rate_hz=0.6, depth=0.4)
    n = _add(g, n, "fade", fade_in=0.8, fade_out=1.0, curve="cosine")
    return _add(g, n, "normalize", peak_db=-6)


SFX: Dict[str, Tuple[str, Callable]] = {
    "whoosh": ("Sweeping air whoosh (transition)", whoosh),
    "laser": ("Descending sci-fi laser zap", laser),
    "explosion": ("Low boom with debris tail", explosion),
    "coin": ("Two-note arcade pickup", coin),
    "riser": ("Building tension riser", riser),
    "impact": ("Cinematic hit with sub drop", impact),
    "wind": ("Gentle wind ambience", wind),
}


def build_sfx(name: str, **kw) -> Tuple[Graph, str]:
    if name not in SFX:
        raise KeyError(f"unknown sfx {name!r}. Available: {', '.join(sorted(SFX))}")
    g = Graph()
    out = SFX[name][1](g, **kw)
    return g, out
