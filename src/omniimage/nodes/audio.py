"""Audio nodes: synthesis, envelopes, filters, dynamics, time/pitch and space.

Everything is plain NumPy/SciPy so the same graph renders identically on any machine. Oscillators are naive
(not band-limited): fine for SFX, use a high sample rate for clean saws/squares."""
from __future__ import annotations
import numpy as np
from scipy import signal

from ..core.registry import node
from ..core.params import P, ParamError
from ..core.ports import PortType as T
from ..core.media import AudioBuf

AA = dict(inputs={"audio": T.AUDIO}, outputs={"audio": T.AUDIO})
SR = P("int", 44100, 8000, 192000, animatable=False)


def _db(x):
    return 10.0 ** (x / 20.0)


def _like(a: AudioBuf, samples) -> AudioBuf:
    return AudioBuf(samples, a.sr)


# ------------------------------------------------------------------ sources
@node("osc", "audio_source", "Oscillator with a pitch glide from freq_start to freq_end (exponential or linear).",
      outputs={"audio": T.AUDIO},
      params={"waveform": P("choice", "sine", choices=("sine", "square", "saw", "triangle")),
              "freq_start": P("float", 440.0, 1, 20000), "freq_end": P("float", 440.0, 1, 20000),
              "glide": P("choice", "exp", choices=("exp", "linear")), "duration": P("float", 1.0, 0.01, 120),
              "gain": P("float", 0.8, 0, 4), "sr": SR})
def osc(inp, p, ctx):
    sr, n = p["sr"], int(p["duration"] * p["sr"])
    t = np.linspace(0, 1, n, endpoint=False)
    f0, f1 = p["freq_start"], p["freq_end"]
    f = f0 * (f1 / f0) ** t if p["glide"] == "exp" else f0 + (f1 - f0) * t
    ph = np.cumsum(2 * np.pi * f / sr)
    w = p["waveform"]
    if w == "sine":
        y = np.sin(ph)
    elif w == "square":
        y = np.sign(np.sin(ph))
    elif w == "saw":
        y = 2 * ((ph / (2 * np.pi)) % 1.0) - 1
    else:
        y = 2 * np.abs(2 * ((ph / (2 * np.pi)) % 1.0) - 1) - 1
    return AudioBuf((y * p["gain"]).astype(np.float32), sr)


@node("noise_audio", "audio_source", "White, pink or brown noise (seeded, deterministic).", outputs={"audio": T.AUDIO},
      params={"color": P("choice", "white", choices=("white", "pink", "brown")), "duration": P("float", 1.0, 0.01, 120),
              "gain": P("float", 0.5, 0, 4), "seed": P("int", 1, 0, 2 ** 31 - 1), "sr": SR})
def noise_audio(inp, p, ctx):
    n = int(p["duration"] * p["sr"])
    rng = np.random.default_rng(p["seed"])
    w = rng.standard_normal(n)
    if p["color"] != "white":
        spec = np.fft.rfft(w)
        f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
        spec = spec / (np.sqrt(f) if p["color"] == "pink" else f)
        w = np.fft.irfft(spec, n)
    w = w / (np.max(np.abs(w)) + 1e-9)
    return AudioBuf((w * p["gain"]).astype(np.float32), p["sr"])


# ------------------------------------------------------------------ shaping
@node("envelope", "audio_dynamics", "ADSR amplitude envelope over the whole clip (times in seconds).", **AA,
      params={"attack": P("float", 0.01, 0, 30), "decay": P("float", 0.1, 0, 30), "sustain": P("float", 0.7, 0, 1),
              "release": P("float", 0.2, 0, 30), "curve": P("choice", "exp", choices=("linear", "exp"))})
def envelope(inp, p, ctx):
    a = inp["audio"]
    n, sr = a.samples.shape[0], a.sr
    A, D, R = int(p["attack"] * sr), int(p["decay"] * sr), int(p["release"] * sr)
    A = min(A, n); R = min(R, max(n - A, 0)); D = min(D, max(n - A - R, 0))
    S = n - A - D - R
    k = 4.0 if p["curve"] == "exp" else 0.0

    def seg(m, v0, v1):
        if m <= 0:
            return np.zeros(0)
        u = np.linspace(0, 1, m, endpoint=False)
        if k:
            u = (np.exp(k * u) - 1) / (np.exp(k) - 1) if v1 > v0 else 1 - (np.exp(k * (1 - u)) - 1) / (np.exp(k) - 1)
        return v0 + (v1 - v0) * u

    env = np.concatenate([seg(A, 0, 1), seg(D, 1, p["sustain"]), np.full(max(S, 0), p["sustain"]),
                          seg(R, p["sustain"], 0)])
    env = np.pad(env, (0, n - env.size))[:n]
    return _like(a, a.samples * env[:, None].astype(np.float32))


@node("gain", "audio_dynamics", "Change level in dB.", **AA, params={"db": P("float", 0.0, -96, 48)})
def gain(inp, p, ctx):
    a = inp["audio"]
    return _like(a, a.samples * np.float32(_db(p["db"])))


@node("fade", "audio_dynamics", "Fade in and/or out (seconds).", **AA,
      params={"fade_in": P("float", 0.0, 0, 60), "fade_out": P("float", 0.0, 0, 60),
              "curve": P("choice", "linear", choices=("linear", "cosine"))})
def fade(inp, p, ctx):
    a = inp["audio"]
    n, sr = a.samples.shape[0], a.sr
    env = np.ones(n, np.float32)
    fi, fo = min(int(p["fade_in"] * sr), n), min(int(p["fade_out"] * sr), n)
    shape = (lambda u: 0.5 - 0.5 * np.cos(np.pi * u)) if p["curve"] == "cosine" else (lambda u: u)
    if fi:
        env[:fi] *= shape(np.linspace(0, 1, fi, endpoint=False)).astype(np.float32)
    if fo:
        env[n - fo:] *= shape(np.linspace(1, 0, fo, endpoint=True)).astype(np.float32)
    return _like(a, a.samples * env[:, None])


@node("normalize", "audio_dynamics", "Scale so the loudest peak hits a target dBFS.", **AA,
      params={"peak_db": P("float", -1.0, -60, 0)})
def normalize(inp, p, ctx):
    a = inp["audio"]
    pk = float(np.max(np.abs(a.samples))) or 1.0
    return _like(a, a.samples * np.float32(_db(p["peak_db"]) / pk))


# ------------------------------------------------------------------ filters
def _rbj(kind, f, q, sr, gain_db=0.0):
    f = float(np.clip(f, 10, sr * 0.49))
    w0 = 2 * np.pi * f / sr
    alpha = np.sin(w0) / (2 * max(q, 0.05))
    c = np.cos(w0)
    if kind == "lowpass":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]; a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "notch":
        b = [1, -2 * c, 1]; a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "peak":
        A = 10 ** (gain_db / 40)
        b = [1 + alpha * A, -2 * c, 1 - alpha * A]; a = [1 + alpha / A, -2 * c, 1 - alpha / A]
    else:
        raise ParamError(f"unknown filter {kind}")
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])


@node("filter", "audio_filter", "Biquad filter (low/high/band-pass, notch). 'order' cascades the section for a steeper slope.",
      **AA, params={"type": P("choice", "lowpass", choices=("lowpass", "highpass", "bandpass", "notch")),
                    "cutoff": P("float", 1000.0, 20, 20000), "q": P("float", 0.707, 0.1, 30), "order": P("int", 1, 1, 4)})
def filter_(inp, p, ctx):
    a = inp["audio"]
    sos = np.tile(_rbj(p["type"], p["cutoff"], p["q"], a.sr), (p["order"], 1))
    return _like(a, signal.sosfilt(sos, a.samples, axis=0).astype(np.float32))


@node("filter_sweep", "audio_filter", "Filter whose cutoff glides over the clip (whooshes, risers, wah).", **AA,
      params={"type": P("choice", "bandpass", choices=("lowpass", "highpass", "bandpass")),
              "cutoff_start": P("float", 300.0, 20, 20000), "cutoff_end": P("float", 4000.0, 20, 20000),
              "q": P("float", 2.0, 0.1, 30), "block": P("int", 128, 16, 4096)})
def filter_sweep(inp, p, ctx):
    a = inp["audio"]
    x = a.samples
    n = x.shape[0]
    out = np.zeros_like(x)
    zi = np.zeros((1, 2, x.shape[1]))
    nb = max(1, int(np.ceil(n / p["block"])))
    for i in range(nb):
        s, e = i * p["block"], min(n, (i + 1) * p["block"])
        fc = p["cutoff_start"] * (p["cutoff_end"] / p["cutoff_start"]) ** (i / max(nb - 1, 1))
        sos = _rbj(p["type"], fc, p["q"], a.sr)
        y, zi = signal.sosfilt(sos, x[s:e], axis=0, zi=zi)
        out[s:e] = y
    return _like(a, out.astype(np.float32))


@node("eq_peak", "audio_filter", "Parametric peaking EQ band.", **AA,
      params={"freq": P("float", 1000.0, 20, 20000), "gain_db": P("float", 3.0, -24, 24), "q": P("float", 1.0, 0.1, 20)})
def eq_peak(inp, p, ctx):
    a = inp["audio"]
    return _like(a, signal.sosfilt(_rbj("peak", p["freq"], p["q"], a.sr, p["gain_db"]), a.samples, axis=0).astype(np.float32))


# ------------------------------------------------------------------ dynamics / distortion
@node("compressor", "audio_dynamics", "Feed-forward compressor with attack/release smoothing.", **AA,
      params={"threshold_db": P("float", -18.0, -80, 0), "ratio": P("float", 4.0, 1, 40), "attack_ms": P("float", 5.0, 0.1, 500),
              "release_ms": P("float", 80.0, 1, 5000), "makeup_db": P("float", 0.0, -24, 24)})
def compressor(inp, p, ctx):
    a = inp["audio"]
    x = a.samples
    lvl = np.max(np.abs(x), axis=1) + 1e-9
    db = 20 * np.log10(lvl)
    over = np.maximum(db - p["threshold_db"], 0)
    target = -over * (1 - 1 / p["ratio"])          # desired gain change in dB (<= 0)
    step = 16                                       # control rate: sr/16
    tc = target[::step]
    ca = np.exp(-1.0 / max(p["attack_ms"] * 1e-3 * a.sr / step, 1e-3))
    cr = np.exp(-1.0 / max(p["release_ms"] * 1e-3 * a.sr / step, 1e-3))
    g = np.zeros_like(tc)
    prev = 0.0
    for i, t in enumerate(tc):
        c = ca if t < prev else cr
        prev = c * prev + (1 - c) * t
        g[i] = prev
    gain_db = np.interp(np.arange(x.shape[0]), np.arange(len(g)) * step, g)
    return _like(a, x * (10 ** ((gain_db + p["makeup_db"]) / 20))[:, None].astype(np.float32))


@node("soft_limiter", "audio_dynamics", "Soft-clip limiter that keeps peaks under a ceiling.", **AA,
      params={"ceiling_db": P("float", -1.0, -24, 0)})
def soft_limiter(inp, p, ctx):
    a = inp["audio"]
    c = _db(p["ceiling_db"])
    return _like(a, (c * np.tanh(a.samples / c)).astype(np.float32))


@node("distortion", "audio_fx", "Waveshaping distortion.", **AA,
      params={"drive": P("float", 4.0, 0.1, 100), "shape": P("choice", "tanh", choices=("tanh", "hard", "fold")),
              "mix": P("float", 1.0, 0, 1)})
def distortion(inp, p, ctx):
    a = inp["audio"]
    x = a.samples * p["drive"]
    if p["shape"] == "tanh":
        y = np.tanh(x)
    elif p["shape"] == "hard":
        y = np.clip(x, -1, 1)
    else:
        y = np.sin(np.clip(x, -50, 50) * np.pi / 2)
    y = y / max(np.max(np.abs(y)), 1e-6) * min(1.0, np.max(np.abs(a.samples)) or 1.0)
    return _like(a, (a.samples * (1 - p["mix"]) + y * p["mix"]).astype(np.float32))


@node("bitcrush", "audio_fx", "Lo-fi: reduce bit depth and sample rate.", **AA,
      params={"bits": P("int", 8, 1, 16), "downsample": P("int", 4, 1, 64)})
def bitcrush(inp, p, ctx):
    a = inp["audio"]
    x = a.samples
    if p["downsample"] > 1:
        x = np.repeat(x[::p["downsample"]], p["downsample"], axis=0)[: a.samples.shape[0]]
    q = 2.0 ** (p["bits"] - 1)
    return _like(a, (np.round(x * q) / q).astype(np.float32))


@node("tremolo", "audio_fx", "Amplitude modulation.", **AA,
      params={"rate_hz": P("float", 5.0, 0.01, 100), "depth": P("float", 0.5, 0, 1)})
def tremolo(inp, p, ctx):
    a = inp["audio"]
    t = np.arange(a.samples.shape[0]) / a.sr
    m = 1 - p["depth"] * 0.5 * (1 + np.sin(2 * np.pi * p["rate_hz"] * t))
    return _like(a, a.samples * m[:, None].astype(np.float32))


# ------------------------------------------------------------------ time / space
@node("delay", "audio_fx", "Echo with feedback. The clip is extended so the tail is not cut off.", **AA,
      params={"time_ms": P("float", 250.0, 1, 4000), "feedback": P("float", 0.4, 0, 0.95), "mix": P("float", 0.5, 0, 1)})
def delay(inp, p, ctx):
    a = inp["audio"]
    d = max(1, int(p["time_ms"] * 1e-3 * a.sr))
    fb = p["feedback"]
    tail = int(min(5.0, d / a.sr * (np.log(1e-3) / np.log(fb) if fb > 0.01 else 1)) * a.sr)
    x = np.pad(a.samples, ((0, tail), (0, 0)))
    b = np.zeros(d + 1); b[d] = 1.0
    den = np.zeros(d + 1); den[0] = 1.0; den[d] = -fb
    wet = signal.lfilter(b, den, x, axis=0)
    return _like(a, (x + wet * p["mix"]).astype(np.float32))


@node("reverb", "audio_fx", "Schroeder reverb (4 combs + 2 all-passes). rt60 = seconds to decay by 60 dB.", **AA,
      params={"rt60": P("float", 1.5, 0.1, 8), "mix": P("float", 0.3, 0, 1), "predelay_ms": P("float", 10.0, 0, 200)})
def reverb(inp, p, ctx):
    a = inp["audio"]
    sr = a.sr
    tail = int(min(p["rt60"], 6.0) * sr)
    x = np.pad(a.samples, ((int(p["predelay_ms"] * 1e-3 * sr), tail), (0, 0)))
    chans = []
    for ch in range(x.shape[1]):
        spread = 1 + 0.013 * ch
        acc = np.zeros(x.shape[0])
        for dl in (1557, 1617, 1491, 1422):
            d = max(1, int(dl * spread * sr / 44100))
            g = 10 ** (-3 * d / (sr * p["rt60"]))
            b = np.zeros(d + 1); b[d] = 1.0
            den = np.zeros(d + 1); den[0] = 1; den[d] = -g
            acc += signal.lfilter(b, den, x[:, ch])
        acc *= 0.25
        for dl, g in ((225, 0.5), (556, 0.5)):
            d = max(1, int(dl * spread * sr / 44100))
            b = np.zeros(d + 1); b[0] = -g; b[d] = 1.0
            den = np.zeros(d + 1); den[0] = 1; den[d] = -g
            acc = signal.lfilter(b, den, acc)
        chans.append(acc)
    wet = np.stack(chans, 1)
    dry = np.pad(a.samples, ((int(p["predelay_ms"] * 1e-3 * sr), tail), (0, 0)))
    return _like(a, (dry * (1 - p["mix"]) + wet * p["mix"] * 1.5).astype(np.float32))


@node("pitch_speed", "audio_fx", "Change playback speed/pitch together by semitones (duration changes accordingly).", **AA,
      params={"semitones": P("float", 0.0, -36, 36)})
def pitch_speed(inp, p, ctx):
    a = inp["audio"]
    ratio = 2.0 ** (p["semitones"] / 12.0)
    n = a.samples.shape[0]
    new_n = max(1, int(n / ratio))
    pos = np.linspace(0, n - 1, new_n)
    out = np.stack([np.interp(pos, np.arange(n), a.samples[:, c]) for c in range(a.channels)], 1)
    return _like(a, out.astype(np.float32))


@node("reverse_audio", "audio_fx", "Play backwards.", **AA)
def reverse_audio(inp, p, ctx):
    a = inp["audio"]
    return _like(a, a.samples[::-1].copy())


@node("trim_audio", "audio_fx", "Keep a section (seconds).", **AA,
      params={"start": P("float", 0.0, 0, 3600), "duration": P("float", 1.0, 0.001, 3600)})
def trim_audio(inp, p, ctx):
    a = inp["audio"]
    s = int(p["start"] * a.sr)
    return _like(a, a.samples[s:s + int(p["duration"] * a.sr)].copy())


@node("pan", "audio_fx", "Place in the stereo field (mono is expanded to stereo).", **AA, params={"pan": P("float", 0.0, -1, 1)})
def pan(inp, p, ctx):
    a = inp["audio"]
    m = a.samples.mean(1, keepdims=True) if a.channels != 2 else None
    x = a.samples if m is None else np.repeat(m, 2, 1)
    ang = (p["pan"] + 1) * np.pi / 4
    return _like(a, np.stack([x[:, 0] * np.cos(ang) * np.sqrt(2), x[:, 1] * np.sin(ang) * np.sqrt(2)], 1).astype(np.float32))


@node("stereo_width", "audio_fx", "Mid/side width control (needs stereo).", **AA, params={"width": P("float", 1.0, 0, 3)})
def stereo_width(inp, p, ctx):
    a = inp["audio"]
    if a.channels != 2:
        raise ParamError("stereo_width needs a stereo input; put a pan node before it to make the signal stereo")
    m, s = (a.samples[:, 0] + a.samples[:, 1]) / 2, (a.samples[:, 0] - a.samples[:, 1]) / 2
    return _like(a, np.stack([m + s * p["width"], m - s * p["width"]], 1).astype(np.float32))


@node("mix_audio", "audio_mix", "Sum two clips with individual gains and an optional start offset for B.",
      inputs={"a": T.AUDIO, "b": T.AUDIO}, outputs={"audio": T.AUDIO},
      params={"gain_a_db": P("float", 0.0, -96, 24), "gain_b_db": P("float", 0.0, -96, 24),
              "offset_b": P("float", 0.0, 0, 3600, doc="seconds before B starts")})
def mix_audio(inp, p, ctx):
    a, b = inp["a"], inp["b"]
    bs = b.samples
    if b.sr != a.sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(a.sr, b.sr)
        bs = resample_poly(bs, a.sr // g, b.sr // g, axis=0).astype(np.float32)
    ch = max(a.channels, bs.shape[1])
    ax = np.repeat(a.samples, ch, 1) if a.channels == 1 and ch > 1 else a.samples
    bx = np.repeat(bs, ch, 1) if bs.shape[1] == 1 and ch > 1 else bs
    off = int(p["offset_b"] * a.sr)
    n = max(ax.shape[0], bx.shape[0] + off)
    out = np.zeros((n, ch), np.float32)
    out[:ax.shape[0]] += ax * np.float32(_db(p["gain_a_db"]))
    out[off:off + bx.shape[0]] += bx * np.float32(_db(p["gain_b_db"]))
    return AudioBuf(out, a.sr)
