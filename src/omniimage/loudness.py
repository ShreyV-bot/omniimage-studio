"""ITU-R BS.1770-4 / EBU R128 loudness measurement (integrated LUFS) and gain helpers."""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter

from .core.media import AudioBuf


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Two-stage K-weighting (high shelf + RLB high-pass), coefficients derived for any sample rate."""
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.9744509555319
    K = np.tan(np.pi * fc / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    y = lfilter(b, a, x, axis=0)
    fc2, Q2 = 38.13547087613982, 0.5003270373253953
    K = np.tan(np.pi * fc2 / sr)
    a0 = 1 + K / Q2 + K * K
    return lfilter([1, -2, 1], [1, 2 * (K * K - 1) / a0, (1 - K / Q2 + K * K) / a0], y, axis=0)


def measure_lufs(buf: AudioBuf) -> float:
    """Integrated loudness in LUFS (400 ms blocks, 75% overlap, -70 LUFS absolute and -10 LU relative gates).
    Channel weights are 1.0 (mono/stereo). Returns ``-inf`` for silence or clips shorter than one block."""
    x = _k_weight(buf.samples.astype(np.float64), buf.sr)
    n = int(round(0.4 * buf.sr))
    step = int(round(0.1 * buf.sr))
    if x.shape[0] < n:
        return float("-inf")
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)], 0)
    starts = np.arange(0, x.shape[0] - n + 1, step)
    z = (cs[starts + n] - cs[starts]) / n                      # mean square per block per channel
    loud = -0.691 + 10 * np.log10(np.maximum(z.sum(1), 1e-12))
    keep = loud > -70.0
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(np.maximum(z[keep].sum(1).mean(), 1e-12)) - 10.0
    keep &= loud > rel
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10 * np.log10(max(z[keep].sum(1).mean(), 1e-12)))


def sample_peak_db(buf: AudioBuf) -> float:
    p = float(np.abs(buf.samples).max()) if buf.samples.size else 0.0
    return 20 * np.log10(p) if p > 0 else float("-inf")


def normalize_lufs(buf: AudioBuf, target: float = -23.0) -> AudioBuf:
    """Apply a single gain so integrated loudness equals ``target`` LUFS (silence is returned unchanged)."""
    cur = measure_lufs(buf)
    if not np.isfinite(cur):
        return buf
    g = 10 ** ((target - cur) / 20)
    return AudioBuf((buf.samples * g).astype(np.float32), buf.sr)
