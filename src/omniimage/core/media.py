"""Small media containers and image helpers shared by all nodes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AudioBuf:
    samples: np.ndarray  # (n, channels) float32 in [-1, 1] (may exceed before limiting)
    sr: int = 44100

    def __post_init__(self):
        s = np.asarray(self.samples, dtype=np.float32)
        if s.ndim == 1:
            s = s[:, None]
        self.samples = s

    @property
    def duration(self) -> float:
        return self.samples.shape[0] / float(self.sr)

    @property
    def channels(self) -> int:
        return self.samples.shape[1]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma weights."""
    return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 3:
        a = np.ones(img.shape[:2] + (1,), np.float32)
        img = np.concatenate([img, a], axis=2)
    return img


def with_rgb(img: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    out[..., :3] = rgb
    out[..., 3] = img[..., 3]
    return out
