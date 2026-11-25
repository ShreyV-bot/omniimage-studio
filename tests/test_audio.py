import numpy as np
import pytest
import omniimage as o
from omniimage.presets import SFX, build_sfx
from omniimage.nodes.io import write_audio_file, read_audio_file


@pytest.mark.parametrize("name", sorted(SFX))
def test_sfx_valid_audible_and_deterministic(name):
    g, out = build_sfx(name)
    a = g.evaluate(out)
    g2, out2 = build_sfx(name)
    b = g2.evaluate(out2)
    assert np.array_equal(a.samples, b.samples)                       # deterministic
    assert np.isfinite(a.samples).all()
    peak = float(np.abs(a.samples).max())
    assert 0.3 < peak <= 1.0, peak                                    # normalised, not silent, no clipping
    assert float(np.sqrt((a.samples ** 2).mean())) > 0.01             # audible energy
    assert 0.2 < a.duration < 8


def test_osc_frequency_is_correct():
    g = o.Graph()
    n = g.add("osc", freq_start=1000, freq_end=1000, duration=1.0)
    a = g.evaluate(n)
    spec = np.abs(np.fft.rfft(a.samples[:, 0]))
    assert abs(int(np.argmax(spec)) - 1000) <= 1


def test_lowpass_attenuates_highs():
    g = o.Graph()
    hi = g.add("osc", freq_start=8000, freq_end=8000, duration=0.5)
    lp = g.add("filter", type="lowpass", cutoff=500, order=2)
    g.connect(hi, lp)
    assert np.abs(g.evaluate(lp).samples[2000:]).max() < 0.02 * np.abs(g.evaluate(hi).samples).max() + 0.01


def test_delay_extends_and_echoes():
    g = o.Graph()
    c = g.add("osc", freq_start=500, freq_end=500, duration=0.05)
    d = g.add("delay", time_ms=100, feedback=0.5, mix=1.0)
    g.connect(c, d)
    out = g.evaluate(d)
    assert out.samples.shape[0] > g.evaluate(c).samples.shape[0] + 4000
    echo = out.samples[int(0.1 * 44100) + 100: int(0.1 * 44100) + 1000]
    assert np.abs(echo).max() > 0.1


def test_compressor_reduces_peaks():
    g = o.Graph()
    s = g.add("osc", duration=0.5, gain=1.0)
    c = g.add("compressor", threshold_db=-20, ratio=10, attack_ms=0.5, makeup_db=0)
    g.connect(s, c)
    assert np.abs(g.evaluate(c).samples[8000:]).max() < 0.5 * np.abs(g.evaluate(s).samples).max()


def test_normalize_hits_target():
    g = o.Graph()
    s = g.add("osc", gain=0.1, duration=0.1)
    n = g.add("normalize", peak_db=-6)
    g.connect(s, n)
    assert float(np.abs(g.evaluate(n).samples).max()) == pytest.approx(10 ** (-6 / 20), rel=1e-3)


def test_wav_roundtrip_all_depths(tmp_path):
    g, out = build_sfx("coin")
    a = g.evaluate(out)
    for bits, tol in (("16", 1e-4), ("24", 1e-6), ("32f", 1e-7)):
        p = str(tmp_path / f"c{bits}.wav")
        write_audio_file(a, p, bits)
        b = read_audio_file(p)
        assert b.sr == a.sr and b.samples.shape == a.samples.shape
        assert np.abs(b.samples - np.clip(a.samples, -1, 1)).max() < tol * 10 + tol


def test_mix_audio_offset_and_resample():
    g = o.Graph()
    a = g.add("osc", duration=0.1, sr=44100)
    b = g.add("osc", duration=0.1, sr=22050)
    m = g.add("mix_audio", offset_b=0.2)
    g.connect(a, m, dst_port="a"); g.connect(b, m, dst_port="b")
    out = g.evaluate(m)
    assert out.sr == 44100 and out.duration == pytest.approx(0.3, abs=0.01)


def test_waveform_image_draws_something():
    g = o.Graph()
    a = g.add("osc", duration=0.2)
    w = g.add("waveform_image", width=64, height=32)
    g.connect(a, w)
    img = g.evaluate(w)
    assert img[..., 3].sum() > 64
