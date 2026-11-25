import json
import shutil
import numpy as np
import pytest
import omniimage as o
from omniimage.core.media import AudioBuf, srgb_to_linear
from omniimage.interchange import export_edl, export_otio, timecode
from omniimage.nodes.io import write_audio_file

W, H = 64, 36


def solid_graph(color, w=W, h=H, stops=None):
    g = o.Graph()
    n = g.add("solid", width=w, height=h, color=color)
    if stops is not None:
        e = g.add("exposure", stops=stops)
        g.connect(n, e)
        n = e
    return g, n


def sine_graph(freq=1000.0, dur=2.0, gain=0.5, sr=48000):
    g = o.Graph()
    n = g.add("osc", freq_start=freq, freq_end=freq, duration=dur, gain=gain, sr=sr)
    return g, n


def tl_basic(**kw):
    return o.Timeline(W, H, 24, 48000, **kw)


# ------------------------------------------------------------ video
def test_clip_placement_and_background():
    tl = tl_basic()
    v = tl.add_track("video")
    g, n = solid_graph("1,0,0")
    v.add(o.Clip.from_graph(g, n, 10, 5))
    assert tl.duration == 15
    assert tl.render_frame(0)[..., :3].max() == 0                     # before the clip: black background
    assert tl.render_frame(12)[0, 0, 0] == pytest.approx(1.0)         # during
    assert tl.render_frame(15)[..., :3].max() == 0                    # after (end is exclusive)


def test_tracks_stack_with_opacity_and_blend_modes():
    tl = tl_basic()
    a, b = tl.add_track("video"), tl.add_track("video")
    g1, n1 = solid_graph("0,0,1"); g2, n2 = solid_graph("1,0,0")
    a.add(o.Clip.from_graph(g1, n1, 0, 4))
    b.add(o.Clip.from_graph(g2, n2, 0, 4, opacity=0.5))
    px = tl.render_frame(1)[0, 0]
    assert px[0] == pytest.approx(0.5, abs=1e-4) and px[2] == pytest.approx(0.5, abs=1e-4)
    b.opacity, b.blend = 1.0, "multiply"
    b.clips[0].opacity = 1.0
    assert tl.render_frame(1)[0, 0, :3].max() == pytest.approx(0.0, abs=1e-5)        # red x blue = black
    b.blend = "add"
    assert np.allclose(tl.render_frame(1)[0, 0, :3], [1, 0, 1], atol=1e-4)


def test_mute_and_solo_video_tracks():
    tl = tl_basic()
    a, b = tl.add_track("video"), tl.add_track("video")
    g1, n1 = solid_graph("0,0,1"); g2, n2 = solid_graph("1,0,0")
    a.add(o.Clip.from_graph(g1, n1, 0, 4)); b.add(o.Clip.from_graph(g2, n2, 0, 4))
    b.mute = True
    assert tl.render_frame(0)[0, 0, 2] == 1
    b.mute = False; a.solo = True
    assert tl.render_frame(0)[0, 0, 2] == 1 and tl.render_frame(0)[0, 0, 0] == 0


def test_speed_and_in_point_map_source_frames():
    """exposure stops = frame/10, so brightness encodes the SOURCE frame the clip asked for."""
    g = o.Graph()
    s = g.add("solid", width=W, height=H, color="0.5,0.5,0.5")
    e = g.add("exposure", stops="=frame/10")
    g.connect(s, e)
    tl = tl_basic()
    v = tl.add_track("video")
    v.add(o.Clip.from_graph(g, e, start=100, duration=20, in_point=5, speed=2.0))
    base = float(srgb_to_linear(np.float32(0.5)))
    for t, src in ((100, 5), (104, 13), (110, 25)):
        val = tl.render_frame(t)[0, 0, 0]
        assert val == pytest.approx(base * 2 ** (src / 10), rel=1e-3), (t, src)


def test_fit_modes():
    g, n = solid_graph("1,1,1", w=32, h=36)                    # tall source on a wide timeline
    for fit, check in (("contain", lambda im: im[0, 0, 3] == 0 and im[18, 32, 3] == 1),
                       ("stretch", lambda im: im[0, 0, 3] == 1 and im[35, 63, 3] == 1),
                       ("cover", lambda im: im[0, 0, 3] == 1)):
        tl = tl_basic(background="0,0,0,0")
        v = tl.add_track("video")
        v.add(o.Clip.from_graph(g, n, 0, 2, fit=fit))
        img = tl._clip_image(v.clips[0], 0)
        assert check(img), fit


@pytest.mark.parametrize("kind", ["dissolve", "dip_to_black", "wipe_left", "wipe_right"])
def test_transitions(kind):
    tl = tl_basic()
    v = tl.add_track("video")
    g1, n1 = solid_graph("1,0,0"); g2, n2 = solid_graph("0,0,1")
    v.add(o.Clip.from_graph(g1, n1, 0, 20))
    v.add(o.Clip.from_graph(g2, n2, 10, 20, transition_in=o.Transition(kind, 10)))
    start, end = tl.render_frame(10), tl.render_frame(20)
    assert start[0, 0, 0] == pytest.approx(1, abs=1e-4)                        # transition starts on A
    assert end[0, 0, 2] == pytest.approx(1, abs=1e-4)                          # and is fully B afterwards
    mid = tl.render_frame(15)[..., :3]
    if kind == "dissolve":
        assert mid[0, 0, 0] == pytest.approx(0.5, abs=0.02) and mid[0, 0, 2] == pytest.approx(0.5, abs=0.02)
    elif kind == "dip_to_black":
        assert mid.max() < 0.05
    else:
        assert mid[0, 0, 0] != mid[0, -1, 0]                                    # split picture: part A, part B


def test_unknown_transition_and_bad_track_clip_rejected():
    with pytest.raises(o.GraphError):
        o.Transition("star_wipe", 5)
    tl = tl_basic()
    a = tl.add_track("audio")
    g, n = solid_graph("1,0,0")
    with pytest.raises(o.GraphError):
        a.add(o.Clip.from_graph(g, n, 0, 4))


def test_animated_clip_opacity_fades():
    tl = tl_basic()
    v = tl.add_track("video")
    g, n = solid_graph("1,1,1")
    v.add(o.Clip.from_graph(g, n, 0, 11, opacity={"keys": [[0, 0], [10, 1]]}))
    assert tl.render_frame(0)[0, 0, 0] == pytest.approx(0, abs=1e-5)
    assert tl.render_frame(5)[0, 0, 0] == pytest.approx(0.5, abs=1e-4)


def test_file_clips_and_effects(tmp_path):
    p = str(tmp_path / "red.png")
    g, n = solid_graph("1,0,0")
    from omniimage.nodes.io import write_image_file
    write_image_file(g.evaluate(n), p)
    c = o.Clip.from_file(p, 0, 4, effects=[("exposure", {"stops": -1})])
    tl = tl_basic()
    tl.add_track("video").add(c)
    assert tl.render_frame(0)[0, 0, 0] == pytest.approx(0.5, abs=1e-3)
    assert c.media == p and c.name == "red.png"


# ------------------------------------------------------------ audio
def test_audio_clip_position_gain_and_fades():
    tl = tl_basic()
    a = tl.add_track("audio")
    g, n = sine_graph(500, dur=2.0, gain=0.5)
    a.add(o.AudioClip.from_graph(g, n, start=24, duration=24, gain_db=-6, fade_in=0.25, fade_out=0.25))
    out = tl.render_audio()
    assert out.sr == 48000 and out.duration == pytest.approx(2.0, abs=0.01)
    assert np.abs(out.samples[:48000]).max() == 0                                   # silent before it starts (1 s)
    body = out.samples[int(1.4 * 48000):int(1.8 * 48000)]
    assert np.abs(body).max() == pytest.approx(0.5 * 10 ** (-6 / 20), rel=0.05)
    assert np.abs(out.samples[48000:48000 + 50]).max() < 0.01                       # faded in from silence
    assert np.abs(out.samples[-50:]).max() < 0.01                                   # faded out


def test_pan_is_hard_and_equal_power():
    tl = tl_basic(); tl.master.ceiling_db = None
    a = tl.add_track("audio", pan=-1.0)
    g, n = sine_graph(500, dur=1.0)
    a.add(o.AudioClip.from_graph(g, n, 0, 24))
    out = tl.render_audio().samples
    assert np.abs(out[:, 1]).max() == 0 and np.abs(out[:, 0]).max() > 0.5
    a.pan = 0.0
    c = tl.render_audio().samples
    assert np.abs(c[:, 0]).max() == pytest.approx(np.abs(c[:, 1]).max())
    assert np.abs(c).max() == pytest.approx(0.5, rel=0.02)                           # centre is unity (no level jump)


def test_mute_solo_audio_tracks_and_sum():
    tl = tl_basic(); tl.master.ceiling_db = None
    t1, t2 = tl.add_track("audio"), tl.add_track("audio")
    g, n = sine_graph(500, dur=1.0, gain=0.2)
    g2, n2 = sine_graph(500, dur=1.0, gain=0.2)
    t1.add(o.AudioClip.from_graph(g, n, 0, 24)); t2.add(o.AudioClip.from_graph(g2, n2, 0, 24))
    both = np.abs(tl.render_audio().samples).max()
    assert both == pytest.approx(0.4, rel=0.02)
    t2.mute = True
    assert np.abs(tl.render_audio().samples).max() == pytest.approx(0.2, rel=0.02)
    t2.mute = False; t1.solo = True
    assert np.abs(tl.render_audio().samples).max() == pytest.approx(0.2, rel=0.02)


def test_gain_automation_ramps():
    tl = tl_basic(); tl.master.ceiling_db = None
    a = tl.add_track("audio", gain_db={"keys": [[0, -60], [1, 0]]})
    g, n = sine_graph(500, dur=1.0, gain=0.5)
    a.add(o.AudioClip.from_graph(g, n, 0, 24))
    s = tl.render_audio().samples[:, 0]
    assert np.abs(s[:2000]).max() < 0.01
    assert np.abs(s[-3000:]).max() > 0.4


def test_bus_gain_and_master_limiter_and_lufs_target():
    tl = tl_basic()
    a = tl.add_track("audio", bus="music")
    g, n = sine_graph(500, dur=1.0, gain=0.8)
    a.add(o.AudioClip.from_graph(g, n, 0, 24))
    tl.master.ceiling_db = None
    base = np.abs(tl.render_audio().samples).max()
    tl.buses["music"] = o.Bus(gain_db=-6)
    assert np.abs(tl.render_audio().samples).max() == pytest.approx(base * 0.5, rel=0.02)
    tl.buses["music"] = o.Bus(gain_db=+12)
    tl.master.ceiling_db = -1.0
    peak = np.abs(tl.render_audio().samples).max()
    assert peak <= 10 ** (-1 / 20) + 1e-4                                            # soft limiter holds the ceiling
    assert peak > 0.8


def test_ducking_lowers_music_only_while_voice_plays():
    tl = tl_basic(); tl.master.ceiling_db = None
    voice = tl.add_track("audio", bus="voice")
    music = tl.add_track("audio", bus="music")
    gv, nv = sine_graph(300, dur=1.0, gain=0.5)
    gm, nm = sine_graph(800, dur=3.0, gain=0.5)
    voice.add(o.AudioClip.from_graph(gv, nv, start=24, duration=24))                 # voice during seconds 1..2
    music.add(o.AudioClip.from_graph(gm, nm, start=0, duration=72))
    tl.buses["music"] = o.Bus(0, o.Duck(key="voice", amount_db=-12, threshold_db=-40, smooth=0.05))
    s = tl.render_audio().samples[:, 0]
    sr = 48000
    quiet_part = np.abs(s[int(0.2 * sr):int(0.6 * sr)]).max()                        # music alone
    ducked = s[int(1.3 * sr):int(1.7 * sr)]
    spec = np.abs(np.fft.rfft(ducked * np.hanning(len(ducked))))
    f = np.fft.rfftfreq(len(ducked), 1 / sr)
    music_level = spec[np.argmin(np.abs(f - 800))]
    ref_spec = np.abs(np.fft.rfft(s[int(0.2 * sr):int(0.6 * sr)] * np.hanning(int(0.4 * sr))))
    ref = ref_spec[np.argmin(np.abs(np.fft.rfftfreq(int(0.4 * sr), 1 / sr) - 800))]
    assert 20 * np.log10(music_level / ref) < -9                                     # about -12 dB while voice speaks
    assert quiet_part > 0.4                                                          # untouched without voice


def test_lufs_reference_values():
    """EBU Tech 3341: a stereo 1 kHz sine at -23 dBFS reads -23 LUFS; -20 dBFS reads -20."""
    sr = 48000
    t = np.arange(sr * 5) / sr
    for db in (-23.0, -20.0, -14.0):
        x = (10 ** (db / 20) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        assert o.measure_lufs(AudioBuf(np.stack([x, x], 1), sr)) == pytest.approx(db, abs=0.15)
    assert o.measure_lufs(AudioBuf(np.zeros((sr * 2, 2), np.float32), sr)) == float("-inf")
    y = o.normalize_lufs(AudioBuf(np.stack([x, x], 1) * 0.1, sr), -23.0)
    assert o.measure_lufs(y) == pytest.approx(-23.0, abs=0.1)


def test_lufs_other_sample_rates_and_gating():
    for sr in (44100, 96000):
        t = np.arange(sr * 4) / sr
        x = (10 ** (-23 / 20) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        assert o.measure_lufs(AudioBuf(np.stack([x, x], 1), sr)) == pytest.approx(-23.0, abs=0.2)
    # a long near-silent tail must not drag the integrated value (relative gate)
    sr = 48000
    t = np.arange(sr * 3) / sr
    x = (10 ** (-23 / 20) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    padded = np.concatenate([x, np.zeros(sr * 6, np.float32) + 1e-5])
    assert o.measure_lufs(AudioBuf(np.stack([padded, padded], 1), sr)) == pytest.approx(-23.0, abs=0.3)


def test_master_target_lufs():
    tl = tl_basic(); tl.master.ceiling_db = None
    a = tl.add_track("audio")
    g, n = sine_graph(1000, dur=5.0, gain=0.05)
    a.add(o.AudioClip.from_graph(g, n, 0, 120))
    tl.master.target_lufs = -16.0
    assert o.measure_lufs(tl.render_audio()) == pytest.approx(-16.0, abs=0.2)


def test_wav_file_clip_resampled(tmp_path):
    p = str(tmp_path / "s.wav")
    g, n = sine_graph(440, dur=1.0, gain=0.5, sr=22050)
    write_audio_file(g.evaluate(n), p, "16")
    tl = tl_basic()
    tl.add_track("audio").add(o.AudioClip.from_file(p, 0, 24))
    out = tl.render_audio()
    assert out.sr == 48000 and out.samples.shape[0] == 48000
    spec = np.abs(np.fft.rfft(out.samples[:, 0]))
    assert abs(int(np.argmax(spec)) - 440) <= 1


# ------------------------------------------------------------ persistence, rendering, interchange
def build_sample():
    from omniimage.presets.timeline_demo import demo_timeline
    return demo_timeline(96, 54, 12)


def test_save_load_roundtrip_is_identical(tmp_path):
    tl = build_sample()
    p = str(tmp_path / "t.json")
    tl.save(p)
    json.load(open(p))
    tl2 = o.Timeline.load(p)
    assert tl2.duration == tl.duration and len(tl2.tracks) == len(tl.tracks)
    for f in (3, 20, 33, 45):
        assert np.allclose(tl.render_frame(f), tl2.render_frame(f), atol=1e-6)
    assert np.allclose(tl.render_audio().samples, tl2.render_audio().samples, atol=1e-6)
    assert tl2.buses["music"].duck.key == "sfx" and tl2.master.target_lufs == -16.0


def test_demo_timeline_mix_is_sane():
    tl = build_sample()
    au = tl.render_audio()
    assert np.isfinite(au.samples).all()
    assert o.measure_lufs(au) == pytest.approx(-16.0, abs=0.5)
    assert np.abs(au.samples).max() <= 10 ** (-1 / 20) + 1e-3


def test_render_image_sequence_from_timeline(tmp_path):
    tl = build_sample()
    out = o.render_timeline(tl, str(tmp_path / "f_{frame}.png"), 0, 3)
    assert len(list(tmp_path.glob("f_*.png"))) == 4


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_video_with_mixed_audio(tmp_path):
    import subprocess
    tl = build_sample()
    p = str(tmp_path / "t.mp4")
    o.render_timeline(tl, p)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", p],
                                     capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(v["nb_frames"]) == tl.duration
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}


def test_empty_timeline_render_errors(tmp_path):
    with pytest.raises(o.GraphError):
        o.render_timeline(tl_basic(), str(tmp_path / "x.mp4"))


def test_timecode():
    assert timecode(0, 24) == "00:00:00:00" and timecode(25, 24) == "00:00:01:01"
    assert timecode(24 * 3600, 24) == "01:00:00:00" and timecode(0, 24, 1) == "01:00:00:00"


def test_edl_cuts_and_dissolve(tmp_path):
    tl = tl_basic(name="My Cut")
    v = tl.add_track("video")
    g1, n1 = solid_graph("1,0,0"); g2, n2 = solid_graph("0,0,1"); g3, n3 = solid_graph("0,1,0")
    v.add(o.Clip.from_graph(g1, n1, 0, 48, name="red", in_point=10))
    v.add(o.Clip.from_graph(g2, n2, 48, 48, name="blue"))
    v.add(o.Clip.from_graph(g3, n3, 90, 30, name="green", transition_in=o.Transition("dissolve", 6)))
    p = export_edl(tl, str(tmp_path / "a.edl"))
    txt = open(p).read().splitlines()
    assert txt[0] == "TITLE: My Cut" and txt[1] == "FCM: NON-DROP FRAME"
    ev = [l for l in txt if l[:3].isdigit()]
    assert ev[0].split()[-4:] == ["00:00:00:10", "00:00:02:10", "01:00:00:00", "01:00:02:00"]   # red: src 10..58 -> rec 0..48
    assert ev[1].split()[-1] == "01:00:04:00"                                                    # blue ends at frame 96
    assert any("D    006" in l for l in ev)                                                      # dissolve with 6 frames
    assert "* FROM CLIP NAME: red" in txt


def test_edl_audio_events(tmp_path):
    tl = tl_basic()
    a = tl.add_track("audio")
    g, n = sine_graph()
    a.add(o.AudioClip.from_graph(g, n, 24, 48, name="beep"))
    txt = open(export_edl(tl, str(tmp_path / "a.edl"), include_audio=True)).read()
    assert " AA " in txt and "01:00:01:00" in txt


def test_otio_structure_and_transition(tmp_path):
    tl = build_sample()
    tl.markers.append({"frame": 5, "name": "m", "color": "red"})
    p = export_otio(tl, str(tmp_path / "t.otio"))
    d = json.load(open(p))
    assert d["OTIO_SCHEMA"] == "Timeline.1"
    kinds = [t["kind"] for t in d["tracks"]["children"]]
    assert kinds.count("Video") == 2 and kinds.count("Audio") == 3
    v1 = d["tracks"]["children"][0]["children"]
    assert [c["OTIO_SCHEMA"] for c in v1] == ["Clip.1", "Transition.1", "Clip.1"]
    # total duration is preserved: clip A + clip B (the transition has zero length of its own)
    total = sum(c["source_range"]["duration"]["value"] for c in v1 if c["OTIO_SCHEMA"] == "Clip.1")
    assert total == tl.video_tracks()[0].clips[-1].end


def test_otio_validates_with_real_library(tmp_path):
    otio = pytest.importorskip("opentimelineio")
    tl = build_sample()
    p = export_otio(tl, str(tmp_path / "t.otio"))
    t = otio.adapters.read_from_file(p)
    assert len(t.video_tracks()) == 2 and len(t.audio_tracks()) == 3
    assert t.duration().rate == 12
    v1 = t.video_tracks()[0]
    assert abs(v1.duration().value - tl.video_tracks()[0].clips[-1].end) < 1e-6


def test_cli_timeline_roundtrip(tmp_path, capsys):
    from omniimage.cli import main
    tj = str(tmp_path / "t.json")
    assert main(["timeline", "demo", "-o", tj]) == 0
    assert main(["timeline", "info", tj]) == 0
    out = capsys.readouterr().out
    assert "LUFS" in out and "video" in out
    assert main(["timeline", "export", tj, "--format", "edl", "-o", str(tmp_path / "x.edl"), "--audio"]) == 0
    assert main(["timeline", "export", tj, "--format", "otio", "-o", str(tmp_path / "x.otio")]) == 0
    assert main(["timeline", "render", tj, "-o", str(tmp_path / "f_{frame}.png"), "--end", "1"]) == 0
