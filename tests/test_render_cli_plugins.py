import shutil
import numpy as np
import pytest
import omniimage as o
from omniimage.cli import main
from omniimage.render import render_frames
from omniimage.presets import build_sfx, neon_sign


def test_render_image_sequence(tmp_path):
    g, out = neon_sign("HI", width=64, height=36, size=30)
    pat = render_frames(g, out, str(tmp_path / "f_{frame}.png"), 0, 3, 24)
    files = sorted(p.name for p in tmp_path.glob("f_*.png"))
    assert files == ["f_00000.png", "f_00001.png", "f_00002.png", "f_00003.png"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_video_with_audio(tmp_path):
    import subprocess, json
    g, out = neon_sign("HI", width=65, height=37, size=30)      # odd size is padded for yuv420p
    sg, so = build_sfx("coin")
    p = str(tmp_path / "v.mp4")
    render_frames(g, out, p, 0, 11, 12, audio=sg.evaluate(so))
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", p],
                                     capture_output=True, text=True).stdout)
    kinds = {s["codec_type"] for s in info["streams"]}
    assert kinds == {"video", "audio"}
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(v["nb_frames"]) == 12


def test_animation_actually_animates(tmp_path):
    g = o.Graph()
    s = g.add("solid", width=16, height=16, color="0.5,0.5,0.5")
    e = g.add("exposure", stops="=frame*0.5")
    g.connect(s, e)
    a, b = g.evaluate(e, ctx=o.Ctx(0))[..., 0].mean(), g.evaluate(e, ctx=o.Ctx(4))[..., 0].mean()
    assert b > a * 3.9


def test_cli_end_to_end(tmp_path, capsys):
    assert main(["nodes"]) == 0
    assert "exposure" in capsys.readouterr().out
    lut = str(tmp_path / "k.cube")
    assert main(["lut", "make", "noir", "-o", lut, "--size", "9"]) == 0
    assert main(["lut", "inspect", lut]) == 0
    wav = str(tmp_path / "s.wav")
    assert main(["sfx", "make", "coin", "-o", wav]) == 0
    g = o.Graph()
    s = g.add("gradient", width=40, height=20)
    g.save(str(tmp_path / "g.json"))
    png = str(tmp_path / "in.png")
    assert main(["run", str(tmp_path / "g.json"), "--target", s, "-o", png]) == 0
    out = str(tmp_path / "out.png")
    assert main(["lut", "apply", png, lut, "-o", out]) == 0
    assert main(["fx", "apply", "vhs", png, "-o", str(tmp_path / "vhs.png")]) == 0
    assert main(["copilot", "make", "a", "sepia", "LUT", "--workdir", str(tmp_path / "w")]) == 0
    assert (tmp_path / "w" / "sepia_tone.cube").exists()
    assert main(["nodes", "--category", "nope"]) == 0


def test_cli_reports_errors_cleanly(tmp_path, capsys):
    assert main(["lut", "make", "not_a_look", "-o", str(tmp_path / "x.cube")]) == 1
    assert "unknown look" in capsys.readouterr().err
    assert main(["sfx", "make", "zzz", "-o", str(tmp_path / "x.wav")]) == 1


def test_bake_via_cli(tmp_path):
    g = o.Graph()
    i = g.add("lut_input", id="i"); l = g.add("srgb_to_linear"); e = g.add("exposure", stops=1); s = g.add("linear_to_srgb", id="o")
    g.chain(i, l, e, s)
    g.save(str(tmp_path / "g.json"))
    assert main(["lut", "bake", str(tmp_path / "g.json"), "--input", "i", "--output", "o", "-o", str(tmp_path / "b.cube"), "--size", "9"]) == 0


def test_plugin_directory_loading(tmp_path, monkeypatch):
    (tmp_path / "my_plugin.py").write_text(
        "from omniimage.core.registry import node\n"
        "from omniimage.core.ports import PortType as T\n"
        "from omniimage.core.params import P\n"
        "@node('plugin_double', 'plugin', 'Doubles an image.', inputs={'image': T.IMAGE}, outputs={'image': T.IMAGE},\n"
        "      params={'k': P('float', 2.0)}, source='plugin')\n"
        "def double(inp, p, ctx):\n    return inp['image'] * p['k']\n")
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.setenv("OMNIIMAGE_PLUGIN_PATH", str(tmp_path))
    monkeypatch.delenv("OMNIIMAGE_NO_PLUGINS")
    with pytest.warns(UserWarning, match="broken.py"):
        loaded = o.load_plugins()
    assert any(p.endswith("my_plugin.py") for p in loaded)
    g = o.Graph()
    s = g.add("solid", width=4, height=4, color="0.5,0.5,0.5")
    d = g.add("plugin_double")
    g.connect(s, d)
    assert g.evaluate(d)[0, 0, 0] == pytest.approx(2 * g.evaluate(s)[0, 0, 0])


def test_node_reference_generation():
    from omniimage.docgen import node_reference
    text = node_reference()
    assert "### `exposure`" in text and "LUT-safe" in text and len(text) > 20000
