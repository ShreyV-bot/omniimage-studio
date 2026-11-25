"""Timeline endpoints of the local GUI server: security (token, JSON, folder confinement) and behaviour."""
import json
import os

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

from test_gui import gui  # noqa: F401  (fixture)


@pytest.fixture()
def tgui(gui):
    call, studio, base, html = gui
    w = studio.workdir
    cv2.imwrite(os.path.join(w, "still.png"), np.full((36, 64, 3), 200, np.uint8))
    t = np.arange(24000) / 48000
    wavfile.write(os.path.join(w, "tone.wav"), 48000, (0.4 * np.sin(2 * np.pi * 330 * t) * 32767).astype(np.int16))
    call("/api/timeline/edit", {"ops": [{"op": "set_timeline", "width": 128, "height": 72},
                                        {"op": "add_track", "kind": "video"}, {"op": "add_track", "kind": "audio"}]})
    return call, studio, base, html


def edit(call, *ops):
    st, d, _ = call("/api/timeline/edit", {"ops": list(ops)})
    return st, d


def test_every_timeline_route_needs_the_token(tgui):
    call, _, _, _ = tgui
    for path in ("/api/timeline", "/api/timeline/frame?t=0", "/api/timeline/audio", "/api/timeline/clip?track=0&clip=0",
                 "/api/timeline/thumb?track=0&clip=0", "/api/timeline/peaks?track=1&clip=0"):
        assert call(path, token=None)[0] == 403, path
        assert call(path, token="wrong")[0] == 403, path
    for path in ("edit", "undo", "redo", "new", "demo", "save", "load", "export", "interchange"):
        assert call("/api/timeline/" + path, {}, token=None)[0] == 403, path


def test_timeline_posts_need_json_content_type_and_local_host(tgui):
    call, _, _, _ = tgui
    st, _, _ = call("/api/timeline/edit", None, headers={"Content-Type": "text/plain"}, method="POST")
    assert st == 415
    assert call("/api/timeline", headers={"Host": "evil.example.com"})[0] == 403


def test_edit_then_view_roundtrip(tgui):
    call, _, _, _ = tgui
    st, d = edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}, "duration": 12})
    assert st == 200 and d["results"][0] == {"track": 0, "clip": 0}
    v = call("/api/timeline")[1]
    assert v["tracks"][0]["clips"][0]["name"] == "still.png" and v["duration"] == 12 and v["can_undo"]
    assert d["view"]["rev"] == v["rev"]


def test_bad_edits_are_400_with_a_message_and_change_nothing(tgui):
    call, _, _, _ = tgui
    st, d = edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}}, {"op": "move_clip", "track": 5, "clip": 0, "start": 1})
    assert st == 400 and "track" in d["error"]
    assert call("/api/timeline")[1]["tracks"][0]["clips"] == []
    assert edit(call, {"op": "nope"})[0] == 400
    assert call("/api/timeline/edit", {"ops": "x"})[0] == 400
    assert call("/api/timeline/edit", {"ops": [1]})[0] == 400


@pytest.mark.parametrize("name", ["../x.png", "/etc/passwd", "a/../../x.png", "%2e%2e/x.png", "missing.png"])
def test_clip_media_is_confined_to_the_work_folder(tgui, name):
    call, studio, _, _ = tgui
    open(os.path.join(os.path.dirname(studio.workdir), "x.png"), "wb").close()
    st, d = edit(call, {"op": "add_clip", "track": 0, "source": {"file": name}})
    assert st in (403, 404) and call("/api/timeline")[1]["tracks"][0]["clips"] == []


def test_symlink_out_of_the_work_folder_is_refused(tgui, tmp_path):
    call, studio, _, _ = tgui
    outside = tmp_path / "outside.png"
    cv2.imwrite(str(outside), np.zeros((8, 8, 3), np.uint8))
    os.symlink(outside, os.path.join(studio.workdir, "link.png"))
    assert edit(call, {"op": "add_clip", "track": 0, "source": {"file": "link.png"}})[0] == 403
    assert edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}})[0] == 200      # positive control


def test_frame_is_a_png_of_the_requested_width(tgui):
    call, _, _, _ = tgui
    edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}})
    st, body, r = call("/api/timeline/frame?t=0&maxw=64", raw=True)
    assert st == 200 and r.headers["Content-Type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (36, 64) and abs(int(img[18, 32, 0]) - 200) <= 2
    assert call("/api/timeline/frame?t=abc")[0] == 400
    assert call("/api/timeline/frame?t=-5&maxw=32", raw=True)[0] == 200


def test_thumb_peaks_clip_and_audio_routes(tgui):
    call, _, _, _ = tgui
    edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}},
         {"op": "add_clip", "track": 1, "source": {"file": "tone.wav"}})
    st, body, r = call("/api/timeline/thumb?track=0&clip=0&maxw=40", raw=True)
    assert st == 200 and r.headers["Content-Type"] == "image/png"
    assert cv2.imdecode(np.frombuffer(body, np.uint8), 1).shape[1] == 40
    st, d, _ = call("/api/timeline/peaks?track=1&clip=0&n=32")
    assert st == 200 and len(d["peaks"]) == 32 and max(d["peaks"]) > 0.3
    assert call("/api/timeline/peaks?track=0&clip=0")[0] == 400                   # picture clip has no peaks
    st, d, _ = call("/api/timeline/clip?track=0&clip=0")
    assert st == 200 and d["node"] == "load_image_1"
    st, body, r = call("/api/timeline/audio", raw=True)
    assert st == 200 and r.headers["Content-Type"] == "audio/wav" and body[:4] == b"RIFF"
    assert call("/api/timeline/clip?track=3&clip=0")[0] == 400
    assert call("/api/timeline/nothing")[0] == 404


def test_undo_redo_new_and_demo(tgui):
    call, _, _, _ = tgui
    edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}})
    st, d, _ = call("/api/timeline/undo", {})
    assert st == 200 and d["ok"] and d["view"]["tracks"][0]["clips"] == []
    assert call("/api/timeline/redo", {})[1]["view"]["tracks"][0]["clips"]
    assert call("/api/timeline/demo", {})[1]["view"]["name"] == "OmniImage demo"
    st, d, _ = call("/api/timeline/undo", {})
    assert d["view"]["name"] != "OmniImage demo"                                  # opening the demo is undoable
    assert call("/api/timeline/new", {})[1]["view"]["tracks"] == []


def test_node_graph_clip_goes_through_the_project(tgui):
    call, studio, _, _ = tgui
    st, d, _ = call("/api/command", {"cmd": {"op": "add_node", "type": "solid", "params": {"width": 64, "height": 36, "color": "#00ff00"}}})
    nid = d["results"][0]["id"]
    st, d = edit(call, {"op": "add_clip", "track": 0, "source": {"node": nid}, "duration": 6})
    assert st == 200
    call("/api/command", {"cmd": {"op": "remove_node", "id": nid}})                # later edits to the project do not break the clip
    st, body, _ = call("/api/timeline/frame?t=2&maxw=64", raw=True)
    img = cv2.imdecode(np.frombuffer(body, np.uint8), 1)
    assert st == 200 and img[18, 32, 1] > 200 and img[18, 32, 2] < 40


def test_save_load_confinement_and_naming(tgui):
    call, studio, _, _ = tgui
    edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}, "duration": 7}, {"op": "set_timeline", "name": "Cut A"})
    for given in ("cut", "cut.json", "cut.timeline", "cut.timeline.json"):
        assert call("/api/timeline/save", {"name": given})[1]["saved"] == "cut.timeline.json", given
    assert call("/api/timeline/save", {"name": "../evil"})[0] == 403
    assert call("/api/timeline/save", {"name": "/tmp/evil"})[0] == 403
    call("/api/timeline/new", {})
    st, d, _ = call("/api/timeline/load", {"name": "cut.timeline.json"})
    assert st == 200 and d["view"]["name"] == "Cut A" and d["view"]["duration"] == 7
    assert call("/api/timeline/load", {"name": "../x.json"})[0] == 403
    assert call("/api/timeline/load", {"name": "nothing.timeline.json"})[0] == 404


def test_loading_garbage_or_hostile_timelines_is_a_400_not_a_crash(tgui):
    call, studio, _, _ = tgui
    w = studio.workdir
    open(os.path.join(w, "junk.timeline.json"), "w").write("{not json")
    open(os.path.join(w, "empty.timeline.json"), "w").write("{}")
    open(os.path.join(w, "list.timeline.json"), "w").write("[1,2]")
    good = {"version": 1, "name": "x", "width": 64, "height": 36, "fps": 24, "sample_rate": 48000, "tracks": []}
    open(os.path.join(w, "huge.timeline.json"), "w").write(json.dumps({**good, "width": 10 ** 8}))
    open(os.path.join(w, "ok.timeline.json"), "w").write(json.dumps(good))
    for n in ("junk", "empty", "list", "huge"):
        st, d, _ = call("/api/timeline/load", {"name": n + ".timeline.json"})
        assert st == 400 and d["error"], n
    assert call("/api/timeline/load", {"name": "ok.timeline.json"})[0] == 200                # positive control
    assert call("/api/timeline")[1]["name"] == "x"


def test_export_is_confined_bounded_and_writes_a_file(tgui):
    call, studio, _, _ = tgui
    assert call("/api/timeline/export", {"name": "o.png"})[0] == 400                          # empty timeline
    edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}, "duration": 3})
    assert call("/api/timeline/export", {"name": "../o.png"})[0] == 403
    assert call("/api/timeline/export", {"name": "o_{frame}.png", "start": 2, "end": 1})[0] == 400
    st, d, _ = call("/api/timeline/export", {"name": "o_{frame}.png", "start": 0, "end": 99})
    assert st == 200
    assert sorted(f for f in os.listdir(studio.workdir) if f.startswith("o_")) == ["o_00000.png", "o_00001.png", "o_00002.png"]
    edit(call, {"op": "move_clip", "track": 0, "clip": 0, "start": 5000})
    assert call("/api/timeline/export", {"name": "big.mp4", "start": 0, "end": 4000})[0] == 400


def test_interchange_exports_edl_and_otio_inside_the_folder(tgui):
    call, studio, _, _ = tgui
    edit(call, {"op": "add_clip", "track": 0, "source": {"file": "still.png"}, "duration": 12})
    st, d, _ = call("/api/timeline/interchange", {"format": "edl"})
    assert st == 200 and d["file"] == "timeline.edl"
    assert "TITLE" in open(os.path.join(studio.workdir, "timeline.edl")).read()
    st, d, _ = call("/api/timeline/interchange", {"format": "otio", "name": "cuts/t.otio"})
    assert st == 200 and json.load(open(os.path.join(studio.workdir, "cuts", "t.otio")))
    assert call("/api/timeline/interchange", {"format": "aaf"})[0] == 400
    assert call("/api/timeline/interchange", {"format": "edl", "name": "../x.edl"})[0] == 403


def test_page_has_the_timeline_workspace(tgui):
    html = tgui[3]
    for marker in ('id="tlpage"', "/api/timeline/edit", 'id="tabTimeline"'):
        assert marker in html


def test_oversized_timeline_file_is_refused_before_parsing(tgui):
    call, studio, _, _ = tgui
    with open(os.path.join(studio.workdir, "big.timeline.json"), "wb") as f:
        f.truncate(17 * 1024 * 1024)
    assert call("/api/timeline/load", {"name": "big.timeline.json"})[0] == 413
