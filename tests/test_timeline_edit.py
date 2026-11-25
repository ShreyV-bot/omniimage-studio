"""Timeline editing layer: every op, atomic batches, undo/redo, limits, and render-level invariants."""
import json
import os

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

from omniimage.core.graph import Graph
from omniimage.timeline import Timeline, Clip, AudioClip
from omniimage.timeline_edit import (EditError, TimelineEditor, check_timeline, MAX_FRAME, MAX_CLIPS, MAX_TRACKS)


@pytest.fixture()
def media(tmp_path):
    """A 40-frame video whose frame i is a flat grey of value 5*i, a still, and a 1 s tone."""
    d = tmp_path / "media"
    d.mkdir()
    vw = cv2.VideoWriter(str(d / "ramp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 36))
    for i in range(40):
        vw.write(np.full((36, 64, 3), 5 * i, np.uint8))
    vw.release()
    cv2.imwrite(str(d / "still.png"), np.full((36, 64, 3), 200, np.uint8))
    t = np.arange(48000) / 48000
    wavfile.write(str(d / "tone.wav"), 48000, (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))
    return d


@pytest.fixture()
def ed(media):
    def resolve(name):
        p = os.path.realpath(os.path.join(media, name))
        if not p.startswith(str(media)):
            raise EditError("escapes")
        return p
    g = Graph()
    g.add("solid", id="red", width=64, height=36, color="#ff0000")
    g.add("osc", id="beep", duration=0.5)
    e = TimelineEditor(Timeline(64, 36, 24.0), resolve_media=resolve, project_graph=lambda: g)
    e.apply([{"op": "add_track", "kind": "video"}, {"op": "add_track", "kind": "audio"}])
    return e


def clips(e, t=0):
    return [(c["name"], c["start"], c["duration"]) for c in e.view()["tracks"][t]["clips"]]


def grey(e, frame):
    """Display value (0..255, sRGB) of the grey at the centre of a timeline frame."""
    from omniimage.core.media import linear_to_srgb
    return float(linear_to_srgb(e.frame(frame, 64)[18, 32, 0]) * 255)


# ------------------------------------------------------------------ building
def test_add_clips_from_file_and_node_sorted_by_start(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}, "start": 50}])
    r = ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "still.png"}, "start": 0}])
    assert r[0]["clip"] == 0                                   # sorted: the still comes first
    assert clips(ed) == [("still.png", 0, 48), ("ramp.mp4", 50, 40)]
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 10}])
    assert ed.view()["tracks"][0]["clips"][-1]["start"] == 90   # default start = end of the track


def test_audio_clip_defaults_to_file_length(ed):
    ed.apply([{"op": "add_clip", "track": 1, "source": {"file": "tone.wav"}}])
    assert clips(ed, 1) == [("tone.wav", 0, 24)]               # 1 s at 24 fps
    ed.apply([{"op": "add_clip", "track": 1, "source": {"node": "beep"}}])
    assert clips(ed, 1)[1][2] == 12


def test_wrong_media_kind_for_track_is_rejected(ed):
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "tone.wav"}}])
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 1, "source": {"file": "still.png"}}])
    with pytest.raises(EditError, match="needs"):
        ed.apply([{"op": "add_clip", "track": 1, "source": {"node": "red"}}])   # picture node on an audio track
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "a.png", "node": "red"}}])


def test_media_names_cannot_escape_the_folder(ed):
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "../secret.png"}}])


def test_node_clip_copies_only_its_upstream_graph(ed):
    g = Graph()
    s = g.add("solid", width=64, height=36)
    b = g.add("blur_gaussian", radius=2)
    g.connect(s, b)
    g.add("noise")                                              # unrelated node
    ed.project_graph = lambda: g
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": b}}])
    cg = ed.clip_graph({"track": 0, "clip": 0})
    assert sorted(n["id"] for n in cg["graph"]["nodes"]) == sorted([s, b]) and cg["node"] == b


def test_broken_node_clip_is_refused_up_front(ed):
    g = Graph()
    g.add("blur_gaussian", id="dangling")                       # required input not connected
    ed.project_graph = lambda: g
    with pytest.raises(Exception):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "dangling"}}])
    assert clips(ed) == []


# ------------------------------------------------------------------ moving, trimming, splitting
def test_move_changes_start_and_can_change_track(ed):
    ed.apply([{"op": "add_track", "kind": "video"}, {"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 10}])
    r = ed.apply([{"op": "move_clip", "track": 0, "clip": 0, "start": 30, "to_track": 2}])
    assert r[0] == {"track": 2, "clip": 0} and clips(ed, 2) == [("red", 30, 10)] and clips(ed, 0) == []
    with pytest.raises(EditError):
        ed.apply([{"op": "move_clip", "track": 2, "clip": 0, "start": 0, "to_track": 1}])   # video -> audio track
    with pytest.raises(EditError):
        ed.apply([{"op": "move_clip", "track": 2, "clip": 0, "start": -1}])


def test_left_trim_keeps_the_picture_in_place(ed):
    """Trimming a head off a video clip must not shift what plays at any remaining frame."""
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}, "start": 10}])
    before = [grey(ed, f) for f in range(10, 50)]
    ed.apply([{"op": "trim_clip", "track": 0, "clip": 0, "edge": "left", "frame": 20}])
    c = ed.view()["tracks"][0]["clips"][0]
    assert (c["start"], c["duration"], c["in_point"]) == (20, 30, 10.0)
    assert [grey(ed, f) for f in range(20, 50)] == before[10:]
    assert grey(ed, 15) == 0                                    # trimmed part is gone (background)


def test_left_trim_with_speed_advances_source_by_speed(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}, "duration": 20},
              {"op": "set_clip", "track": 0, "clip": 0, "speed": 2.0}])
    ed.apply([{"op": "trim_clip", "track": 0, "clip": 0, "edge": "left", "frame": 5}])
    assert ed.view()["tracks"][0]["clips"][0]["in_point"] == 10.0


def test_trim_limits(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "start": 10, "duration": 20}])
    with pytest.raises(EditError, match="before the start"):
        ed.apply([{"op": "trim_clip", "track": 0, "clip": 0, "edge": "left", "frame": 5}])    # would need negative in-point
    with pytest.raises(EditError, match="one frame"):
        ed.apply([{"op": "trim_clip", "track": 0, "clip": 0, "edge": "right", "frame": 10}])
    with pytest.raises(EditError, match="one frame"):
        ed.apply([{"op": "trim_clip", "track": 0, "clip": 0, "edge": "left", "frame": 30}])
    ed.apply([{"op": "trim_clip", "track": 0, "clip": 0, "edge": "right", "frame": 15}])
    assert clips(ed) == [("red", 10, 5)]


def test_split_preserves_picture_and_makes_independent_graphs(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}, "start": 0}])
    before = [grey(ed, f) for f in range(40)]
    r = ed.apply([{"op": "split_clip", "track": 0, "clip": 0, "frame": 15}])
    assert r[0] == {"track": 0, "clip": 1, "first": 0}
    assert clips(ed) == [("ramp.mp4", 0, 15), ("ramp.mp4", 15, 25)]
    assert [grey(ed, f) for f in range(40)] == before           # cutting is invisible
    a, b = ed.tl.tracks[0].clips
    assert a.graph is not b.graph
    with pytest.raises(EditError):
        ed.apply([{"op": "split_clip", "track": 0, "clip": 0, "frame": 15}])                  # not inside the first clip
    with pytest.raises(EditError):
        ed.apply([{"op": "split_clip", "track": 0, "clip": 0, "frame": 0}])


def test_split_shifts_opacity_keyframes_so_the_fade_is_continuous(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 20},
              {"op": "set_clip", "track": 0, "clip": 0, "opacity": {"keys": [[0, 0.0], [20, 1.0]], "interp": "linear"}}])
    before = [ed.frame(f, 64)[18, 32, 0] for f in range(20)]          # red over black: the fade shows in the red channel
    assert before[0] < 0.05 and before[19] > 0.8 and before == sorted(before)
    ed.apply([{"op": "split_clip", "track": 0, "clip": 0, "frame": 8}])
    assert np.allclose([ed.frame(f, 64)[18, 32, 0] for f in range(20)], before, atol=1e-6)


def test_split_audio_keeps_fades_on_the_outer_ends_and_aligns_source(ed):
    ed.apply([{"op": "add_clip", "track": 1, "source": {"file": "tone.wav"}},
              {"op": "set_clip", "track": 1, "clip": 0, "fade_in": 0.1, "fade_out": 0.2, "gain_db": -6}])
    ed.apply([{"op": "split_clip", "track": 1, "clip": 0, "frame": 12}])
    a, b = ed.tl.tracks[1].clips
    assert (a.fade_in, a.fade_out, b.fade_in, b.fade_out) == (0.1, 0.0, 0.0, 0.2)
    assert b.in_point == pytest.approx(0.5) and a.gain_db == b.gain_db == -6


def test_delete_and_ripple_delete(ed):
    for i in range(3):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 10, "name": f"c{i}"}])
    ed.apply([{"op": "delete_clip", "track": 0, "clip": 1}])
    assert clips(ed) == [("c0", 0, 10), ("c2", 20, 10)]
    ed.apply([{"op": "ripple_delete", "track": 0, "clip": 0}])
    assert clips(ed) == [("c2", 10, 10)]


# ------------------------------------------------------------------ properties
def test_set_clip_validates_and_is_kind_specific(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}},
              {"op": "add_clip", "track": 1, "source": {"node": "beep"}}])
    ed.apply([{"op": "set_clip", "track": 0, "clip": 0, "opacity": 0.5, "fit": "cover", "speed": 1.5, "name": "x"}])
    c = ed.view()["tracks"][0]["clips"][0]
    assert (c["opacity"], c["fit"], c["speed"], c["name"]) == (0.5, "cover", 1.5, "x")
    for bad in ({"opacity": 1.5}, {"opacity": "=sin(t)"}, {"speed": 0}, {"speed": float("nan")}, {"fit": "zoom"},
                {"gain_db": 3}, {"opacity": {"keys": []}}, {"opacity": {"keys": [[0, 2]]}}):
        with pytest.raises(EditError):
            ed.apply([{"op": "set_clip", "track": 0, "clip": 0, **bad}])
    ed.apply([{"op": "set_clip", "track": 1, "clip": 0, "gain_db": -3, "fade_in": 0.1}])
    with pytest.raises(EditError):
        ed.apply([{"op": "set_clip", "track": 1, "clip": 0, "opacity": 0.5}])
    with pytest.raises(EditError):
        ed.apply([{"op": "set_clip", "track": 1, "clip": 0, "fade_in": 9999}])


def test_set_track_validates_and_is_kind_specific(ed):
    ed.apply([{"op": "set_track", "track": 0, "mute": True, "opacity": 0.4, "blend": "screen", "name": "Top"},
              {"op": "set_track", "track": 1, "solo": True, "gain_db": -6, "pan": 0.5, "bus": "music"}])
    v = ed.view()["tracks"]
    assert (v[0]["mute"], v[0]["opacity"], v[0]["blend"], v[0]["name"]) == (True, 0.4, "screen", "Top")
    assert (v[1]["solo"], v[1]["gain_db"], v[1]["pan"], v[1]["bus"]) == (True, -6, 0.5, "music")
    for bad in ({"track": 0, "pan": 0.2}, {"track": 1, "blend": "over"}, {"track": 0, "blend": "burn"},
                {"track": 0, "opacity": 2}, {"track": 1, "pan": 5}, {"track": 0, "mute": "yes"}, {"track": 9}):
        with pytest.raises(EditError):
            ed.apply([{"op": "set_track", **bad}])


def test_mute_changes_the_picture(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    assert ed.frame(0, 64)[18, 32, 0] > 0.9
    ed.apply([{"op": "set_track", "track": 0, "mute": True}])
    assert ed.frame(0, 64)[18, 32, 0] == 0


def test_remove_track_drops_its_clips(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}, {"op": "remove_track", "track": 0}])
    assert [t["kind"] for t in ed.view()["tracks"]] == ["audio"] and ed.tl.duration == 0


def test_set_timeline_bounds(ed):
    ed.apply([{"op": "set_timeline", "name": "Cut", "width": 128, "height": 72, "fps": 30}])
    v = ed.view()
    assert (v["name"], v["width"], v["height"], v["fps"]) == ("Cut", 128, 72, 30)
    for bad in ({"width": 4}, {"width": 100000}, {"fps": 0}, {"fps": 1000}, {"height": 3.5}, {"name": ""}):
        with pytest.raises(EditError):
            ed.apply([{"op": "set_timeline", **bad}])


# ------------------------------------------------------------------ transitions
def _two_clips(ed, overlap):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "start": 0, "duration": 20},
              {"op": "add_clip", "track": 0, "source": {"node": "red"}, "start": 20 - overlap, "duration": 20}])


def test_transition_with_enough_overlap(ed):
    _two_clips(ed, 8)
    ed.apply([{"op": "set_transition", "track": 0, "clip": 1, "type": "dissolve", "duration": 8}])
    assert ed.view()["tracks"][0]["clips"][1]["transition"] == {"type": "dissolve", "duration": 8}
    ed.apply([{"op": "set_transition", "track": 0, "clip": 1, "type": None}])
    assert ed.view()["tracks"][0]["clips"][1]["transition"] is None


def test_transition_without_overlap_errors_or_is_fixed_on_request(ed):
    _two_clips(ed, 0)
    with pytest.raises(EditError, match="overlap"):
        ed.apply([{"op": "set_transition", "track": 0, "clip": 1, "type": "dissolve", "duration": 6}])
    assert clips(ed)[1][1] == 20                                # failed edit changed nothing
    r = ed.apply([{"op": "set_transition", "track": 0, "clip": 1, "type": "dissolve", "duration": 6, "fix": True}])
    assert clips(ed)[r[0]["clip"]][1] == 14                     # pulled earlier to overlap by exactly 6


def test_transition_edge_cases(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 20}])
    with pytest.raises(EditError, match="before"):
        ed.apply([{"op": "set_transition", "track": 0, "clip": 0, "type": "dissolve"}])       # first clip has nothing before it
    _two_clips(ed, 4)
    for bad in ({"type": "spin"}, {"duration": 0}, {"duration": 99}):
        with pytest.raises(EditError):
            ed.apply([{"op": "set_transition", "track": 0, "clip": 2, "type": "dissolve", **bad}])
    with pytest.raises(EditError):
        ed.apply([{"op": "set_transition", "track": 1, "clip": 0, "type": "dissolve"}])        # audio track


def test_transition_renders_a_blend(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "still.png"}, "duration": 20},
              {"op": "add_clip", "track": 0, "source": {"node": "red"}, "start": 10, "duration": 20}])
    ed.apply([{"op": "set_transition", "track": 0, "clip": 1, "type": "dissolve", "duration": 10}])
    mid = ed.frame(15, 64)[18, 32]
    assert 0.05 < mid[1] < 0.5 and mid[0] > 0.5                  # between the grey still and pure red


# ------------------------------------------------------------------ effects
def test_add_effect_changes_the_picture_and_remove_restores_it(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "still.png"}}])
    plain = ed.frame(0, 64)
    ed.apply([{"op": "add_effect", "track": 0, "clip": 0, "type": "invert"}])
    assert not np.allclose(ed.frame(0, 64), plain)
    ed.apply([{"op": "remove_effect", "track": 0, "clip": 0, "node": ed.clip_graph({"track": 0, "clip": 0})["node"]}])
    assert np.allclose(ed.frame(0, 64), plain)
    assert list(ed.tl.tracks[0].clips[0].graph.nodes) == ["load_image_1"]


def test_effect_chain_order_and_param_edit(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    r = ed.apply([{"op": "add_effect", "track": 0, "clip": 0, "type": "blur_gaussian", "params": {"radius": 1}},
                  {"op": "add_effect", "track": 0, "clip": 0, "type": "invert"}])
    order = ed.clip_graph({"track": 0, "clip": 0})["order"]
    assert order[0] == "red" and order[-1] == r[1]["node"]
    ed.apply([{"op": "set_clip_param", "track": 0, "clip": 0, "node": r[0]["node"], "name": "radius", "value": 4}])
    assert ed.tl.tracks[0].clips[0].graph.nodes[r[0]["node"]].params["radius"] == 4
    with pytest.raises(EditError, match="last effect"):
        ed.apply([{"op": "remove_effect", "track": 0, "clip": 0, "node": r[0]["node"]}])
    with pytest.raises(Exception):
        ed.apply([{"op": "set_clip_param", "track": 0, "clip": 0, "node": r[0]["node"], "name": "nope", "value": 1}])


def test_effect_must_be_a_one_click_picture_effect(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    for typ in ("composite_over", "osc", "not_a_node"):
        with pytest.raises(Exception):
            ed.apply([{"op": "add_effect", "track": 0, "clip": 0, "type": typ}])
    assert list(ed.tl.tracks[0].clips[0].graph.nodes) == ["red"]
    with pytest.raises(Exception):
        ed.apply([{"op": "add_effect", "track": 0, "clip": 0, "type": "blur_gaussian", "params": {"radius": "huge"}}])
    assert list(ed.tl.tracks[0].clips[0].graph.nodes) == ["red"]      # failed batch left the graph alone


def test_source_node_cannot_be_removed_as_an_effect(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    with pytest.raises(EditError, match="source"):
        ed.apply([{"op": "remove_effect", "track": 0, "clip": 0, "node": "red"}])


# ------------------------------------------------------------------ markers, master
def test_markers_and_master(ed):
    ed.apply([{"op": "add_marker", "frame": 12, "name": "beat"}, {"op": "set_master", "target_lufs": -14, "gain_db": 1}])
    v = ed.view()
    assert v["markers"][0]["frame"] == 12 and v["master"]["target_lufs"] == -14
    ed.apply([{"op": "remove_marker", "index": 0}, {"op": "set_master", "target_lufs": None}])
    assert ed.view()["markers"] == [] and ed.view()["master"]["target_lufs"] is None
    with pytest.raises(EditError):
        ed.apply([{"op": "remove_marker", "index": 0}])
    with pytest.raises(EditError):
        ed.apply([{"op": "set_master", "target_lufs": 5}])


# ------------------------------------------------------------------ atomicity, history, limits
def test_batch_is_atomic(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 10}])
    rev = ed.rev
    with pytest.raises(EditError):
        ed.apply([{"op": "move_clip", "track": 0, "clip": 0, "start": 50}, {"op": "move_clip", "track": 0, "clip": 7, "start": 1}])
    assert clips(ed) == [("red", 0, 10)] and ed.rev == rev and ed.undo()   # nothing half-applied, one undo step only


def test_undo_redo_roundtrip(ed):
    assert ed.undo() and not ed.undo()                           # only the track setup in history
    ed.apply([{"op": "add_track", "kind": "video"}])
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": 10}])
    ed.apply([{"op": "move_clip", "track": 0, "clip": 0, "start": 9}])
    assert ed.view()["can_undo"] and not ed.view()["can_redo"]
    ed.undo()
    assert clips(ed) == [("red", 0, 10)] and ed.view()["can_redo"]
    ed.redo()
    assert clips(ed) == [("red", 9, 10)]
    ed.undo(); ed.apply([{"op": "set_clip", "track": 0, "clip": 0, "name": "n"}])
    assert not ed.view()["can_redo"]                              # a new edit clears redo


def test_noop_edit_does_not_pollute_history(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    rev, n = ed.rev, len(ed._undo)
    ed.apply([{"op": "set_clip", "track": 0, "clip": 0, "name": "red"}])
    assert (ed.rev, len(ed._undo)) == (rev, n)


def test_undo_restores_a_working_timeline(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}}])
    ed.apply([{"op": "split_clip", "track": 0, "clip": 0, "frame": 10}])
    ed.undo()
    assert len(ed.tl.tracks[0].clips) == 1 and grey(ed, 12) == pytest.approx(60, abs=6)


def test_unknown_and_malformed_ops(ed):
    for bad in ([], [{"op": "explode"}], [{}], [{"op": "__init__"}], [{"op": "_do"}], ["move"], [{"op": 5}]):
        with pytest.raises(EditError):
            ed.apply(bad)
    with pytest.raises(EditError):
        ed.apply([{"op": "set_timeline"}] * 201)
    for v in (True, "3", 2.5, float("inf"), None, [1]):
        with pytest.raises(EditError):
            ed.apply([{"op": "move_clip", "track": 0, "clip": v, "start": 0}])


def test_limits_on_frames_tracks_and_clips(ed):
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "start": MAX_FRAME + 1}])
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}, "duration": MAX_FRAME + 1}])
    for _ in range(MAX_TRACKS - 2):
        ed.apply([{"op": "add_track", "kind": "video"}])
    with pytest.raises(EditError):
        ed.apply([{"op": "add_track", "kind": "video"}])
    ed.tl.tracks[0].clips = [Clip(Graph(), "x", i, 1) for i in range(MAX_CLIPS)]
    with pytest.raises(EditError):
        ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])


def test_check_timeline_rejects_hostile_loaded_numbers():
    tl = Timeline(64, 36)
    tl.width = 10 ** 7
    with pytest.raises(EditError):
        check_timeline(tl)
    tl = Timeline(64, 36)
    tl.add_track("video").add(Clip(Graph(), "x", 0, 10 ** 9))
    with pytest.raises(EditError):
        check_timeline(tl)
    tl = Timeline(64, 36, fps=0.1)
    with pytest.raises(EditError):
        check_timeline(tl)
    check_timeline(Timeline(1280, 720))


def test_replace_is_undoable_and_validated(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    other = Timeline(128, 72, name="Other")
    ed.replace(other)
    assert ed.view()["name"] == "Other"
    ed.undo()
    assert clips(ed) == [("red", 0, 48)]
    bad = Timeline(64, 36)
    bad.width = 1
    with pytest.raises(EditError):
        ed.replace(bad)


# ------------------------------------------------------------------ previews
def test_frame_proxy_width_and_full_size(ed):
    ed.apply([{"op": "set_timeline", "width": 128, "height": 72}, {"op": "add_clip", "track": 0, "source": {"node": "red"}}])
    assert ed.frame(0, 32).shape[:2] == (18, 32)
    assert ed.frame(0, 2048).shape[:2] == (72, 128)             # never upscales beyond the timeline
    assert ed.tl.width == 128                                    # the proxy is a copy: the timeline is untouched
    assert ed.frame(10 ** 6, 64).shape[:2] == (36, 64)           # past the end: just background, no error


def test_peaks_reflect_gain_fades_and_are_cached(ed):
    ed.apply([{"op": "add_clip", "track": 1, "source": {"file": "tone.wav"}}, {"op": "set_clip", "track": 1, "clip": 0, "fade_out": 0.5}])
    p = ed.peaks({"track": 1, "clip": 0}, 40)
    assert len(p) == 40 and max(p) == pytest.approx(0.5, abs=0.02) and p[-1] < 0.1 and p[5] > 0.4
    assert ed.peaks({"track": 1, "clip": 0}, 40) is ed.peaks({"track": 1, "clip": 0}, 40)       # cached
    ed.apply([{"op": "set_clip", "track": 1, "clip": 0, "gain_db": -20}])
    assert max(ed.peaks({"track": 1, "clip": 0}, 40)) < 0.1                                      # new settings, new peaks
    with pytest.raises(EditError):
        ed.peaks({"track": 0, "clip": 0})


def test_mix_wav_is_valid_cached_and_limited(ed, tmp_path):
    with pytest.raises(EditError, match="no audio"):
        ed.mix_wav()
    ed.apply([{"op": "add_clip", "track": 1, "source": {"file": "tone.wav"}}])
    data = ed.mix_wav()
    assert data[:4] == b"RIFF" and ed.mix_wav() is data
    p = tmp_path / "m.wav"; p.write_bytes(data)
    sr, pcm = wavfile.read(str(p))
    assert sr == 48000 and pcm.shape == (48000, 2) and np.abs(pcm).max() > 1000
    ed.apply([{"op": "move_clip", "track": 1, "clip": 0, "start": 24 * 400}])
    with pytest.raises(EditError, match="limited"):
        ed.mix_wav()


def test_view_is_json_safe_and_roundtrips_through_save(ed, tmp_path):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}},
              {"op": "add_clip", "track": 1, "source": {"file": "tone.wav"}},
              {"op": "add_effect", "track": 0, "clip": 0, "type": "invert"}])
    json.dumps(ed.view(), allow_nan=False)
    p = tmp_path / "t.timeline.json"
    ed.tl.save(str(p))
    back = TimelineEditor(Timeline.load(str(p)))
    assert [c["name"] for t in back.view()["tracks"] for c in t["clips"]] == ["ramp.mp4", "tone.wav"]
    assert np.allclose(back.frame(3, 64), ed.frame(3, 64))


def test_sig_changes_only_when_the_thumbnail_would(ed):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "ramp.mp4"}, "start": 5}])
    s0 = ed.view()["tracks"][0]["clips"][0]["sig"]
    ed.apply([{"op": "move_clip", "track": 0, "clip": 0, "start": 30}])
    assert ed.view()["tracks"][0]["clips"][0]["sig"] == s0       # moving does not change the first picture
    ed.apply([{"op": "add_effect", "track": 0, "clip": 0, "type": "invert"}])
    assert ed.view()["tracks"][0]["clips"][0]["sig"] != s0


def test_clip_file_param_cannot_point_outside_the_media_folder(ed, media):
    ed.apply([{"op": "add_clip", "track": 0, "source": {"file": "still.png"}}])
    for bad in ("../outside.png", "/etc/hostname", "missing.png"):
        with pytest.raises(EditError):
            ed.apply([{"op": "set_clip_param", "track": 0, "clip": 0, "node": "load_image_1", "name": "path", "value": bad}])
    assert ed.tl.tracks[0].clips[0].graph.nodes["load_image_1"].params["path"].endswith("still.png")
    cv2.imwrite(str(media / "other.png"), np.full((36, 64, 3), 10, np.uint8))                       # positive control
    ed.apply([{"op": "set_clip_param", "track": 0, "clip": 0, "node": "load_image_1", "name": "path", "value": "other.png"}])
    assert ed.tl.tracks[0].clips[0].graph.nodes["load_image_1"].params["path"] == str(media / "other.png")


def test_thumbnail_of_a_transparent_clip_is_over_black(ed):
    g = Graph()
    g.add("solid", id="ghost", width=64, height=36, color="1,1,1,0")          # white, fully transparent
    ed.project_graph = lambda: g
    ed.apply([{"op": "add_clip", "track": 0, "source": {"node": "ghost"}}])
    th = ed.thumb({"track": 0, "clip": 0})
    assert th[..., 3].min() == 1.0 and th[..., :3].max() == 0.0
