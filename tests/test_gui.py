import json
import re
import threading
import urllib.error
import urllib.request

import numpy as np
import cv2
import pytest

from omniimage.gui.server import Studio, StudioServer


@pytest.fixture()
def gui(tmp_path):
    studio = Studio(workdir=str(tmp_path / "work"))
    srv = StudioServer(studio, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    html = urllib.request.urlopen(base + "/").read().decode()
    token = re.search(r'const TOKEN="([^"]+)"', html).group(1)

    def call(path, body=None, token=token, headers=None, raw=False, method=None):
        h = {"X-Omni-Token": token} if token is not None else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
        h.update(headers or {})
        req = urllib.request.Request(base + path, data=data, headers=h, method=method)
        try:
            r = urllib.request.urlopen(req)
            out = r.read()
            return r.status, (out if raw else json.loads(out)), r
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}"), e

    yield call, studio, base, html
    srv.shutdown(); srv.server_close()


def test_page_is_served_with_token(gui):
    call, studio, base, html = gui
    assert "OmniImage Studio" in html and "__OMNI_TOKEN__" not in html


def test_api_requires_token_and_json(gui):
    call, studio, base, html = gui
    assert call("/api/project", token="wrong")[0] == 403
    assert call("/api/project", token=None)[0] == 403
    # a cross-site HTML form can only send text/plain or form encodings, so even a valid token needs real JSON
    assert call("/api/undo", {}, headers={"Content-Type": "text/plain"})[0] == 415
    assert call("/api/undo", {}, headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415


def test_bad_host_header_rejected(gui):
    call, studio, base, html = gui
    req = urllib.request.Request(base + "/", headers={"Host": "evil.example.com"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403
    code, body, _ = call("/api/project", headers={"Host": "attacker.test:80"})
    assert code == 403


def test_node_catalog(gui):
    call, *_ = gui
    code, cat, _ = call("/api/nodes")
    assert code == 200 and len(cat) >= 100
    ex = next(n for n in cat if n["type"] == "exposure")
    assert ex["inputs"] == {"image": "image"} and "stops" in ex["params"]


def test_edit_undo_redo_roundtrip(gui):
    call, studio, *_ = gui
    c, d, _ = call("/api/command", {"cmd": {"op": "add_node", "type": "gradient", "id": "bg", "params": {"width": 64, "height": 36}}})
    assert c == 200 and d["results"][0]["id"] == "bg"
    call("/api/command", {"cmds": [{"op": "add_node", "type": "exposure", "id": "ex"},
                                    {"op": "connect", "src": "bg", "dst": "ex"},
                                    {"op": "set_param", "node": "ex", "name": "stops", "value": 1.0}]})
    c, d, _ = call("/api/project")
    assert {n["id"] for n in d["graph"]["nodes"]} == {"bg", "ex"} and d["problems"] == [] and d["can_undo"]
    assert call("/api/undo", {})[1]["ok"]                       # the 3-command batch is one undo step
    assert {n["id"] for n in call("/api/project")[1]["graph"]["nodes"]} == {"bg"}
    assert call("/api/redo", {})[1]["ok"]


def test_errors_are_400_with_message(gui):
    call, *_ = gui
    c, d, _ = call("/api/command", {"cmd": {"op": "add_node", "type": "does_not_exist"}})
    assert c == 400 and "error" in d
    c, d, _ = call("/api/command", {"cmd": {"op": "connect", "src": "a", "dst": "b"}})
    assert c == 400
    assert call("/api/command", {"cmd": "nope"})[0] == 400


def test_failed_batch_rolls_back(gui):
    call, *_ = gui
    c, d, _ = call("/api/command", {"cmds": [{"op": "add_node", "type": "solid", "id": "a"},
                                              {"op": "add_node", "type": "nope"}]})
    assert c == 400
    assert call("/api/project")[1]["graph"]["nodes"] == []


def test_render_picture_and_scopes(gui):
    call, *_ = gui
    call("/api/command", {"cmd": {"op": "add_node", "type": "gradient", "id": "bg", "params": {"width": 96, "height": 54}}})
    for scope in ("none", "waveform", "parade", "vectorscope", "histogram"):
        c, png, r = call(f"/api/render?node=bg&scope={scope}&maxw=128", raw=True)
        assert c == 200 and png[:4] == b"\x89PNG"
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.std() > 1, scope
    assert call("/api/render?node=bg&scope=bogus")[0] == 400
    assert call("/api/render?node=ghost")[0] == 404


def test_render_audio_and_mask_nodes(gui):
    call, *_ = gui
    call("/api/command", {"cmds": [{"op": "add_node", "type": "osc", "id": "o", "params": {"duration": 0.2}},
                                    {"op": "add_node", "type": "shape_mask", "id": "m"}]})
    c, png, r = call("/api/render?node=o&maxw=200", raw=True)
    assert c == 200 and r.headers["X-Omni-Kind"] == "audio"
    assert call("/api/render?node=m", raw=True)[0] == 200


def test_animated_frames_differ(gui):
    call, *_ = gui
    call("/api/command", {"cmds": [{"op": "add_node", "type": "solid", "id": "s", "params": {"color": "0.2,0.2,0.2"}},
                                    {"op": "add_node", "type": "exposure", "id": "e", "params": {"stops": "=frame*0.5"}},
                                    {"op": "connect", "src": "s", "dst": "e"}]})
    a = call("/api/render?node=e&frame=0", raw=True)[1]
    b = call("/api/render?node=e&frame=6", raw=True)[1]
    assert a != b


def test_save_load_and_path_confinement(gui, tmp_path):
    call, studio, *_ = gui
    call("/api/command", {"cmd": {"op": "add_node", "type": "solid", "id": "s"}})
    call("/api/layout", {"layout": {"s": [10, 20], "ghost": [1, 2]}})
    assert call("/api/save", {"name": "proj"})[1]["saved"] == "proj.json"
    call("/api/new", {})
    assert call("/api/project")[1]["graph"]["nodes"] == []
    d = call("/api/load", {"name": "proj.json"})[1]
    assert [n["id"] for n in d["graph"]["nodes"]] == ["s"] and d["layout"] == {"s": [10.0, 20.0]}
    for bad in ("../evil.json", "/etc/passwd", "a/../../evil.json"):
        assert call("/api/save", {"name": bad})[0] == 403, bad
        assert call("/api/file?name=" + bad.replace("/", "%2F"))[0] == 403, bad
    assert not (tmp_path / "evil.json").exists()
    assert call("/api/file?name=missing.png")[0] == 404


def test_copilot_over_http_shares_project_and_files(gui):
    call, studio, *_ = gui
    c, d, _ = call("/api/copilot", {"prompt": "make a teal and orange LUT"})
    assert c == 200 and d["files"] == ["teal_orange.cube"]
    c, body, r = call("/api/file?name=teal_orange.cube", raw=True)
    assert c == 200 and b"LUT_3D_SIZE" in body
    c, d, _ = call("/api/copilot", {"prompt": "give me a laser sound effect"})     # builds nodes in the shared project
    assert c == 200 and d["graph"]["nodes"] and d["can_undo"]
    name = d["files"][0]
    assert call("/api/undo", {})[1]["ok"] and call("/api/project")[1]["graph"]["nodes"] == []
    assert call("/api/copilot", {"prompt": "  "})[0] == 400
    assert name in {f["name"] for f in call("/api/files")[1]}


def test_export_frames(gui):
    call, studio, *_ = gui
    call("/api/command", {"cmd": {"op": "add_node", "type": "gradient", "id": "g", "params": {"width": 32, "height": 18}}})
    c, d, _ = call("/api/export", {"node": "g", "start": 0, "end": 2, "name": "seq/f_{frame}.png"})
    assert c == 200
    names = {f["name"] for f in call("/api/files")[1]}
    assert {"seq/f_00000.png", "seq/f_00002.png"} <= names
    assert call("/api/export", {"node": "g", "start": 0, "end": 99999, "name": "x.png"})[0] == 400
    assert call("/api/export", {"node": "g", "start": 0, "end": 1, "name": "../x_{frame}.png"})[0] == 403


def test_gui_cli_registered():
    from omniimage.cli import build_parser
    a = build_parser().parse_args(["gui", "--port", "0", "--no-browser"])
    assert a.fn.__name__ == "cmd_gui"
