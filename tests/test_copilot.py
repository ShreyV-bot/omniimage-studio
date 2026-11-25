import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import omniimage as o
from omniimage.ai_assistant import Copilot, OpenAICompatBackend, Toolbox, RuleBackend


def test_rule_backend_makes_lut_sfx_and_effect(tmp_path):
    c = Copilot(workdir=str(tmp_path))
    r = c.ask("make a teal and orange LUT")
    assert (tmp_path / "teal_orange.cube").exists() and r.files
    r = c.ask("give me a laser sound effect")
    assert (tmp_path / "laser.wav").exists()
    r = c.ask('create a neon sign saying "HELLO"')
    assert (tmp_path / "neon.png").exists()
    r = c.ask("apply a strong VHS effect")
    assert (tmp_path / "vhs.png").exists()
    assert not o.Graph.validate(c.project.graph)


def test_each_request_is_one_undo_step(tmp_path):
    c = Copilot(workdir=str(tmp_path))
    c.ask("give me a laser sound effect")
    n1 = len(c.project.graph.nodes)
    assert n1 > 3
    c.ask("give me a whoosh sound effect")
    assert len(c.project.graph.nodes) > n1
    assert c.undo()
    assert len(c.project.graph.nodes) == n1


def test_unknown_request_gets_help_not_garbage(tmp_path):
    c = Copilot(workdir=str(tmp_path))
    r = c.ask("write me a poem")
    assert "Try:" in r.text and not c.project.graph.nodes


def test_copilot_cannot_write_outside_workdir(tmp_path):
    tb = Toolbox(o.Project(), str(tmp_path / "work"))
    assert tb.call("add_node", {"type": "solid", "id": "s", "params": {"width": 4, "height": 4}})["ok"]
    assert tb.call("add_node", {"type": "write_image", "id": "w"})["ok"]
    res = tb.call("set_param", {"node": "w", "name": "path", "value": "../../evil.png"})
    assert not res["ok"] and "only write inside" in res["error"]
    res = tb.call("set_param", {"node": "w", "name": "path", "value": "/etc/evil.png"})
    assert not res["ok"]
    assert not (tmp_path / "evil.png").exists()
    assert tb.call("set_param", {"node": "w", "name": "path", "value": "ok/inside.png"})["ok"]


def test_toolbox_errors_are_reported_not_raised(tmp_path):
    tb = Toolbox(o.Project(), str(tmp_path))
    assert not tb.call("add_node", {"type": "nope"})["ok"]
    assert not tb.call("connect", {"src": "a", "dst": "b"})["ok"]
    assert not tb.call("__init__", {})["ok"]
    assert not tb.call("does_not_exist", {})["ok"]
    assert not tb.call("add_node", {"bad_arg": 1})["ok"]


def test_bake_lut_tool_and_refusal(tmp_path):
    tb = Toolbox(o.Project(), str(tmp_path))
    for spec in ({"type": "lut_input", "id": "i"}, {"type": "exposure", "id": "e", "params": {"stops": 1}}):
        tb.call("add_node", spec)
    tb.call("connect", {"src": "i", "dst": "e"})
    assert tb.call("bake_lut", {"output": "e", "input_node": "i", "path": "x.cube", "size": 9})["ok"]
    assert (tmp_path / "x.cube").exists()
    tb.call("add_node", {"type": "blur_gaussian", "id": "b"})
    tb.call("connect", {"src": "e", "dst": "b"})
    res = tb.call("bake_lut", {"output": "b", "input_node": "i", "path": "y.cube"})
    assert not res["ok"] and "cannot bake" in res["error"]


class FakeLLM(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: first answers with tool calls, then with a final message."""
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.seen.append(body)
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if not has_tool_result:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "add_node", "arguments": json.dumps(
                    {"type": "gradient", "id": "bg", "params": {"width": 64, "height": 36}})}},
                {"id": "c2", "type": "function", "function": {"name": "add_node", "arguments": json.dumps(
                    {"type": "glow", "id": "g"})}},
                {"id": "c3", "type": "function", "function": {"name": "connect", "arguments": json.dumps(
                    {"src": "bg", "dst": "g"})}},
                {"id": "c4", "type": "function", "function": {"name": "render_preview", "arguments": json.dumps(
                    {"node": "g", "path": "glow.png"})}}]}
        else:
            msg = {"role": "assistant", "content": "Built a glow graph."}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_openai_compatible_backend_tool_loop(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = OpenAICompatBackend(f"http://127.0.0.1:{srv.server_port}/v1", "fake", timeout=10)
        c = Copilot(backend=be, workdir=str(tmp_path))
        r = c.ask("add a glow to a gradient")
    finally:
        srv.shutdown()
    assert r.text == "Built a glow graph."
    assert set(c.project.graph.nodes) == {"bg", "g"}
    assert (tmp_path / "glow.png").exists()
    first = FakeLLM.seen[0]
    assert first["tools"] and first["messages"][0]["role"] == "system"
    assert {t["function"]["name"] for t in first["tools"]} >= {"add_node", "connect", "set_param", "bake_lut"}


def test_unreachable_server_gives_clear_error(tmp_path):
    c = Copilot(backend=OpenAICompatBackend("http://127.0.0.1:9/v1", "x", timeout=2), workdir=str(tmp_path))
    with pytest.raises(ConnectionError, match="cannot reach"):
        c.ask("hello")
    assert not c.project.graph.nodes
