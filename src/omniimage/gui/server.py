"""A small, dependency-free HTTP server that exposes a Project to a browser UI.

Security model (the server can render files and run the Copilot, so it is locked down):
  * binds to 127.0.0.1 only, and rejects requests whose Host header is not localhost (DNS-rebinding defence);
  * every /api call needs the random per-process token that is embedded in the page it served;
  * POSTs must be application/json, which a cross-site HTML form cannot send;
  * all file access (save/load/serve/export/Copilot writes) is confined to the work folder.
"""
from __future__ import annotations
import hmac
import json
import os
import re
import secrets
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Optional
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

from ..core.graph import Graph, GraphError, Ctx
from ..core.params import ParamError
from ..core.project import Project
from ..core.registry import REGISTRY
from ..core.ports import PortType as T
from ..core.media import linear_to_srgb
from ..ai_assistant import Copilot, RuleBackend, Backend
from ..timeline import Timeline
from ..timeline_edit import TimelineEditor, EditError

SCOPES = {"waveform": "scope_waveform", "parade": "scope_parade", "vectorscope": "scope_vectorscope",
          "histogram": "scope_histogram"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".wav": "audio/wav",
        ".mp4": "video/mp4", ".webm": "video/webm", ".cube": "text/plain", ".json": "application/json",
        ".exr": "application/octet-stream"}
MAX_BODY = 8 * 1024 * 1024
MAX_EXPORT_FRAMES = 2000
MAX_TIMELINE_FILE = 16 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


class Studio:
    """Server-side state: one project, its Copilot, node layout, and the work folder."""

    def __init__(self, project: Optional[Project] = None, workdir: str = "omni_out", backend: Optional[Backend] = None):
        self.project = project or Project()
        self.workdir = os.path.realpath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.copilot = Copilot(self.project, backend or RuleBackend(), workdir=self.workdir)
        self.layout: dict = {}
        self.lock = threading.RLock()
        self.tl = TimelineEditor(resolve_media=self.media_path, project_graph=lambda: self.project.graph)

    # ---- files -----------------------------------------------------------
    def safe_path(self, name: str) -> str:
        if not name or os.path.isabs(name) or "\x00" in name:
            raise ApiError("file name must be relative to the work folder", 403)
        p = os.path.realpath(os.path.join(self.workdir, name))
        if p != self.workdir and not p.startswith(self.workdir + os.sep):
            raise ApiError("path escapes the work folder", 403)
        return p

    def media_path(self, name: str) -> str:
        p = self.safe_path(name)
        if not os.path.isfile(p):
            raise ApiError(f"no such file in the work folder: {name}", 404)
        return p

    def list_files(self):
        out = []
        for root, _, files in os.walk(self.workdir):
            for f in files:
                full = os.path.join(root, f)
                out.append({"name": os.path.relpath(full, self.workdir).replace(os.sep, "/"), "size": os.path.getsize(full)})
        return sorted(out, key=lambda d: d["name"])[:500]

    # ---- project ---------------------------------------------------------
    def snapshot(self) -> dict:
        p = self.project
        return {"graph": p.graph.to_dict(), "layout": self.layout, "problems": p.graph.validate(),
                "can_undo": bool(p._undo), "can_redo": bool(p._redo)}

    def command(self, cmds):
        with self.project.transaction():
            results = [self.project.apply(c) for c in cmds]
        live = set(self.project.graph.nodes)
        self.layout = {k: v for k, v in self.layout.items() if k in live}
        return results

    def save(self, name):
        path = self.safe_path(name if name.endswith(".json") else name + ".json")
        d = self.project.graph.to_dict()
        d["layout"] = self.layout
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        return os.path.relpath(path, self.workdir)

    def load(self, name):
        path = self.safe_path(name)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.project.replace_graph(Graph.from_dict(d))
        self.layout = d.get("layout", {}) if isinstance(d.get("layout"), dict) else {}

    # ---- rendering -------------------------------------------------------
    def evaluate(self, node, port, frame, fps):
        g = self.project.graph
        if node not in g.nodes:
            raise ApiError(f"no node {node!r}", 404)
        spec = REGISTRY[g.nodes[node].type]
        if not spec.outputs:
            raise ApiError(f"{node} ({spec.type}) has no output to view")
        port = port or next(iter(spec.outputs))
        if port not in spec.outputs:
            raise ApiError(f"{node} has no output {port!r}")
        return spec.outputs[port], g.evaluate(node, port, Ctx(frame, fps))

    def render_png(self, node, port=None, frame=0, fps=24.0, scope="none", maxw=640):
        with self.lock:
            kind, val = self.evaluate(node, port, frame, fps)
            if kind == T.AUDIO:
                tmp = Graph(); src = tmp.add("osc"); w = tmp.add("waveform_image", width=maxw, height=max(64, maxw // 4))
                tmp.connect(src, w)
                img = tmp.evaluate(w, overrides={src: {"audio": val}}, override_token=uuid.uuid4().hex)
                return _encode(img, maxw), "audio"
            if kind == T.MESH:
                from ..raster import render as render3d, RenderSettings
                w = max(32, min(int(maxw), 960))
                r = render3d(val, RenderSettings(width=w, height=max(18, w * 9 // 16), ssaa=1, floor="checker", shadow_res=512))
                return _encode(r["image"], maxw), "mesh"
            if kind == T.MASK:
                m = np.clip(np.asarray(val, np.float32), 0, 1)
                val = np.dstack([m, m, m, np.ones_like(m)])
                kind = T.IMAGE
            if kind != T.IMAGE:
                raise ApiError(f"output type {kind.value!r} cannot be previewed as a picture")
            if scope != "none":
                if scope not in SCOPES:
                    raise ApiError(f"unknown scope {scope!r}")
                tmp = Graph(); src = tmp.add("solid"); dims = {"size": 256} if scope == "vectorscope" else {"width": min(maxw, 640), "height": 256}
                sc = tmp.add(SCOPES[scope], **dims)
                tmp.connect(src, sc)
                val = tmp.evaluate(sc, overrides={src: {"image": val}}, override_token=uuid.uuid4().hex)
                return _encode(val, maxw, checker=False), "scope"
            return _encode(val, maxw), "image"

    def export(self, node, start, end, fps, name):
        from ..render import render_frames
        if end < start or end - start + 1 > MAX_EXPORT_FRAMES:
            raise ApiError(f"frame range must be 1..{MAX_EXPORT_FRAMES} frames")
        path = self.safe_path(name)
        with self.lock:
            out = render_frames(self.project.graph, node, path, start, end, fps)
        return os.path.relpath(out, self.workdir).replace(os.sep, "/")

    # ---- timeline --------------------------------------------------------
    def tl_edit(self, ops):
        results = self.tl.apply(ops)
        return {"results": results, "view": self.tl.view()}

    def tl_save(self, name):
        base = name[:-5] if name.endswith(".json") else name
        base = base[:-9] if base.endswith(".timeline") else base
        path = self.safe_path(base + ".timeline.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tl.tl.save(path)
        return os.path.relpath(path, self.workdir).replace(os.sep, "/")

    def tl_load(self, name):
        path = self.safe_path(name)
        if not os.path.isfile(path):
            raise ApiError("no such file", 404)
        if os.path.getsize(path) > MAX_TIMELINE_FILE:
            raise ApiError("timeline file is too large", 413)
        try:
            tl = Timeline.load(path)
        except (KeyError, TypeError, AttributeError, IndexError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise EditError(f"not a valid timeline file ({type(e).__name__}: {e})")
        self.tl.replace(tl)

    def tl_export(self, name, start, end):
        from ..timeline import render_timeline
        tl = self.tl.tl
        if tl.duration < 1:
            raise ApiError("the timeline is empty")
        end = tl.duration - 1 if end is None else min(end, tl.duration - 1)
        if start < 0 or end < start or end - start + 1 > MAX_EXPORT_FRAMES:
            raise ApiError(f"frame range must be 1..{MAX_EXPORT_FRAMES} frames inside the timeline")
        out = render_timeline(tl, self.safe_path(name), start, end)
        return os.path.relpath(out, self.workdir).replace(os.sep, "/")

    def tl_interchange(self, fmt, name):
        from ..interchange import export_edl, export_otio
        if fmt not in ("edl", "otio"):
            raise ApiError("format must be edl or otio")
        path = self.safe_path(name or ("timeline." + fmt))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = export_edl(self.tl.tl, path, include_audio=True) if fmt == "edl" else export_otio(self.tl.tl, path)
        return os.path.relpath(out, self.workdir).replace(os.sep, "/")

    def ask_copilot(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError("empty prompt")
        with self.lock:
            r = self.copilot.ask(prompt.strip())
            files = [os.path.relpath(f, self.workdir).replace(os.sep, "/") if os.path.isabs(f) else f for f in r.files]
            return {"text": r.text, "actions": r.actions, "files": files, **self.snapshot()}


def _encode(img, maxw, checker=True):
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    if w > maxw:
        a = cv2.resize(a, (maxw, max(1, int(round(h * maxw / w)))), interpolation=cv2.INTER_AREA)
        h, w = a.shape[:2]
    rgb = np.clip(linear_to_srgb(np.clip(a[..., :3], 0, None)), 0, 1)
    alpha = np.clip(a[..., 3:4], 0, 1)
    if checker:
        yy, xx = np.mgrid[0:h, 0:w]
        bg = (((yy // 8 + xx // 8) % 2) * 0.12 + 0.28).astype(np.float32)[..., None]
        rgb = rgb * alpha + bg * (1 - alpha)
    ok, buf = cv2.imencode(".png", (rgb[..., ::-1] * 255 + 0.5).astype(np.uint8))
    if not ok:
        raise ApiError("could not encode preview", 500)
    return buf.tobytes()


def _node_catalog():
    return [s.describe() for s in sorted(REGISTRY.values(), key=lambda s: (s.category, s.type))]


class Handler(BaseHTTPRequestHandler):
    server_version = "OmniImageGUI/0.8"

    def log_message(self, *a):
        pass

    # ---- plumbing --------------------------------------------------------
    @property
    def studio(self) -> Studio:
        return self.server.studio

    def _host_ok(self):
        m = re.fullmatch(r"(localhost|127\.0\.0\.1|\[::1\])(:\d+)?", self.headers.get("Host") or "")
        return m is not None

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline' blob: data:")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # the browser cancelled the request (scrubbing, closing the tab)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _fail(self, e):
        if isinstance(e, ApiError):
            return self._json({"error": str(e)}, e.code)
        if isinstance(e, (GraphError, ParamError, ValueError, KeyError, FileNotFoundError, ConnectionError)):
            return self._json({"error": str(e.args[0] if isinstance(e, KeyError) and e.args else e)}, 400)
        return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _guard(self, post):
        if not self._host_ok():
            raise ApiError("bad Host header", 403)
        tok = self.headers.get("X-Omni-Token", "")
        if not hmac.compare_digest(tok, self.server.token):
            raise ApiError("missing or wrong token", 403)
        if post and (self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json"):
            raise ApiError("POST needs Content-Type: application/json", 415)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ApiError("request too large", 413)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            d = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise ApiError("invalid JSON body")
        if not isinstance(d, dict):
            raise ApiError("JSON body must be an object")
        return d

    # ---- routes ----------------------------------------------------------
    def do_GET(self):
        try:
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                if not self._host_ok():
                    raise ApiError("bad Host header", 403)
                html = resources.files("omniimage.gui").joinpath("static/index.html").read_text(encoding="utf-8")
                return self._send(200, html.replace("__OMNI_TOKEN__", self.server.token).encode(), "text/html; charset=utf-8")
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not u.path.startswith("/api/"):
                raise ApiError("not found", 404)
            self._guard(False)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            s = self.studio
            if u.path == "/api/nodes":
                return self._json(_node_catalog())
            if u.path == "/api/project":
                with s.lock:
                    return self._json(s.snapshot())
            if u.path == "/api/files":
                return self._json(s.list_files())
            if u.path == "/api/render":
                png, kind = s.render_png(q.get("node", ""), q.get("port") or None, int(q.get("frame", 0)),
                                         float(q.get("fps", 24)), q.get("scope", "none"), max(32, min(2048, int(q.get("maxw", 640)))))
                return self._send(200, png, "image/png", {"X-Omni-Kind": kind})
            if u.path.startswith("/api/timeline"):
                with s.lock:
                    return self._timeline_get(u.path[len("/api/timeline"):], q)
            if u.path == "/api/file":
                p = s.safe_path(q.get("name", ""))
                if not os.path.isfile(p):
                    raise ApiError("no such file", 404)
                with open(p, "rb") as f:
                    return self._send(200, f.read(), MIME.get(os.path.splitext(p)[1].lower(), "application/octet-stream"))
            raise ApiError("not found", 404)
        except Exception as e:
            self._fail(e)

    def _timeline_get(self, sub, q):
        ed = self.studio.tl
        ref = lambda: {"track": int(q.get("track", -1)), "clip": int(q.get("clip", -1))}
        if sub in ("", "/"):
            return self._json(ed.view())
        if sub == "/frame":
            img = ed.frame(int(q.get("t", 0)), max(32, min(2048, int(q.get("maxw", 640)))))
            return self._send(200, _encode(img, 2048, checker=False), "image/png")
        if sub == "/thumb":
            return self._send(200, _encode(ed.thumb(ref()), max(32, min(512, int(q.get("maxw", 160)))), checker=False), "image/png")
        if sub == "/peaks":
            return self._json({"peaks": ed.peaks(ref(), int(q.get("n", 200)))})
        if sub == "/clip":
            return self._json(ed.clip_graph(ref()))
        if sub == "/audio":
            return self._send(200, ed.mix_wav(), "audio/wav")
        raise ApiError("not found", 404)

    def do_POST(self):
        try:
            u = urlparse(self.path)
            self._guard(True)
            d = self._body()
            s = self.studio
            with s.lock:
                if u.path == "/api/command":
                    cmds = d.get("cmds") if "cmds" in d else [d.get("cmd")]
                    if not isinstance(cmds, list) or not all(isinstance(c, dict) for c in cmds):
                        raise ApiError("send {cmd: {...}} or {cmds: [{...}]}")
                    res = s.command(cmds)
                    return self._json({"results": res, **s.snapshot()})
                if u.path == "/api/undo":
                    return self._json({"ok": s.project.undo(), **s.snapshot()})
                if u.path == "/api/redo":
                    return self._json({"ok": s.project.redo(), **s.snapshot()})
                if u.path == "/api/layout":
                    lay = d.get("layout", {})
                    if not isinstance(lay, dict):
                        raise ApiError("layout must be an object")
                    s.layout = {str(k): [float(v[0]), float(v[1])] for k, v in lay.items() if k in s.project.graph.nodes}
                    return self._json({"ok": True})
                if u.path == "/api/save":
                    return self._json({"saved": s.save(str(d.get("name", "project.json")))})
                if u.path == "/api/load":
                    s.load(str(d.get("name", "")))
                    return self._json(s.snapshot())
                if u.path == "/api/new":
                    s.project.replace_graph(Graph()); s.layout = {}
                    return self._json(s.snapshot())
                if u.path == "/api/timeline/edit":
                    ops = d.get("ops") if "ops" in d else [d.get("op")]
                    if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
                        raise ApiError("send {op: {...}} or {ops: [{...}]}")
                    return self._json(s.tl_edit(ops))
                if u.path == "/api/timeline/undo":
                    return self._json({"ok": s.tl.undo(), "view": s.tl.view()})
                if u.path == "/api/timeline/redo":
                    return self._json({"ok": s.tl.redo(), "view": s.tl.view()})
                if u.path == "/api/timeline/new":
                    s.tl.replace(Timeline())
                    return self._json({"view": s.tl.view()})
                if u.path == "/api/timeline/demo":
                    from ..presets.timeline_demo import demo_timeline
                    s.tl.replace(demo_timeline())
                    return self._json({"view": s.tl.view()})
                if u.path == "/api/timeline/save":
                    return self._json({"saved": s.tl_save(str(d.get("name", "timeline")))})
                if u.path == "/api/timeline/load":
                    s.tl_load(str(d.get("name", "")))
                    return self._json({"view": s.tl.view()})
                if u.path == "/api/timeline/export":
                    end = d.get("end")
                    return self._json({"file": s.tl_export(str(d.get("name", "timeline.mp4")), int(d.get("start", 0)),
                                                           None if end is None else int(end))})
                if u.path == "/api/timeline/interchange":
                    return self._json({"file": s.tl_interchange(str(d.get("format", "edl")), str(d.get("name", "")))})
                if u.path == "/api/copilot":
                    return self._json(s.ask_copilot(d.get("prompt")))
                if u.path == "/api/export":
                    out = s.export(str(d.get("node", "")), int(d.get("start", 0)), int(d.get("end", 0)),
                                   float(d.get("fps", 24)), str(d.get("name", "render.mp4")))
                    return self._json({"file": out})
            raise ApiError("not found", 404)
        except Exception as e:
            self._fail(e)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, studio: Studio, host="127.0.0.1", port=8765):
        super().__init__((host, port), Handler)
        self.studio = studio
        self.token = secrets.token_urlsafe(24)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/"


def serve(studio: Optional[Studio] = None, port=8765, open_browser=True):
    srv = StudioServer(studio or Studio(), port=port)
    print(f"OmniImage Studio GUI: {srv.url}   (work folder: {srv.studio.workdir})   Ctrl+C to stop")
    if open_browser:
        try:
            webbrowser.open(srv.url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
