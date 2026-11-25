"""Built-in open-source AI Media Copilot.

The Copilot never edits pixels or samples itself. It turns a request into *tool calls* (add a node, connect, set a
parameter, bake a LUT ...) that run through the same validated, undoable command bus as every other part of the
app. Two interchangeable backends:

* ``RuleBackend``         - fully offline, deterministic keyword intent matcher (no model needed, works anywhere).
* ``OpenAICompatBackend`` - any local or remote server speaking the OpenAI chat-completions + tool-calling protocol:
                            Ollama, llama.cpp ``llama-server``, vLLM, LM Studio, LocalAI ... Only the Python standard
                            library is used, so nothing extra to install.

Safety: files the Copilot writes are confined to ``workdir``; every Copilot request is one undo step and is rolled
back completely if any tool call raises.
"""
from __future__ import annotations
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .core.graph import Ctx, Graph, GraphError
from .core.params import ParamError
from .core.project import Project
from .core.registry import REGISTRY, get_spec, list_specs
from .core.media import AudioBuf
from .lut import bake_lut
from .presets.looks import LOOKS, look_graph
from .presets.sfx import SFX, build_sfx
from .presets.vfx import EFFECTS, apply_effect, neon_sign
from .nodes.io import write_image_file, write_audio_file
from .mesh import Mesh

MAX_RESULT_CHARS = 6000


# =============================================================================================== tools
class Toolbox:
    """The actions the Copilot may take. Every method returns a JSON-serialisable dict with an ``ok`` flag."""

    def __init__(self, project: Project, workdir: str = "omni_out"):
        self.project = project
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

    # ---- helpers
    def _out_path(self, path: str) -> str:
        p = os.path.abspath(os.path.join(self.workdir, path))
        if os.path.commonpath([p, self.workdir]) != self.workdir:
            raise PermissionError(f"the Copilot may only write inside {self.workdir}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def _merge(self, sub: Graph, prefix: str) -> Dict[str, str]:
        g = self.project.graph
        ids = {}
        for n in sub.nodes.values():
            nid = f"{prefix}_{n.id}"
            while nid in g.nodes:
                nid += "_"
            ids[n.id] = nid
            g.add(n.type, id=nid, **n.params)
        for n in sub.nodes.values():
            for port, (s, sp) in n.inputs.items():
                g.connect(ids[s], ids[n.id], sp, port)
        return ids

    # ---- inspection
    def list_node_types(self, category: Optional[str] = None) -> dict:
        items = [{"type": s.type, "category": s.category, "doc": s.doc[:110]} for s in list_specs(category)]
        cats = sorted({s.category for s in REGISTRY.values()})
        return {"ok": True, "categories": cats, "count": len(items), "nodes": items}

    def describe_node_type(self, type: str) -> dict:
        return {"ok": True, **get_spec(type).describe()}

    def show_graph(self) -> dict:
        return {"ok": True, "graph": self.project.graph.describe(), "issues": self.project.graph.validate()}

    def validate(self) -> dict:
        issues = self.project.graph.validate()
        return {"ok": not issues, "issues": issues}

    def list_presets(self) -> dict:
        return {"ok": True, "looks": {k: v[0] for k, v in LOOKS.items()}, "effects": {k: v[0] for k, v in EFFECTS.items()},
                "sfx": {k: v[0] for k, v in SFX.items()}}

    # ---- editing
    def add_node(self, type: str, params: Optional[dict] = None, id: Optional[str] = None) -> dict:
        params = dict(params or {})
        if "path" in params and get_spec(type).sink:      # file-writing nodes may only write inside the work folder
            params["path"] = self._out_path(str(params["path"]))
        r = self.project.apply({"op": "add_node", "type": type, "id": id, "params": params or {}})
        return {"ok": True, "id": r["id"]}

    def connect(self, src: str, dst: str, src_port: Optional[str] = None, dst_port: Optional[str] = None) -> dict:
        self.project.apply({"op": "connect", "src": src, "dst": dst, "src_port": src_port, "dst_port": dst_port})
        return {"ok": True}

    def disconnect(self, dst: str, dst_port: str) -> dict:
        self.project.apply({"op": "disconnect", "dst": dst, "dst_port": dst_port})
        return {"ok": True}

    def set_param(self, node: str, name: str, value: Any) -> dict:
        n = self.project.graph.nodes.get(node)
        if n and get_spec(n.type).sink and name == "path":
            value = self._out_path(str(value))
        self.project.apply({"op": "set_param", "node": node, "name": name, "value": value})
        return {"ok": True}

    def remove_node(self, id: str) -> dict:
        self.project.apply({"op": "remove_node", "id": id})
        return {"ok": True}

    def clear_graph(self) -> dict:
        self.project.replace_graph(Graph())
        return {"ok": True}

    # ---- presets
    def apply_effect(self, name: str, source_node: str, amount: float = 1.0) -> dict:
        with self.project.transaction():
            sub = Graph()
            # build effect on a stand-in node, then splice onto the real source
            stand = sub.add("solid", id="__src__")
            last = apply_effect(name, sub, stand, float(amount))
            ids = self._merge_with_source(sub, stand, source_node, name)
        return {"ok": True, "output": ids[last]}

    def _merge_with_source(self, sub: Graph, stand: str, source: str, prefix: str) -> Dict[str, str]:
        g = self.project.graph
        if source not in g.nodes:
            raise GraphError(f"source node {source!r} does not exist")
        ids = {}
        for n in sub.nodes.values():
            if n.id == stand:
                ids[n.id] = source
                continue
            nid = f"{prefix}_{n.id}"
            while nid in g.nodes:
                nid += "_"
            ids[n.id] = nid
            g.add(n.type, id=nid, **n.params)
        for n in sub.nodes.values():
            if n.id == stand:
                continue
            for port, (s, sp) in n.inputs.items():
                g.connect(ids[s], ids[n.id], sp, port)
        return ids

    def load_sfx(self, name: str) -> dict:
        with self.project.transaction():
            sub, out = build_sfx(name)
            ids = self._merge(sub, f"sfx_{name}")
        return {"ok": True, "output": ids[out]}

    def create_neon_sign(self, text: str = "OMNI", color: str = "#ff2fd0") -> dict:
        with self.project.transaction():
            sub, out = neon_sign(text=text[:24], color=color, width=960, height=540)
            ids = self._merge(sub, "neon")
        return {"ok": True, "output": ids[out]}

    def create_3d_scene(self, shape: str = "torus", color: str = "#e63946", metallic: float = 0.0, roughness: float = 0.4,
                        floor: bool = True, turntable: bool = False, texture: str = "none") -> dict:
        """Primitive -> material -> [texture] -> render_mesh. ``turntable`` animates the camera orbit (azimuth = frame * 4
        degrees); ``texture`` paints a generated image onto the surface: checker | noise | gradient as colour, bump as a
        procedural normal map (noise -> normal_from_height) that makes the surface look rough."""
        shapes = {"cube": "mesh_cube", "box": "mesh_cube", "sphere": "mesh_sphere", "ball": "mesh_sphere", "torus": "mesh_torus",
                  "donut": "mesh_torus", "cylinder": "mesh_cylinder", "cone": "mesh_cone", "icosphere": "mesh_icosphere",
                  "plane": "mesh_plane"}
        if shape not in shapes:
            return {"ok": False, "error": f"unknown shape {shape!r}; choose from {sorted(set(shapes))}"}
        texture = (texture or "none").lower()
        if texture not in ("none", "checker", "noise", "gradient", "bump"):
            return {"ok": False, "error": f"unknown texture {texture!r}; choose from checker, noise, gradient, bump or none"}
        with self.project.transaction():
            a = self.project.apply({"op": "add_node", "type": shapes[shape]})["id"]
            m = self.project.apply({"op": "add_node", "type": "mesh_material",
                                   "params": {"color": color, "metallic": metallic, "roughness": roughness}})["id"]
            r = self.project.apply({"op": "add_node", "type": "render_mesh", "params": {
                "floor": "checker" if floor else "none", "azimuth": "=frame*4" if turntable else 30.0}})["id"]
            last = a
            if texture != "none":
                if shape == "icosphere":                            # the only primitive without built-in UVs
                    uv = self.project.apply({"op": "add_node", "type": "mesh_uv_project", "params": {"mode": "spherical"}})["id"]
                    self.project.apply({"op": "connect", "src": last, "dst": uv})
                    last = uv
                img_params = {"width": 512, "height": 512}
                if texture == "checker":
                    img_params["size"] = 64
                if texture == "bump":
                    img_params["scale"] = 28.0
                img = self.project.apply({"op": "add_node", "type": "noise" if texture == "bump" else texture,
                                          "params": img_params})["id"]
                port = "image"
                if texture == "bump":
                    nm = self.project.apply({"op": "add_node", "type": "normal_from_height", "params": {"strength": 10.0}})["id"]
                    self.project.apply({"op": "connect", "src": img, "dst": nm})
                    img, port = nm, "normal"
                tx = self.project.apply({"op": "add_node", "type": "mesh_texture",
                                         "params": {"scale_u": 3.0, "scale_v": 2.0} if texture == "bump" else {}})["id"]
                self.project.apply({"op": "connect", "src": last, "dst": m})
                self.project.apply({"op": "connect", "src": m, "dst": tx, "dst_port": "mesh"})
                self.project.apply({"op": "connect", "src": img, "dst": tx, "dst_port": port})
                self.project.apply({"op": "connect", "src": tx, "dst": r})
            else:
                self.project.apply({"op": "connect", "src": a, "dst": m})
                self.project.apply({"op": "connect", "src": m, "dst": r})
        return {"ok": True, "output": r}

    def create_demo_source(self, width: int = 960, height: int = 540) -> dict:
        from .presets.vfx import demo_scene
        with self.project.transaction():
            sub, out = demo_scene(int(width), int(height))
            ids = self._merge(sub, "demo")
        return {"ok": True, "output": ids[out]}

    # ---- outputs
    def export_look_lut(self, name: str, path: Optional[str] = None, size: int = 33, strength: float = 1.0) -> dict:
        from .presets.looks import make_look_lut
        out = self._out_path(path or f"{name}.cube")
        lut = make_look_lut(name, int(size), float(strength))
        lut.write_cube(out, f"OmniImage {name}")
        return {"ok": True, "path": out, "size": lut.size}

    def bake_lut(self, output: str, input_node: str, path: str = "baked.cube", size: int = 33) -> dict:
        lut = bake_lut(self.project.graph, output, input_node, size=int(size))
        out = self._out_path(path)
        lut.write_cube(out)
        return {"ok": True, "path": out, "size": lut.size}

    def render_preview(self, node: str, path: Optional[str] = None, frame: int = 0) -> dict:
        val = self.project.graph.evaluate(node, None, Ctx(int(frame), 24.0))
        if isinstance(val, np.ndarray) and val.ndim == 3:
            out = self._out_path(path or f"{node}.png")
            write_image_file(val, out)
            rgb = val[..., :3]
            return {"ok": True, "kind": "image", "path": out, "size": [val.shape[1], val.shape[0]],
                    "mean_linear_rgb": [round(float(x), 4) for x in rgb.reshape(-1, 3).mean(0)]}
        if isinstance(val, AudioBuf):
            out = self._out_path(path or f"{node}.wav")
            write_audio_file(val, out)
            return {"ok": True, "kind": "audio", "path": out, "duration_s": round(val.duration, 3),
                    "peak": round(float(np.abs(val.samples).max()), 4), "sample_rate": val.sr}
        if isinstance(val, Mesh):
            from .raster import render as render3d, RenderSettings
            img = render3d(val, RenderSettings(width=960, height=540, floor="checker"))["image"]
            out = self._out_path(path or f"{node}.png")
            write_image_file(img, out)
            return {"ok": True, "kind": "mesh", "path": out, "vertices": val.nverts, "triangles": val.nfaces}
        return {"ok": True, "kind": type(val).__name__, "value": str(val)[:200]}

    # ---- dispatch
    def call(self, name: str, args: Optional[dict] = None) -> dict:
        fn = getattr(self, name, None)
        if name.startswith("_") or name == "call" or not callable(fn):
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            res = fn(**(args or {}))
        except (GraphError, ParamError, KeyError, PermissionError, TypeError, ValueError, FileNotFoundError, OSError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return res

    def schemas(self) -> List[dict]:
        def fn(name, desc, props=None, req=()):
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
                "type": "object", "properties": props or {}, "required": list(req)}}}
        s, n, o_ = {"type": "string"}, {"type": "number"}, {"type": "object"}
        return [
            fn("list_node_types", "List available node types (optionally by category).", {"category": s}),
            fn("describe_node_type", "Get ports and parameters of a node type before using it.", {"type": s}, ["type"]),
            fn("show_graph", "Show the current node graph and validation issues."),
            fn("validate", "Check the graph for missing inputs, cycles and type errors."),
            fn("list_presets", "List built-in looks (LUTs), image effects and sound effects."),
            fn("create_3d_scene", "Build a rendered 3D scene: a primitive (cube, sphere, torus, cylinder, cone, icosphere, plane) "
                                  "with a material and a render node. turntable=true orbits the camera over time.",
               {"shape": s, "color": s, "metallic": n, "roughness": n, "floor": {"type": "boolean"}, "turntable": {"type": "boolean"},
                "texture": {"type": "string", "enum": ["none", "checker", "noise", "gradient", "bump"]}}),
            fn("add_node", "Add a node. params is an object of parameter values; any number may be animated with an "
                           "expression string like '=sin(t*6)'.", {"type": s, "params": o_, "id": s}, ["type"]),
            fn("connect", "Connect src output to dst input. Ports default to the first compatible ones.",
               {"src": s, "dst": s, "src_port": s, "dst_port": s}, ["src", "dst"]),
            fn("disconnect", "Remove a connection.", {"dst": s, "dst_port": s}, ["dst", "dst_port"]),
            fn("set_param", "Set one parameter.", {"node": s, "name": s, "value": {}}, ["node", "name", "value"]),
            fn("remove_node", "Delete a node.", {"id": s}, ["id"]),
            fn("clear_graph", "Delete every node."),
            fn("apply_effect", "Apply a built-in effect chain after source_node; returns the output node.",
               {"name": s, "source_node": s, "amount": n}, ["name", "source_node"]),
            fn("load_sfx", "Add a built-in procedural sound-effect graph; returns the output node.", {"name": s}, ["name"]),
            fn("create_neon_sign", "Create a neon-sign title graph.", {"text": s, "color": s}),
            fn("create_demo_source", "Create a synthetic test image source node.", {"width": n, "height": n}),
            fn("export_look_lut", "Bake a built-in look to a .cube file in the output folder.",
               {"name": s, "path": s, "size": n, "strength": n}, ["name"]),
            fn("bake_lut", "Bake the color-only chain between a lut_input node and output into a .cube LUT.",
               {"output": s, "input_node": s, "path": s, "size": n}, ["output", "input_node"]),
            fn("render_preview", "Render a node to a PNG (images) or WAV (audio) in the output folder.",
               {"node": s, "path": s, "frame": n}, ["node"]),
        ]


# =============================================================================================== backends
class Backend:
    def complete(self, messages: List[dict], tools: List[dict]) -> dict:  # -> {"content": str|None, "tool_calls": [...]}
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """Talk to any OpenAI-compatible chat-completions endpoint (Ollama, llama.cpp server, vLLM, LM Studio...)."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:7b-instruct",
                 api_key: str = "", timeout: float = 120.0, temperature: float = 0.2):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.temperature = timeout, temperature

    def complete(self, messages, tools):
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": self.temperature}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(f"cannot reach the model server at {self.base_url}: {e}") from e
        msg = data["choices"][0]["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name", ""), "arguments": args})
        return {"content": msg.get("content"), "tool_calls": calls}


_WORDS = {
    "sfx": {"whoosh": ["whoosh", "swoosh", "swish"], "laser": ["laser", "zap", "pew"], "explosion": ["explosion", "boom", "blast"],
            "coin": ["coin", "pickup", "pick-up", "arcade"], "riser": ["riser", "build up", "buildup", "tension"],
            "impact": ["impact", "hit", "slam"], "wind": ["wind", "breeze", "gust"]},
    "looks": {"teal_orange": ["teal", "orange and teal", "blockbuster"], "bleach_bypass": ["bleach"],
              "noir": ["noir", "black and white", "b&w", "black & white", "monochrome"],
              "vintage_fade": ["vintage", "retro", "nostalg", "faded"], "cinematic_warm": ["cinematic", "warm"],
              "cool_moody": ["moody", "cool", "cold", "blue"], "cross_process": ["cross process", "cross-process"],
              "sepia_tone": ["sepia"], "matrix_green": ["matrix", "green terminal", "hacker"],
              "vaporwave": ["vaporwave", "synthwave", "pink and cyan"], "golden_hour": ["golden hour", "sunset", "sunrise"],
              "day_for_night": ["day for night", "night"], "high_contrast_bw": ["high contrast"],
              "faded_film": ["faded film", "film print"]},
    "effects": {"vhs": ["vhs", "analog", "camcorder"], "film": ["film look", "filmic", "film grain"],
                "dreamy": ["dreamy", "bloom", "soft glow"], "glitch": ["glitch", "datamosh"],
                "toon": ["toon", "cartoon", "comic"], "hologram": ["hologram"], "heat_haze": ["heat haze", "shimmer", "mirage"]},
}


class RuleBackend(Backend):
    """Offline intent matcher: maps plain-language requests onto the same tool calls a language model would make."""

    HELP = ("I can build looks (LUTs), image effects and sound effects offline. Try: 'make a warm cinematic LUT', "
            "'give me a laser sound effect', 'create a neon sign saying HELLO', 'apply a VHS effect', "
            "'render a gold 3D torus', 'list presets', or connect an LLM backend for free-form requests.")

    def complete(self, messages, tools):
        if any(m["role"] == "tool" for m in messages):
            return {"content": self._summarise(messages), "tool_calls": []}
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        plan = self._plan(prompt)
        return {"content": None, "tool_calls": plan} if plan else {"content": self.HELP, "tool_calls": []}

    @staticmethod
    def _amount(p: str) -> float:
        if re.search(r"subtle|slight|light|gentle|soft", p):
            return 0.5
        if re.search(r"strong|heavy|intense|extreme|very|a lot", p):
            return 1.6
        return 1.0

    def _plan(self, prompt: str) -> List[dict]:
        p = prompt.lower()
        n = [0]

        def call(tool, /, **args):
            n[0] += 1
            return {"id": f"rule_{n[0]}", "name": tool, "arguments": args}

        if re.search(r"\blist\b|presets|what can you", p):
            return [call("list_presets")]
        amt = self._amount(p)
        quoted = re.search(r"[\"“'‘](.+?)[\"”'’]", prompt)
        calls: List[dict] = []
        shape = re.search(r"\b(sphere|ball|cube|box|torus|donut|cylinder|cone|icosphere)\b", p)
        if shape and re.search(r"\b3d\b|render|three.?d|turntable|model", p):
            colors = {"red": "#d62828", "green": "#2a9d4b", "blue": "#2a6fdb", "gold": "#e0a82e", "yellow": "#f2c230",
                      "orange": "#f07f1a", "purple": "#8a3ec8", "white": "#eeeeee", "black": "#151515", "pink": "#f06fa8",
                      "silver": "#c8c8cc", "chrome": "#d8d8dc", "metal": "#c0c0c8"}
            col = next((v for k, v in colors.items() if k in p), "#e63946")
            metal = 1.0 if re.search(r"gold|silver|chrome|metal", p) else 0.0
            rough = 0.2 if metal else 0.4
            tex = "bump" if re.search(r"bump|normal map|rough surface|rocky", p) else \
                "checker" if re.search(r"check|tile|textur", p) else "noise" if re.search(r"noise|marble|rock", p) else \
                "gradient" if re.search(r"gradient", p) else "none"
            if tex not in ("none", "bump") and not any(k in p for k in colors):
                col = "#ffffff"                                       # let the texture's own colours show
            calls.append(call("create_3d_scene", shape=shape.group(1), color=col, metallic=metal, roughness=rough,
                              turntable=bool(re.search(r"turntable|spin|rotat|orbit", p)), texture=tex))
            calls.append(call("render_preview", node="{OUT}", path="scene3d.png", frame=0))
            return calls
        if "neon" in p:
            text = quoted.group(1) if quoted else (re.search(r"saying\s+(\w+)", prompt, re.I) or [None, "OMNI"])[1]
            col = "#00e5ff" if "cyan" in p or "blue" in p else "#39ff14" if "green" in p else "#ff2fd0"
            calls.append(call("create_neon_sign", text=text.upper(), color=col))
            calls.append(call("render_preview", node="{OUT}", path="neon.png", frame=3))
            return calls
        for name, words in _WORDS["sfx"].items():
            if any(w in p for w in words) and re.search(r"sound|sfx|audio|effect|whoosh|laser|explosion|riser|impact|wind|coin", p):
                return [call("load_sfx", name=name), call("render_preview", node="{OUT}", path=f"{name}.wav")]
        want_lut = bool(re.search(r"\blut\b|\blook\b|grade|grading|color|colour|cinematic|noir|sepia", p))
        for name, words in _WORDS["effects"].items():
            if any(w in p for w in words) and not (name == "film" and want_lut and "film look" not in p):
                return [call("create_demo_source", width=960, height=540),
                        call("apply_effect", name=name, source_node="{OUT}", amount=amt),
                        call("render_preview", node="{OUT}", path=f"{name}.png", frame=2)]
        for name, words in _WORDS["looks"].items():
            if any(w in p for w in words):
                return [call("export_look_lut", name=name, path=f"{name}.cube", strength=min(amt, 1.0) if amt < 1 else 1.0)]
        return []

    def _summarise(self, messages) -> str:
        outs = []
        for m in messages:
            if m["role"] == "tool":
                try:
                    r = json.loads(m["content"])
                except Exception:
                    continue
                if not r.get("ok", False):
                    outs.append(f"A step failed: {r.get('error')}")
                elif r.get("path"):
                    outs.append(f"wrote {r['path']}")
                elif r.get("looks"):
                    outs.append("Looks: " + ", ".join(r["looks"]) + ". Effects: " + ", ".join(r["effects"]) +
                                ". SFX: " + ", ".join(r["sfx"]) + ".")
        return "Done. " + " ".join(outs) if outs else "Done."


# =============================================================================================== copilot
SYSTEM_PROMPT = """You are the OmniImage Copilot, an assistant inside an open-source node-based media application.
You do not draw pixels or synthesise audio yourself: you build and edit a node graph with the provided tools.
Rules:
- Inspect before you build: use list_node_types / describe_node_type if unsure about ports or parameter names.
- Connections are typed (image, mask, audio, lut, mesh, value); only matching types connect. 3D: mesh_* nodes build and modify
  meshes (Y up), mesh_material sets colour/roughness/metallic, render_mesh turns a mesh into an image (plus depth/normal passes).
- Images are scene-linear RGBA. Color nodes work in linear light; use srgb_to_linear / linear_to_srgb around
  display-referred operations. A LUT can only contain color-only nodes (no blur, grain, glow, vignette).
- Any number parameter may be animated with an expression string beginning with '=' (variables t, frame; e.g. '=0.5+0.5*sin(t*6)').
- Prefer built-in presets (list_presets, apply_effect, load_sfx, export_look_lut) when they fit, then tune parameters.
- Finish by validating the graph and rendering a preview. Tell the user briefly what you built and where files are.
- If a tool returns ok:false, read the error, fix the call and retry; do not repeat the same failing call."""


@dataclass
class CopilotResult:
    text: str
    actions: List[dict] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


class Copilot:
    def __init__(self, project: Optional[Project] = None, backend: Optional[Backend] = None,
                 workdir: str = "omni_out", max_steps: int = 14):
        self.project = project or Project()
        self.backend = backend or RuleBackend()
        self.tools = Toolbox(self.project, workdir)
        self.max_steps = max_steps
        self.history: List[dict] = []

    def undo(self) -> bool:
        return self.project.undo()

    def ask(self, prompt: str) -> CopilotResult:
        """Run one request. All graph edits it makes form a single undo step; failures roll everything back."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history,
                    {"role": "user", "content": prompt}]
        actions: List[dict] = []
        files: List[str] = []
        schemas = self.tools.schemas()
        final = ""
        with self.project.transaction():
            for _ in range(self.max_steps):
                resp = self.backend.complete(messages, schemas)
                calls = resp.get("tool_calls") or []
                if not calls:
                    final = resp.get("content") or "Done."
                    break
                messages.append({"role": "assistant", "content": resp.get("content"), "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]})
                last_output = None
                for c in calls:
                    args = dict(c["arguments"])
                    # the offline planner uses "{OUT}" as a placeholder for the previous step's output node
                    for k, v in list(args.items()):
                        if isinstance(v, str) and "{OUT}" in v and last_output:
                            args[k] = v.replace("{OUT}", last_output)
                    res = self.tools.call(c["name"], args)
                    if isinstance(res, dict) and res.get("output"):
                        last_output = res["output"]
                    actions.append({"tool": c["name"], "args": args, "ok": res.get("ok", False)})
                    if res.get("path"):
                        files.append(res["path"])
                    txt = json.dumps(res, default=str)
                    if len(txt) > MAX_RESULT_CHARS:
                        txt = txt[:MAX_RESULT_CHARS] + '..."truncated"}'
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": txt})
            else:
                final = "I stopped after the maximum number of steps; the graph so far is kept (undo to revert)."
        self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": final}]
        self.history = self.history[-12:]
        return CopilotResult(final, actions, files)
