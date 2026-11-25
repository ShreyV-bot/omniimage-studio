"""Command line interface:  python -m omniimage <command>   (or ``omni`` when installed)."""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .core.graph import Ctx, Graph, GraphError
from .core.media import AudioBuf
from .core.registry import REGISTRY, list_specs
from .core.project import Project
from .lut import LUT3D, bake_lut


def _eval_to_file(g: Graph, target: str, out: str, frame: int, fps: float):
    from .nodes.io import write_image_file, write_audio_file
    v = g.evaluate(target, None, Ctx(frame, fps))
    if isinstance(v, AudioBuf):
        return write_audio_file(v, out)
    if isinstance(v, np.ndarray) and v.ndim == 3:
        return write_image_file(v, out)
    raise GraphError(f"node {target!r} produced {type(v).__name__}, which cannot be written to a file")


def cmd_nodes(a):
    specs = list_specs(a.category)
    if a.json:
        print(json.dumps([s.describe() for s in specs], indent=2))
        return 0
    cat = None
    for s in specs:
        if s.category != cat:
            cat = s.category
            print(f"\n[{cat}]")
        flag = " (LUT-safe)" if s.pointwise else ""
        print(f"  {s.type:22s} {s.doc.split('. ')[0][:78]}{flag}")
    print(f"\n{len(specs)} node types")
    return 0


def cmd_run(a):
    g = Graph.load(a.graph)
    issues = g.validate()
    if issues:
        print("graph has problems:\n  " + "\n  ".join(issues), file=sys.stderr)
        return 2
    path = _eval_to_file(g, a.target, a.out, a.frame, a.fps)
    print("wrote", path)
    return 0


def cmd_render(a):
    from .render import render_frames
    from .nodes.io import read_audio_file
    g = Graph.load(a.graph)
    audio = None
    if a.audio_node:
        audio = g.evaluate(a.audio_node)
    elif a.audio_file:
        audio = read_audio_file(a.audio_file)
    last = [0]

    def prog(i, n):
        if i == n or i - last[0] >= max(1, n // 10):
            last[0] = i
            print(f"  frame {i}/{n}", flush=True)

    out = render_frames(g, a.target, a.out, a.start, a.end, a.fps, audio=audio, crf=a.crf, progress=prog)
    print("wrote", out)
    return 0


def cmd_lut(a):
    from .presets.looks import LOOKS, make_look_lut
    if a.action == "list":
        for k, (d, _) in LOOKS.items():
            print(f"  {k:18s} {d}")
        return 0
    if a.action == "make":
        lut = make_look_lut(a.name, a.size, a.strength)
        lut.write_cube(a.out, f"OmniImage {a.name}")
        print(f"wrote {a.out} ({lut.size}^3)")
        return 0
    if a.action == "bake":
        g = Graph.load(a.graph)
        lut = bake_lut(g, a.output, a.input, size=a.size)
        lut.write_cube(a.out)
        print(f"wrote {a.out} ({lut.size}^3)")
        return 0
    if a.action == "apply":
        from .nodes.io import read_image_file, write_image_file
        g = Graph()
        src = g.add("load_image", path=a.image)
        lut = g.add("lut_apply", path=a.lut, intensity=a.intensity)
        g.connect(src, lut)
        write_image_file(g.evaluate(lut), a.out)
        print("wrote", a.out)
        return 0
    if a.action == "inspect":
        lut = LUT3D.read_cube(a.lut)
        ident = LUT3D.identity(lut.size).table
        print(f"{a.lut}: title={lut.title!r} size={lut.size}^3 domain={lut.domain_min.tolist()}..{lut.domain_max.tolist()}")
        print(f"  max deviation from identity: {float(np.abs(lut.table - ident).max()):.4f}")
        return 0
    return 1


def cmd_sfx(a):
    from .presets.sfx import SFX, build_sfx
    from .nodes.io import write_audio_file
    if a.action == "list":
        for k, (d, _) in SFX.items():
            print(f"  {k:12s} {d}")
        return 0
    g, out = build_sfx(a.name)
    audio = g.evaluate(out)
    write_audio_file(audio, a.out, a.bits)
    print(f"wrote {a.out} ({audio.duration:.2f}s, {audio.sr} Hz)")
    return 0


def cmd_fx(a):
    from .presets.vfx import EFFECTS, apply_effect
    from .nodes.io import write_image_file
    if a.action == "list":
        for k, (d, _) in EFFECTS.items():
            print(f"  {k:12s} {d}")
        return 0
    g = Graph()
    src = g.add("load_image", path=a.image)
    last = apply_effect(a.name, g, src, a.amount)
    write_image_file(g.evaluate(last, ctx=Ctx(a.frame)), a.out)
    print("wrote", a.out)
    return 0


def cmd_copilot(a):
    from .ai_assistant import Copilot, OpenAICompatBackend, RuleBackend
    backend = RuleBackend() if a.backend == "rule" else OpenAICompatBackend(a.base_url, a.model, a.api_key or os.environ.get("OPENAI_API_KEY", ""))
    pilot = Copilot(Project(), backend, workdir=a.workdir)

    def one(q):
        r = pilot.ask(q)
        print(r.text)
        for f in r.files:
            print("  ->", f)

    if a.prompt:
        one(" ".join(a.prompt))
        return 0
    print("OmniImage Copilot (empty line to quit). Backend:", a.backend)
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        if q == "undo":
            print("undone" if pilot.undo() else "nothing to undo")
            continue
        try:
            one(q)
        except Exception as e:
            print("error:", e)
    return 0


def cmd_mesh(a):
    from . import mesh as M
    if a.mcmd == "make":
        kw = {"cube": {}, "plane": {}, "sphere": {}, "icosphere": {}, "cylinder": {}, "cone": {}, "torus": {}}
        if a.shape not in kw:
            raise ValueError(f"unknown shape {a.shape!r}; choose from {sorted(kw)}")
        m = M.PRIMITIVES[a.shape]()
        for _ in range(a.subdivide):
            m = M.subdivide(m, 1)
        M.write_mesh(m, a.out)
        print(f"wrote {a.out} ({m.nverts} vertices, {m.nfaces} triangles)")
    elif a.mcmd == "info":
        m = M.read_mesh(a.file)
        lo, hi = m.bounds()
        print(f"{a.file}: {m.nverts} vertices, {m.nfaces} triangles, watertight={m.is_watertight()}, "
              f"area={m.surface_area():.4g}, volume={m.volume():.4g}")
        print("bounds:", [round(float(x), 4) for x in lo], "->", [round(float(x), 4) for x in hi])
        print(f"uv: {'yes' if m.has_uv else 'no'}, textures: {len(m.textures)}")
    elif a.mcmd == "render":
        from .raster import render, RenderSettings
        from .nodes.io import write_image_file
        s = RenderSettings(width=a.width, height=a.height, ssaa=a.quality, azimuth=a.azimuth, elevation=a.elevation,
                           floor="checker" if a.floor else "none", wireframe=a.wireframe)
        m = M.read_mesh(a.file)
        write_image_file(render(m, s)["image"], a.out)
        print(f"wrote {a.out}")
    return 0


def cmd_gui(a):
    from .gui import Studio, serve
    from .ai_assistant import OpenAICompatBackend, RuleBackend
    backend = RuleBackend() if a.backend == "rule" else OpenAICompatBackend(
        a.base_url, a.model, a.api_key or os.environ.get("OPENAI_API_KEY", ""))
    studio = Studio(Project.load(a.project) if a.project else None, a.workdir, backend)
    serve(studio, port=a.port, open_browser=not a.no_browser)
    return 0


def cmd_demo(a):
    from .demo import run_demo
    run_demo(a.out, quick=a.quick)
    return 0


def cmd_docs(a):
    from .docgen import node_reference
    text = node_reference()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", a.out)
    else:
        print(text)
    return 0


def cmd_timeline(a):
    from .timeline import Timeline, render_timeline
    from .loudness import measure_lufs, sample_peak_db
    if a.tcmd == "demo":
        from .presets.timeline_demo import demo_timeline
        demo_timeline().save(a.out)
        print("wrote", a.out)
        return 0
    tl = Timeline.load(a.file)
    if a.tcmd == "info":
        print(f"{tl.name}: {tl.width}x{tl.height} @ {tl.fps:g} fps, {tl.duration} frames ({tl.seconds(tl.duration):.2f}s)")
        for t in tl.tracks:
            print(f"  {t.kind:5s} {t.name}: {len(t.clips)} clip(s)" + (f", bus={t.bus}" if t.kind == "audio" else ""))
        if tl.audio_tracks():
            au = tl.render_audio()
            print(f"  mix: {measure_lufs(au):.1f} LUFS, peak {sample_peak_db(au):.1f} dBFS")
        return 0
    if a.tcmd == "render":
        out = render_timeline(tl, a.out, a.start, a.end, a.crf)
        print("wrote", out)
        return 0
    if a.tcmd == "export":
        from .interchange import export_edl, export_otio
        if a.format == "edl":
            print("wrote", export_edl(tl, a.out, include_audio=a.audio))
        else:
            print("wrote", export_otio(tl, a.out))
        return 0
    return 1


def build_parser():
    p = argparse.ArgumentParser(prog="omni", description="OmniImage: open-source node-based media engine")
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("nodes", help="list node types"); s.add_argument("--category"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_nodes)
    s = sp.add_parser("run", help="evaluate one node of a graph file to an image/audio file")
    s.add_argument("graph"); s.add_argument("--target", required=True); s.add_argument("-o", "--out", required=True)
    s.add_argument("--frame", type=int, default=0); s.add_argument("--fps", type=float, default=24.0); s.set_defaults(fn=cmd_run)
    s = sp.add_parser("render", help="render frames to video (with optional audio) or an image sequence")
    s.add_argument("graph"); s.add_argument("--target", required=True); s.add_argument("-o", "--out", required=True)
    s.add_argument("--start", type=int, default=0); s.add_argument("--end", type=int, default=47)
    s.add_argument("--fps", type=float, default=24.0); s.add_argument("--crf", type=int, default=18)
    s.add_argument("--audio-node"); s.add_argument("--audio-file"); s.set_defaults(fn=cmd_render)

    s = sp.add_parser("lut", help="LUT tools")
    ls = s.add_subparsers(dest="action", required=True)
    ls.add_parser("list")
    m = ls.add_parser("make"); m.add_argument("name"); m.add_argument("-o", "--out", required=True)
    m.add_argument("--size", type=int, default=33); m.add_argument("--strength", type=float, default=1.0)
    b = ls.add_parser("bake"); b.add_argument("graph"); b.add_argument("--input", required=True)
    b.add_argument("--output", required=True); b.add_argument("-o", "--out", required=True); b.add_argument("--size", type=int, default=33)
    ap = ls.add_parser("apply"); ap.add_argument("image"); ap.add_argument("lut"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--intensity", type=float, default=1.0)
    i = ls.add_parser("inspect"); i.add_argument("lut")
    s.set_defaults(fn=cmd_lut)

    s = sp.add_parser("sfx", help="procedural sound effects")
    ss = s.add_subparsers(dest="action", required=True)
    ss.add_parser("list")
    m = ss.add_parser("make"); m.add_argument("name"); m.add_argument("-o", "--out", required=True)
    m.add_argument("--bits", default="16", choices=["16", "24", "32f"])
    s.set_defaults(fn=cmd_sfx)

    s = sp.add_parser("fx", help="image effects")
    fs = s.add_subparsers(dest="action", required=True)
    fs.add_parser("list")
    m = fs.add_parser("apply"); m.add_argument("name"); m.add_argument("image"); m.add_argument("-o", "--out", required=True)
    m.add_argument("--amount", type=float, default=1.0); m.add_argument("--frame", type=int, default=0)
    s.set_defaults(fn=cmd_fx)

    s = sp.add_parser("copilot", help="AI Media Copilot (offline rules or any OpenAI-compatible local LLM)")
    s.add_argument("prompt", nargs="*"); s.add_argument("--backend", choices=["rule", "openai"], default="rule")
    s.add_argument("--base-url", default="http://localhost:11434/v1"); s.add_argument("--model", default="qwen2.5:7b-instruct")
    s.add_argument("--api-key", default=""); s.add_argument("--workdir", default="omni_out"); s.set_defaults(fn=cmd_copilot)

    s = sp.add_parser("mesh", help="3D: make primitives, inspect and render .obj/.stl meshes")
    ms = s.add_subparsers(dest="mcmd", required=True)
    m = ms.add_parser("make"); m.add_argument("shape"); m.add_argument("-o", "--out", required=True); m.add_argument("--subdivide", type=int, default=0)
    m = ms.add_parser("info"); m.add_argument("file")
    m = ms.add_parser("render"); m.add_argument("file"); m.add_argument("-o", "--out", required=True)
    m.add_argument("--width", type=int, default=960); m.add_argument("--height", type=int, default=540)
    m.add_argument("--azimuth", type=float, default=30.0); m.add_argument("--elevation", type=float, default=20.0)
    m.add_argument("--quality", type=int, default=2); m.add_argument("--floor", action="store_true"); m.add_argument("--wireframe", action="store_true")
    s.set_defaults(fn=cmd_mesh)
    s = sp.add_parser("gui", help="open the local web GUI: node editor, viewer with scopes, inspector, Copilot")
    s.add_argument("project", nargs="?", help="optional graph .json to open")
    s.add_argument("--port", type=int, default=8765); s.add_argument("--workdir", default="omni_out")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--backend", choices=["rule", "openai"], default="rule")
    s.add_argument("--base-url", default="http://localhost:11434/v1"); s.add_argument("--model", default="qwen2.5:7b")
    s.add_argument("--api-key", default="")
    s.set_defaults(fn=cmd_gui)
    s = sp.add_parser("demo", help="generate a showcase folder: looks, LUTs, SFX, effects and a short video")
    s.add_argument("-o", "--out", default="omni_demo"); s.add_argument("--quick", action="store_true"); s.set_defaults(fn=cmd_demo)
    s = sp.add_parser("docs", help="print/write the node reference (markdown)"); s.add_argument("-o", "--out"); s.set_defaults(fn=cmd_docs)
    s = sp.add_parser("timeline", help="timeline editing: demo / info / render / export (EDL, OTIO)")
    ts = s.add_subparsers(dest="tcmd", required=True)
    m = ts.add_parser("demo"); m.add_argument("-o", "--out", default="demo_timeline.json")
    m = ts.add_parser("info"); m.add_argument("file")
    m = ts.add_parser("render"); m.add_argument("file"); m.add_argument("-o", "--out", required=True)
    m.add_argument("--start", type=int, default=0); m.add_argument("--end", type=int); m.add_argument("--crf", type=int, default=18)
    m = ts.add_parser("export"); m.add_argument("file"); m.add_argument("--format", choices=["edl", "otio"], required=True)
    m.add_argument("-o", "--out", required=True); m.add_argument("--audio", action="store_true", help="EDL: include audio events")
    s.set_defaults(fn=cmd_timeline)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except (GraphError, KeyError, FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
