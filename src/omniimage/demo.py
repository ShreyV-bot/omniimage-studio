"""``omni demo``: render a showcase folder so a new user sees everything working in about a minute."""
from __future__ import annotations
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw

from .core.graph import Ctx, Graph
from .core.media import AudioBuf
from .nodes.io import write_image_file, write_audio_file
from .presets.looks import LOOKS, make_look_lut
from .presets.sfx import SFX, build_sfx
from .presets.vfx import EFFECTS, apply_effect, demo_scene, neon_sign
from .render import render_frames


def _label(img_lin: np.ndarray, text: str) -> np.ndarray:
    from .core.media import linear_to_srgb
    u8 = (np.clip(linear_to_srgb(img_lin[..., :3]), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(u8)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    from .core.media import srgb_to_linear, ensure_rgba
    return ensure_rgba(srgb_to_linear(np.asarray(im, np.float32) / 255.0))


def _sheet(tiles, cols):
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)


def run_demo(out: str = "omni_demo", quick: bool = False):
    os.makedirs(out, exist_ok=True)
    w, h = (320, 180) if quick else (480, 270)
    print("1/6 source image")
    sg, sout = demo_scene(w, h)
    base = sg.evaluate(sout)
    write_image_file(base, f"{out}/source.png")

    print("2/6 LUT looks (baked from node graphs) + contact sheet")
    os.makedirs(f"{out}/luts", exist_ok=True)
    tiles = [_label(base, "original")]
    for name in LOOKS:
        lut = make_look_lut(name, 17 if quick else 33)
        lut.write_cube(f"{out}/luts/{name}.cube", f"OmniImage {name}")
        g = Graph()
        src = g.add("load_image", path=f"{out}/source.png")
        la = g.add("lut_apply", path=f"{out}/luts/{name}.cube")
        g.connect(src, la)
        tiles.append(_label(g.evaluate(la), name))
    write_image_file(_sheet(tiles, 5), f"{out}/looks_contact_sheet.png")

    print("3/6 image effects")
    tiles = [_label(base, "original")]
    for name in EFFECTS:
        if name == "heat_haze":
            continue
        g, s = demo_scene(w, h)
        last = apply_effect(name, g, s)
        tiles.append(_label(g.evaluate(last, ctx=Ctx(6)), name))
    ng, nout = neon_sign("OMNI", width=w, height=h, size=h // 3)
    tiles.append(_label(ng.evaluate(nout, ctx=Ctx(3)), "neon sign"))
    write_image_file(_sheet(tiles, 4), f"{out}/effects_contact_sheet.png")

    print("4/6 sound effects")
    os.makedirs(f"{out}/sfx", exist_ok=True)
    for name in SFX:
        g, o = build_sfx(name)
        write_audio_file(g.evaluate(o), f"{out}/sfx/{name}.wav")

    print("5/6 animated video with synthesized soundtrack (keyframes + expressions)")
    fps, frames = 24, (24 if quick else 72)
    vg, vout = neon_sign("OMNI", width=w, height=h, size=h // 3)
    vg.set_param("tube", "x", "=0.5+0.15*sin(t*2)")
    vg.set_param("glow", "radius", {"keys": [[0, 8], [frames // 2, 30], [frames, 12]], "interp": "smooth"})
    sg2, sn = build_sfx("riser", duration=frames / fps)
    audio = sg2.evaluate(sn)
    try:
        render_frames(vg, vout, f"{out}/neon_animation.mp4", 0, frames - 1, fps, audio=audio)
    except Exception as e:  # ffmpeg missing etc.: keep going, the rest of the demo is still useful
        print("   (video skipped:", e, ")")

    print("6/6 Copilot (offline rule backend)")
    from .ai_assistant import Copilot
    print("3D showcase render")
    from .presets.scene3d import showcase_3d
    g3, r3 = showcase_3d(w * 2, h * 2, quality=1 if quick else 2)
    write_image_file(g3.evaluate(r3), f"{out}/showcase_3d.png")
    from .presets.scene3d import textured_showcase_3d
    g4, r4 = textured_showcase_3d(w * 2, h * 2, quality=1 if quick else 2)
    write_image_file(g4.evaluate(r4), f"{out}/showcase_3d_textured.png")
    from .presets.scene3d import pbr_showcase_3d
    g5, r5 = pbr_showcase_3d(w * 2, h * 2, quality=1 if quick else 2)
    write_image_file(g5.evaluate(r5), f"{out}/showcase_3d_pbr.png")
    c = Copilot(workdir=f"{out}/copilot")
    for q in ["make a golden hour LUT", "give me a laser sound effect", "apply a strong VHS effect"]:
        r = c.ask(q)
        print("  ", q, "->", r.text)
    print(f"\nDone. Open the folder: {os.path.abspath(out)}")
