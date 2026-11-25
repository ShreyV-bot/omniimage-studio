"""Phase 1 showcase: scopes, a hue-qualified secondary grade, and a Figma-style UI card. Writes phase1_sheet.png."""
import numpy as np
import omniimage as o
from omniimage.nodes.io import write_image_file
from omniimage.presets.vfx import demo_scene

g, scene = demo_scene(640, 360)
# secondary grade: push only the warm (hue ~20) areas toward teal
q = g.add("hsl_qualifier", hue=25, hue_width=70, sat_min=0.2)
shift = g.add("hue_shift", degrees=150)
g.connect(scene, shift)
mg = g.add("merge", mode="over")
g.connect(shift, mg, dst_port="a"); g.connect(scene, mg, dst_port="b"); g.connect(q, mg, dst_port="mask")
g.connect(scene, q)
wf = g.add("scope_parade", width=320, height=180); g.connect(mg, wf)
vs = g.add("scope_vectorscope", size=180); g.connect(mg, vs)
# UI card
card = g.add("vector_shape", width=320, height=180, kind="rounded_rect", x=160, y=90, w=260, h=120, radius=18,
             gradient=True, fill="#5b8cff", fill2="#a45cff", stroke="#ffffff", stroke_width=2)
sh = g.add("drop_shadow", dy=10, blur=22, opacity=0.4); g.connect(card, sh)
bgc = g.add("solid", width=320, height=180, color="#14161f")
ui = g.add("merge"); g.connect(sh, ui, dst_port="a"); g.connect(bgc, ui, dst_port="b")

tiles = [g.evaluate(x) for x in (scene, mg)]
small = [g.evaluate(x) for x in (wf, vs, ui)]
import cv2
def fit(im, w, h): return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
top = np.concatenate([fit(tiles[0], 320, 180), fit(tiles[1], 320, 180), fit(small[2], 320, 180)], 1)
bot = np.concatenate([fit(small[0], 320, 180), fit(small[1], 320, 180), np.zeros((180, 320, 4), np.float32)], 1)
write_image_file(np.concatenate([top, bot], 0), "phase1_sheet.png")
print("wrote phase1_sheet.png")
