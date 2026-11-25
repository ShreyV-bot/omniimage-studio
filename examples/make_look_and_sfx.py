"""Build a graph by hand, bake it to a .cube LUT, and synthesise a sound effect."""
import omniimage as o
from omniimage.lut import bake_lut
from omniimage.presets import build_sfx
from omniimage.nodes.io import write_audio_file

g = o.Graph()
src = g.add("lut_input", id="src")
lin = g.add("srgb_to_linear")
warm = g.add("temperature_tint", temperature=0.25)
sat = g.add("saturation", amount=1.2)
out = g.add("linear_to_srgb", id="out")
g.chain(src, lin, warm, sat, out)
bake_lut(g, out, src, size=33, title="My warm look").write_cube("my_warm_look.cube")

sg, so = build_sfx("whoosh")
write_audio_file(sg.evaluate(so), "whoosh.wav", "16")
print("wrote my_warm_look.cube and whoosh.wav")
