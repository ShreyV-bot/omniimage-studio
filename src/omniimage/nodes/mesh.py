"""3D nodes: primitives, modifiers, materials, OBJ/STL I/O and a software renderer that turns a mesh into an
image (plus depth and normal passes) so 3D flows straight into the 2D compositing, colour and scope nodes."""
from __future__ import annotations
import os

import numpy as np

from ..core.registry import node
from ..core.params import P, ParamError
from ..core.ports import PortType as T
from ..core.media import srgb_to_linear
from .. import mesh as M
from ..raster import RenderSettings, render
from .io import _mtime

OUT = dict(outputs={"mesh": T.MESH})
MM = dict(inputs={"mesh": T.MESH}, outputs={"mesh": T.MESH})


def _lin(c):
    return tuple(float(x) for x in srgb_to_linear(np.array(c[:3], np.float32)))


# ---------------------------------------------------------------- primitives
@node("mesh_cube", "mesh", "Box with sharp edges. size is per-axis; subdivisions split each face into a grid.", **OUT,
      params={"size_x": P("float", 1.0, 1e-4, 1000), "size_y": P("float", 1.0, 1e-4, 1000), "size_z": P("float", 1.0, 1e-4, 1000),
              "subdivisions": P("int", 1, 1, 64)})
def mesh_cube(inp, p, ctx):
    return M.cube((p["size_x"], p["size_y"], p["size_z"]), p["subdivisions"])


@node("mesh_plane", "mesh", "Flat grid in the XZ plane facing up (+Y).", **OUT,
      params={"size_x": P("float", 2.0, 1e-4, 10000), "size_z": P("float", 2.0, 1e-4, 10000),
              "subdivisions_x": P("int", 1, 1, 256), "subdivisions_z": P("int", 1, 1, 256)})
def mesh_plane(inp, p, ctx):
    return M.plane(p["size_x"], p["size_z"], p["subdivisions_x"], p["subdivisions_z"])


@node("mesh_sphere", "mesh", "UV sphere (poles on the Y axis).", **OUT,
      params={"radius": P("float", 1.0, 1e-4, 1000), "segments": P("int", 32, 3, 256), "rings": P("int", 16, 2, 256)})
def mesh_sphere(inp, p, ctx):
    return M.uv_sphere(p["radius"], p["segments"], p["rings"])


@node("mesh_icosphere", "mesh", "Sphere made of evenly sized triangles (better for sculpt-like displacement).", **OUT,
      params={"radius": P("float", 1.0, 1e-4, 1000), "subdivisions": P("int", 2, 0, 5)})
def mesh_icosphere(inp, p, ctx):
    return M.icosphere(p["radius"], p["subdivisions"])


@node("mesh_cylinder", "mesh", "Cylinder along Y. Set radius_top different from radius for a truncated cone.", **OUT,
      params={"radius": P("float", 0.5, 1e-4, 1000), "radius_top": P("float", 0.5, 0, 1000), "height": P("float", 1.0, 1e-4, 1000),
              "segments": P("int", 32, 3, 256), "caps": P("bool", True)})
def mesh_cylinder(inp, p, ctx):
    return M.cylinder(p["radius"], p["height"], p["segments"], p["caps"], p["radius_top"])


@node("mesh_cone", "mesh", "Cone along Y.", **OUT,
      params={"radius": P("float", 0.5, 1e-4, 1000), "height": P("float", 1.0, 1e-4, 1000), "segments": P("int", 32, 3, 256),
              "cap": P("bool", True)})
def mesh_cone(inp, p, ctx):
    return M.cone(p["radius"], p["height"], p["segments"], p["cap"])


@node("mesh_torus", "mesh", "Donut around the Y axis.", **OUT,
      params={"major_radius": P("float", 1.0, 1e-4, 1000), "minor_radius": P("float", 0.3, 1e-4, 1000),
              "segments": P("int", 48, 3, 512), "tube_segments": P("int", 16, 3, 256)})
def mesh_torus(inp, p, ctx):
    return M.torus(p["major_radius"], p["minor_radius"], p["segments"], p["tube_segments"])


@node("mesh_load", "mesh", "Load a .obj, .stl, .glb or .gltf file. Polygons are triangulated; UVs are kept (OBJ and glTF), "
      "and glTF also brings its PBR colour/roughness/metallic/emission and base-colour textures.", **OUT,
      params={"path": P("str", "")}, fingerprint=_mtime)
def mesh_load(inp, p, ctx):
    if not p["path"]:
        raise ParamError("mesh_load needs a path")
    return M.read_mesh(p["path"])


# ---------------------------------------------------------------- modifiers
@node("mesh_transform", "mesh", "Move / rotate (degrees, X then Y then Z) / scale a mesh. All values can be animated.", **MM,
      params={"tx": P("float", 0.0, -1e6, 1e6), "ty": P("float", 0.0, -1e6, 1e6), "tz": P("float", 0.0, -1e6, 1e6),
              "rx": P("float", 0.0, -1e6, 1e6), "ry": P("float", 0.0, -1e6, 1e6), "rz": P("float", 0.0, -1e6, 1e6),
              "sx": P("float", 1.0, -1e6, 1e6), "sy": P("float", 1.0, -1e6, 1e6), "sz": P("float", 1.0, -1e6, 1e6),
              "uniform_scale": P("float", 1.0, -1e6, 1e6)})
def mesh_transform(inp, p, ctx):
    u = p["uniform_scale"]
    return inp["mesh"].transformed(M.trs_matrix((p["tx"], p["ty"], p["tz"]), (p["rx"], p["ry"], p["rz"]),
                                                (p["sx"] * u, p["sy"] * u, p["sz"] * u)))


@node("mesh_material", "mesh", "Set the surface look: base colour (sRGB), roughness, metallic, and emission (glow).", **MM,
      params={"color": P("color", [0.8, 0.8, 0.8, 1.0]), "roughness": P("float", 0.5, 0, 1), "metallic": P("float", 0.0, 0, 1),
              "emission_color": P("color", [1.0, 1.0, 1.0, 1.0]), "emission_strength": P("float", 0.0, 0, 100)})
def mesh_material(inp, p, ctx):
    return inp["mesh"].with_material(_lin(p["color"]), p["roughness"], p["metallic"],
                                     np.array(_lin(p["emission_color"])) * p["emission_strength"])


@node("mesh_subdivide", "mesh", "Split every triangle into four. smooth=true uses Loop subdivision (rounds the shape); "
      "false only adds detail. The mesh is welded first, so sharp cube edges become soft.", **MM,
      params={"levels": P("int", 1, 0, 4), "smooth": P("bool", True)})
def mesh_subdivide(inp, p, ctx):
    return M.subdivide(inp["mesh"], p["levels"], p["smooth"])


@node("mesh_smooth", "mesh", "Relax the surface by moving each vertex toward the average of its neighbours.", **MM,
      params={"iterations": P("int", 3, 0, 100), "factor": P("float", 0.5, 0, 1), "keep_boundary": P("bool", True)})
def mesh_smooth(inp, p, ctx):
    return M.smooth_laplacian(inp["mesh"], p["iterations"], p["factor"], p["keep_boundary"])


@node("mesh_displace", "mesh", "Push vertices along their normals with fractal noise (rocks, terrain, organic wobble). "
      "Animate 'evolution' for moving noise.", **MM,
      params={"amount": P("float", 0.2, -100, 100), "scale": P("float", 2.0, 0.01, 100), "octaves": P("int", 3, 1, 8),
              "seed": P("int", 0, 0, 100000), "evolution": P("float", 0.0, -1e6, 1e6)})
def mesh_displace(inp, p, ctx):
    return M.displace_noise(inp["mesh"], p["amount"], p["scale"], p["octaves"], p["seed"], (0, p["evolution"], 0))


@node("mesh_twist", "mesh", "Twist around an axis; the angle grows linearly along it.", **MM,
      params={"angle": P("float", 90.0, -36000, 36000), "axis": P("choice", "y", choices=("x", "y", "z"))})
def mesh_twist(inp, p, ctx):
    return M.twist(inp["mesh"], p["angle"], p["axis"])


@node("mesh_array", "mesh", "Repeat a mesh in a line (instancing). offset is the step between copies.", **MM,
      params={"count": P("int", 3, 1, 64), "offset_x": P("float", 1.5, -1e4, 1e4), "offset_y": P("float", 0.0, -1e4, 1e4),
              "offset_z": P("float", 0.0, -1e4, 1e4)})
def mesh_array(inp, p, ctx):
    return M.array(inp["mesh"], p["count"], (p["offset_x"], p["offset_y"], p["offset_z"]))


@node("mesh_mirror", "mesh", "Mirror across an axis plane through the origin and join with the original.", **MM,
      params={"axis": P("choice", "x", choices=("x", "y", "z")), "keep_original": P("bool", True)})
def mesh_mirror(inp, p, ctx):
    return M.mirror(inp["mesh"], p["axis"], p["keep_original"])


@node("mesh_shading", "mesh", "Choose smooth (interpolated normals) or flat (faceted) shading.", **MM,
      params={"smooth": P("bool", True)})
def mesh_shading(inp, p, ctx):
    out = inp["mesh"].copy()
    out.smooth = p["smooth"]
    return out


@node("mesh_join", "mesh", "Combine two meshes into one scene (materials are kept per vertex).",
      inputs={"a": T.MESH, "b": T.MESH}, outputs={"mesh": T.MESH}, params={})
def mesh_join(inp, p, ctx):
    return inp["a"].merged(inp["b"])


@node("mesh_write", "mesh", "Write the mesh to .obj, .stl or .glb (by file extension). .glb keeps UVs, materials and textures.",
      inputs={"mesh": T.MESH}, sink=True,
      params={"path": P("str", "mesh.obj")})
def mesh_write(inp, p, ctx):
    M.write_mesh(inp["mesh"], p["path"])
    return {}


# ---------------------------------------------------------------- UVs and textures
@node("mesh_uv_project", "mesh", "Generate UV coordinates from positions so a texture can be applied. planar looks along an axis; "
      "cylindrical and spherical wrap around Y; box picks the best axis per face. Cylindrical, spherical and box un-weld the mesh "
      "(each face gets its own vertices) so seams stay clean.", **MM,
      params={"mode": P("choice", "box", choices=("planar", "cylindrical", "spherical", "box")),
              "axis": P("choice", "y", choices=("x", "y", "z")), "scale": P("float", 1.0, 1e-4, 1000),
              "offset_u": P("float", 0.0, -1000, 1000), "offset_v": P("float", 0.0, -1000, 1000)})
def mesh_uv_project(inp, p, ctx):
    return M.uv_project(inp["mesh"], p["mode"], p["axis"], p["scale"], (p["offset_u"], p["offset_v"]))


def _map_size(*imgs):
    for im in imgs:
        if im is not None:
            return im.shape[:2]
    return (1, 1)


def _gray_map(img, size):
    """Single-channel data map (mean of RGB, no colour management) resized to ``size`` = (h, w)."""
    import cv2
    g = np.asarray(img, np.float32)[..., :3].mean(-1)
    if g.shape != tuple(size):
        g = cv2.resize(g, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return g


@node("mesh_texture", "mesh", "Paint images onto a mesh. image = base colour (multiplied with the mesh colour). The optional maps "
      "are DATA, so load them with load_image decode=linear: normal (tangent space, +Y up like glTF; see normal_from_height), "
      "roughness, metallic, emissive and occlusion (ambient-light shadowing). Connecting a roughness / metallic / emissive map "
      "replaces the mesh's value with the matching scale / strength parameter, which the map then multiplies. The mesh needs "
      "UVs: the cube, plane, sphere, cylinder, cone and torus have them; others need a mesh_uv_project node first. Tile with "
      "scale, shift with offset; textures repeat and are mip-mapped.",
      inputs={"mesh": T.MESH, "image": T.IMAGE, "normal": T.IMAGE, "roughness": T.IMAGE, "metallic": T.IMAGE,
              "emissive": T.IMAGE, "occlusion": T.IMAGE},
      optional=("image", "normal", "roughness", "metallic", "emissive", "occlusion"), outputs={"mesh": T.MESH},
      params={"scale_u": P("float", 1.0, 1e-4, 1000), "scale_v": P("float", 1.0, 1e-4, 1000),
              "offset_u": P("float", 0.0, -1000, 1000), "offset_v": P("float", 0.0, -1000, 1000),
              "normal_strength": P("float", 1.0, 0.0, 10.0, doc="scales the XY tilt of the normal map"),
              "roughness_scale": P("float", 1.0, 0.0, 4.0), "metallic_scale": P("float", 1.0, 0.0, 4.0),
              "emissive_strength": P("float", 1.0, 0.0, 1000.0), "occlusion_strength": P("float", 1.0, 0.0, 1.0)})
def mesh_texture(inp, p, ctx):
    m = inp["mesh"]
    if not m.has_uv:
        raise ParamError("mesh_texture: this mesh has no UVs - add a mesh_uv_project node before it")
    img, nrm, rou, met, emi, occ = (inp.get(k) for k in ("image", "normal", "roughness", "metallic", "emissive", "occlusion"))
    if all(x is None for x in (img, nrm, rou, met, emi, occ)):
        raise ParamError("mesh_texture: connect at least one image (colour, normal, roughness, metallic, emissive or occlusion)")
    rm = None
    if rou is not None or met is not None:
        size = _map_size(rou, met)
        rm = np.ones(size + (4,), np.float32)
        if rou is not None:
            rm[..., 1] = _gray_map(rou, size)
        if met is not None:
            rm[..., 2] = _gray_map(met, size)
    occ_map = None
    if occ is not None:
        g = _gray_map(occ, occ.shape[:2])
        occ_map = np.stack([g, g, g, np.ones_like(g)], -1)
    tex = M.Texture(img, (p["scale_u"], p["scale_v"]), (p["offset_u"], p["offset_v"]),
                    normal=None if nrm is None else np.asarray(nrm, np.float32)[..., :3], normal_scale=p["normal_strength"],
                    rough_metal=rm, emissive=None if emi is None else np.asarray(emi, np.float32)[..., :3], occlusion=occ_map,
                    occlusion_strength=p["occlusion_strength"])
    out = m.with_texture(tex)
    if rou is not None:
        out.rough[:] = p["roughness_scale"]
    if met is not None:
        out.metal[:] = p["metallic_scale"]
    if emi is not None:
        base = out.emit if out.emit.max() > 0 else np.ones_like(out.emit)
        out.emit = (base * p["emissive_strength"]).astype(np.float32)
    return out


# ---------------------------------------------------------------- renderer
@node("render_mesh", "mesh",
      "Render a mesh to an image with a camera, 3-point lighting, shadows, an optional floor and anti-aliasing. "
      "Outputs: image (scene-linear RGBA), depth (mask, near=bright) and normal (image) passes for compositing. "
      "azimuth/elevation/zoom orbit the camera around the object (animate azimuth for a turntable); in auto mode the "
      "light angles are relative to the camera.", inputs={"mesh": T.MESH}, outputs={"image": T.IMAGE, "depth": T.MASK, "normal": T.IMAGE},
      params={"width": P("int", 960, 16, 8192), "height": P("int", 540, 16, 8192),
              "quality": P("int", 2, 1, 4, doc="supersampling factor (anti-aliasing)"),
              "camera": P("choice", "auto", choices=("auto", "manual")),
              "azimuth": P("float", 30.0, -1e6, 1e6), "elevation": P("float", 20.0, -89, 89), "zoom": P("float", 1.0, 0.05, 50),
              "fov": P("float", 40.0, 5, 120), "orthographic": P("bool", False),
              "eye_x": P("float", 3.0, -1e5, 1e5), "eye_y": P("float", 2.0, -1e5, 1e5), "eye_z": P("float", 4.0, -1e5, 1e5),
              "target_x": P("float", 0.0, -1e5, 1e5), "target_y": P("float", 0.0, -1e5, 1e5), "target_z": P("float", 0.0, -1e5, 1e5),
              "key_azimuth": P("float", 70.0, -360, 360), "key_elevation": P("float", 42.0, -89, 89),
              "key_intensity": P("float", 1.6, 0, 50), "key_color": P("color", [1.0, 0.96, 0.9, 1.0]),
              "fill_intensity": P("float", 0.25, 0, 50), "rim_intensity": P("float", 0.45, 0, 50), "ambient": P("float", 0.5, 0, 20),
              "sky_top": P("color", [0.42, 0.55, 0.75, 1.0]), "sky_bottom": P("color", [0.6, 0.58, 0.56, 1.0]),
              "background": P("choice", "sky", choices=("sky", "transparent", "color")),
              "background_color": P("color", [0.05, 0.05, 0.06, 1.0]),
              "shadows": P("bool", True), "floor": P("choice", "none", choices=("none", "plane", "checker")),
              "exposure": P("float", 0.0, -10, 10), "tonemap": P("choice", "aces", choices=("aces", "none")),
              "wireframe": P("bool", False), "wire_width": P("float", 1.0, 0.2, 10)})
def render_mesh(inp, p, ctx):
    s = RenderSettings(width=p["width"], height=p["height"], ssaa=p["quality"], camera_mode=p["camera"],
                       azimuth=p["azimuth"], elevation=p["elevation"], zoom=p["zoom"], fov=p["fov"], ortho=p["orthographic"],
                       eye=(p["eye_x"], p["eye_y"], p["eye_z"]), target=(p["target_x"], p["target_y"], p["target_z"]),
                       key_azimuth=p["key_azimuth"], key_elevation=p["key_elevation"], key_intensity=p["key_intensity"],
                       key_color=_lin(p["key_color"]), fill_intensity=p["fill_intensity"], rim_intensity=p["rim_intensity"],
                       ambient=p["ambient"], sky_top=_lin(p["sky_top"]), sky_bottom=_lin(p["sky_bottom"]),
                       background=p["background"], background_color=_lin(p["background_color"]), shadows=p["shadows"],
                       floor=p["floor"], exposure=p["exposure"], tonemap=p["tonemap"], wireframe=p["wireframe"],
                       wire_width=p["wire_width"])
    r = render(inp["mesh"], s)
    return {"image": r["image"], "depth": r["depth"], "normal": r["normal"]}
