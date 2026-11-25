"""A ready-made 3D showcase graph: glossy sphere, gold torus and blue cube on a checker floor."""
from __future__ import annotations
from ..core.graph import Graph


def showcase_3d(width: int = 960, height: int = 540, turntable: bool = False, quality: int = 2):
    """-> (graph, render_node_id). With ``turntable`` the camera orbits 4 degrees per frame."""
    g = Graph()
    sphere = g.add("mesh_sphere", radius=0.8, segments=48, rings=24)
    sphere_m = g.add("mesh_material", color="#e0453a", roughness=0.3)
    sphere_t = g.add("mesh_transform", ty=0.3)
    torus = g.add("mesh_torus", major_radius=1.0, minor_radius=0.28)
    torus_m = g.add("mesh_material", color="#f2c14e", metallic=1.0, roughness=0.25)
    torus_t = g.add("mesh_transform", ty=-0.6)
    cube = g.add("mesh_cube", size_x=0.8, size_y=0.8, size_z=0.8)
    cube_m = g.add("mesh_material", color="#3a7bd5", roughness=0.6)
    cube_t = g.add("mesh_transform", tx=-1.6, ty=-0.2, tz=0.4, ry=35)
    j1, j2 = g.add("mesh_join"), g.add("mesh_join")
    r = g.add("render_mesh", width=width, height=height, quality=quality, floor="checker", elevation=22,
              azimuth="=25+frame*4" if turntable else 25)
    g.chain(sphere, sphere_m, sphere_t); g.chain(torus, torus_m, torus_t); g.chain(cube, cube_m, cube_t)
    g.connect(sphere_t, j1, dst_port="a"); g.connect(torus_t, j1, dst_port="b")
    g.connect(j1, j2, dst_port="a"); g.connect(cube_t, j2, dst_port="b")
    g.connect(j2, r)
    return g, r


def textured_showcase_3d(width: int = 960, height: int = 540, quality: int = 2):
    """-> (graph, render_node_id). Three UV-textured objects: a checker sphere, a noise-marbled torus (colourised through a
    gradient map) and a box-projected cube wearing a gradient - the same pipeline works on any imported .obj / .glb."""
    g = Graph()
    chk = g.add("checker", width=512, height=512, size=64, color_a=[0.95, 0.85, 0.2, 1], color_b=[0.75, 0.1, 0.15, 1])
    sph = g.add("mesh_sphere", radius=0.8, segments=64, rings=32)
    sph_t = g.add("mesh_texture", scale_u=2.0, scale_v=1.0)
    sph_x = g.add("mesh_transform", ty=0.3)
    nz = g.add("noise", width=512, height=512, scale=96.0, octaves=5)
    nz_c = g.add("gradient_map", colors=["#10243f", "#2e6f9e", "#e8e2d0"])
    grad = g.add("gradient", width=512, height=512, kind="linear", color_a=[0.95, 0.4, 0.1, 1], color_b=[0.1, 0.3, 0.9, 1], angle=45)
    tor = g.add("mesh_torus", major_radius=1.0, minor_radius=0.28, segments=96, tube_segments=32)
    tor_t = g.add("mesh_texture", scale_u=3.0, scale_v=1.0)
    tor_m = g.add("mesh_material", roughness=0.3)
    tor_x = g.add("mesh_transform", ty=-0.6)
    cube = g.add("mesh_cube", size_x=0.8, size_y=0.8, size_z=0.8)
    cube_t = g.add("mesh_texture")
    cube_x = g.add("mesh_transform", tx=-1.6, ty=-0.2, tz=0.4, ry=35)
    j1, j2 = g.add("mesh_join"), g.add("mesh_join")
    r = g.add("render_mesh", width=width, height=height, quality=quality, floor="checker", elevation=22, azimuth=25)
    g.connect(sph, sph_t, dst_port="mesh"); g.connect(chk, sph_t, dst_port="image"); g.connect(sph_t, sph_x)
    g.connect(tor, tor_m); g.connect(tor_m, tor_t, dst_port="mesh"); g.connect(nz, nz_c); g.connect(nz_c, tor_t, dst_port="image"); g.connect(tor_t, tor_x)
    g.connect(cube, cube_t, dst_port="mesh"); g.connect(grad, cube_t, dst_port="image"); g.connect(cube_t, cube_x)
    g.connect(sph_x, j1, dst_port="a"); g.connect(tor_x, j1, dst_port="b")
    g.connect(j1, j2, dst_port="a"); g.connect(cube_x, j2, dst_port="b")
    g.connect(j2, r)
    return g, r


def pbr_showcase_3d(width: int = 960, height: int = 540, quality: int = 2):
    """-> (graph, render_node_id). One procedural height map drives three PBR looks: a rocky sphere (normal + base-colour
    maps), a brushed metal torus (roughness map on a fully metallic material) and a cube whose emissive checker glows."""
    g = Graph()
    nz = g.add("noise", width=512, height=512, scale=40.0, octaves=5)
    nrm = g.add("normal_from_height", strength=14.0)
    rock = g.add("gradient_map", colors=["#2a211b", "#6b5a48", "#b9a98f"])
    sph = g.add("mesh_sphere", radius=0.85, segments=96, rings=48)
    sph_t = g.add("mesh_texture", scale_u=3.0, scale_v=2.0, normal_strength=1.2)
    sph_x = g.add("mesh_transform", tx=0.2)
    g.connect(nz, nrm); g.connect(nz, rock)
    g.connect(sph, sph_t, dst_port="mesh"); g.connect(rock, sph_t, dst_port="image"); g.connect(nrm, sph_t, dst_port="normal")
    g.connect(sph_t, sph_x)
    tor = g.add("mesh_torus", major_radius=0.8, minor_radius=0.24, segments=96, tube_segments=32)
    tor_m = g.add("mesh_material", color=[0.95, 0.72, 0.35], metallic=1.0, roughness=0.5)
    tor_t = g.add("mesh_texture", scale_u=6.0, scale_v=2.0, roughness_scale=0.75)
    tor_x = g.add("mesh_transform", tx=1.9, ty=0.115, tz=0.3, rx=65)
    g.connect(tor, tor_m); g.connect(tor_m, tor_t, dst_port="mesh"); g.connect(nz, tor_t, dst_port="roughness"); g.connect(tor_t, tor_x)
    chk = g.add("checker", width=256, height=256, size=32, color_a=[1, 0.3, 0.05, 1], color_b=[0, 0, 0, 1])
    cube = g.add("mesh_cube", size_x=0.9, size_y=0.9, size_z=0.9)
    cube_m = g.add("mesh_material", color=[0.06, 0.06, 0.08], roughness=0.35)
    cube_t = g.add("mesh_texture", scale_u=1.0, emissive_strength=2.5)
    cube_x = g.add("mesh_transform", tx=-1.8, ty=-0.4, tz=0.4, ry=35)
    g.connect(cube, cube_m); g.connect(cube_m, cube_t, dst_port="mesh"); g.connect(chk, cube_t, dst_port="emissive"); g.connect(cube_t, cube_x)
    j1, j2 = g.add("mesh_join"), g.add("mesh_join")
    r = g.add("render_mesh", width=width, height=height, quality=quality, floor="checker", elevation=20, azimuth=20)
    g.connect(sph_x, j1, dst_port="a"); g.connect(tor_x, j1, dst_port="b"); g.connect(j1, j2, dst_port="a"); g.connect(cube_x, j2, dst_port="b")
    g.connect(j2, r)
    return g, r
