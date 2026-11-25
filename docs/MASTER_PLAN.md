# Master Plan: toward Blender + DaVinci Resolve + Maya + Figma in one open-source app

Honest framing: those four products represent ~100+ combined engineer-years each. "God level" is a direction, not a milestone. The strategy is to share ONE engine (typed node graph, cache, undo, Copilot) across all four domains, so every feature built for one is reusable in the others, and to ship in phases that each leave a working, tested product.

Engine pillars (already built in v0.1): typed node graph · lazy cached evaluation · keyframes/expressions · command bus + undo · plugin system · Copilot tool API · CLI · tests/CI.

## Phase 1: Foundations (v0.2: DONE except OCIO config loading and component/variant system)  [Resolve + Figma + Fusion basics]
- Colour management: scene-referred working space, Rec.709/Rec.2020/P3/ACEScg matrices, log curves (LogC, S-Log3, ACEScct, sRGB, gamma)
- Scopes: waveform, RGB parade, vectorscope, histogram (as nodes producing images)
- Secondary grading: HSL qualifier, power windows (circle/rect/gradient), soft edges
- Video clips as nodes (decode any ffmpeg-readable clip by frame)
- Vector design layer (Figma-like): rect/rounded rect/ellipse/polygon/line, fills, strokes, gradients, drop shadow, blur, auto-layout stack, masks
## Phase 2: Timeline + Editing (Resolve Edit/Cut/Fairlight): CORE DONE in v0.3 (still open: trim/roll/slip edit ops, multicam, nested timelines, FCPXML, audio effects per track, frame-blended retime)
- Timeline model (tracks, clips, transitions, speed, markers), render from timeline, EDL/FCPXML export
- Multitrack audio mixer, buses, automation, loudness (LUFS) metering
## Phase 3: Viewer + GUI (all four): FIRST SLICE DONE in v0.4 (local web GUI: node editor, viewer + scopes, inspector, Copilot panel, save/load, render). Second slice done in v0.8 (timeline workspace: multitrack editing, transitions, clip effects, waveforms, playback, undo). Open: keyframe curves and nested clips in the timeline, design canvas, GPU backend, desktop packaging
- Node editor, viewer with scopes, inspector, timeline UI, design canvas; cross-platform toolkit; GPU (wgpu) backend behind same node API
## Phase 4: 3D (Blender + Maya): FIRST SLICE DONE in v0.5 (mesh type, 7 primitives, modifiers, materials, OBJ/STL, software renderer with shadows and depth/normal passes). Second slice done in v0.6 (UVs on primitives + projection modes, textures with mip-mapped trilinear sampling, OBJ vt, glTF/GLB read + GLB write). Third slice done in v0.7 (normal / roughness-metallic / emissive / occlusion maps in the renderer, nodes and glTF; `normal_from_height`). Open: glTF skins and animation, booleans, sculpt, rigging/skeletons, physics, GPU path tracing
- Mesh data model + primitives/modifiers, OBJ/glTF I/O, software then GPU renderer (PBR, shadows), materials as shader nodes, camera/lights, animation curves, rigging (skeleton, skinning, IK), cloth/rigid-body sims
## Phase 5: Design/prototyping depth (Figma)
- Components/variants, constraints, auto-layout, prototyping flows, SVG/PDF import-export, design tokens, multiplayer (CRDT)
## Phase 6: AI layer
- ONNX nodes (upscale, interpolate, segment, depth, Whisper, Demucs), Copilot "match this reference", text-to-graph for all domains
## Phase 7: Ecosystem
- Plugin marketplace, Python/Lua scripting console, OpenColorIO/OpenEXR/USD/OTIO interchange, docs and tutorials

Rule for every phase: ships with tests, node reference docs regenerated, honest README status table.
