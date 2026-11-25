# Architecture

```
          +-------------------+
 Copilot  | ai_assistant.py   |  rules or LLM -> tool calls
          +---------+---------+
                    | Toolbox (sandboxed, one undo step per request)
          +---------v---------+
          | Project           |  command bus, snapshots, undo/redo
          +---------+---------+
                    |
          +---------v---------+       +-----------------+
          | Graph             |<----->| NodeSpec registry|<-- built-in nodes, plugins
          | typed ports, lazy |       +-----------------+
          | content-hash cache|
          +----+---------+----+
               |         |
        render.py       lut.py (bake_lut: identity lattice -> graph -> .cube)
```

* **Data types** (`core/ports.py`): IMAGE (float32 RGBA, scene-linear, premultiplication explicit), MASK (H x W float), AUDIO (`AudioBuf`), LUT, VALUE, ANY. Connections are type-checked when made; cycles are rejected.
* **Evaluation**: pull-based. Asking for a node evaluates only its upstream. Results are cached by a blake2b hash of (node type, resolved params, input hashes, frame if the node `uses_time`).
* **Animation**: any numeric param can be a constant, a keyframe dict (`linear|smooth|hold`) or an expression string starting with `=`. Expressions are parsed with `ast` and a whitelist, never `eval`.
* **LUT baking**: nodes flagged `pointwise` depend only on the pixel they are given, so a graph made only of them equals a 3D LUT. `bake_lut` replaces the `lut_input` node with an identity lattice and samples the graph. Spatial nodes (blur, grain...) are refused.
* **Copilot safety**: the model can only call `Toolbox` methods; file writes are confined to the work folder; every request runs in a transaction that rolls back on error.
* **Plugins**: `omniimage.nodes` entry points and `*.py` files in plugin dirs; a broken plugin warns and is skipped.

## GUI (v0.4)
`omni gui` starts `gui/server.py`, a stdlib `ThreadingHTTPServer` holding one `Studio` (Project + Copilot + node layout + work folder). The browser UI (`gui/static/index.html`, plain JS + SVG, no build step) talks JSON: `/api/command` (the same command bus as the CLI and Copilot, so edits are undoable), `/api/render` (PNG preview, optionally through a scope node), `/api/copilot`, `/api/save|load|export`. Safety: 127.0.0.1 only, Host-header check, per-process token, JSON-only POSTs, all paths confined to the work folder.

## 3D (v0.5, UVs/textures/glTF in v0.6)
`mesh.py` holds the `Mesh` data type (triangles + per-vertex colour/roughness/metallic/emission; Y up, CCW outward), primitives, Loop subdivision, smoothing, noise displacement, OBJ/STL I/O. `raster.py` is a CPU renderer: a per-triangle z-buffer rasteriser stores triangle id and barycentrics, then shading is fully vectorised (GGX/Lambert, hemisphere environment, 3 lights, PCF shadow map for the key light). `nodes/mesh.py` exposes all of it as `mesh_*` nodes and `render_mesh` (image + depth + normal outputs). The new `MESH` port type connects only to mesh inputs.

**UVs and textures (v0.6).** `Mesh` also carries optional per-vertex `uv` (origin top-left, the glTF convention), a per-face `face_tex` index (-1 = none) and a list of `Texture` objects (linear RGBA image + scale/offset). Primitives duplicate vertices on UV seams and sharp edges, so `vertex_normals(crease_deg)` blends coincident vertices only when their normals agree, which keeps seams invisible and cube edges sharp. Subdivision keeps vertices unwelded for attributes (UVs, materials, per-face textures) while Loop positions are computed on the position-welded topology and mapped back, so seams never tear. The rasteriser samples a texture per pixel: perspective-correct UVs plus analytic screen-space derivatives (the same barycentric formula evaluated one pixel to the right and below) give the mip level; a lazily built box-filtered pyramid is sampled trilinearly with repeat wrapping. `gltf.py` is a dependency-free reader/writer (struct + json + OpenCV for image decode); it validates every accessor and bufferView range, confines external files to the model's folder, and refuses what it cannot represent (sparse accessors, Draco, meshopt) instead of importing it wrongly.

**PBR maps (v0.7).** A `Texture` is a *set* of maps sharing one UV transform: base colour (`image`), tangent-space `normal` (with a strength), glTF-style `rough_metal` (G roughness, B metallic), `emissive` and `occlusion` (with a strength). Only colour maps are colour-managed; the rest are data and are never converted. `Mesh.tangent_frame()` builds per-vertex tangent and bitangent vectors from the UVs (T = d/du, B = image-up = -d/dv, area-weighted), including the texture's scale so mirrored mappings stay correct. In the renderer the per-pixel frame is the interpolated T and B, orthogonalised against the interpolated normal, so mirrored UV islands work without a handedness flag. The mapped normal is flipped for back faces (decided from the geometric normal first), which matches glTF's double-sided rule. Each map is sampled with its own mip level (its own resolution), roughness / metallic / emission multiply the per-vertex values like glTF factors, and occlusion scales only the ambient and environment terms.

**glTF as an untrusted input.** `gltf.py` range-checks every offset, count and index; refuses accessors over 64M elements and images over 64 megapixels (read from the header, before decoding); resolves external files through `realpath` and requires them to stay inside the model's folder (so `..`, absolute paths, encoded dots and symlinks all fail); caps node-tree depth; turns NaN / inf into zeros; and converts every malformed-structure failure into `GltfError` rather than a stray `KeyError` or `MemoryError`. Each guard has a test that goes red when the guard is removed.

**Timeline editing (v0.8).** `timeline_edit.py` wraps a `Timeline` in a `TimelineEditor`. Every change is a plain dict (`move_clip`, `trim_clip`, `split_clip`, `set_transition`, `add_effect`, ...), validated, and applied in a batch that is atomic: the timeline is serialised before the batch and restored if any op fails, so a rejected edit never leaves half a change. The same snapshots give undo and redo. Clips on a track stay sorted by start frame and are addressed as (track, index); ops that can reorder return the clip's new index. Cutting is invisible by construction: a left trim or split advances the source offset by `frames * speed`, shifts opacity keyframes, and gives each half its own copy of the clip graph. The GUI server exposes the editor under `/api/timeline/*`; previews are rendered on demand (a proxy copy of the timeline at the requested width, never a change to the real one), thumbnails and waveforms are fetched lazily and cached by a content signature, and the audio mix is a cached WAV that the browser plays while frames follow its clock.

**Limits and trust in the timeline GUI.** Media names go through the same work-folder sandbox as everything else (realpath, no `..`, no symlinks out), including when a clip's `path` parameter is edited. Frame numbers, track and clip counts, picture size, fps and sample rate are bounded in every op and again when a timeline file is loaded, so a hostile `.timeline.json` cannot ask for gigabytes; loading also refuses files over 16 MB, audio preview over 5 minutes and one render request over 2000 frames. Each of these limits has a test that goes red when the check is removed.

