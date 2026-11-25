# Roadmap

**Current: v0.8.** Phase 3 second slice done (timeline workspace in the GUI: tracks, drag/trim/split, transitions, clip effects, audio waveforms, playback, undo). Phase 4 third slice done in v0.7 (PBR maps: normal, roughness/metallic, emissive, occlusion, on top of UVs, textures and glTF). Phases 1-2 done (colour management, scopes, qualifier/windows, vector shapes, timeline, audio mixer). Phase 3 first slice done: `omni gui`. Next: Copilot that edits the timeline, keyframe curves and nested clips in the timeline UI, glTF animation/skin import, booleans and rigging (Phase 4), Figma-style components (Phase 5), ONNX AI nodes (Phase 6). Full plan: `docs/MASTER_PLAN.md`.

The 1,503-feature blueprint (`docs/Media_Studio_Feature_Blueprint.docx`) is the long-term plan. This is an honest account of where the code is relative to it.

## v0.1 (this release): headless engine
- [x] Typed node graph, lazy evaluation, caching, keyframes and safe expressions, undo/redo
- [x] ~95 nodes: generators, colour, filters, compositing, keying, image/audio I/O, audio DSP
- [x] LUT3D: read/write `.cube`, apply, **bake any colour-only node graph to a LUT**
- [x] 14 look LUTs, 7 VFX presets, 7 procedural SFX
- [x] Copilot: offline rules + any OpenAI-compatible local LLM (Ollama, llama.cpp, vLLM, LM Studio), sandboxed tools
- [x] CLI, ffmpeg video render, plugin system, tests, CI

## v0.2 (this release)
- [x] Built-in colour management (ACES, LogC3, S-Log3, Rec.2020, P3), scopes, HSL qualifier, power windows, vector shapes, drop shadow, auto-layout
- [ ] OpenColorIO config loading
- [ ] More keying/roto tools, motion blur, optical flow retime
- [x] Video clip node, timeline, transitions, audio mixer, LUFS, EDL/OTIO export (v0.3)
- [ ] Spectrogram view, convolution reverb with user IRs

## v0.3: AI nodes (optional extras, models downloaded by the user)
- [ ] ONNX Runtime node: upscaling, frame interpolation, segmentation, depth
- [ ] Whisper captions, Demucs stem separation
- [ ] Copilot "look from reference image" (match a LUT to a frame)

## v0.4+: GUI
- [ ] Node editor, viewer, scopes (waveform/vectorscope), timeline, inspector
- [ ] GPU backend (wgpu/Vulkan) behind the same node API
- [ ] Scripting console, plugin marketplace

See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for the full multi-phase plan toward Blender / Resolve / Maya / Figma scope.
