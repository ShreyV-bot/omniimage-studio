# Contributing

Thanks for helping. The project is small enough that the rules fit on one page.

1. `pip install -e ".[dev]" && pytest -q` must pass.
2. **Adding a node** = one decorated function in `src/omniimage/nodes/`. Give it a clear `doc`, typed ports, bounded params (`P(type, default, min, max)`), and flag it `pointwise=True` only if each output pixel depends on that input pixel alone (this is what lets it be baked into a LUT). `tests/test_nodes.py` smoke-tests every built-in node automatically; add a behaviour test for anything non-trivial.
3. **Plugins** live outside the core: put a `.py` file in `~/.omniimage/plugins` or publish a package with an `omniimage.nodes` entry point.
4. Keep the engine deterministic: seeded noise, no wall-clock dependence, no network access in nodes.
5. Anything the Copilot can do must go through `Toolbox` so it stays undoable and sandboxed to its work folder.
6. By contributing you agree your work is licensed under Apache-2.0.
