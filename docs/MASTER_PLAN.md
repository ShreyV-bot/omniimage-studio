# Master Plan: one in all open-source app

Honest framing: those four products represent ~100+ combined engineer-years each. "God level" is a direction, not a milestone. The strategy is to share ONE engine (typed node graph, cache, undo, Copilot) across all four domains, so every feature built for one is reusable in the others, and to ship in phases that each leave a working, tested product.

Engine pillars (already built in v0.1): typed node graph · lazy cached evaluation · keyframes/expressions · command bus + undo · plugin system · Copilot tool API · CLI · tests/CI.
