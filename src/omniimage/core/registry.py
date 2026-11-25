"""Node registry. Every node type is a plain function plus a typed spec.

    @node("exposure", "color", "Brighten or darken in stops.",
          inputs={"image": PortType.IMAGE}, outputs={"image": PortType.IMAGE},
          params={"stops": P("float", 0.0, -10, 10)}, pointwise=True)
    def exposure(inp, p, ctx):
        ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from .ports import PortType
from .params import P


@dataclass
class NodeSpec:
    type: str
    category: str
    doc: str
    fn: Callable
    inputs: Dict[str, PortType] = field(default_factory=dict)
    outputs: Dict[str, PortType] = field(default_factory=dict)
    params: Dict[str, P] = field(default_factory=dict)
    optional: Tuple[str, ...] = ()
    pointwise: bool = False      # color-only per-pixel op: safe to bake into a 3D LUT
    uses_time: bool = False      # result depends on ctx.frame beyond animated params
    sink: bool = False           # has side effects (writes a file); never cached
    fingerprint: Optional[Callable] = None  # extra cache key (e.g. file mtime)
    source: str = "builtin"

    def describe(self) -> dict:
        return {
            "type": self.type, "category": self.category, "doc": self.doc,
            "inputs": {k: v.value for k, v in self.inputs.items()},
            "optional_inputs": list(self.optional),
            "outputs": {k: v.value for k, v in self.outputs.items()},
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "lut_safe": self.pointwise,
        }


REGISTRY: Dict[str, NodeSpec] = {}


def node(type: str, category: str, doc: str, inputs=None, outputs=None, params=None, optional=(),
         pointwise=False, uses_time=False, sink=False, fingerprint=None, source="builtin"):
    def deco(fn):
        if type in REGISTRY and REGISTRY[type].source == "builtin" and source == "builtin":
            raise RuntimeError(f"duplicate node type {type!r}")
        REGISTRY[type] = NodeSpec(type, category, doc, fn, dict(inputs or {}), dict(outputs or {}),
                                  dict(params or {}), tuple(optional), pointwise, uses_time, sink,
                                  fingerprint, source)
        return fn
    return deco


def get_spec(type: str) -> NodeSpec:
    try:
        return REGISTRY[type]
    except KeyError:
        close = [t for t in REGISTRY if type.lower() in t.lower() or t.lower() in type.lower()][:5]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown node type {type!r}.{hint}")


def list_specs(category: Optional[str] = None):
    specs = sorted(REGISTRY.values(), key=lambda s: (s.category, s.type))
    return [s for s in specs if category in (None, s.category)]
