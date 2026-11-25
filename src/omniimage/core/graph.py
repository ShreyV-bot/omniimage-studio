"""The node graph: typed, validated, lazily evaluated and cached."""
from __future__ import annotations
import hashlib
import json
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .params import ParamError, coerce, resolve
from .ports import PortType, compatible
from .registry import REGISTRY, get_spec


class GraphError(Exception):
    pass


@dataclass
class Ctx:
    """Evaluation context: which frame of which timeline we are computing."""
    frame: int = 0
    fps: float = 24.0

    @property
    def time(self) -> float:
        return self.frame / float(self.fps)


@dataclass
class NodeInst:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # dst_port -> (src_id, src_port)


def _h(*parts) -> str:
    m = hashlib.blake2b(digest_size=12)
    for p in parts:
        m.update(json.dumps(p, sort_keys=True, default=str).encode())
        m.update(b"|")
    return m.hexdigest()


class Graph:
    def __init__(self, cache_limit: int = 96):
        self.nodes: Dict[str, NodeInst] = {}
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self.cache_limit = cache_limit
        self._counter = 0

    # ---- editing -------------------------------------------------------
    def add(self, node_type: str, /, *, id: Optional[str] = None, **params) -> str:
        """Add a node. ``params`` are the node's parameters (a parameter may itself be called ``type``)."""
        type = node_type
        spec = get_spec(type)
        if id is None:
            while True:
                self._counter += 1
                id = f"{type}_{self._counter}"
                if id not in self.nodes:
                    break
        if id in self.nodes:
            raise GraphError(f"node id {id!r} already exists")
        n = NodeInst(id, type)
        self.nodes[id] = n
        try:
            for k, v in params.items():
                self.set_param(id, k, v)
        except Exception:
            del self.nodes[id]
            raise
        return id

    def remove(self, id: str):
        self._need(id)
        del self.nodes[id]
        for n in self.nodes.values():
            for port in [p for p, (s, _) in n.inputs.items() if s == id]:
                del n.inputs[port]

    def set_param(self, id: str, name: str, value):
        n = self._need(id)
        spec = get_spec(n.type)
        if name not in spec.params:
            raise ParamError(f"node type {n.type!r} has no parameter {name!r}. Valid: {sorted(spec.params)}")
        n.params[name] = coerce(name, spec.params[name], value)

    def connect(self, src: str, dst: str, src_port: Optional[str] = None, dst_port: Optional[str] = None):
        s, d = self._need(src), self._need(dst)
        sspec, dspec = get_spec(s.type), get_spec(d.type)
        if not sspec.outputs:
            raise GraphError(f"node {src!r} ({s.type}) has no outputs")
        if src_port is None:
            src_port = next(iter(sspec.outputs))
        if src_port not in sspec.outputs:
            raise GraphError(f"node {src!r} has no output {src_port!r}; outputs: {list(sspec.outputs)}")
        stype = sspec.outputs[src_port]
        if dst_port is None:
            cands = [p for p, t in dspec.inputs.items() if compatible(stype, t)]
            free = [p for p in cands if p not in d.inputs]
            if not cands:
                raise GraphError(f"node {dst!r} ({d.type}) has no input accepting {stype.value}")
            dst_port = (free or cands)[0]
        if dst_port not in dspec.inputs:
            raise GraphError(f"node {dst!r} has no input {dst_port!r}; inputs: {list(dspec.inputs)}")
        if not compatible(stype, dspec.inputs[dst_port]):
            raise GraphError(f"type mismatch: {src}.{src_port} is {stype.value}, "
                             f"{dst}.{dst_port} expects {dspec.inputs[dst_port].value}")
        if src == dst or dst in self._upstream(src):
            raise GraphError(f"connecting {src} -> {dst} would create a cycle")
        d.inputs[dst_port] = (src, src_port)

    def disconnect(self, dst: str, dst_port: str):
        n = self._need(dst)
        if dst_port not in n.inputs:
            raise GraphError(f"{dst}.{dst_port} is not connected")
        del n.inputs[dst_port]

    def chain(self, *ids: str) -> str:
        for a, b in zip(ids, ids[1:]):
            self.connect(a, b)
        return ids[-1]

    def _need(self, id: str) -> NodeInst:
        if id not in self.nodes:
            raise GraphError(f"no node with id {id!r}. Existing: {list(self.nodes)[:20]}")
        return self.nodes[id]

    def _upstream(self, id: str) -> set:
        seen, stack = set(), [id]
        while stack:
            cur = stack.pop()
            for s, _ in self.nodes[cur].inputs.values():
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return seen

    # ---- inspection ----------------------------------------------------
    def validate(self) -> List[str]:
        issues = []
        for n in self.nodes.values():
            if n.type not in REGISTRY:
                issues.append(f"{n.id}: unknown node type {n.type!r}")
                continue
            spec = REGISTRY[n.type]
            for port in spec.inputs:
                if port not in n.inputs and port not in spec.optional:
                    issues.append(f"{n.id} ({n.type}): required input {port!r} is not connected")
            for port, (s, sp) in n.inputs.items():
                if s not in self.nodes:
                    issues.append(f"{n.id}.{port}: source node {s!r} missing")
        try:
            self.topological_order()
        except GraphError as e:
            issues.append(str(e))
        return issues

    def topological_order(self) -> List[str]:
        indeg = {i: 0 for i in self.nodes}
        out: Dict[str, List[str]] = {i: [] for i in self.nodes}
        for n in self.nodes.values():
            for s, _ in n.inputs.values():
                if s in self.nodes:
                    indeg[n.id] += 1
                    out[s].append(n.id)
        q = deque(i for i, d in indeg.items() if d == 0)
        order = []
        while q:
            i = q.popleft()
            order.append(i)
            for j in out[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)
        if len(order) != len(self.nodes):
            raise GraphError("graph contains a cycle")
        return order

    def describe(self) -> str:
        if not self.nodes:
            return "(empty graph)"
        lines = []
        for i in self.topological_order():
            n = self.nodes[i]
            ins = ", ".join(f"{p}<-{s}.{sp}" for p, (s, sp) in n.inputs.items())
            ps = ", ".join(f"{k}={v}" for k, v in n.params.items())
            lines.append(f"{i} [{n.type}] params({ps}) inputs({ins})")
        return "\n".join(lines)

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": 1, "nodes": [
            {"id": n.id, "type": n.type, "params": n.params,
             "inputs": {p: list(v) for p, v in n.inputs.items()}} for n in self.nodes.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls()
        for nd in d.get("nodes", []):
            g.add(nd["type"], id=nd["id"], **nd.get("params", {}))
        for nd in d.get("nodes", []):
            for port, (s, sp) in nd.get("inputs", {}).items():
                g.connect(s, nd["id"], sp, port)
        return g

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Graph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def copy(self) -> "Graph":
        return Graph.from_dict(json.loads(json.dumps(self.to_dict())))

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, target: str, port: Optional[str] = None, ctx: Optional[Ctx] = None,
                 overrides: Optional[Dict[str, dict]] = None, override_token: str = ""):
        """Compute one output. Only the ancestors of ``target`` are evaluated (lazy), results are cached by
        content key so unchanged branches are never recomputed."""
        ctx = ctx or Ctx()
        overrides = overrides or {}
        self._need(target)
        memo: Dict[str, Tuple[str, dict]] = {}

        def run(nid: str):
            if nid in memo:
                return memo[nid]
            n = self.nodes[nid]
            if nid in overrides:
                res = ("ovr:" + _h(nid, override_token), overrides[nid])
                memo[nid] = res
                return res
            spec = get_spec(n.type)
            invals, inkeys = {}, {}
            for p in spec.inputs:
                if p in n.inputs:
                    sid, sp = n.inputs[p]
                    key, outs = run(sid)
                    if sp not in outs:
                        raise GraphError(f"{sid} produced no output {sp!r}")
                    invals[p] = outs[sp]
                    inkeys[p] = [key, sp]
                elif p not in spec.optional:
                    raise GraphError(f"node {nid} ({n.type}): required input {p!r} is not connected")
            params = {}
            for name, ps in spec.params.items():
                v = resolve(n.params.get(name, ps.default), ctx)
                params[name] = coerce(name, ps, v)
            fp = spec.fingerprint(params) if spec.fingerprint else None
            key = _h(n.type, params, inkeys, ctx.frame if spec.uses_time else None, fp)
            if not spec.sink and key in self._cache:
                self._cache.move_to_end(key)
                res = (key, self._cache[key])
            else:
                try:
                    out = spec.fn(invals, params, ctx)
                except (GraphError, ParamError):
                    raise
                except Exception as e:
                    raise GraphError(f"node {nid} ({n.type}) failed: {type(e).__name__}: {e}") from e
                if out is None:
                    out = {}
                elif not isinstance(out, dict):
                    if len(spec.outputs) != 1:
                        raise GraphError(f"node type {n.type!r} returned a bare value but has {len(spec.outputs)} outputs")
                    out = {next(iter(spec.outputs)): out}
                if not spec.sink:
                    self._cache[key] = out
                    while len(self._cache) > self.cache_limit:
                        self._cache.popitem(last=False)
                res = (key, out)
            memo[nid] = res
            return res

        _, outs = run(target)
        if port is None:
            if not outs:
                return None
            port = next(iter(outs))
        if port not in outs:
            raise GraphError(f"{target} has no output {port!r}; outputs: {list(outs)}")
        return outs[port]

    def clear_cache(self):
        self._cache.clear()
