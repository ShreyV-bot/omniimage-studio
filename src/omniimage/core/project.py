"""Project = graph + command bus + undo/redo.

Every edit (from the CLI, a script, a GUI or the AI Copilot) is a command dict applied through
``Project.apply``; the whole history is therefore undoable and replayable."""
from __future__ import annotations
import contextlib
import json
from typing import Any, Dict, List, Optional

from .graph import Graph, GraphError


class Project:
    def __init__(self, graph: Optional[Graph] = None):
        self.graph = graph or Graph()
        self._undo: List[str] = []
        self._redo: List[str] = []
        self._depth = 0
        self.log: List[dict] = []

    def _snapshot(self) -> str:
        return json.dumps(self.graph.to_dict())

    def _restore(self, snap: str):
        self.graph = Graph.from_dict(json.loads(snap))

    @contextlib.contextmanager
    def transaction(self):
        """Group several commands into a single undo step (the Copilot does this per request)."""
        before = self._snapshot()
        self._depth += 1
        try:
            yield self
        except Exception:
            self._depth -= 1
            self._restore(before)
            raise
        self._depth -= 1
        if self._depth == 0 and self._snapshot() != before:
            self._undo.append(before)
            self._redo.clear()

    def apply(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        op = cmd.get("op")
        g = self.graph
        with self.transaction():
            if op == "add_node":
                nid = g.add(cmd["type"], id=cmd.get("id"), **cmd.get("params", {}))
                res = {"id": nid}
            elif op == "remove_node":
                g.remove(cmd["id"]); res = {}
            elif op == "connect":
                g.connect(cmd["src"], cmd["dst"], cmd.get("src_port"), cmd.get("dst_port")); res = {}
            elif op == "disconnect":
                g.disconnect(cmd["dst"], cmd["dst_port"]); res = {}
            elif op == "set_param":
                g.set_param(cmd["node"], cmd["name"], cmd["value"]); res = {}
            else:
                raise GraphError(f"unknown command op {op!r}")
        self.log.append(cmd)
        return res

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def replace_graph(self, graph: Graph):
        with self.transaction():
            self.graph = graph

    def save(self, path: str):
        self.graph.save(path)

    @classmethod
    def load(cls, path: str) -> "Project":
        return cls(Graph.load(path))
