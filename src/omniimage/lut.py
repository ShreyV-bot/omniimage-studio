"""3D/1D LUT support: read/write .cube, apply with trilinear interpolation, bake a node graph into a LUT.

Only *color-only, per-pixel* nodes (``pointwise=True``) can be baked. Blur, grain, glow, vignette etc.
are spatial effects: a LUT cannot represent them, and ``bake_lut`` refuses them with a clear message.
"""
from __future__ import annotations
import uuid
from typing import Callable, Optional

import numpy as np

from .core.graph import Graph, GraphError
from .core.registry import get_spec


class LUT3D:
    """table[b, g, r, 3]; the .cube file order is red fastest, then green, then blue."""

    def __init__(self, table: np.ndarray, title: str = "OmniImage LUT",
                 domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0)):
        t = np.asarray(table, dtype=np.float32)
        if t.ndim != 4 or t.shape[0] != t.shape[1] or t.shape[1] != t.shape[2] or t.shape[3] != 3:
            raise ValueError("LUT table must have shape (N, N, N, 3)")
        self.table = t
        self.title = title
        self.domain_min = np.asarray(domain_min, np.float32)
        self.domain_max = np.asarray(domain_max, np.float32)

    @property
    def size(self) -> int:
        return self.table.shape[0]

    @classmethod
    def identity(cls, size: int = 33) -> "LUT3D":
        n = size
        r = np.linspace(0, 1, n, dtype=np.float32)
        b, g, rr = np.meshgrid(r, r, r, indexing="ij")
        return cls(np.stack([rr, g, b], axis=-1))

    @classmethod
    def from_function(cls, fn: Callable[[np.ndarray], np.ndarray], size: int = 33, title="OmniImage LUT") -> "LUT3D":
        ident = cls.identity(size).table.reshape(-1, 3)
        out = np.asarray(fn(ident), np.float32).reshape(size, size, size, 3)
        return cls(out, title)

    # ---- application -----------------------------------------------------------------
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Trilinear interpolation. ``rgb`` is (..., 3) in the LUT's input domain."""
        n = self.size
        x = (np.asarray(rgb, np.float32) - self.domain_min) / (self.domain_max - self.domain_min)
        x = np.clip(x, 0.0, 1.0) * (n - 1)
        i0 = np.floor(x).astype(np.int32)
        i0 = np.minimum(i0, n - 2) if n > 1 else i0
        f = x - i0
        i1 = np.minimum(i0 + 1, n - 1)
        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
        fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
        t = self.table
        c00 = t[b0, g0, r0] * (1 - fr) + t[b0, g0, r1] * fr
        c01 = t[b0, g1, r0] * (1 - fr) + t[b0, g1, r1] * fr
        c10 = t[b1, g0, r0] * (1 - fr) + t[b1, g0, r1] * fr
        c11 = t[b1, g1, r0] * (1 - fr) + t[b1, g1, r1] * fr
        c0 = c00 * (1 - fg) + c01 * fg
        c1 = c10 * (1 - fg) + c11 * fg
        return (c0 * (1 - fb) + c1 * fb).astype(np.float32)

    def blend(self, amount: float) -> "LUT3D":
        ident = LUT3D.identity(self.size).table
        return LUT3D(ident + (self.table - ident) * float(amount), self.title)

    # ---- .cube I/O ---------------------------------------------------------------------
    def write_cube(self, path: str, title: Optional[str] = None):
        n = self.size
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'TITLE "{(title or self.title).replace(chr(34), "")}"\n')
            f.write("# Created by OmniImage\n")
            f.write(f"LUT_3D_SIZE {n}\n")
            f.write("DOMAIN_MIN %.6f %.6f %.6f\n" % tuple(self.domain_min))
            f.write("DOMAIN_MAX %.6f %.6f %.6f\n" % tuple(self.domain_max))
            flat = self.table.reshape(-1, 3)
            for r, g, b in flat:
                f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    @classmethod
    def read_cube(cls, path: str) -> "LUT3D":
        title, size3, size1 = "LUT", None, None
        dmin, dmax = [0.0] * 3, [1.0] * 3
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                key = line.split()[0]
                if key == "TITLE":
                    title = line[5:].strip().strip('"')
                elif key == "LUT_3D_SIZE":
                    size3 = int(line.split()[1])
                elif key == "LUT_1D_SIZE":
                    size1 = int(line.split()[1])
                elif key == "DOMAIN_MIN":
                    dmin = [float(v) for v in line.split()[1:4]]
                elif key == "DOMAIN_MAX":
                    dmax = [float(v) for v in line.split()[1:4]]
                elif key[0].isalpha():
                    continue  # unsupported keyword (e.g. LUT_3D_INPUT_RANGE): ignore
                else:
                    rows.append([float(v) for v in line.split()[:3]])
        arr = np.asarray(rows, np.float32)
        if size3:
            if arr.shape[0] != size3 ** 3:
                raise ValueError(f"{path}: expected {size3 ** 3} rows, found {arr.shape[0]}")
            return cls(arr.reshape(size3, size3, size3, 3), title, dmin, dmax)
        if size1:
            return LUT3D._from_1d(arr, size1, title, dmin, dmax)
        raise ValueError(f"{path}: no LUT_3D_SIZE or LUT_1D_SIZE found")

    @classmethod
    def _from_1d(cls, arr, size1, title, dmin, dmax) -> "LUT3D":
        """Expand a 1D LUT into an equivalent 33^3 3D LUT (per-channel curves)."""
        if arr.shape[0] != size1:
            raise ValueError("1D LUT row count mismatch")
        n = 33
        grid = np.linspace(0, 1, n, dtype=np.float32)
        xs = np.linspace(0, 1, size1)
        curves = [np.interp(grid, xs, arr[:, c]) for c in range(3)]
        b, g, r = np.meshgrid(curves[2], curves[1], curves[0], indexing="ij")
        return cls(np.stack([r, g, b], -1), title, dmin, dmax)


def bake_lut(graph: Graph, output: str, input_node: str, size: int = 33, port: Optional[str] = None,
             title: str = "OmniImage baked look") -> LUT3D:
    """Evaluate ``graph`` on an identity lattice to produce a 3D LUT.

    ``input_node`` must be a ``lut_input`` node; every node between it and ``output`` must be color-only.
    """
    if input_node not in graph.nodes or output not in graph.nodes:
        raise GraphError("bake_lut: input_node or output not found in graph")
    if graph.nodes[input_node].type != "lut_input":
        raise GraphError(f"bake_lut: {input_node!r} must be a 'lut_input' node (is {graph.nodes[input_node].type!r})")
    # nodes that depend on the input lattice
    dependents = set()
    for nid in graph.topological_order():
        n = graph.nodes[nid]
        if nid == input_node or any(s in dependents or s == input_node for s, _ in n.inputs.values()):
            dependents.add(nid)
    needed = graph._upstream(output) | {output}
    if input_node not in needed:
        raise GraphError(f"bake_lut: {output!r} does not depend on {input_node!r}")
    bad = [nid for nid in dependents & needed
           if nid != input_node and not get_spec(graph.nodes[nid].type).pointwise]
    if bad:
        detail = ", ".join(f"{b} ({graph.nodes[b].type})" for b in bad)
        raise GraphError("cannot bake into a LUT: these nodes are spatial or time-based effects, which a 3D LUT "
                         f"cannot represent: {detail}. Remove them or bake only the color nodes.")
    n = size
    ident = LUT3D.identity(n).table.reshape(-1, 3)              # order: b, g, r (r fastest)
    img = np.concatenate([ident, np.ones((ident.shape[0], 1), np.float32)], axis=1).reshape(n * n, n, 4)
    res = graph.evaluate(output, port, overrides={input_node: {"image": img}}, override_token=uuid.uuid4().hex)
    if not isinstance(res, np.ndarray) or res.shape[:2] != (n * n, n):
        raise GraphError("bake_lut: output must be an image of the same size as the lattice")
    return LUT3D(res[..., :3].reshape(n, n, n, 3), title)
