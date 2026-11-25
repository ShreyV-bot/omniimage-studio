"""Parameter specs, validation, keyframes and safe expressions.

Any numeric parameter may be:
  * a plain value                         3.5
  * a keyframe dict                       {"keys": [[0, 0.0], [24, 1.0]], "interp": "linear"|"smooth"|"hold"}
  * an expression string starting with =  "=0.5 + 0.5*sin(t*6.28)"   (t = seconds, frame, fps)
"""
from __future__ import annotations
import ast
import math
from dataclasses import dataclass, field
from typing import Any, Optional


class ParamError(ValueError):
    pass


@dataclass
class P:
    type: str = "float"          # float | int | bool | str | choice | color | list | any
    default: Any = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    choices: tuple = ()
    doc: str = ""
    animatable: bool = True

    def to_dict(self) -> dict:
        d = {"type": self.type, "default": self.default, "doc": self.doc}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.choices:
            d["choices"] = list(self.choices)
        return d


_ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": pow, "floor": math.floor, "ceil": math.ceil, "exp": math.exp,
    "log": math.log, "pi": math.pi, "tau": math.tau,
    "clamp": lambda x, a, b: max(a, min(b, x)),
    "mix": lambda a, b, k: a + (b - a) * k,
    "smoothstep": lambda a, b, x: (lambda t: t * t * (3 - 2 * t))(max(0.0, min(1.0, (x - a) / (b - a) if b != a else 0.0))),
    "pulse": lambda t, hz=1.0: 1.0 if (t * hz) % 1.0 < 0.5 else 0.0,
}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call, ast.Load,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv,
                  ast.IfExp, ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq, ast.BoolOp,
                  ast.And, ast.Or)


def safe_eval(expr: str, variables: dict) -> float:
    """Evaluate a small math expression with a whitelist of syntax and functions."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ParamError(f"bad expression {expr!r}: {e}")
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED_NODES):
            raise ParamError(f"expression {expr!r} uses disallowed syntax: {type(n).__name__}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in _ALLOWED_FUNCS):
            raise ParamError(f"expression {expr!r} calls an unknown function")
        if isinstance(n, ast.Name) and n.id not in _ALLOWED_FUNCS and n.id not in variables:
            raise ParamError(f"expression {expr!r} uses unknown name {n.id!r}")
    env = dict(_ALLOWED_FUNCS)
    env.update(variables)
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)


def eval_keys(keys, frame: float, interp: str = "linear") -> float:
    ks = sorted((float(a), float(b)) for a, b in keys)
    if not ks:
        raise ParamError("empty keyframe list")
    if frame <= ks[0][0]:
        return ks[0][1]
    if frame >= ks[-1][0]:
        return ks[-1][1]
    for (f0, v0), (f1, v1) in zip(ks, ks[1:]):
        if f0 <= frame <= f1:
            if interp == "hold":
                return v0
            u = (frame - f0) / (f1 - f0)
            if interp == "smooth":
                u = u * u * (3 - 2 * u)
            return v0 + (v1 - v0) * u
    return ks[-1][1]


def is_dynamic(value) -> bool:
    return (isinstance(value, dict) and "keys" in value) or (isinstance(value, str) and value.startswith("="))


def resolve(value, ctx):
    """Turn keyframes / expressions into a plain value for the current frame."""
    if isinstance(value, dict) and "keys" in value:
        return eval_keys(value["keys"], ctx.frame, value.get("interp", "linear"))
    if isinstance(value, str) and value.startswith("="):
        return safe_eval(value[1:], {"t": ctx.time, "frame": ctx.frame, "fps": ctx.fps})
    return value


def coerce(name: str, spec: P, value):
    """Validate/convert a *static* value against its spec."""
    if is_dynamic(value):
        if not spec.animatable or spec.type not in ("float", "int"):
            raise ParamError(f"parameter {name!r} cannot be animated")
        return value
    t = spec.type
    try:
        if t == "float":
            value = float(value)
        elif t == "int":
            value = int(round(float(value)))
        elif t == "bool":
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            else:
                value = bool(value)
        elif t == "str":
            value = str(value)
        elif t == "choice":
            value = str(value)
            if value not in spec.choices:
                raise ParamError(f"parameter {name!r} must be one of {list(spec.choices)}, got {value!r}")
        elif t == "color":
            if isinstance(value, str):
                value = parse_color(value)
            value = [float(c) for c in value]
            if len(value) == 3:
                value.append(1.0)
            if len(value) != 4:
                raise ParamError(f"parameter {name!r} needs 3 or 4 color components")
        elif t == "list":
            value = list(value)
        # "any": untouched
    except (TypeError, ValueError) as e:
        raise ParamError(f"parameter {name!r}: cannot convert {value!r} to {t} ({e})")
    if t in ("float", "int"):
        if spec.min is not None and value < spec.min:
            value = type(value)(spec.min)
        if spec.max is not None and value > spec.max:
            value = type(value)(spec.max)
    return value


def parse_color(s: str):
    """'#rrggbb', '#rrggbbaa' or 'r,g,b[,a]' (0..1). Hex values are sRGB-encoded and returned in the same space."""
    s = s.strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        if len(h) not in (6, 8):
            raise ParamError(f"bad hex color {s!r}")
        vals = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
        return vals + ([1.0] if len(vals) == 3 else [])
    parts = [float(x) for x in s.split(",")]
    return parts
