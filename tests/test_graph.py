import numpy as np
import pytest
import omniimage as o
from omniimage.core.params import safe_eval, ParamError
from omniimage.core.registry import node
from omniimage.core.ports import PortType as T
from omniimage.core.params import P


def small(g, w=32, h=24):
    return g.add("gradient", width=w, height=h)


def test_type_mismatch_rejected():
    g = o.Graph()
    a = g.add("osc", duration=0.1)
    b = g.add("exposure")
    with pytest.raises(o.GraphError, match="no input accepting audio"):
        g.connect(a, b)


def test_explicit_port_type_mismatch():
    g = o.Graph()
    a = g.add("osc", duration=0.1)
    b = g.add("merge")
    with pytest.raises(o.GraphError, match="type mismatch"):
        g.connect(a, b, dst_port="a")


def test_cycle_detected():
    g = o.Graph()
    a, b = g.add("exposure"), g.add("contrast")
    g.connect(a, b)
    with pytest.raises(o.GraphError, match="cycle"):
        g.connect(b, a)


def test_unknown_param_and_clamping():
    g = o.Graph()
    with pytest.raises(ParamError, match="no parameter"):
        g.add("exposure", wrong=1)
    n = g.add("exposure", stops=999)
    assert g.nodes[n].params["stops"] == 12  # clamped to spec max


def test_choice_validated():
    g = o.Graph()
    with pytest.raises(ParamError):
        g.add("merge", mode="nonsense")


def test_missing_input_reported():
    g = o.Graph()
    n = g.add("exposure")
    assert any("required input" in i for i in g.validate())
    with pytest.raises(o.GraphError, match="not connected"):
        g.evaluate(n)


def test_lazy_and_cached_evaluation():
    calls = {"n": 0}

    @node("test_counter", "test", "counts calls", inputs={"image": T.IMAGE}, outputs={"image": T.IMAGE},
          params={"k": P("float", 1.0)}, source="plugin")
    def counter(inp, p, ctx):
        calls["n"] += 1
        return inp["image"] * p["k"]

    g = o.Graph()
    src = small(g)
    used = g.add("test_counter")
    unused = g.add("test_counter")
    g.connect(src, used)
    g.connect(src, unused)
    g.evaluate(used)
    assert calls["n"] == 1            # lazy: the unused branch was never computed
    g.evaluate(used)
    assert calls["n"] == 1            # cached: same inputs and params
    g.set_param(used, "k", 2.0)
    g.evaluate(used)
    assert calls["n"] == 2            # param change invalidates only this node


def test_serialization_roundtrip(tmp_path):
    g = o.Graph()
    a = small(g)
    b = g.add("exposure", stops=1.5)
    g.connect(a, b)
    path = tmp_path / "g.json"
    g.save(str(path))
    g2 = o.Graph.load(str(path))
    assert np.allclose(g.evaluate(b), g2.evaluate(b))


def test_keyframes_and_expressions():
    g = o.Graph()
    a = small(g)
    b = g.add("exposure", stops={"keys": [[0, 0], [10, 2]], "interp": "linear"})
    g.connect(a, b)
    v0 = g.evaluate(b, ctx=o.Ctx(0))[0, -1, 0]
    v10 = g.evaluate(b, ctx=o.Ctx(10))[0, -1, 0]
    v5 = g.evaluate(b, ctx=o.Ctx(5))[0, -1, 0]
    assert v10 == pytest.approx(v0 * 4, rel=1e-4)       # +2 stops = x4
    assert v5 == pytest.approx(v0 * 2, rel=1e-4)        # +1 stop
    g.set_param(b, "stops", "=frame/10")
    assert g.evaluate(b, ctx=o.Ctx(10))[0, -1, 0] == pytest.approx(v0 * 2, rel=1e-4)


@pytest.mark.parametrize("expr", ["__import__('os').system('echo hi')", "(1).__class__", "open('x')", "a+1", "[1,2][0]"])
def test_expressions_are_sandboxed(expr):
    with pytest.raises(ParamError):
        safe_eval(expr, {"t": 0.0})


def test_expression_math():
    assert safe_eval("sin(0) + max(2, 3) * 2", {}) == 6
    assert safe_eval("clamp(t*10, 0, 1)", {"t": 5}) == 1


def test_project_undo_redo_and_transaction_rollback():
    pr = o.Project()
    pr.apply({"op": "add_node", "type": "solid", "params": {"width": 8, "height": 8}})
    pr.apply({"op": "add_node", "type": "exposure"})
    assert len(pr.graph.nodes) == 2
    assert pr.undo() and len(pr.graph.nodes) == 1
    assert pr.redo() and len(pr.graph.nodes) == 2
    with pytest.raises(Exception):
        with pr.transaction():
            pr.apply({"op": "add_node", "type": "invert"})
            pr.apply({"op": "add_node", "type": "does_not_exist"})
    assert len(pr.graph.nodes) == 2                      # whole transaction rolled back
    with pr.transaction():
        pr.apply({"op": "add_node", "type": "invert"})
        pr.apply({"op": "add_node", "type": "sepia"})
    assert len(pr.graph.nodes) == 4
    pr.undo()
    assert len(pr.graph.nodes) == 2                      # grouped as one undo step


def test_remove_node_drops_edges():
    g = o.Graph()
    a = small(g)
    b = g.add("exposure")
    g.connect(a, b)
    g.remove(a)
    assert not g.nodes[b].inputs
