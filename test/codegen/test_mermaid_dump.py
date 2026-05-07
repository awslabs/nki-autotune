"""Unit tests for the dump_forest_mermaid helper."""

from dataclasses import dataclass, field

from nkigym.codegen.graph import DimInfo, OpGraph
from nkigym.codegen.mermaid import dump_forest_mermaid

"""Stand-in for an NKIOp subclass — only __name__ is consulted."""
_FakeOpClass = type("FakeOp", (), {})


@dataclass
class _FakeOp:
    """Stand-in for ParsedOp — only op_cls is consulted by dump_forest_mermaid."""

    op_cls: type = field(default_factory=lambda: _FakeOpClass)
    idx: int = 0


def _empty_op_graph() -> OpGraph:
    """An OpGraph with no ops, no tensors — used by the empty-forest test."""
    return OpGraph(func_name="f_empty", param_names=[], return_name="", tensors={}, dims={}, ops=[])


def test_dump_empty_forest_emits_graph_td_header_only() -> None:
    """An empty forest renders to just the 'graph TD' header line and nothing else."""
    src = dump_forest_mermaid(forest=[], op_graph=_empty_op_graph())
    assert src.splitlines() == ["graph TD"]


def test_dump_single_loop_with_leaf_child() -> None:
    """One LoopNode with one BodyLeaf child renders as two nodes + one edge."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.ops.base import AxisRole

    leaf = BodyLeaf(op_idx=0, phase="main")
    node = LoopNode(dim_id="d0", trip_count=16, role=AxisRole.PARALLEL, children=[leaf], name="i_d0_0")
    forest = [node]
    og = OpGraph(
        func_name="f",
        param_names=[],
        return_name="",
        tensors={},
        dims={"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)},
        ops=[_FakeOp()],
    )
    src = dump_forest_mermaid(forest=forest, op_graph=og)
    lines = src.splitlines()
    assert lines[0] == "graph TD"
    joined = "\n".join(lines[1:])
    assert 'L_0["L(0,)<br/>dim=d0 trip=16<br/>role=PARALLEL name=i_d0_0"]' in joined
    assert 'B_0_0(["B(0, 0)<br/>op=FakeOp phase=main"])' in joined
    assert "L_0 -- 0 --> B_0_0" in joined


def test_dump_canonical_rmsnorm_matmul_forest_is_well_formed() -> None:
    """End-to-end smoke: dump the real canonical rmsnorm+matmul forest."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest

    op_graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(op_graph)
    src = dump_forest_mermaid(forest=forest, op_graph=op_graph)
    assert src.startswith("graph TD\n")
    assert len(forest) == 8
    assert src.count("\nL_0[") == 1
    assert src.count("\nL_1[") == 1
    assert src.count("\nL_7[") == 1
    assert "NKILoad" in src
    assert "NKIMatmul" in src
    assert "phase=psum_init" in src
    assert "phase=compute" in src
    assert "phase=drain" in src
    assert "reduce_op=add" in src
