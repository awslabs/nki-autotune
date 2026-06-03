"""Tests for nkigym.transforms._domain_solve."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split
from nkigym.transforms._domain_solve import dim_loops_of_block, enclosing_dim_loops
from nkigym.transforms._tree_ops import _block_local_descendants


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _leaf_in(ir, block_nid: int) -> int:
    for d in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(d), ISANode):
            return d
    raise AssertionError("no ISA leaf")


def test_dim_loops_of_block_canonical_matmul():
    """The canonical matmul block owns d0,d1,d2 loops, each trip 16/16/4."""
    ir = build_canonical_ir()
    mm = _block_for_op(ir, "NKIMatmul")
    loops = dim_loops_of_block(ir.tree, mm)
    assert set(loops) == {"d0", "d1", "d2"}
    assert [e for _v, e in loops["d0"]] == [16]
    assert [e for _v, e in loops["d1"]] == [16]
    assert [e for _v, e in loops["d2"]] == [4]


def _dim_for_node_in_block(ir, block_nid: int, dim: str) -> int:
    loops = dim_loops_of_block(ir.tree, block_nid)
    target_loop_var = loops[dim][0][0]
    for nid in _block_local_descendants(ir.tree, block_nid):
        data = ir.tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var == target_loop_var:
            return nid
    raise AssertionError(f"no {dim} ForNode in block {block_nid}")


def test_dim_loops_of_block_tiled_dim_outer_to_inner():
    """A split dim's loops list outer->inner: (i_d0_0, 2), (i_d0_1, 8)."""
    ir = build_canonical_ir()
    mm = _block_for_op(ir, "NKIMatmul")
    d0_nid = _dim_for_node_in_block(ir, mm, "d0")
    sp = Split()
    opt = next(o for o in sp.analyze(ir) if o.target_nid == d0_nid and o.target_axis is None and o.factors == (2, 8))
    ir2 = sp.apply(ir, opt)
    mm2 = _block_for_op(ir2, "NKIMatmul")
    assert dim_loops_of_block(ir2.tree, mm2)["d0"] == [("i_d0_0", 2), ("i_d0_1", 8)]


def test_enclosing_dim_loops_of_matmul_inner_loop():
    """The dims covered at/above the matmul's innermost loop are d0,d1,d2."""
    ir = build_canonical_ir()
    mm = _block_for_op(ir, "NKIMatmul")
    leaf = _leaf_in(ir, mm)
    innermost = ir.tree.ancestors(leaf)[-1]
    assert isinstance(ir.tree.data(innermost), ForNode)
    enclosing = enclosing_dim_loops(ir.tree, innermost)
    assert set(enclosing) == {"d0", "d1", "d2"}


def test_solve_iter_domains_full_cover():
    """Moved d1 trip 16, target enclosing d1 trip 16 -> covered, residual 1."""
    from nkigym.transforms._domain_solve import solve_iter_domains

    moved = {"d1": [("i_d1_0", 16)]}
    target = {"d1": [("i_d1_0", 16)], "d0": [("i_d0_0", 16)]}
    solved = solve_iter_domains(moved, target)
    assert solved["d1"].residual_extent == 1
    assert solved["d1"].target_loops == [("i_d1_0", 16)]


def test_solve_iter_domains_partial_cover_residual():
    """Moved d1 trip 16, target covers trip 4 -> residual 4."""
    from nkigym.transforms._domain_solve import solve_iter_domains

    moved = {"d1": [("i_d1_0", 16)]}
    target = {"d1": [("i_d1_0", 4)]}
    solved = solve_iter_domains(moved, target)
    assert solved["d1"].residual_extent == 4
    assert solved["d1"].target_loops == [("i_d1_0", 4)]


def test_solve_iter_domains_uncovered_dim_all_residual():
    """A moved dim the target does not iterate stays fully residual."""
    from nkigym.transforms._domain_solve import solve_iter_domains

    moved = {"d2": [("i_d2_0", 4)]}
    target = {"d1": [("i_d1_0", 16)]}
    solved = solve_iter_domains(moved, target)
    assert solved["d2"].residual_extent == 4
    assert solved["d2"].target_loops == []


def test_solve_iter_domains_indivisible_raises():
    """A target coverage that does not divide the moved extent is illegal."""
    import pytest

    from nkigym.transforms._domain_solve import DomainSolveError, solve_iter_domains

    moved = {"d1": [("i_d1_0", 16)]}
    target = {"d1": [("i_d1_0", 5)]}
    with pytest.raises(DomainSolveError, match="divide"):
        solve_iter_domains(moved, target)


def test_regen_and_rebind_full_cover_drops_all_loops():
    """Full-coverage move: the moved block keeps no ForNodes (all covered)."""
    from nkigym.ir.tree import ForNode
    from nkigym.transforms._domain_solve import DimDomain, dim_loops_of_block, regen_and_rebind

    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")  # first load (lhs_T): dims d0, d1
    solved = {
        dim: DimDomain(target_loops=[(f"i_{dim}_0", e[0][1])], residual_extent=1)
        for dim, e in dim_loops_of_block(ir.tree, load).items()
    }
    regen_and_rebind(ir.tree, load, solved)
    remaining = [d for d in ir.tree.descendants(load) if isinstance(ir.tree.data(d), ForNode)]
    assert remaining == []


def test_regen_and_rebind_residual_keeps_one_loop():
    """Partial cover (residual 4 on d1) leaves exactly one residual ForNode on d1."""
    from nkigym.ir.tree import ForNode
    from nkigym.transforms._domain_solve import DimDomain, dim_loops_of_block, regen_and_rebind

    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    loops = dim_loops_of_block(ir.tree, load)
    solved = {
        "d0": DimDomain(target_loops=[("i_d0_0", 16)], residual_extent=1),
        "d1": DimDomain(target_loops=[("i_d1_0", 4)], residual_extent=4),
    }
    regen_and_rebind(ir.tree, load, solved)
    remaining = [ir.tree.data(d) for d in ir.tree.descendants(load) if isinstance(ir.tree.data(d), ForNode)]
    assert len(remaining) == 1
    assert remaining[0].extent == 4
