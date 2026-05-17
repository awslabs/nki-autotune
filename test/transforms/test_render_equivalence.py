"""Render-equivalence regression: applied transforms produce expected NKI body.

This test mirrors the ``kernel_1`` lhs_T-load region in ``kernel_transforms.py``:
applying a tensorize :class:`~nkigym.transforms.Split` on the lhs_T load's free
axis (``F``) with factors ``(16, 128)`` must lower to a 128-wide inner F slice
controlled by a new trip-16 inner loop on the ``d1`` dimension.
"""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
from nkigym.transforms import Split, SplitOption


def _find_lhs_t_load(ir: KernelIR) -> int:
    """Return the nid of the ``ISANode`` ``dma_copy`` that reads ``lhs_T``."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode) and data.op_cls.NAME == "dma_copy" and "lhs_T" in data.reads:
            return nid
    raise AssertionError("lhs_T load not found in canonical IR")


def test_split_lhs_t_F_tensorize_renders_expected_load_region() -> None:
    """Tensorize Split on lhs_T-load F axis renders 128-wide inner-loop slices.

    The canonical IR has two loops above the lhs_T load: ``i_d0_0 in range(16)``
    on K and ``i_d1_0 in range(1)`` on F (with tensorize=2048). After
    ``Split(factors=(16, 128), target_axis='F')`` the F tensorize size shrinks to
    128 and a new inner trip-16 loop ``i_d1_1`` appears as cardinal 1 on d1. The
    rendered slice for the F axis becomes ``(i_d1_0*16 + i_d1_1*1)*128:(i_d1_0*16 + i_d1_1*1+1)*128``,
    matching ``kernel_1``'s lhs_T-load region in ``kernel_transforms.py``.
    """
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    new_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(16, 128), target_axis="F"))
    src = render(new_ir)

    """The new inner trip-16 loop on d1 (cardinal 1) appears in the rendered source."""
    assert "for i_d1_1 in range(16):" in src

    """The K-axis outer load loop is unchanged."""
    assert "for i_d0_0 in range(16):" in src

    """The pre-existing trip-1 outer F loop is preserved at cardinal 0."""
    assert "for i_d1_0 in range(1):" in src

    """The lhs_T HBM read uses a 128-wide F slice indexed by both cardinals."""
    expected_lhs_t_slice = (
        "lhs_T[(i_d0_0*1)*128:(i_d0_0*1+1)*128, " "(i_d1_0*16 + i_d1_1*1)*128:(i_d1_0*16 + i_d1_1*1+1)*128]"
    )
    assert expected_lhs_t_slice in src

    """The lhs_T SBUF write uses the same 128-wide F slice on the contiguous axis."""
    expected_sbuf_slice = "sbuf_lhs_T[0:128, i_d0_0*1, " "(i_d1_0*16 + i_d1_1*1)*128:(i_d1_0*16 + i_d1_1*1+1)*128]"
    assert expected_sbuf_slice in src
