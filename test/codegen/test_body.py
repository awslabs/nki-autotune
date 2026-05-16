"""Tests for :func:`nkigym.codegen.emit_body` indentation and line-shape contract.

The body emitter writes inside the ``def f(...):`` scope, so every line
it produces must carry the 4-space function-scope indent. The output
must end with ``\\n`` so it composes cleanly with :func:`emit_return`
(which begins with ``    <name> = nl.ndarray(...)``).
"""

import ast

import pytest

from nkigym.codegen import emit_body, emit_header, emit_return
from nkigym.ir import build_initial_ir
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _identity(x):
    """Trivial fixture with a few NKIAlloc leaves."""
    sbuf_x = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    hbm_y = NKIAlloc(location="shared_hbm", shape=(128, 512), dtype="bfloat16")()
    NKILoad()(src=x, dst=sbuf_x)
    NKIStore()(src=sbuf_x, dst=hbm_y)
    return hbm_y


_INPUT_SPECS: dict[str, tuple[int, ...]] = {"x": (128, 512)}


_MK, _MM, _MN = 2048, 2048, 2048
_MATMUL_INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (_MK, _MM), "rhs": (_MK, _MN)}


@nkigym_kernel
def _matmul(lhs_T, rhs):
    """Matmul fixture exercising every compute op (load/memset/matmul/copy/store)."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(_MK, _MM), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(_MK, _MN), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(_MM, _MN), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(_MM, _MN), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(_MM, _MN), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def test_body_lines_are_indented_one_level() -> None:
    """Every non-empty body line begins with the 4-space function-scope indent."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    body = emit_body(ir)
    for line in body.splitlines():
        if not line.strip():
            continue
        assert line.startswith("    "), f"body line missing function-scope indent: {line!r}"


def test_body_ends_with_newline() -> None:
    """Body output ends with ``\\n`` so it composes with :func:`emit_return` cleanly."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    body = emit_body(ir)
    assert body == "" or body.endswith("\n"), "body must end with a newline (or be empty)"


def test_render_output_parses_as_python() -> None:
    """``header + body + ret`` parses as valid Python — proves indentation is consistent."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    src = emit_header(ir) + emit_body(ir) + emit_return(ir)
    ast.parse(src)


def test_render_output_inside_function_scope() -> None:
    """Body lines (alloc assignments, etc.) appear inside the ``def`` block, not at module scope."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    src = emit_header(ir) + emit_body(ir) + emit_return(ir)
    module = ast.parse(src)
    func_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(func_defs) == 1
    func = func_defs[0]
    body_targets = {
        t.id for node in func.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)
    }
    assert {"sbuf_x", "hbm_y"} <= body_targets, f"alloc assignments missing from function body: {body_targets}"
    for node in module.body:
        assert isinstance(node, (ast.Import, ast.FunctionDef)), f"unexpected top-level node: {type(node).__name__}"


def test_emit_body_seam_with_emit_return() -> None:
    """The last body line is followed by the first ``emit_return`` line — no fused tokens."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    header = emit_header(ir)
    body = emit_body(ir)
    ret = emit_return(ir)
    fused = header + body + ret
    assert "buffer=nl.sbuf)    " not in fused
    assert ")    hbm_y" not in fused


def test_isa_calls_use_nisa_dotted_name() -> None:
    """Every compute leaf renders as ``nisa.<NAME>(...)`` — never as a Python class repr."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    assert "<class 'nkigym" not in body, "emit_isa_call still falling back to op_cls repr"
    for isa_name in ("dma_copy", "memset", "nc_matmul", "tensor_copy"):
        assert f"nisa.{isa_name}(" in body, f"missing nisa.{isa_name}( in body"


def test_isa_calls_carry_call_kwargs() -> None:
    """``NKIMemset(value=0.0)`` survives as ``value=0.0`` on the rendered call."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    memset_lines = [line for line in body.splitlines() if "nisa.memset(" in line]
    assert memset_lines, "no nisa.memset( call rendered"
    for line in memset_lines:
        assert "value=0.0" in line, f"memset line missing value kwarg: {line!r}"


def test_isa_calls_emit_loop_var_indices() -> None:
    """ISA call slice expressions reference the enclosing for-loop variables."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    matmul_lines = [line for line in body.splitlines() if "nisa.nc_matmul(" in line]
    assert matmul_lines, "no nisa.nc_matmul( call rendered"
    for line in matmul_lines:
        for dim in ("d0", "d1", "d2"):
            assert dim in line, f"nc_matmul slice missing loop var {dim}: {line!r}"


def test_render_matmul_parses_as_python() -> None:
    """The full matmul fixture renders to syntactically valid Python."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    src = emit_header(ir) + emit_body(ir) + emit_return(ir)
    ast.parse(src)


def test_loop_vars_use_i_dim_cardinal_naming() -> None:
    """Every ``for`` line spells the iter-var as ``i_<dim>_<cardinal>``.

    Canonical trees have no same-dim ancestor on any path, so every
    cardinal is ``0``.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    for_lines = [line.strip() for line in body.splitlines() if line.strip().startswith("for ")]
    assert for_lines, "matmul fixture should produce for-loops"
    for line in for_lines:
        assert line.startswith(("for i_d0_0 ", "for i_d1_0 ", "for i_d2_0 ")), f"non-canonical loop var: {line!r}"


def test_isa_calls_reference_i_dim_cardinal_loop_vars() -> None:
    """ISA-call slices reference ``i_<dim>_<cardinal>`` for the enclosing for-loops."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    matmul_lines = [line for line in body.splitlines() if "nisa.nc_matmul(" in line]
    assert matmul_lines
    for line in matmul_lines:
        for var in ("i_d0_0", "i_d1_0", "i_d2_0"):
            assert var in line, f"nc_matmul slice missing {var}: {line!r}"


def _split_d0_into_outer_8_inner_2(ir):
    """Hand-rewire the matmul fixture's K-loop into two same-dim loops.

    The original ``d0`` loop has trip 16; after the rewire, the path
    above the matmul ISA carries an outer ``d0`` loop of trip 8 and an
    inner ``d0`` loop of trip 2 (8 * 2 == 16 — the original tile range,
    flattened across two ancestor cardinals). Returns the matmul ISA
    node id so callers can read its ``axis_map``.
    """
    tree = ir.tree
    matmul_isa_nid = next(
        nid for nid in tree.preorder() if isinstance(tree.data(nid), ISANode) and tree.data(nid).op_cls is NKIMatmul
    )
    matmul_loops_path = [nid for nid in tree.ancestors(matmul_isa_nid) if isinstance(tree.data(nid), ForNode)]
    d0_loop_nid = next(nid for nid in matmul_loops_path if tree.data(nid).dim == "d0")
    assert tree.data(d0_loop_nid).trip == 16, "fixture precondition: d0 trip is 16"
    parent_nid = tree.parent(d0_loop_nid)
    assert parent_nid is not None
    children_under_d0 = list(tree.children(d0_loop_nid))

    outer_nid = tree.add_node(ForNode(dim="d0", trip=8, loop_type=AxisRole.PARALLEL), parent=parent_nid)
    inner_nid = tree.add_node(ForNode(dim="d0", trip=2, loop_type=AxisRole.PARALLEL), parent=outer_nid)
    for grandchild in children_under_d0:
        tree.graph.remove_edge(d0_loop_nid, grandchild)
        tree.graph.add_edge(inner_nid, grandchild)
    tree.graph.remove_node(d0_loop_nid)
    return matmul_isa_nid


def test_split_loop_increments_cardinal_for_same_dim_ancestor() -> None:
    """When the same dim appears twice on a path, cardinals number by ancestor count.

    The outer ``d0`` loop renders as ``i_d0_0``; the inner same-dim
    loop renders as ``i_d0_1``. The matmul ISA call's slice expression
    must reference both — the operand still spans the original
    ``8 * 2 == 16`` tiles, so dropping the outer cardinal would shrink
    the operand range to just 2 tiles.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    matmul_isa_nid = _split_d0_into_outer_8_inner_2(ir)
    matmul_isa = ir.tree.data(matmul_isa_nid)

    body = emit_body(ir)
    for_lines = [line.strip() for line in body.splitlines() if line.strip().startswith("for ")]
    assert "for i_d0_0 in range(8):" in for_lines, f"missing outer split loop: {for_lines!r}"
    assert "for i_d0_1 in range(2):" in for_lines, f"missing inner split loop: {for_lines!r}"

    matmul_lines = [line for line in body.splitlines() if "nisa.nc_matmul(" in line]
    assert matmul_lines, "split fixture should still emit a matmul call"
    for line in matmul_lines:
        assert "i_d0_0" in line, f"matmul slice should reference the outer d0 loop: {line!r}"
        assert "i_d0_1" in line, f"matmul slice should reference the inner d0 loop: {line!r}"
        assert (
            f"i_{matmul_isa.axis_map['M']}_0" in line
        ), f"matmul slice should still reference the M-axis loop with cardinal 0: {line!r}"


def test_same_dim_ancestor_emits_linear_combination_in_slice() -> None:
    """Multiple same-dim ancestors emit ``i_d_0 * t_inner + i_d_1`` as the tile coord.

    Outer trip 8, inner trip 2: the flattened tile coord is
    ``i_d0_0 * 2 + i_d0_1``. Without the multiplication by the inner
    trip the operand only covers 2 tiles instead of the original 16.
    Asserting on a whitespace-normalised line so the test survives
    formatting changes.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    _split_d0_into_outer_8_inner_2(ir)

    body = emit_body(ir)
    matmul_lines = [line for line in body.splitlines() if "nisa.nc_matmul(" in line]
    assert matmul_lines
    normalized = matmul_lines[0].replace(" ", "")
    assert "i_d0_0*2+i_d0_1" in normalized, (
        f"matmul slice should combine outer d0 (stride = inner trip 2) and inner d0: " f"{matmul_lines[0]!r}"
    )


def test_emit_block_handles_alloc_inside_loop() -> None:
    """``_emit_block`` is generic: an ``NKIAlloc`` leaf nested under a ``ForNode`` renders correctly.

    The canonical builder hoists allocs to root, but rewrites
    (``compute_at``-style) may sink an alloc inside a loop. Hand-sink
    ``psum_acc`` under the matmul's outer for-loop and confirm the
    alloc line renders at the inner indent. The matmul's outer loop
    fully covers M, so axis coverage still holds and ``emit_body``
    does not raise.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    tree = ir.tree
    psum_alloc_nid = next(
        nid
        for nid in tree.preorder()
        if isinstance(tree.data(nid), ISANode)
        and tree.data(nid).op_cls is NKIAlloc
        and tree.data(nid).location == "psum"
    )
    matmul_isa_nid = next(
        nid for nid in tree.preorder() if isinstance(tree.data(nid), ISANode) and tree.data(nid).op_cls is NKIMatmul
    )
    matmul_outer_for_nid = next(nid for nid in tree.children(tree.root) if matmul_isa_nid in tree.descendants(nid))
    tree.graph.remove_edge(tree.root, psum_alloc_nid)
    tree.graph.add_edge(matmul_outer_for_nid, psum_alloc_nid)

    body = emit_body(ir)
    alloc_lines = [line for line in body.splitlines() if "buffer=nl.psum" in line]
    assert alloc_lines, f"psum alloc missing from rendered body:\n{body}"
    assert all(
        line.startswith("        ") for line in alloc_lines
    ), f"psum alloc must be indented inside the matmul outer loop:\n{alloc_lines!r}"


def test_axis_extent_mismatch_raises() -> None:
    """A loop nest that doesn't fully cover an axis must abort codegen.

    Mutate the matmul fixture's K-loop trip from 16 to 8: the product
    of enclosing trips (8) times the matmul tile size (128) is 1024,
    but the K-axis extent is 2048. Codegen must raise rather than
    silently emit out-of-bounds slices.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    tree = ir.tree
    matmul_isa_nid = next(
        nid for nid in tree.preorder() if isinstance(tree.data(nid), ISANode) and tree.data(nid).op_cls is NKIMatmul
    )
    d0_loop_nid = next(
        nid
        for nid in tree.ancestors(matmul_isa_nid)
        if isinstance(tree.data(nid), ForNode) and tree.data(nid).dim == "d0"
    )
    tree.graph.nodes[d0_loop_nid]["data"] = ForNode(dim="d0", trip=8, loop_type=AxisRole.PARALLEL)

    with pytest.raises(AssertionError, match="d0"):
        emit_body(ir)


def test_alloc_partial_extent_coverage_raises() -> None:
    """An ``NKIAlloc`` whose tile size doesn't span the dim extent must abort codegen.

    Shrink the ``psum_acc`` alloc's P tile from full extent (2048) to
    128 while leaving it at root with no enclosing P-loop. The trip
    product (1) times tile size (128) is 128, but the dim extent is
    2048, so axis coverage fails and ``emit_body`` must raise.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    tree = ir.tree
    psum_alloc_nid = next(
        nid
        for nid in tree.preorder()
        if isinstance(tree.data(nid), ISANode)
        and tree.data(nid).op_cls is NKIAlloc
        and tree.data(nid).location == "psum"
    )
    psum_alloc = tree.data(psum_alloc_nid)
    tree.graph.nodes[psum_alloc_nid]["data"] = ISANode(
        op_cls=psum_alloc.op_cls,
        reads=psum_alloc.reads,
        writes=psum_alloc.writes,
        rmw=psum_alloc.rmw,
        tensorize_sizes={"P": 128, "F": psum_alloc.tensorize_sizes["F"]},
        axis_map=psum_alloc.axis_map,
        kwargs=psum_alloc.kwargs,
        location=psum_alloc.location,
        dtype=psum_alloc.dtype,
    )

    with pytest.raises(AssertionError, match="NKIAlloc"):
        emit_body(ir)
