"""Tests for :func:`nkigym.codegen.emit_body` indentation and line-shape contract.

The body emitter writes inside the ``def f(...):`` scope, so every line
it produces must carry the 4-space function-scope indent. The output
must end with ``\\n`` so it composes cleanly with :func:`emit_return`
(which begins with ``    <name> = nl.ndarray(...)``).
"""

import ast

from nkigym.codegen import emit_body, emit_header, emit_return
from nkigym.ir import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _identity(x):
    """Trivial fixture with a few NKIAlloc leaves."""
    sbuf_x = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    hbm_y = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
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
    hbm_out = NKIAlloc(location="hbm", shape=(_MM, _MN), dtype="bfloat16")()
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


def test_loop_vars_use_i_dim_naming() -> None:
    """Every ``for`` line spells the iter-var as ``i_<dim>``."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    for_lines = [line.strip() for line in body.splitlines() if line.strip().startswith("for ")]
    assert for_lines, "matmul fixture should produce for-loops"
    for line in for_lines:
        assert line.startswith(("for i_d0 ", "for i_d1 ", "for i_d2 ")), f"non-canonical loop var: {line!r}"


def test_isa_calls_reference_i_dim_loop_vars() -> None:
    """ISA-call slices reference ``i_<dim>`` (the renamed loop var), not the bare dim id."""
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    body = emit_body(ir)
    matmul_lines = [line for line in body.splitlines() if "nisa.nc_matmul(" in line]
    assert matmul_lines
    for line in matmul_lines:
        for var in ("i_d0", "i_d1", "i_d2"):
            assert var in line, f"nc_matmul slice missing {var}: {line!r}"
