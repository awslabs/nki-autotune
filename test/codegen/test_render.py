"""Tests for :func:`nkigym.codegen.render` — the top-level codegen entry."""

from nkigym.codegen import emit_header, emit_return, render
from nkigym.ir import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore

_INPUT_SPECS: dict[str, tuple[int, ...]] = {"x": (128, 512)}


@nkigym_kernel
def _identity(x):
    """Trivial fixture: load x, store back to a fresh HBM buffer."""
    sbuf_x = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    hbm_y = NKIAlloc(location="shared_hbm", shape=(128, 512), dtype="bfloat16")()
    NKILoad()(src=x, dst=sbuf_x)
    NKIStore()(src=sbuf_x, dst=hbm_y)
    return hbm_y


def test_render_starts_with_header() -> None:
    """``render`` produces a string that begins with the ``emit_header`` output."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    src = render(ir)
    assert src.startswith(emit_header(ir))


def test_render_ends_with_return_block() -> None:
    """``render`` closes with the ``emit_return`` output."""
    ir = build_initial_ir(_identity, _INPUT_SPECS)
    src = render(ir)
    assert src.endswith(emit_return(ir))
