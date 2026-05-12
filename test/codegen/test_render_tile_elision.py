"""Regression test: renderer elides innermost tile loops.

After canonical builds outer+inner ForNodes per axis, the renderer must
emit only the outer as a python `for`. The inner's extent is consumed as
slice width on the ISA call.
"""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.memset import NKIMemset


@nkigym_kernel
def _memset_only():
    psum = NKIAlloc(location="psum", shape=(128, 2048), dtype="float32")()
    NKIMemset(value=0.0)(dst=psum)
    return psum


def test_render_emits_only_outer_tile_loop():
    """Memset P=128 has MAX=128 → trip=1 outer + tile=128 inner; F=None → trip=1 outer + tile=2048 inner.

    Renderer must emit:
    - `for i_<P>_0 in range(1):` (outer P)
    - `for i_<F>_0 in range(1):` (outer F)
    No `for ... in range(128)` nor `for ... in range(2048)` — both are inner tile loops.
    """
    module = build_canonical_module(_memset_only, input_specs={})
    source = render(module)
    assert "range(128)" not in source, f"inner P-tile loop not elided:\n{source}"
    assert "range(2048)" not in source, f"inner F-tile loop not elided:\n{source}"


def test_render_slice_uses_inner_tile_extent():
    """Memset's dst slice should cover the full inner tile per axis.

    SBUF/PSUM physical shape is ``(P_tile, num_P_slots, F_tile * num_F_tiles)``.
    Partition (dim 0) emits ``0:P_tile`` — here ``0:128``. F-axis (last
    dim) emits ``(outer) * F_tile : (outer) * F_tile + F_tile`` — here
    the ``+ 2048`` tail encodes the inner F-tile width.
    """
    module = build_canonical_module(_memset_only, input_specs={})
    source = render(module)
    assert "0:128" in source, f"P partition slice (width 128) missing from:\n{source}"
    assert "+ 2048" in source, f"F inner-tile slice width 2048 missing from:\n{source}"
