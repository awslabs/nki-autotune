"""End-to-end test: Fuse then Split on the same axis renders + CPU-sims.

This test currently fails at render time with
``KeyError: 'iter var N is neither live nor recorded in fused_iter_var_map'``
because Fuse leaves stale retired iter-var ids in BufferAccess patterns
and relies on fused_iter_var_map to resolve them — but Split then retires
the fused iter-var without updating the map.

After the eager-Fuse-rewrite refactor, Fuse rewrites access patterns
directly so no side-table is needed; subsequent Split works normally.
"""

import nki
import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.fuse import Fuse
from nkigym.tune.split import Split
from nkigym.tune.verify import _rewrite_to_fp32


@nkigym_kernel
def _matmul(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a)
    NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s)
    NKIStore()(src=s, dst=h)
    return h


_SPECS = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}


def _find_lhs_d1_pair(module):
    d1 = module.axis_id_by_name("d1")

    def walk(node, anc):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKILoad":
            for _s, ba in node.writes.items():
                if ba.tensor_name == "a":
                    return anc
        if isinstance(node, ForNode):
            new_anc = anc + [node.iter_var] if node.iter_var.axis_id == d1 else anc
            for c in node.children:
                r = walk(c, new_anc)
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None and len(r) == 2:
            return r[0].var_id, r[1].var_id
    raise AssertionError("could not locate d1 pair")


def _find_d1_2048_path_above_lhs(module):
    d1 = module.axis_id_by_name("d1")

    def has_lhs(n):
        if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
            for _s, ba in n.writes.items():
                if ba.tensor_name == "a":
                    return True
        if isinstance(n, ForNode):
            return any(has_lhs(c) for c in n.children)
        return False

    def walk(node, path):
        if isinstance(node, ForNode) and node.iter_var.axis_id == d1 and node.iter_var.extent == 2048:
            if has_lhs(node):
                return path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None

    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise AssertionError("could not locate fused d1 path")


def test_fuse_then_split_renders_and_cpu_sims():
    """Canonical -> Fuse(lhs_T d1 outer+inner) -> Split(fused, factor=128).

    After all atoms, render must succeed and CPU-sim must match numpy golden.
    """
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    outer_id, inner_id = _find_lhs_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)
    fused_path = _find_d1_2048_path_above_lhs(module)
    module = Split(loop_path=fused_path, factor=128).apply(module)

    source = render(module)
    sim_src = _rewrite_to_fp32(source)
    ns = {}
    exec(sim_src, ns)
    fn = ns["_matmul"]
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    rhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = np.asarray(nki.simulate(fn)(lhs, rhs))
    expected = lhs.T @ rhs
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3), f"max_abs={float(np.abs(actual - expected).max()):.3e}"


def test_same_axis_fuse_drops_outer_from_access_patterns():
    """After same-axis Fuse(outer.extent=1, inner), retired outer var_id is
    absent from every BufferAccess.iter_var_coeffs in affected subtree."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    outer_id, inner_id = _find_lhs_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)

    def all_sblocks(module):
        def walk(node):
            if isinstance(node, SBlock):
                yield node
            if isinstance(node, ForNode):
                for c in node.children:
                    yield from walk(c)

        for root in module.body:
            yield from walk(root)

    """outer_id (extent-1, retired) must not appear in any access pattern."""
    for sblock in all_sblocks(module):
        for access_map in (sblock.reads, sblock.writes, sblock.reads_writes):
            for ba in access_map.values():
                assert outer_id not in ba.iter_var_ids
                for ar in ba.pattern:
                    assert outer_id not in dict(ar.iter_var_coeffs)
