"""End-to-end tests for eager-Fuse access-pattern rewriting.

Originally the Fuse atom relied on a ``fused_iter_var_map`` side-table;
subsequent Split retired the fused iter-var without updating the map and
render failed with ``KeyError``. The eager-Fuse-rewrite refactor rewrites
access patterns directly at apply time so no side-table is needed.

Post canonical-drop-trip-1-loops: canonical no longer emits trip-1 outers
on unbounded axes, so a legal same-axis Fuse pair does not arise from
canonical alone. These tests inject a synthetic trip-1 outer via
``_inject_trip1_outer`` to reach the (1, N) pair state that exercises
the eager-rewrite path.
"""

from dataclasses import replace

import nki
import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, IterVar, KernelModule, SBlock, replace_at_path, resolve_node
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


def _find_lhs_load_d1_fornode_path(module: KernelModule) -> tuple[int, ...]:
    """Return the path to the single d1 ForNode above lhs_T's NKILoad."""
    d1 = module.axis_id_by_name("d1")

    def walk(node, path):
        if isinstance(node, ForNode) and node.iter_var.axis_id == d1:

            def has_lhs(n):
                if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
                    for _slot, ba in n.writes.items():
                        if ba.tensor_name == "a":
                            return True
                if isinstance(n, ForNode):
                    return any(has_lhs(c) for c in n.children)
                return False

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
    raise AssertionError("could not locate d1 ForNode above lhs_T load")


def _inject_trip1_outer(module: KernelModule, fornode_path: tuple[int, ...]) -> tuple[KernelModule, int, int]:
    """Wrap the ForNode at ``fornode_path`` in a fresh trip-1 ForNode on the same axis.

    Canonical no longer emits trip-1 outers, so same-axis Fuse legality
    (``outer.extent == 1``) is unreachable without manual construction.
    This helper restores that state for testing Fuse semantics. Updates
    every SBlock under the target to include the new outer iter-var just
    before the original iter-var in its ``iter_vars`` list.

    Returns ``(module, outer_var_id, inner_var_id)``.
    """
    target = resolve_node(module.body, fornode_path)
    assert isinstance(target, ForNode)
    inner_iv = target.iter_var
    outer_iv = module.allocate_iter_var(axis_id=inner_iv.axis_id, extent=1, role=inner_iv.role)

    def rewrite(node):
        if isinstance(node, SBlock):
            new_ivs: list[IterVar] = []
            for iv in node.iter_vars:
                if iv.var_id == inner_iv.var_id:
                    new_ivs.append(outer_iv)
                new_ivs.append(iv)
            return SBlock(
                iter_vars=new_ivs,
                reads=node.reads,
                writes=node.writes,
                reads_writes=node.reads_writes,
                body=node.body,
                annotations=dict(node.annotations),
            )
        return ForNode(
            iter_var=node.iter_var,
            children=[rewrite(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )

    rewritten_target = rewrite(target)
    wrapper = ForNode(iter_var=outer_iv, children=[rewritten_target], name=None, annotations={})
    new_body = replace_at_path(module.body, fornode_path, wrapper)
    return replace(module, body=new_body), outer_iv.var_id, inner_iv.var_id


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
    """Canonical -> inject trip-1 outer -> Fuse(lhs_T d1 outer+inner) -> Split(fused, factor=128).

    After all atoms, render must succeed and CPU-sim must match numpy golden.
    """
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    inner_path = _find_lhs_load_d1_fornode_path(module)
    module, outer_id, inner_id = _inject_trip1_outer(module, inner_path)
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
    inner_path = _find_lhs_load_d1_fornode_path(module)
    module, outer_id, inner_id = _inject_trip1_outer(module, inner_path)
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
