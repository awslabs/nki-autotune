"""Layer-B structural oracle: drive TVM's OWN TensorIR schedule and read back loop shape.

Test-only; never imported by shipped ``nkigym/`` code. The TVM (TIRx fork) dependency
lives exclusively here so later tasks can diff our Split/Fuse/Reorder against TVM's own
``schedule.split`` (etc.) without leaking TVM into the package.

API surface used (TIRx fork, namespace ``tvm.s_tir`` / ``tvm.tirx`` — NOT classic ``tvm.tir``):
    - ``tvm.te.create_prim_func`` builds a schedulable s_tir ``PrimFunc`` from a TE compute
      (avoids ``T.prim_func`` source-introspection fragility).
    - ``tvm.s_tir.Schedule(func)`` constructs the schedule.
    - ``Schedule.get_sblock(name)`` / ``Schedule.get_loops(block)`` navigate to the loop(s).
    - ``Schedule.split(loop, factors=[...])`` returns the new loops (outer -> inner).
    - ``Schedule.reorder(*loops)`` permutes a perfect chain in place; re-fetching via
      ``get_loops`` reads back the new outer -> inner order.
    - ``Schedule.get(loop_rv).extent`` reads back each loop's extent.
    - The recovered binding of the original iter var lives on the block's
      ``SBlockRealize.iter_values[i]`` (found via ``stmt_functor.post_order_visit``).

Binding normalization (``_canonical_binding``) renames each *loop* var to its positional
``i0, i1, ...`` (outer -> inner) name and TVM-simplifies. It therefore assumes the block's
bindings are **affine** in the loop vars (the only regime our IR emits, and the only regime
TVM's ``reorder``/``split`` accept — both require affine, data-parallel/reduction bindings);
a non-affine binding would simplify to an opaque expression and the string compare would
no longer be a meaningful structural diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

import tvm.tirx as TX
from tvm.arith import Analyzer
from tvm.s_tir import Schedule
from tvm.s_tir.schedule import LoopRV, SBlockRV
from tvm.te import compute, create_prim_func, placeholder
from tvm.tirx import stmt_functor

_F32 = "float32"


@dataclass(frozen=True)
class LoopNest:
    """Normalized summary of a loop nest produced by a TVM schedule transform.

    Attributes
    ----------
    extents:
        The loop extents, outer -> inner.
    binding:
        Canonical string of the original iter var's recovered binding, with the new
        loop vars renamed positionally to ``i0, i1, ...`` (outer -> inner) and
        ``*`` rendered tight (e.g. ``"i0*4 + i1"``).
    """

    extents: list[int]
    binding: str


@dataclass(frozen=True)
class FusedNest:
    """Normalized summary of the single loop produced by TVM's ``schedule.fuse``.

    ``fuse`` collapses a perfect nest into ONE loop of product extent, so unlike a
    split (one source iter -> several loops) it leaves SEVERAL source iters bound to
    ONE fused loop var. The structural signal is therefore a single ``extent`` plus
    the *list* of recovered per-iter bindings.

    Attributes
    ----------
    extent:
        The single fused loop's extent (the product of the source extents).
    bindings:
        Canonical strings of each original iter var's recovered binding, in
        source outer -> inner order, with the single fused loop var renamed to ``i0``
        and ``*`` rendered tight. For a 2-deep ``[4, 4]`` fuse these are
        ``["i0 // 4", "i0 % 4"]`` (TVM gives ``fused // inner`` and ``fused % inner``).
    """

    extent: int
    bindings: list[str]


def _find_block_realize(func_body: TX.Stmt, name: str) -> TX.SBlockRealize:
    """Locate the ``SBlockRealize`` for the named block within a function body."""
    matches: list[TX.SBlockRealize] = []
    stmt_functor.post_order_visit(
        func_body,
        lambda node: (
            matches.append(node) if isinstance(node, TX.SBlockRealize) and node.block.name_hint == name else None
        ),
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one realize of block {name!r}, found {len(matches)}")
    return matches[0]


def _canonical_binding(binding: TX.PrimExpr, loop_vars: list[TX.Var]) -> str:
    """Rename loop vars to positional ``i0, i1, ...`` outer->inner, simplify, stringify tight."""
    vmap = {lv: TX.Var(f"i{idx}", lv.dtype) for idx, lv in enumerate(loop_vars)}
    remapped = stmt_functor.substitute(binding, vmap)
    simplified = Analyzer().simplify(remapped)
    return str(simplified).replace(" * ", "*")


def _perfect_nest_schedule(extents: list[int]) -> tuple[Schedule, SBlockRV, list[LoopRV]]:
    """Build a perfect loop nest over a single flattened buffer and return its schedule.

    Constructs ``B[i0, ..., i_{n-1}] = A[flat] * 2.0`` where ``flat`` is the row-major
    affine fold of the per-axis indices, builds the ``Schedule``, and navigates to block
    ``B`` and its loops. Both the Reorder and Fuse oracles share this construction (and the
    row-major fold defined here) before applying their distinct transform.

    Parameters
    ----------
    extents:
        The source loop extents, outer -> inner (one loop per entry).

    Returns
    -------
    tuple[Schedule, SBlockRV, list[LoopRV]]
        The schedule, the realized block ``B``, and its loops in outer -> inner order.
    """
    total = reduce(mul, extents, 1)
    a = placeholder((total,), name="A", dtype=_F32)

    def _body(*idx: TX.Var) -> TX.PrimExpr:
        """Row-major affine fold of the per-axis indices into a flat offset."""
        flat: TX.PrimExpr = idx[0]
        for extent, index in zip(extents[1:], idx[1:]):
            flat = flat * extent + index
        return a[flat] * 2.0

    b = compute(tuple(extents), _body, name="B")
    func = create_prim_func([a, b])

    sch = Schedule(func)
    block = sch.get_sblock("B")
    loops = list(sch.get_loops(block))
    return sch, block, loops


def tvm_split_loopnest(extent: int, factors: list[int]) -> LoopNest:
    """Run TVM's own ``schedule.split`` on a single ``extent``-loop and read back the nest.

    Builds a minimal ``B[i] = A[i] * 2.0`` compute over one loop of size ``extent``,
    splits that loop by ``factors``, then extracts the resulting nested loops' extents
    (outer -> inner) and the original iter var's recovered binding (normalized to a
    deterministic canonical string).

    Parameters
    ----------
    extent:
        Size of the single source loop.
    factors:
        The split factors; their product must equal ``extent`` (at most one ``None`` is
        TVM-inferred, but this harness takes concrete ints).

    Returns
    -------
    LoopNest
        ``extents`` outer->inner and the canonical ``binding`` string (e.g. ``"i0*4 + i1"``).
    """
    a = placeholder((extent,), name="A", dtype=_F32)
    b = compute((extent,), lambda i: a[i] * 2.0, name="B")
    func = create_prim_func([a, b])

    sch = Schedule(func)
    block = sch.get_sblock("B")
    (loop,) = sch.get_loops(block)
    new_loops = sch.split(loop, factors=list(factors))

    fors = [sch.get(lp) for lp in new_loops]
    extents = [int(f.extent) for f in fors]
    loop_vars = [f.loop_var for f in fors]

    realize = _find_block_realize(sch.mod["main"].body, "B")
    binding = _canonical_binding(realize.iter_values[0], loop_vars)

    return LoopNest(extents=extents, binding=binding)


def tvm_reorder_loopnest(extents: list[int], order: list[int]) -> LoopNest:
    """Run TVM's own ``schedule.reorder`` on a perfect nest and read back the new order.

    Builds a perfect ``len(extents)``-deep nest over a single flattened buffer
    (``B[i0,...,i_{n-1}] = A[flat] * 2.0`` with ``flat`` the row-major affine fold),
    permutes the loops so the new outer->inner order is ``[loops[order[0]], ...]``,
    then reads back the resulting loop extents (outer->inner) via ``get_loops``.

    ``order`` is a permutation of ``range(len(extents))`` naming, per new position,
    the index of the *original* loop that lands there. E.g. ``order=[1, 0]`` swaps a
    2-deep nest; ``order=[0, 2, 1]`` swaps the inner pair of a 3-deep nest.

    Reorder permutes loop *position* only: each block iter var stays bound to its own
    loop var, so the meaningful structural signal is ``extents`` (the new loop order).
    ``binding`` is populated with the post-reorder loop-var order rendered as the
    canonical positional names (outer->inner), i.e. it echoes which original axis sits
    at each depth after the permutation.

    Parameters
    ----------
    extents:
        The source loop extents, outer -> inner (one loop per entry).
    order:
        A permutation of ``range(len(extents))``; ``order[k]`` is the original loop
        index placed at new depth ``k``.

    Returns
    -------
    LoopNest
        ``extents`` in the new outer->inner order and ``binding`` the canonical
        positional loop-var ordering after the permutation.
    """
    if sorted(order) != list(range(len(extents))):
        raise ValueError(f"order {order!r} must be a permutation of range({len(extents)})")

    sch, block, loops = _perfect_nest_schedule(extents)
    sch.reorder(*[loops[i] for i in order])

    new_loops = list(sch.get_loops(block))
    fors = [sch.get(lp) for lp in new_loops]
    new_extents = [int(f.extent) for f in fors]

    new_loop_vars = [f.loop_var for f in fors]
    binding = ", ".join(_canonical_binding(lv, new_loop_vars) for lv in new_loop_vars)

    return LoopNest(extents=new_extents, binding=binding)


def tvm_fuse_loopnest(extents: list[int]) -> FusedNest:
    """Run TVM's own ``schedule.fuse`` on a perfect nest and read back the single fused loop.

    Builds a perfect ``len(extents)``-deep nest over a single flattened buffer
    (``B[i0,...,i_{n-1}] = A[flat] * 2.0`` with ``flat`` the row-major affine fold),
    fuses every loop into one, then reads back the single fused loop's extent (= the
    product of ``extents``) and the recovered binding of each original iter var.

    ``fuse`` is the inverse of ``split``: it collapses the nest into ONE loop while
    keeping the source iters, binding each to a ``floordiv`` / ``floormod`` slice of
    the fused loop var. For ``extents=[4, 4]`` TVM gives the fused extent ``16`` and
    bindings ``[fused // 4, fused % 4]`` -- after renaming the fused var to ``i0`` (the
    same positional canonicalization the Split/Reorder oracles use), ``["i0 // 4",
    "i0 % 4"]``. The recovered ELEMENT index those bindings recombine to is the
    row-major fold ``(i0 // 4) * 4 + (i0 % 4)``, which is the identity ``i0`` over
    ``[0, prod)`` -- TVM's ``IterMapSimplifyBlockBinding`` equivalent of our IR
    collapsing a fused-then-split dim back to a single ``Var`` binding.

    Parameters
    ----------
    extents:
        The source loop extents, outer -> inner (one loop per entry); ``len >= 2``.

    Returns
    -------
    FusedNest
        ``extent`` the single fused loop's extent (= ``prod(extents)``) and
        ``bindings`` the per-iter recovered bindings (source outer -> inner) with the
        fused loop var renamed to ``i0``.
    """
    if len(extents) < 2:
        raise ValueError(f"tvm_fuse_loopnest needs at least 2 extents to fuse; got {extents!r}")

    sch, _block, loops = _perfect_nest_schedule(extents)
    fused = sch.fuse(*loops)

    fused_for = sch.get(fused)
    fused_extent = int(fused_for.extent)
    fused_var = [fused_for.loop_var]

    realize = _find_block_realize(sch.mod["main"].body, "B")
    bindings = [_canonical_binding(iv, fused_var) for iv in realize.iter_values]

    return FusedNest(extent=fused_extent, bindings=bindings)
