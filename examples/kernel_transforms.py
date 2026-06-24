"""Canonical matmul -> the hand ``kernel_target``, by DRIVING the shipped transforms.

d0 = K (reduction), d1 = M, d2 = N. From the canonical SSA matmul ``f_nkigym``
(== kernel_0), ``_build_ladder`` drives ONE shipped-transform atom per rung to a
``kernel_target``-equivalent — tiling AND the two-stage psum->sbuf fold in a
SINGLE IR. Run with ``--cache <dir>``: ``_main`` renders every rung, CPU-sims it
against ``lhs_T.T @ rhs``, and compiles + profiles each on real Trn2 hardware.

THE SINGLE LADDER (``_build_ladder``, k0..k15), all CPU-sim clean on gym-1:
  k1      Split        K-trip 16 -> ko(2), ki(8)
  k2      RFactor(ko)  one-stage -> two-stage accumulation: per-ko PSUM partial,
                       ko folded in SBUF via ``tensor_tensor`` (the fused
                       single-accumulator form; psum stays per-output-tile, so a
                       later Split/Reorder of M never corrupts it)
  k3-k5   Reorder x3   bubble N(i_d2_0) above M, sink ki(i_d0_1) innermost
                       -> matmul nest ``ko > N > M > ki``
  k6-k8   Split x3     tensorize d2 (2048 -> 4x512) of the per-ko memset / drain
                       ``tensor_copy`` / ``tensor_tensor`` fold
  k9-k11  Reorder x3   each of those blocks to ``[N, M]`` (the matmul's tile-prefix
                       order, so the next moves are same-prefix legal)
  k12-k13 ReverseComputeAt  sink memset(psum) + drain ``tensor_copy`` under the
                       matmul's ``i_d1_0`` -> memset/matmul/copy CO-LOCATED per
                       (N, M) output tile. ``compact_shapes`` then shrinks
                       ``psum_prod`` (128,16,2048) -> **(128, 1, 512)** — the
                       per-tile PSUM the HW needs.
  k14-k15 ComputeAt    sink the rhs / lhs_T loads to per-(N) / per-(N,M) scope
                       (stream operands instead of one 32 MiB up-front load).
k15 is the HW-runnable ``kernel_target`` equivalent: ``ko > N > M`` with a per-tile
``memset -> ki-matmul -> tensor_copy``, a per-(N,M) ``tensor_tensor`` ko-fold into
``sbuf_prod``, and streamed loads. ``kernel_target`` (the hand 90.7%-MFU goal) is
profiled beside it as the reference.

MEASURED (gym-1): all k0..k15 + kernel_target CPU-sim PASS (~1.4e-4). On HW,
k0..k13 (k14 too: it still holds a wide load buffer) hold a full-extent buffer
neuronx-cc's BIR verifier rejects (exit 70) for most rungs; the runnable rungs are
**k13 37.7% -> k15 46.2% MFU** (both load sinks); ``kernel_target`` 90.7%. The
~45pp gap k15 -> kernel_target is PERF, not correctness: kernel_target inlines the
``tensor_tensor`` fold into the matmul's innermost loop (vs k15's separate ``[N,M]``
fold sweep, which serializes the two accumulation stages and keeps ``sbuf_rfactor``
full-extent) and reloads lhs_T as ``(128,1,512)`` slabs (vs k15's full-width
reload). Inlining the fold is blocked by the reduction-axis-coverage guard (the
fold carries ``ko`` as ACCUMULATION) — follow-on work.
"""

import argparse
import ast
import importlib.util
import os
import shutil
import sys
import tempfile

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "nkigym", "src"), os.path.join(_REPO_ROOT, "autotune", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autotune.runner.api import profile
from autotune.runner.types import KernelJob
from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    ComputeAt,
    ComputeAtOption,
    Reorder,
    ReorderOption,
    ReverseComputeAt,
    ReverseComputeAtOption,
    RFactor,
    RFactorOption,
    Split,
    SplitOption,
)

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
SEED = 0
NEURON_PLATFORM_TARGET = "trn2"
"""Hand-placed PSUM: the neuronx-cc scheduler + linear-scan allocator must be OFF
(scheduler-on OOMs PSUM even with per-iteration alloc)."""
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — the canonical matmul (== kernel_0)."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


@nki.jit
def kernel_98(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(psum_prod[i_d1_0][0:128, 0, 0:512], 0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                    rhs[
                        i_d0_0 * 1024 + i_d0_1 * 128 : i_d0_0 * 1024 + (i_d0_1 + 1) * 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                )
            for i_d1_0 in range(4):
                sbuf_lhs_T = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        sbuf_lhs_T[i_d0_1][0:128, 0, 0:512],
                        lhs_T[
                            i_d0_0 * 1024 + i_d0_1 * 128 : i_d0_0 * 1024 + (i_d0_1 + 1) * 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                    )
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : (i_d1_1 + 1) * 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(sbuf_prod[i_d1_0][0:128, 0, 0:512], psum_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                hbm_out[i_d1_0 * 128 : (i_d1_0 + 1) * 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                sbuf_prod[i_d1_0][0:128, 0, 0:512],
            )

    return hbm_out


@nki.jit
def kernel_target(lhs_T, rhs):
    """The HAND-written 90.6%-MFU goal = SHAPE ladder (k15) + ACCUMULATION ladder
    (accum_2) combined, which the transforms cannot yet do in one IR.

    The shape ladder (k15) gives the tiling + per-N-block drain/store; the accum
    ladder (accum_2) gives the two-stage psum->sbuf ``tensor_tensor`` fold. This
    kernel is both at once, plus per-output-tile PSUM and flat Python lists of
    separate ``(128, 1, 512)`` tiles. See the module docstring for the merge blocker.
    """
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(sbuf_prod[i_d1_0][0:128, 0, 0:512], 0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                    rhs[
                        i_d0_0 * 1024 + i_d0_1 * 128 : i_d0_0 * 1024 + (i_d0_1 + 1) * 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                )
            for i_d1_0 in range(4):
                sbuf_lhs_T = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        sbuf_lhs_T[i_d0_1][0:128, 0, 0:512],
                        lhs_T[
                            i_d0_0 * 1024 + i_d0_1 * 128 : i_d0_0 * 1024 + (i_d0_1 + 1) * 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                    )
                for i_d1_1 in range(4):
                    psum_prod = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_prod[0:128, 0, 0:512], 0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            dst=psum_prod[0:128, 0, 0:512],
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : (i_d1_1 + 1) * 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                        )
                    sbuf_rfactor = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(sbuf_rfactor[0:128, 0, 0:512], psum_prod[0:128, 0, 0:512])
                    nisa.tensor_tensor(
                        sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        sbuf_rfactor[0:128, 0, 0:512],
                        op=nl.add,
                    )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                hbm_out[i_d1_0 * 128 : (i_d1_0 + 1) * 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                sbuf_prod[i_d1_0][0:128, 0, 0:512],
            )

    return hbm_out


def _mm_leaf(ir: object) -> int:
    """Node id of the matmul ISA leaf."""
    return next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )


def _loop(ir: object, loop_var: str) -> int:
    """Node id of the matmul-enclosing ForNode whose ``loop_var`` matches."""
    return next(
        a
        for a in ir.tree.ancestors(_mm_leaf(ir))
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == loop_var
    )


def _load_blk(ir: object, tensor: str) -> int:
    """Node id of the single-leaf load block whose ``NKILoad`` reads ``tensor``."""

    def _reads(nid: int) -> bool:
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        return (
            len(leaves) == 1
            and ir.tree.data(leaves[0]).op_cls.__name__ == "NKILoad"
            and ir.tree.data(leaves[0]).operand_bindings["src"].tensor == tensor
        )

    return next(nid for nid in ir.tree.blocks() if _reads(nid))


def _load_leaf(ir: object, tensor: str) -> int:
    """Node id of the ``NKILoad`` ISA leaf that reads ``tensor`` (the Split target)."""
    return next(d for d in ir.tree.descendants(_load_blk(ir, tensor)) if isinstance(ir.tree.data(d), ISANode))


def _load_for(ir: object, tensor: str, loop_var: str) -> int:
    """Node id of the ForNode with ``loop_var`` inside the ``tensor`` load block's nest."""
    return next(
        d
        for d in ir.tree.descendants(_load_blk(ir, tensor))
        if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == loop_var
    )


def _op_blk(ir: object, op_name: str) -> int:
    """Node id of the single-leaf block whose ISA leaf is op ``op_name`` (e.g. NKITensorCopy)."""

    def _is(nid: int) -> bool:
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        return len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name

    return next(nid for nid in ir.tree.blocks() if _is(nid))


def _op_leaf(ir: object, op_name: str) -> int:
    """Node id of the ISA leaf for op ``op_name`` (the Split tensorize target)."""
    return next(d for d in ir.tree.descendants(_op_blk(ir, op_name)) if isinstance(ir.tree.data(d), ISANode))


def _psum_memset_leaf(ir: object) -> int:
    """Node id of the memset ISA leaf that writes the PSUM accumulator (not sbuf_prod).

    After RFactor there are two memsets — ``init_two_stage_0`` zeros the SBUF
    accumulator, ``init_two_stage_1`` zeros the per-``ko`` PSUM partial. This finds
    the PSUM one (the Split/ReverseComputeAt target for the per-tile PSUM shrink).
    """
    return next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode)
        and ir.tree.data(n).op_cls.NAME == "memset"
        and ir.tree.data(n).operand_bindings["dst"].tensor.startswith("psum")
    )


def _psum_memset_blk(ir: object) -> int:
    """Node id of the single-leaf block owning the PSUM-zeroing memset leaf."""
    leaf = _psum_memset_leaf(ir)
    return next(
        b
        for b in ir.tree.blocks()
        if b != ir.tree.root
        and leaf in ir.tree.descendants(b)
        and not any(isinstance(ir.tree.data(d), ISANode) for d in ir.tree.descendants(b) if d != leaf)
    )


def _blk_loop(ir: object, blk_nid: int, loop_var: str) -> int:
    """Node id of the ForNode with ``loop_var`` inside block ``blk_nid``'s subtree."""
    return next(
        d
        for d in ir.tree.descendants(blk_nid)
        if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == loop_var
    )


def _reorder_blk_to_nm(ir: object, blk_nid: int) -> object:
    """Reorder a per-``ko`` single-leaf block's ``[M, N]`` loop pair to ``[N, M]``.

    The post-Split memset / drain / fold blocks iterate ``i_d1_0(M) > i_d2_0(N)``;
    the matmul's tile prefix is ``i_d2_0(N) > i_d1_0(M)``. Matching that order makes
    the subsequent same-prefix ``ReverseComputeAt`` under the matmul ``i_d1_0`` legal.
    """
    return Reorder().apply(
        ir, ReorderOption(outer_nid=_blk_loop(ir, blk_nid, "i_d1_0"), inner_nid=_blk_loop(ir, blk_nid, "i_d2_0"))
    )


def _blk_m_loop(ir: object, op_name: str) -> int:
    """Node id of the ``i_d1_0`` (M) ForNode inside op ``op_name``'s single-leaf block."""
    return _blk_loop(ir, _op_blk(ir, op_name), "i_d1_0")


def _build_ladder() -> list[tuple[str, object]]:
    """Drive the shipped transforms from canonical ``f_nkigym`` to a HW-runnable
    ``kernel_target`` equivalent (k0..k19), with the ko-fold INLINED into the
    matmul's innermost M loop. Returns ``[(name, ir), ...]``.

    Every locator is SEMANTIC (matmul loop_var, op-class block, PSUM-writer leaf),
    so it tracks node ids across the structural change each ``apply`` makes — no
    hard-coded nids. The N-OUTERMOST ladder (probed sim-clean on gym-1; the
    fold-inlining the two coverage-guard refinements unblock — see the module
    docstring):
    k1-k2  Split K(->ko,ki) + Split M(->Mo,Mi); k3-k8 Reorder x6 -> nest
    ``N > ko > Mo > Mi > ki``; k9 RFactor(ko) -> two-stage fold; k10-k15 Split each
    of memset/copy/fold on d2(4x512) AND M(4x4) to the matmul's tile prefix;
    k16-k18 ReverseComputeAt memset + drain ``tensor_copy`` + ``tensor_tensor``
    fold under the matmul's ``i_d1_1`` -> all CO-LOCATED per (N, Mo, Mi) tile, fold
    INLINED in the matmul's innermost body. ``compact_shapes`` then shrinks
    ``psum_prod`` and ``sbuf_rfactor`` to per-tile ``(128, 1, 512)``. k19-k20 Split
    the rhs/lhs_T loads on their FREE axis (rhs d2, lhs_T d1, each 4x512) so each
    load is one N-/M-tile wide, NOT the full 2048; k21-k22 ComputeAt the tiled rhs
    load under ``i_d2_0`` (N) and lhs_T under ``i_d1_0`` (Mo) -> ``sbuf_rhs``
    ``(128, 16, 512)`` and ``sbuf_lhs_T`` ``(128, 8, 512)`` streamed per-tile. k22
    is the fold-inlined, tiled-load ``kernel_target`` reproduction.
    """
    steps = [
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_1"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_1"))),
        lambda ir: RFactor().apply(ir, RFactorOption(target_loop_nid=_loop(ir, "i_d0_0"), factor_axis=0)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2")),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_blk_loop(ir, _psum_memset_blk(ir), "i_d1_0"), factors=(4, 4), target_axis=None)
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_blk_m_loop(ir, "NKITensorCopy"), factors=(4, 4), target_axis=None)
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorTensor"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_blk_m_loop(ir, "NKITensorTensor"), factors=(4, 4), target_axis=None)
        ),
        lambda ir: ReverseComputeAt().apply(
            ir, ReverseComputeAtOption(block_nid=_psum_memset_blk(ir), target_loop_nid=_loop(ir, "i_d1_1"), index=0)
        ),
        lambda ir: ReverseComputeAt().apply(
            ir,
            ReverseComputeAtOption(
                block_nid=_op_blk(ir, "NKITensorCopy"), target_loop_nid=_loop(ir, "i_d1_1"), index=-1
            ),
        ),
        lambda ir: ReverseComputeAt().apply(
            ir,
            ReverseComputeAtOption(
                block_nid=_op_blk(ir, "NKITensorTensor"), target_loop_nid=_loop(ir, "i_d1_1"), index=-1
            ),
        ),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_leaf(ir, "rhs"), factors=(4, 512), target_axis="d2")),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_0"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: ComputeAt().apply(
            ir, ComputeAtOption(block_nid=_load_blk(ir, "rhs"), target_loop_nid=_loop(ir, "i_d2_0"), index=0)
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_leaf(ir, "lhs_T"), factors=(4, 512), target_axis="d1")
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "lhs_T", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "lhs_T", "i_d0_1"), inner_nid=_load_for(ir, "lhs_T", "i_d1_0"))
        ),
        lambda ir: ComputeAt().apply(
            ir, ComputeAtOption(block_nid=_load_blk(ir, "lhs_T"), target_loop_nid=_loop(ir, "i_d1_0"), index=0)
        ),
    ]
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    ladder = [("kernel_0", ir)]
    for step in steps:
        ir = step(ir)
        ladder.append((f"kernel_{len(ladder)}", ir))
    return ladder


def _func_name(source: str) -> str:
    """The top-level function name defined in a standalone kernel module string."""
    return next(ln[len("def ") :].split("(", 1)[0] for ln in source.splitlines() if ln.startswith("def "))


def _kernel_source(name: str) -> str:
    """Standalone NKI module string for the hand kernel ``name`` (AST-extracted).

    ``inspect.getsource`` does not work through the ``@nki.jit`` wrapper, so the
    single ``def {name}`` is pulled from THIS file by AST and the three ``nki``
    imports prepended, giving ``profile`` a module it can compile in isolation.
    """
    this_src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(this_src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = ast.get_source_segment(this_src, fn)
    if body is None:
        raise RuntimeError(f"could not extract source for {name}")
    return "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n\n" + body + "\n"


def _sim_source(name: str, source: str, func_name: str, inputs: dict, expected: np.ndarray) -> None:
    """Write ``source``, import it, CPU-sim ``func_name`` against the golden; print PASS/FAIL."""
    path = os.path.join(tempfile.gettempdir(), f"kt_sim_{name}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f"kt_sim_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(getattr(module, func_name))(**inputs))
    ok = np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
    print(f"[sim] {name}: max_abs={float(np.abs(actual - expected).max()):.3e} pass={ok}")


def _main() -> None:
    """Build the ladder, render every rung, sim-check it, then profile all on Trn2."""
    parser = argparse.ArgumentParser(description="Drive canonical->target transforms; CPU-sim + Trn2 HW profile.")
    parser.add_argument("--cache", required=True, help="absolute cache dir (under the box's $HOME on Kaizen)")
    args = parser.parse_args()
    cache_dir = args.cache
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    inputs = {nm: rng.standard_normal(shape).astype(np.float32) for nm, (shape, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]

    """Each entry: (name, standalone-module source). The single merged ladder
    (k0..k13) comes from rendering the transform-driven IR states; kernel_target is
    the AST-extracted hand kernel, profiled beside k13 as the perf reference."""
    sources = [(name, render(ir)) for name, ir in _build_ladder()]
    sources.append(("kernel_target", _kernel_source("kernel_target")))

    """Save every rendered kernel as <cache>/<name>.py for inspection."""
    for name, source in sources:
        with open(os.path.join(cache_dir, f"{name}.py"), "w", encoding="utf-8") as handle:
            handle.write(source)
    print(f"[save] wrote {len(sources)} kernels to {cache_dir}/<name>.py")

    jobs: dict[str, KernelJob] = {}
    for name, source in sources:
        func_name = _func_name(source)
        _sim_source(name, source, func_name, inputs, expected)
        jobs[name] = KernelJob(
            source=source,
            func_name=func_name,
            output_shape=(M, N),
            input_specs=INPUT_SPECS,
            neuronx_cc_args=SCHEDULER_OFF_ARGS,
        )

    print(
        f"\n[hw] compiling + profiling {len(jobs)} kernels on {NEURON_PLATFORM_TARGET} (scheduler + linear-scan OFF)\n"
    )
    output = profile(
        jobs,
        cache_dir=cache_dir,
        seed=SEED,
        neuron_platform_target=NEURON_PLATFORM_TARGET,
        collect_detailed_profile=False,
    )
    print(output)


if __name__ == "__main__":
    _main()
