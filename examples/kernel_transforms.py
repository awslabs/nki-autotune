"""Reproduce the ``manual_transforms.py`` ladder by DRIVING the shipped transforms.

GOAL: drive the shipped transforms (Split / Reorder / CodeMotion /
BufferCompaction / BufferLayout) from the canonical SSA matmul ``f_nkigym`` (==
``manual_transforms.py`` kernel_0), ONE transform per rung, so the machine-driven
ladder reproduces the hand ladder RUNG-FOR-RUNG, BYTE-EXACT. ``_main`` renders every
rung, asserts it AST-matches the corresponding ``manual_transforms.kernel_i``
(``assert_matches_hand``), CPU-sims it against ``lhs_T.T @ rhs``, and compiles +
profiles each on real Trn2 hardware.

d0 = K (reduction), d1 = M, d2 = N. ``_build_ladder`` is 32 transforms
(k0..k32):
  k1-k2   Reorder x2   bubble N(i_d2_0) outermost (each an adjacent-swap; the manual
                       "# Reorder" from K>M>N to N>K>M is two atomic swaps -> k1, k2)
  k3-k4   Split x2     K -> ko(2),ki(8); M -> Mo(4),Mi(4)
  k5-k6   Reorder x2   -> matmul nest ``N > ko > Mo > Mi > ki``
  k7      BufferLayout psum_prod -> list-of-16
  k8-k10  drain tensor_copy: Split d2, Reorder to N-outer, CodeMotion sink under N
  k11-k13 store: Split d2, Reorder, structural CodeMotion sink under N
  k14-k15 BufferCompaction sbuf_prod, then BufferLayout -> list-of-16
  k16-k18 psum memset: Split d2, Reorder, structural CodeMotion sink under N
  k19     BufferCompaction psum_prod
  k20-k24 rhs load: Split d0(2,8), Split d2, Reorder x2, structural CodeMotion
  k25-k26 BufferCompaction sbuf_rhs, then BufferLayout -> list-of-8
  k27-k30 lhs_T load: Split d1, Split d0(2,8), Reorder, structural CodeMotion
  k31-k32 BufferCompaction sbuf_lhs_T, then BufferLayout -> list-of-8

RFactor is the manual k33 rung and is intentionally deferred.

Every locator is SEMANTIC
(matmul loop_var, op-class block, load/PSUM leaf), tracking node ids across each
transform's structural change.
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

from test.transforms._ladder_compare import assert_matches_hand

from autotune.runner.types import KernelJob
from examples import manual_transforms
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
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    Reorder,
    ReorderOption,
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
    the PSUM one (the Split/CodeMotion target for the per-tile PSUM shrink).
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
    the subsequent same-prefix ``CodeMotion`` under the matmul ``i_d1_0`` legal.
    """
    return Reorder().apply(
        ir, ReorderOption(outer_nid=_blk_loop(ir, blk_nid, "i_d1_0"), inner_nid=_blk_loop(ir, blk_nid, "i_d2_0"))
    )


def _blk_m_loop(ir: object, op_name: str) -> int:
    """Node id of the ``i_d1_0`` (M) ForNode inside op ``op_name``'s single-leaf block."""
    return _blk_loop(ir, _op_blk(ir, op_name), "i_d1_0")


def _build_ladder() -> list[tuple[str, object]]:
    """Drive the shipped transforms from canonical ``f_nkigym`` to ``manual_transforms``
    k0..k32, ONE transform per rung, in MANUAL rung order. Returns 33 named states.

    Every locator is SEMANTIC (matmul loop_var, op-class block, load/PSUM leaf), so it
    tracks node ids across the structural change each ``apply`` makes. CodeMotion is
    structural-only; k14/k19/k25/k31 explicitly compact the moved buffer, and the
    following BufferLayout rungs re-factorize it into list form. Manual k33's RFactor
    is out of scope.

    k1-k6   Reorder x2 (N-outer, two atomic swaps) + Split K + Split M + Reorder x2
            -> matmul nest ``N > ko > Mo > Mi > ki`` (manual k6's packed nest)
    k7      BufferLayout psum_prod -> list-of-16
    k8-k10  drain tensor_copy: Split d2, Reorder to N-outer, CodeMotion sink under N
    k11-k13 store: Split d2, Reorder to N-outer, CodeMotion sink under N
    k14-k15 BufferCompaction + BufferLayout sbuf_prod
    k16-k19 psum memset: Split, Reorder, CodeMotion, BufferCompaction
    k20-k26 rhs load: Split x2, Reorder x2, CodeMotion, BufferCompaction, BufferLayout
    k27-k32 lhs_T load: Split x2, Reorder, CodeMotion, BufferCompaction, BufferLayout
    """
    steps = [
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_1"))),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16)),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: _reorder_blk_to_nm(ir, _op_blk(ir, "NKITensorCopy")),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_op_blk(ir, "NKITensorCopy"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_op_leaf(ir, "NKIStore"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: _reorder_blk_to_nm(ir, _op_blk(ir, "NKIStore")),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_op_blk(ir, "NKIStore"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_prod", list_len=16)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2")),
        lambda ir: _reorder_blk_to_nm(ir, _psum_memset_blk(ir)),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_psum_memset_blk(ir), target_loop_nid=_loop(ir, "i_d2_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_prod")),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "rhs", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_leaf(ir, "rhs"), factors=(4, 512), target_axis="d2")),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_1"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_0"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_load_blk(ir, "rhs"), target_loop_nid=_loop(ir, "i_d0_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_rhs")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_leaf(ir, "lhs_T"), factors=(4, 512), target_axis="d1")
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "lhs_T", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "lhs_T", "i_d0_1"), inner_nid=_load_for(ir, "lhs_T", "i_d1_0"))
        ),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_load_blk(ir, "lhs_T"), target_loop_nid=_loop(ir, "i_d1_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_lhs_T")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_lhs_T", list_len=8)),
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
    from autotune.runner.api import profile

    parser = argparse.ArgumentParser(description="Drive canonical->target transforms; CPU-sim + Trn2 HW profile.")
    parser.add_argument("--cache", required=True, help="absolute cache dir (under the box's $HOME on Kaizen)")
    args = parser.parse_args()
    cache_dir = args.cache
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    inputs = {nm: rng.standard_normal(shape).astype(np.float32) for nm, (shape, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]

    """Each entry: (name, standalone-module source). The driven ladder (k0..k32)
    comes from rendering the transform-driven IR states; kernel_target is the
    AST-extracted hand kernel, profiled beside the endpoint as the perf reference."""
    ladder = _build_ladder()

    """Byte-exact gate: each driven rung must match manual_transforms.kernel_i
    AST-canonically. A mismatch prints the got-vs-want diff and aborts."""
    for name, ir in ladder:
        manual_fn = getattr(manual_transforms, name)
        assert_matches_hand(render(ir), manual_fn)
        print(f"[byte-exact] {name}: OK")

    sources = [(name, render(ir)) for name, ir in ladder]
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
