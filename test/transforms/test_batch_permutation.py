"""Structural and numerical coverage for batched permutation tensorization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir import Const, ISANode, KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    BatchPermutation,
    BufferLayout,
    BufferLayoutOption,
    Split,
    SplitOption,
    TransformLegalityError,
)


@nkigym_kernel
def f_wide_transpose(source):
    """Load, transpose, and store a matrix wider than one partition tile."""
    loaded = NKILoad()(src=source)
    transposed = NKIDMATranspose()(src=loaded)
    output = NKIStore()(src=transposed)
    return output


def _build(free_extent: int) -> KernelIR:
    """Build a wide-transpose fixture."""
    return build_initial_ir(f_wide_transpose, {"source": ((128, free_extent), "bfloat16")})


def _load_source(source: str, path: Path) -> ModuleType:
    """Load rendered NKI from ``path``."""
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_permutation_renders_and_preserves_transpose(tmp_path: Path) -> None:
    """Four adjacent transposes become one equivalent rank-four DMA transpose."""
    ir = _build(512)
    transform = BatchPermutation()
    options = transform.analyze(ir)
    assert len(options) == 1
    assert ir.tree.loop(options[0].loop_nid).extent == 4

    batched = transform.apply(ir, options[0])
    source = render(batched)
    expected = (
        "nisa.dma_transpose("
        "src=loaded[0].ap(pattern=[[512, 128], [1, 1], [128, 4], [1, 128]], offset=0), "
        "dst=transposed[0].ap(pattern=[[512, 128], [1, 1], [128, 4], [1, 128]], offset=0), "
        "axes=(3, 1, 2, 0))"
    )
    assert expected in source
    assert transform.analyze(batched) == []

    module = _load_source(source, tmp_path / "batched_transpose.py")
    rng = np.random.default_rng(0)
    source_array = rng.standard_normal((128, 512), dtype=np.float32)
    actual = np.asarray(simulate_fp32(module.nki_f_wide_transpose)(source=source_array))
    np.testing.assert_allclose(actual, source_array.T, atol=1e-6, rtol=1e-6)


def test_split_then_batch_permutation_preserves_outer_groups() -> None:
    """Split chooses batch size while BatchPermutation absorbs only the inner loop."""
    ir = _build(1024)
    direct = BatchPermutation().analyze(ir)
    assert len(direct) == 1
    split = Split().apply(ir, SplitOption(target_nid=direct[0].loop_nid, factors=(2, 4), target_axis=None))
    inner = BatchPermutation().analyze(split)
    assert len(inner) == 1
    assert split.tree.loop(inner[0].loop_nid).extent == 4

    batched = BatchPermutation().apply(split, inner[0])
    source = render(batched)
    assert "for i_d1_0 in range(2):" in source
    assert "pattern=[[1024, 128], [1, 1], [128, 4], [1, 128]]" in source
    assert "offset=i_d1_0 * 512" in source


def test_discontiguous_list_layout_rejects_batching() -> None:
    """A Python list of separate destination allocations cannot form one view."""
    ir = _build(512)
    option = BatchPermutation().analyze(ir)[0]
    listed = BufferLayout().apply(ir, BufferLayoutOption(tensor="transposed", list_len=4))
    assert BatchPermutation().analyze(listed) == []
    with pytest.raises(TransformLegalityError, match="not an eligible permutation batch"):
        BatchPermutation().apply(listed, option)


def test_batch_permutation_accepts_pipeline_versioned_contiguous_output() -> None:
    """Batching preserves the physical stride of every contiguous pipeline slot."""
    ir = _build(512)
    object.__setattr__(ir.buffer("transposed"), "versions", 2)
    options = BatchPermutation().analyze(ir)
    assert len(options) == 1

    batched = BatchPermutation().apply(ir, options[0])
    leaf = next(
        node
        for nid in batched.tree.preorder()
        if isinstance((node := batched.tree.data(nid)), ISANode) and node.op_cls.NAME == "dma_transpose"
    )
    destination = leaf.access_patterns["dst"]
    assert destination.pattern[0] == (Const(value=1024), Const(value=128))
