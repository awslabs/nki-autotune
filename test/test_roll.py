"""Tests for loop rolling: nl.affine_range detection and rendering."""

from golden.nki_codegen import MATMUL_TANH_KERNEL, MATMUL_TANH_RENDERED, NORMALIZE_AFTER

from nkigym.codegen.types import (
    _NO_ARITH,
    NKIBlock,
    NKIKernel,
    _ArithProg,
    _cluster_by_pattern,
    _collect_varying,
    _extract_skeleton,
    _group_blocks,
    _locally_normalize,
    _verify_arithmetic,
)
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.tensor_copy import NKITensorCopy


class TestLocallyNormalize:
    """Per-block tensor renumbering."""

    def test_already_canonical(self) -> None:
        """Block 0 (tensor_0..tensor_6) is unchanged by normalization."""
        body = MATMUL_TANH_KERNEL.blocks[0].body
        assert _locally_normalize(body) == body

    def test_renumbers_to_match(self) -> None:
        """Block 0 and block 1 normalize to the same local names."""
        norm_0 = _locally_normalize(MATMUL_TANH_KERNEL.blocks[0].body)
        norm_1 = _locally_normalize(MATMUL_TANH_KERNEL.blocks[1].body)
        skel_0 = _extract_skeleton(norm_0, {"a", "b", "output"})
        skel_1 = _extract_skeleton(norm_1, {"a", "b", "output"})
        assert skel_0 == skel_1


class TestSkeletonEquality:
    """Skeleton extraction replaces HBM slices with sentinels."""

    def test_same_skeleton_different_hbm_slices(self) -> None:
        """All 4 blocks produce identical skeletons after normalization."""
        skeletons = []
        for block in MATMUL_TANH_KERNEL.blocks:
            norm = _locally_normalize(block.body)
            skeletons.append(_extract_skeleton(norm, {"a", "b", "output"}))
        assert all(s == skeletons[0] for s in skeletons[1:])


class TestGroupBlocks:
    """Grouping blocks by skeleton."""

    def test_four_blocks_one_group(self) -> None:
        """4-block matmul+tanh kernel forms a single group of 4."""
        normalized = [_locally_normalize(b.body) for b in MATMUL_TANH_KERNEL.blocks]
        groups = _group_blocks(normalized, {"a", "b", "output"})
        assert len(groups) == 1
        indices = list(groups.values())[0]
        assert indices == [0, 1, 2, 3]


class TestAffineDetection:
    """Arithmetic progression detection from value patterns."""

    def test_simple_progression(self) -> None:
        """Values (0, 0, 128, 128) for outer loop → size=2, stride=128."""
        normalized = [_locally_normalize(b.body) for b in MATMUL_TANH_KERNEL.blocks]
        varying = _collect_varying(normalized, {"a", "b", "output"})
        clusters = _cluster_by_pattern(varying, normalized)
        outer_positions, outer_pattern = clusters[0]
        prog = _verify_arithmetic(outer_pattern)
        assert prog == _ArithProg(size=2, stride=128, base=0)
        assert len(outer_positions) == 3

    def test_inner_loop(self) -> None:
        """Values (0, 128, 0, 128) for inner loop → size=2, stride=128."""
        normalized = [_locally_normalize(b.body) for b in MATMUL_TANH_KERNEL.blocks]
        varying = _collect_varying(normalized, {"a", "b", "output"})
        clusters = _cluster_by_pattern(varying, normalized)
        inner_positions, inner_pattern = clusters[1]
        prog = _verify_arithmetic(inner_pattern)
        assert prog == _ArithProg(size=2, stride=128, base=0)
        assert len(inner_positions) == 3

    def test_non_arithmetic_returns_sentinel(self) -> None:
        """Non-arithmetic values return _NO_ARITH sentinel."""
        assert _verify_arithmetic((0, 50, 128, 300)) == _NO_ARITH

    def test_single_value_returns_sentinel(self) -> None:
        """Single unique value returns _NO_ARITH (nothing to roll)."""
        assert _verify_arithmetic((128, 128, 128)) == _NO_ARITH


class TestRenderRolled:
    """End-to-end rolled rendering."""

    def test_matmul_tanh_rolled(self) -> None:
        """4-block kernel renders with nested nl.affine_range(2) loops."""
        assert MATMUL_TANH_KERNEL.render() == MATMUL_TANH_RENDERED

    def test_has_affine_range(self) -> None:
        """Rolled output contains affine_range loop headers."""
        rendered = MATMUL_TANH_KERNEL.render()
        assert "for i_p0 in nl.affine_range(2):" in rendered
        assert "for i_p1 in nl.affine_range(2):" in rendered

    def test_has_affine_expressions(self) -> None:
        """Rolled output contains affine slice expressions."""
        rendered = MATMUL_TANH_KERNEL.render()
        assert "i_p0 * 128:i_p0 * 128 + 128" in rendered
        assert "i_p1 * 128:i_p1 * 128 + 128" in rendered

    def test_no_block_functions(self) -> None:
        """Rolled output has _rolled_block_0_1_2_3 helper, no _block_N helpers."""
        rendered = MATMUL_TANH_KERNEL.render()
        assert "def _block_" not in rendered
        assert "def _rolled_block_0_1_2_3(" in rendered


class TestRenderSingleBlock:
    """Single-block kernels render as before (no rolling)."""

    def test_singleton_has_helper(self) -> None:
        """Single-block kernel renders with _block_0 helper function."""
        rendered = NORMALIZE_AFTER.render()
        assert "def _block_0(" in rendered
        assert "_block_0(a, output)" in rendered
        assert "nl.affine_range" not in rendered


MIXED_KERNEL = NKIKernel(
    name="mixed",
    params=("a",),
    input_shapes=((256, 128),),
    dtype="nl.float16",
    output_shape=(256, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (256, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (256, 128), ((128, 256), (0, 128))),
                    src=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_2",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKITensorCopy(
                    dst=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (256, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)


class TestRenderMixed:
    """Kernel with some rollable and some non-rollable blocks."""

    def test_has_loop_and_helper(self) -> None:
        """Rolled blocks produce loop with helper, singleton has separate helper."""
        rendered = MIXED_KERNEL.render()
        assert "nl.affine_range(2)" in rendered
        assert "def _rolled_block_0_1(" in rendered
        assert "_block_2(a, output)" in rendered
        assert "def _block_2(a, output):" in rendered

    def test_rolled_blocks_not_as_helpers(self) -> None:
        """Blocks 0 and 1 are rolled into _rolled_block_0, not separate helpers."""
        rendered = MIXED_KERNEL.render()
        assert "def _block_0(" not in rendered
        assert "def _block_1(" not in rendered

    def test_affine_expressions_in_mixed(self) -> None:
        """Loop body uses affine expressions for varying a/output slices."""
        rendered = MIXED_KERNEL.render()
        assert "i_p0 * 128:i_p0 * 128 + 128" in rendered
