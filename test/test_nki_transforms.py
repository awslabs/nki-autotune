"""Tests for NKIKernel transforms: data reuse and operand merge."""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.transforms.base import TransformOption
from nkigym.transforms.data_reuse import DataReuseTransform
from nkigym.transforms.operand_merge import OperandMergeTransform

_T = (128, 128)
_S = ((0, 128), (0, 128))


def _r(name: str, shape: tuple[int, ...], slices: tuple[tuple[int, int], ...]) -> TensorRef:
    """Create a TensorRef shorthand."""
    return TensorRef(name, shape, slices)


_DATA_REUSE_KERNEL = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=(_T, _T),
    dtype="nl.float16",
    output_shape=_T,
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_0", _T, _S), src=_r("a", _T, _S)),
                NKIAlloc(dst="tensor_1", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_1", _T, _S), src=_r("b", _T, _S)),
                NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_0", _T, _S)),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_2", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_2", _T, _S), src=_r("a", _T, _S)),
                NKIAlloc(dst="tensor_3", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_3", _T, _S), src=_r("b", _T, _S)),
                NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_2", _T, _S)),
            ),
        ),
    ),
)


_ALLOC_WIDEN_KERNEL = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 256),),
    dtype="nl.float16",
    output_shape=_T,
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_0", _T, _S), src=_r("a", _T, _S)),
                NKIAlloc(dst="tensor_1", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_1", _T, _S), src=_r("a", _T, ((0, 128), (128, 256)))),
                NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_0", _T, _S)),
            ),
        ),
    ),
)


_N_AXIS_MATMUL_KERNEL = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=(_T, (128, 256)),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_1", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_1", _T, _S), src=_r("a", _T, _S)),
                NKIAlloc(dst="tensor_2", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=_r("tensor_2", (128, 256), ((0, 128), (0, 256))), src=_r("b", (128, 256), ((0, 128), (0, 256)))
                ),
                NKIMatmul(
                    dst=_r("tensor_0", _T, _S),
                    stationary=_r("tensor_1", _T, _S),
                    moving=_r("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_3", shape=_T, dtype="nl.float32", buffer="psum"),
                NKIMatmul(
                    dst=_r("tensor_3", _T, _S),
                    stationary=_r("tensor_1", _T, _S),
                    moving=_r("tensor_2", (128, 128), ((0, 128), (128, 256))),
                ),
            ),
        ),
    ),
)


class TestDataReuse:
    """Tests for DataReuseTransform."""

    def test_analyze_finds_duplicate_loads(self) -> None:
        """Duplicate a[0:128,0:128] loads across blocks are detected."""
        transform = DataReuseTransform()
        options = transform.analyze(_DATA_REUSE_KERNEL)
        a_options = [o for o in options if _is_dma_to_hbm(o, _DATA_REUSE_KERNEL, "a")]
        assert len(a_options) >= 1

    def test_analyze_no_false_positives(self) -> None:
        """Single-load kernel produces no options."""
        kernel = NKIKernel(
            name="test",
            params=("a",),
            input_shapes=(_T,),
            dtype="nl.float16",
            output_shape=_T,
            blocks=(
                NKIBlock(
                    name="_block_0",
                    params=("a", "output"),
                    body=(
                        NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float16", buffer="sbuf"),
                        NKIDmaCopy(dst=_r("tensor_0", _T, _S), src=_r("a", _T, _S)),
                        NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_0", _T, _S)),
                    ),
                ),
            ),
        )
        transform = DataReuseTransform()
        assert transform.analyze(kernel) == []

    def test_apply_removes_duplicate(self) -> None:
        """Applying reuse removes one DMA load from the merged block."""
        transform = DataReuseTransform()
        options = transform.analyze(_DATA_REUSE_KERNEL)
        result = transform.apply(_DATA_REUSE_KERNEL, options[0])
        total_dma = sum(
            1
            for b in result.blocks
            for s in b.body
            if isinstance(s, NKIDmaCopy) and s.src.name == "a" and s.src.slices == _S
        )
        assert total_dma < 2

    def test_apply_preserves_output_store(self) -> None:
        """Output DMA store is never removed by data reuse."""
        transform = DataReuseTransform()
        options = transform.analyze(_DATA_REUSE_KERNEL)
        result = transform.apply(_DATA_REUSE_KERNEL, options[0])
        output_stores = [
            s for b in result.blocks for s in b.body if isinstance(s, NKIDmaCopy) and s.dst.name == "output"
        ]
        assert len(output_stores) >= 1


class TestAllocWiden:
    """Tests for OperandMergeTransform alloc widen."""

    def test_analyze_finds_adjacent_allocs(self) -> None:
        """Two allocs with adjacent DMA sources are detected."""
        transform = OperandMergeTransform()
        options = transform.analyze(_ALLOC_WIDEN_KERNEL)
        alloc_options = [
            o for o in options if isinstance(_ALLOC_WIDEN_KERNEL.blocks[0].body[o.ref_a.stmt_idx], NKIAlloc)
        ]
        assert len(alloc_options) >= 1

    def test_apply_widens_alloc_shape(self) -> None:
        """After alloc widen, kept alloc has widened shape."""
        transform = OperandMergeTransform()
        options = transform.analyze(_ALLOC_WIDEN_KERNEL)
        alloc_options = [
            o for o in options if isinstance(_ALLOC_WIDEN_KERNEL.blocks[0].body[o.ref_a.stmt_idx], NKIAlloc)
        ]
        result = transform.apply(_ALLOC_WIDEN_KERNEL, alloc_options[0])
        allocs = [s for s in result.blocks[0].body if isinstance(s, NKIAlloc)]
        widened = [a for a in allocs if a.shape[1] == 256]
        assert len(widened) == 1

    def test_apply_removes_absorbed_alloc(self) -> None:
        """After alloc widen, one fewer alloc exists."""
        transform = OperandMergeTransform()
        options = transform.analyze(_ALLOC_WIDEN_KERNEL)
        alloc_options = [
            o for o in options if isinstance(_ALLOC_WIDEN_KERNEL.blocks[0].body[o.ref_a.stmt_idx], NKIAlloc)
        ]
        result = transform.apply(_ALLOC_WIDEN_KERNEL, alloc_options[0])
        before_allocs = sum(1 for s in _ALLOC_WIDEN_KERNEL.blocks[0].body if isinstance(s, NKIAlloc))
        after_allocs = sum(1 for s in result.blocks[0].body if isinstance(s, NKIAlloc))
        assert after_allocs == before_allocs - 1


class TestComputeMerge:
    """Tests for OperandMergeTransform compute merge."""

    def test_analyze_finds_n_axis_merge(self) -> None:
        """Two matmuls with adjacent moving on N axis are detected."""
        transform = OperandMergeTransform()
        options = transform.analyze(_N_AXIS_MATMUL_KERNEL)
        compute_options = [
            o for o in options if isinstance(_N_AXIS_MATMUL_KERNEL.blocks[0].body[o.ref_a.stmt_idx], NKIMatmul)
        ]
        assert len(compute_options) >= 1

    def test_apply_merges_matmuls(self) -> None:
        """After compute merge, one fewer matmul exists."""
        transform = OperandMergeTransform()
        options = transform.analyze(_N_AXIS_MATMUL_KERNEL)
        compute_options = [
            o for o in options if isinstance(_N_AXIS_MATMUL_KERNEL.blocks[0].body[o.ref_a.stmt_idx], NKIMatmul)
        ]
        result = transform.apply(_N_AXIS_MATMUL_KERNEL, compute_options[0])
        before_mats = sum(1 for s in _N_AXIS_MATMUL_KERNEL.blocks[0].body if isinstance(s, NKIMatmul))
        after_mats = sum(1 for s in result.blocks[0].body if isinstance(s, NKIMatmul))
        assert after_mats == before_mats - 1

    def test_merged_matmul_has_wider_moving(self) -> None:
        """Merged matmul's moving operand spans full N range."""
        transform = OperandMergeTransform()
        options = transform.analyze(_N_AXIS_MATMUL_KERNEL)
        compute_options = [
            o for o in options if isinstance(_N_AXIS_MATMUL_KERNEL.blocks[0].body[o.ref_a.stmt_idx], NKIMatmul)
        ]
        result = transform.apply(_N_AXIS_MATMUL_KERNEL, compute_options[0])
        matmuls = [s for s in result.blocks[0].body if isinstance(s, NKIMatmul)]
        assert len(matmuls) == 1
        assert matmuls[0].moving.slices[1] == (0, 256)

    def test_tile_limit_rejects_m_axis(self) -> None:
        """M-axis merge to 256 exceeds TILE_LIMITS[M]=128, rejected."""
        kernel = NKIKernel(
            name="test",
            params=("a", "b"),
            input_shapes=((128, 256), _T),
            dtype="nl.float16",
            output_shape=(256, 128),
            blocks=(
                NKIBlock(
                    name="_block_0",
                    params=("a", "b", "output"),
                    body=(
                        NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float32", buffer="psum"),
                        NKIAlloc(dst="tensor_1", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                        NKIDmaCopy(
                            dst=_r("tensor_1", (128, 256), ((0, 128), (0, 256))),
                            src=_r("a", (128, 256), ((0, 128), (0, 256))),
                        ),
                        NKIAlloc(dst="tensor_2", shape=_T, dtype="nl.float16", buffer="sbuf"),
                        NKIDmaCopy(dst=_r("tensor_2", _T, _S), src=_r("b", _T, _S)),
                        NKIMatmul(
                            dst=_r("tensor_0", _T, _S),
                            stationary=_r("tensor_1", _T, ((0, 128), (0, 128))),
                            moving=_r("tensor_2", _T, _S),
                        ),
                        NKIAlloc(dst="tensor_3", shape=_T, dtype="nl.float32", buffer="psum"),
                        NKIMatmul(
                            dst=_r("tensor_3", _T, _S),
                            stationary=_r("tensor_1", _T, ((0, 128), (128, 256))),
                            moving=_r("tensor_2", _T, _S),
                        ),
                    ),
                ),
            ),
        )
        transform = OperandMergeTransform()
        options = transform.analyze(kernel)
        compute_options = [o for o in options if isinstance(kernel.blocks[0].body[o.ref_a.stmt_idx], NKIMatmul)]
        assert len(compute_options) == 0

    def test_accumulation_rejected(self) -> None:
        """Two matmuls accumulating to same dst are not merge candidates."""
        kernel = NKIKernel(
            name="test",
            params=("a", "b"),
            input_shapes=(_T, _T),
            dtype="nl.float16",
            output_shape=_T,
            blocks=(
                NKIBlock(
                    name="_block_0",
                    params=("a", "b", "output"),
                    body=(
                        NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float32", buffer="psum"),
                        NKIAlloc(dst="tensor_1", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                        NKIDmaCopy(
                            dst=_r("tensor_1", (128, 256), ((0, 128), (0, 256))),
                            src=_r("a", (128, 256), ((0, 128), (0, 256))),
                        ),
                        NKIAlloc(dst="tensor_2", shape=_T, dtype="nl.float16", buffer="sbuf"),
                        NKIDmaCopy(dst=_r("tensor_2", _T, _S), src=_r("b", _T, _S)),
                        NKIMatmul(
                            dst=_r("tensor_0", _T, _S),
                            stationary=_r("tensor_1", _T, ((0, 128), (0, 128))),
                            moving=_r("tensor_2", _T, _S),
                        ),
                        NKIMatmul(
                            dst=_r("tensor_0", _T, _S),
                            stationary=_r("tensor_1", _T, ((0, 128), (128, 256))),
                            moving=_r("tensor_2", _T, _S),
                        ),
                    ),
                ),
            ),
        )
        transform = OperandMergeTransform()
        options = transform.analyze(kernel)
        compute_options = [o for o in options if isinstance(kernel.blocks[0].body[o.ref_a.stmt_idx], NKIMatmul)]
        assert len(compute_options) == 0


def _is_dma_to_hbm(option: TransformOption, kernel: NKIKernel, param: str) -> bool:
    """Check if a TransformOption references DMA loads from a given param."""
    result = False
    for block in kernel.blocks:
        if block.name == option.ref_a.block_name:
            stmt = block.body[option.ref_a.stmt_idx]
            if isinstance(stmt, NKIDmaCopy) and stmt.src.name == param:
                result = True
    return result
