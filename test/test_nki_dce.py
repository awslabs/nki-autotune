"""Tests for NKIKernel dead code elimination."""

from nkigym.codegen.dce import dce
from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy

_T = (128, 128)
_S = ((0, 128), (0, 128))


def _r(name: str, shape: tuple[int, ...], slices: tuple[tuple[int, int], ...]) -> TensorRef:
    """Create a TensorRef shorthand."""
    return TensorRef(name, shape, slices)


_LIVE_KERNEL = NKIKernel(
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

_DEAD_KERNEL = NKIKernel(
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
                NKIAlloc(dst="tensor_dead", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_dead", _T, _S), src=_r("a", _T, _S)),
                NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_0", _T, _S)),
            ),
        ),
    ),
)


class TestDCE:
    """Dead code elimination on NKIKernel."""

    def test_no_dead_code(self) -> None:
        """All-live kernel is unchanged."""
        result = dce(_LIVE_KERNEL)
        assert result == _LIVE_KERNEL

    def test_removes_dead_alloc_and_load(self) -> None:
        """Dead alloc + DMA load pair is removed."""
        result = dce(_DEAD_KERNEL)
        assert result == _LIVE_KERNEL

    def test_output_store_preserved(self) -> None:
        """DMA copy to output is never removed."""
        result = dce(_DEAD_KERNEL)
        last_stmt = result.blocks[0].body[-1]
        assert isinstance(last_stmt, NKIDmaCopy)
        assert last_stmt.dst.name == "output"

    def test_empty_block_dropped(self) -> None:
        """Block with only dead code is removed entirely."""
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
                    body=(NKIDmaCopy(dst=_r("output", _T, _S), src=_r("a", _T, _S)),),
                ),
                NKIBlock(
                    name="_block_1",
                    params=("a", "output"),
                    body=(
                        NKIAlloc(dst="tensor_dead", shape=_T, dtype="nl.float16", buffer="sbuf"),
                        NKIDmaCopy(dst=_r("tensor_dead", _T, _S), src=_r("a", _T, _S)),
                    ),
                ),
            ),
        )
        result = dce(kernel)
        assert len(result.blocks) == 1
        assert result.blocks[0].name == "_block_0"
