"""End-to-end pipeline tests: tile -> data reuse -> operand merge.

Run with: pytest test/test_pipeline.py -v
"""

import numpy as np
import pytest
from conftest import assert_programs_numerically_equal

import nkigym  # type: ignore[import]
from nkigym.function_to_program import source_to_program, tile_program  # type: ignore[import]
from nkigym.transforms import DataReuseTransform  # type: ignore[import]
from nkigym.transforms.operand_merge import OperandMergeTransform  # type: ignore[import]
from nkigym.utils import callable_to_source  # type: ignore[import]

_reuse = DataReuseTransform()
_merge = OperandMergeTransform()


def _make_kwargs(shapes: dict[str, tuple[int, ...]], dtype: type) -> dict[str, np.ndarray]:
    """Create zero-filled kernel kwargs from shapes and dtype."""
    return {name: np.zeros(shape, dtype=dtype) for name, shape in shapes.items()}


@pytest.mark.parametrize("a_shape,b_shape", [((128, 128), (128, 256)), ((128, 256), (128, 256))], ids=["1x2", "2x2"])
def test_e2e_pipeline(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> None:
    """Full pipeline: tile_program -> DataReuseTransform -> OperandMergeTransform.

    Args:
        a_shape: Shape of the a input.
        b_shape: Shape of the b input.
    """

    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute matrix multiplication."""
        return nkigym.nc_matmul(a, b)  # type: ignore[attr-defined]

    source = callable_to_source(matmul)
    before = source_to_program(source, _make_kwargs({"a": a_shape, "b": b_shape}, np.float32))
    program = tile_program(before)

    while True:
        pairs = _reuse.analyze_ir(program)
        if not pairs:
            break
        program = _reuse.transform_ir(program, pairs[0])

    while True:
        opps = _merge.analyze_ir(program)
        if not opps:
            break
        program = _merge.transform_ir(program, opps[0])

    assert_programs_numerically_equal(before, program, rtol=1e-5, atol=1e-5)
