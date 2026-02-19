"""Golden fixture functions for data reuse analysis and transform tests.

This module defines pre-tiled fixture functions that exercise the data reuse
transform. Each function uses the ``tensor_N`` naming convention from codegen
output and operates on ``nkigym`` primitives.

Before functions represent the tiled program state before any data reuse
merges. After functions represent the expected state after applying a
specific number of merge iterations.

Fixture functions:
- ``before_matmul_1x1``: Single tile, no reuse possible.
- ``before_matmul_2x1``: Two tiles on M, b loaded twice identically.
- ``after_matmul_2x1``: After 1 merge: tensor_1 shared.
- ``before_matmul_1x2``: Two tiles on N, a loaded twice identically.
- ``after_matmul_1x2``: After 1 merge: tensor_0 shared.
- ``before_matmul_2x2``: Four tiles (2x2), 4 reuse pairs.
- ``after_matmul_2x2_partial``: After 1 merge: tensor_0 shared for first pair.
- ``after_matmul_2x2_full``: After all 4 merges: all redundant loads eliminated.
- ``before_matmul_4x1``: Four tiles on M, b loaded 4 times identically.
- ``after_matmul_4x1_partial``: After 1 merge: tensor_1 shared for first pair.
- ``after_matmul_4x1_full``: After all 3 iterative merges: tensor_1 shared.
- ``before_double_matmul_2x1``: Double matmul tiled on b's N dimension.
- ``after_double_matmul_2x1``: After all 2 merges: a and c loads shared.
"""

from typing import NamedTuple

import numpy as np

import nkigym


class DataReuseCase(NamedTuple):
    """A single data reuse test case.

    Attributes:
        id: Test case identifier for pytest parametrize.
        before: Golden fixture function representing the pre-merge state.
        params: Parameter names in sorted order.
        input_shapes: Mapping from parameter names to shape tuples.
        output_dtype: Numpy dtype type for output allocation.
        expected_pairs: Exact reuse pairs returned by analyze_ir on the before program.
        merge_count: Number of iterative merge passes to apply.
        after: Golden fixture function representing the expected post-merge state.
    """

    id: str
    before: object
    params: tuple[str, ...]
    input_shapes: dict[str, tuple[int, ...]]
    output_dtype: type
    expected_pairs: list[tuple[str, str]]
    merge_count: int
    after: object


def before_matmul_1x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1x1 matmul: single tile, no reuse possible."""
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    return output


def before_matmul_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x1 matmul: 2 tiles on M, b loaded twice identically."""
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[128:256, 0:128] = tensor_5
    return output


def after_matmul_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x1 matmul after 1 merge: tensor_4 removed, tensor_1 shared."""
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_1)
    output[128:256, 0:128] = tensor_5
    return output


def before_matmul_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1x2 matmul: 2 tiles on N, a loaded twice identically."""
    output = np.empty((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5
    return output


def after_matmul_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1x2 matmul after 1 merge: tensor_3 removed, tensor_0 shared."""
    output = np.empty((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_0, tensor_4)
    output[0:128, 128:256] = tensor_5
    return output


def before_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x2 matmul: 4 tiles, 4 reuse pairs."""
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5
    tensor_6 = a[0:128, 128:256]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_7)
    output[128:256, 0:128] = tensor_8
    tensor_9 = a[0:128, 128:256]
    tensor_10 = b[0:128, 128:256]
    tensor_11 = nkigym.nc_matmul(tensor_9, tensor_10)
    output[128:256, 128:256] = tensor_11
    return output


def after_matmul_2x2_partial(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x2 matmul after 1 merge: tensor_3 removed, tensor_0 shared."""
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_0, tensor_4)
    output[0:128, 128:256] = tensor_5
    tensor_6 = a[0:128, 128:256]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_7)
    output[128:256, 0:128] = tensor_8
    tensor_9 = a[0:128, 128:256]
    tensor_10 = b[0:128, 128:256]
    tensor_11 = nkigym.nc_matmul(tensor_9, tensor_10)
    output[128:256, 128:256] = tensor_11
    return output


def after_matmul_2x2_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x2 matmul after all 4 merges: all redundant loads eliminated."""
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_0, tensor_4)
    output[0:128, 128:256] = tensor_5
    tensor_6 = a[0:128, 128:256]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_1)
    output[128:256, 0:128] = tensor_8
    tensor_11 = nkigym.nc_matmul(tensor_6, tensor_4)
    output[128:256, 128:256] = tensor_11
    return output


def before_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4x1 matmul: 4 tiles on M, b loaded 4 times identically."""
    output = np.empty((512, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[128:256, 0:128] = tensor_5
    tensor_6 = a[0:128, 256:384]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_7)
    output[256:384, 0:128] = tensor_8
    tensor_9 = a[0:128, 384:512]
    tensor_10 = b[0:128, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9, tensor_10)
    output[384:512, 0:128] = tensor_11
    return output


def after_matmul_4x1_partial(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4x1 matmul after 1 merge: tensor_4 removed, tensor_1 shared."""
    output = np.empty((512, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_1)
    output[128:256, 0:128] = tensor_5
    tensor_6 = a[0:128, 256:384]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_7)
    output[256:384, 0:128] = tensor_8
    tensor_9 = a[0:128, 384:512]
    tensor_10 = b[0:128, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9, tensor_10)
    output[384:512, 0:128] = tensor_11
    return output


def after_matmul_4x1_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4x1 matmul after all 3 iterative merges: tensor_1 shared across all."""
    output = np.empty((512, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_1)
    output[128:256, 0:128] = tensor_5
    tensor_6 = a[0:128, 256:384]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_1)
    output[256:384, 0:128] = tensor_8
    tensor_9 = a[0:128, 384:512]
    tensor_11 = nkigym.nc_matmul(tensor_9, tensor_1)
    output[384:512, 0:128] = tensor_11
    return output


def before_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul 2x1: tiled on b's N dimension, a and c each loaded twice."""
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    tensor_3 = c[0:128, 0:128]
    tensor_4 = nkigym.nc_matmul(tensor_2, tensor_3)
    output[0:128, 0:128] = tensor_4
    tensor_5 = a[0:128, 0:128]
    tensor_6 = b[0:128, 128:256]
    tensor_7 = nkigym.nc_matmul(tensor_5, tensor_6)
    tensor_8 = c[0:128, 0:128]
    tensor_9 = nkigym.nc_matmul(tensor_7, tensor_8)
    output[128:256, 0:128] = tensor_9
    return output


def after_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul 2x1 after all 2 merges: a and c loads shared."""
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    tensor_3 = c[0:128, 0:128]
    tensor_4 = nkigym.nc_matmul(tensor_2, tensor_3)
    output[0:128, 0:128] = tensor_4
    tensor_6 = b[0:128, 128:256]
    tensor_7 = nkigym.nc_matmul(tensor_0, tensor_6)
    tensor_9 = nkigym.nc_matmul(tensor_7, tensor_3)
    output[128:256, 0:128] = tensor_9
    return output


CASES: list[DataReuseCase] = [
    DataReuseCase(
        id="matmul_1x1_no_reuse",
        before=before_matmul_1x1,
        params=("a", "b"),
        input_shapes={"a": (128, 128), "b": (128, 128)},
        output_dtype=np.float32,
        expected_pairs=[],
        merge_count=0,
        after=before_matmul_1x1,
    ),
    DataReuseCase(
        id="matmul_2x1_merge_b",
        before=before_matmul_2x1,
        params=("a", "b"),
        input_shapes={"a": (128, 256), "b": (128, 128)},
        output_dtype=np.float32,
        expected_pairs=[("tensor_1", "tensor_4")],
        merge_count=1,
        after=after_matmul_2x1,
    ),
    DataReuseCase(
        id="matmul_1x2_merge_a",
        before=before_matmul_1x2,
        params=("a", "b"),
        input_shapes={"a": (128, 128), "b": (128, 256)},
        output_dtype=np.float32,
        expected_pairs=[("tensor_0", "tensor_3")],
        merge_count=1,
        after=after_matmul_1x2,
    ),
    DataReuseCase(
        id="matmul_2x2_partial",
        before=before_matmul_2x2,
        params=("a", "b"),
        input_shapes={"a": (128, 256), "b": (128, 256)},
        output_dtype=np.float32,
        expected_pairs=[
            ("tensor_0", "tensor_3"),
            ("tensor_1", "tensor_7"),
            ("tensor_4", "tensor_10"),
            ("tensor_6", "tensor_9"),
        ],
        merge_count=1,
        after=after_matmul_2x2_partial,
    ),
    DataReuseCase(
        id="matmul_2x2_full",
        before=before_matmul_2x2,
        params=("a", "b"),
        input_shapes={"a": (128, 256), "b": (128, 256)},
        output_dtype=np.float32,
        expected_pairs=[
            ("tensor_0", "tensor_3"),
            ("tensor_1", "tensor_7"),
            ("tensor_4", "tensor_10"),
            ("tensor_6", "tensor_9"),
        ],
        merge_count=4,
        after=after_matmul_2x2_full,
    ),
    DataReuseCase(
        id="matmul_4x1_partial",
        before=before_matmul_4x1,
        params=("a", "b"),
        input_shapes={"a": (128, 512), "b": (128, 128)},
        output_dtype=np.float32,
        expected_pairs=[
            ("tensor_1", "tensor_4"),
            ("tensor_1", "tensor_7"),
            ("tensor_1", "tensor_10"),
            ("tensor_4", "tensor_7"),
            ("tensor_4", "tensor_10"),
            ("tensor_7", "tensor_10"),
        ],
        merge_count=1,
        after=after_matmul_4x1_partial,
    ),
    DataReuseCase(
        id="matmul_4x1_full",
        before=before_matmul_4x1,
        params=("a", "b"),
        input_shapes={"a": (128, 512), "b": (128, 128)},
        output_dtype=np.float32,
        expected_pairs=[
            ("tensor_1", "tensor_4"),
            ("tensor_1", "tensor_7"),
            ("tensor_1", "tensor_10"),
            ("tensor_4", "tensor_7"),
            ("tensor_4", "tensor_10"),
            ("tensor_7", "tensor_10"),
        ],
        merge_count=3,
        after=after_matmul_4x1_full,
    ),
    DataReuseCase(
        id="double_matmul_2x1_full",
        before=before_double_matmul_2x1,
        params=("a", "b", "c"),
        input_shapes={"a": (128, 128), "b": (128, 256), "c": (128, 128)},
        output_dtype=np.float32,
        expected_pairs=[("tensor_0", "tensor_5"), ("tensor_3", "tensor_8")],
        merge_count=2,
        after=after_double_matmul_2x1,
    ),
]
