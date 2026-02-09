"""Golden fixture functions for operand merge analysis tests.

This module defines pre-tiled fixture functions that exercise the operand
merge transform. Each function uses the ``tensor_N`` naming convention
from codegen output and operates on ``nkigym`` primitives.

Since these functions are defined in a regular source file,
``inspect.getsource()`` works and the ``get_source`` utility can parse them
directly without needing a ``__source__`` attribute.

Fixture functions:
- ``tiled_adjacent_loads_2x``: Two loads from same tensor, adjacent on free dim.
- ``tiled_no_adjacent_loads``: Two loads from same tensor, non-adjacent on free dim.
- ``tiled_single_subgraph``: Single subgraph, no merging possible.
- ``tiled_adjacent_4x``: Four adjacent loads from same tensor on free dim.
- ``tiled_different_source_tensors``: Loads from different source tensors.
- ``tiled_different_partition_slices``: Loads with different partition dim slices.
- ``tiled_matmul_post_reuse_1x2``: Post-data-reuse 1x2 with shared a, adjacent b.
- ``tiled_matmul_post_reuse_1x4``: Post-data-reuse 1x4 with shared a, 4 adjacent b.
- ``tiled_matmul_post_reuse_2x2``: Post-data-reuse 2x2 with shared loads on both a and b.
- ``tiled_matmul_exceeds_n_limit``: 5 adjacent b loads, merged N=640 exceeds 512 limit.
- ``tiled_subscript_loads_2x``: Two adjacent loads consumed via subscript (not bare Name).
- ``tiled_matmul_n_at_limit``: 2 nc_matmul with merged N=512, exactly at hardware limit.
- ``tiled_matmul_m_dim_merge``: Same RHS, adjacent LHS on M dimension (64+64=128).
- ``tiled_matmul_m_exceeds_limit``: Same RHS, adjacent LHS merged M=192 exceeds 128 limit.
- ``tiled_tensor_tensor_2x``: Two tensor_tensor ops with same op, adjacent first arg.
- ``tiled_tensor_tensor_diff_ops``: Two tensor_tensor ops with different op kwargs.
- ``tiled_activation_2x``: Two activation ops with same op kwarg, adjacent input slices.
- ``tiled_tensor_scalar_2x``: Two tensor_scalar ops with same kwargs, adjacent input slices.
- ``tiled_activation_single``: Single activation op, no merging possible.
"""

import numpy as np

import nkigym


def tiled_adjacent_loads_2x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two loads from same tensor b, adjacent on free dimension."""
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5
    return output


def tiled_no_adjacent_loads(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two loads from same tensor b, non-adjacent free dimension (gap between slices)."""
    output = nkigym.ndarray((128, 384), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 256:384]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[0:128, 256:384] = tensor_5
    return output


def tiled_single_subgraph(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single subgraph, no merging possible."""
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    return output


def tiled_adjacent_4x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Four adjacent loads from tensor b on free dimension."""
    output = nkigym.ndarray((128, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 256:384]
    tensor_8 = nkigym.nc_matmul(tensor_6, tensor_7)
    output[0:128, 256:384] = tensor_8
    tensor_9 = a[0:128, 0:128]
    tensor_10 = b[0:128, 384:512]
    tensor_11 = nkigym.nc_matmul(tensor_9, tensor_10)
    output[0:128, 384:512] = tensor_11
    return output


def tiled_different_source_tensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two loads from different source tensors (a and b), no merging."""
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    return output


def tiled_different_partition_slices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two loads from same tensor but different partition dimension slices."""
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[128:256, 0:128] = tensor_5
    return output


def tiled_matmul_post_reuse_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Post-data-reuse: 1x2 matmul with shared a load, adjacent b loads."""
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_0, tensor_4)
    output[0:128, 128:256] = tensor_5
    return output


def tiled_matmul_post_reuse_1x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Post-data-reuse: 1x4 matmul with shared a, 4 adjacent b loads."""
    output = nkigym.ndarray((128, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = b[0:128, 128:256]
    tensor_4 = nkigym.nc_matmul(tensor_0, tensor_3)
    output[0:128, 128:256] = tensor_4
    tensor_5 = b[0:128, 256:384]
    tensor_6 = nkigym.nc_matmul(tensor_0, tensor_5)
    output[0:128, 256:384] = tensor_6
    tensor_7 = b[0:128, 384:512]
    tensor_8 = nkigym.nc_matmul(tensor_0, tensor_7)
    output[0:128, 384:512] = tensor_8
    return output


def tiled_matmul_exceeds_n_limit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """5 adjacent b loads: merged N=640 exceeds nc_matmul N limit of 512."""
    output = nkigym.ndarray((128, 640), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = b[0:128, 128:256]
    tensor_4 = nkigym.nc_matmul(tensor_0, tensor_3)
    output[0:128, 128:256] = tensor_4
    tensor_5 = b[0:128, 256:384]
    tensor_6 = nkigym.nc_matmul(tensor_0, tensor_5)
    output[0:128, 256:384] = tensor_6
    tensor_7 = b[0:128, 384:512]
    tensor_8 = nkigym.nc_matmul(tensor_0, tensor_7)
    output[0:128, 384:512] = tensor_8
    tensor_9 = b[0:128, 512:640]
    tensor_10 = nkigym.nc_matmul(tensor_0, tensor_9)
    output[0:128, 512:640] = tensor_10
    return output


def tiled_subscript_loads_2x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two adjacent loads from b consumed via subscript in nc_matmul.

    Loads are consumed as ``tensor_1[0:128, 0:128]`` (subscripted) rather
    than bare ``tensor_1``, so load merging is safe.
    """
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_0, tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5
    return output


def tiled_matmul_n_at_limit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two nc_matmul with merged N=512, exactly at hardware limit.

    Each matmul uses a 256-wide b slice. Merged N = 256+256 = 512, which
    is the nc_matmul N limit. Should produce one nc_matmul opportunity.
    """
    output = nkigym.ndarray((128, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:256]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:256] = tensor_2
    tensor_3 = b[0:128, 256:512]
    tensor_4 = nkigym.nc_matmul(tensor_0, tensor_3)
    output[0:128, 256:512] = tensor_4
    return output


def tiled_matmul_m_dim_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same RHS, adjacent LHS on M dimension (64+64=128, within M limit).

    nc_matmul(lhs[K,M], rhs[K,N]) -> [M,N]. Two matmuls share the same b
    load (K=128, N=128) but differ on the a load's M dimension (dim 1):
    a[0:128, 0:64] (K=128, M=64) vs a[0:128, 64:128] (K=128, M=64).
    Merged M = 128.
    """
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:64]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:64, 0:128] = tensor_2
    tensor_3 = a[0:128, 64:128]
    tensor_4 = nkigym.nc_matmul(tensor_3, tensor_1)
    output[64:128, 0:128] = tensor_4
    return output


def tiled_matmul_m_exceeds_limit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same RHS, adjacent LHS merged M=192 exceeds nc_matmul M limit of 128.

    nc_matmul(lhs[K,M], rhs[K,N]). Two matmuls share the same b load but
    differ on a's M dimension (dim 1): a[0:128, 0:128] (M=128) and
    a[0:128, 128:192] (M=64). Merged M = 192 > 128.
    """
    output = nkigym.ndarray((192, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:192]
    tensor_4 = nkigym.nc_matmul(tensor_3, tensor_1)
    output[128:192, 0:128] = tensor_4
    return output


def tiled_tensor_tensor_2x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two tensor_tensor ops with same op (np.add), differing first arg.

    tensor_0 and tensor_3 are adjacent slices of a on free dimension.
    tensor_1 is a shared load from b. The second call reuses tensor_1
    via bare Name for the shared arg, and tensor_3 for the differing arg.
    """
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.tensor_tensor(tensor_0, tensor_1, op=np.add)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_5 = nkigym.tensor_tensor(tensor_3, tensor_1, op=np.add)
    output[0:128, 128:256] = tensor_5
    return output


def tiled_tensor_tensor_diff_ops(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two tensor_tensor ops with different op kwargs (np.add vs np.multiply).

    Despite adjacent first-arg slices and shared second arg, different op
    kwargs prevent merging.
    """
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.tensor_tensor(tensor_0, tensor_1, op=np.add)
    output[0:128, 0:128] = tensor_2
    tensor_3 = a[0:128, 128:256]
    tensor_5 = nkigym.tensor_tensor(tensor_3, tensor_1, op=np.multiply)
    output[0:128, 128:256] = tensor_5
    return output


def tiled_activation_2x(a: np.ndarray) -> np.ndarray:
    """Two activation ops with same op kwarg, adjacent input slices.

    tensor_0 and tensor_2 are adjacent slices of a on free dimension,
    passed as bare Name args. After merging: single activation with
    a widened load.
    """
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = nkigym.activation(tensor_0, op=np.tanh)
    output[0:128, 0:128] = tensor_1
    tensor_2 = a[0:128, 128:256]
    tensor_3 = nkigym.activation(tensor_2, op=np.tanh)
    output[0:128, 128:256] = tensor_3
    return output


def tiled_tensor_scalar_2x(a: np.ndarray) -> np.ndarray:
    """Two tensor_scalar ops with same kwargs, adjacent input slices.

    tensor_0 and tensor_2 are adjacent slices of a on free dimension,
    passed as bare Name args. After merging: single tensor_scalar with
    a widened load.
    """
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = nkigym.tensor_scalar(tensor_0, op0=np.multiply, operand0=2.0)
    output[0:128, 0:128] = tensor_1
    tensor_2 = a[0:128, 128:256]
    tensor_3 = nkigym.tensor_scalar(tensor_2, op0=np.multiply, operand0=2.0)
    output[0:128, 128:256] = tensor_3
    return output


def tiled_matmul_post_reuse_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Post-data-reuse: 2x2 matmul with shared loads on both a and b.

    The 2x2 tile grid produces four nc_matmul calls. After data reuse,
    tensor_0 (a row 0) is shared by [0,0] and [0,1], tensor_1 (b col 0)
    is shared by [0,0] and [1,0], tensor_4 (b col 1) is shared by [0,1]
    and [1,1], and tensor_6 (a row 1) is shared by [1,0] and [1,1].
    """
    output = nkigym.ndarray((256, 256), dtype=np.float32)
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


def tiled_activation_single(a: np.ndarray) -> np.ndarray:
    """Single activation op, no merging possible."""
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = nkigym.activation(tensor_0, op=np.tanh)
    output[0:128, 0:128] = tensor_1
    return output
