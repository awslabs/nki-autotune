"""Rolled golden functions for loop rolling tests."""

import numpy as np

import nkigym


def roll1_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 2x1 matmul after 1 roll (1 loop over cols)."""
    output = np.empty((128, 256), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, 0:128]
        tensor_1 = b[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[0:128, i_0 * 128 : (i_0 + 1) * 128] = tensor_2[0:128, 0:128]
    return output


def roll1_3x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 3x1 matmul after 1 roll (1 loop over cols)."""
    output = np.empty((128, 384), dtype=np.float32)
    for i_0 in range(3):
        tensor_0 = a[0:128, 0:128]
        tensor_1 = b[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[0:128, i_0 * 128 : (i_0 + 1) * 128] = tensor_2[0:128, 0:128]
    return output


def roll1_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 4x1 matmul after 1 roll (1 loop over cols)."""
    output = np.empty((128, 512), dtype=np.float32)
    for i_0 in range(4):
        tensor_0 = a[0:128, 0:128]
        tensor_1 = b[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[0:128, i_0 * 128 : (i_0 + 1) * 128] = tensor_2[0:128, 0:128]
    return output


def roll1_1x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 1x4 matmul after 1 roll (1 loop over rows)."""
    output = np.empty((512, 128), dtype=np.float32)
    for i_0 in range(4):
        tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128 : (i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output


def roll1_1x5(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 1x5 matmul after 1 roll (1 loop over rows)."""
    output = np.empty((640, 128), dtype=np.float32)
    for i_0 in range(5):
        tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128 : (i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output


def roll1_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 2x2 matmul after 1 roll (outer loop, inner unrolled)."""
    output = np.empty((256, 256), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128 : (i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
        tensor_3 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_4 = b[0:128, 128:256]
        tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
        output[i_0 * 128 : (i_0 + 1) * 128, 128:256] = tensor_5[0:128, 0:128]
    return output


def roll2_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 2x2 matmul after 2 rolls (nested loops)."""
    output = np.empty((256, 256), dtype=np.float32)
    for i_0 in range(2):
        for i_1 in range(2):
            tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128 : (i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i_0 * 128 : (i_0 + 1) * 128, i_1 * 128 : (i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output


def roll2_3x5(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 3x5 matmul after 2 rolls (nested loops)."""
    output = np.empty((640, 384), dtype=np.float32)
    for i_0 in range(5):
        for i_1 in range(3):
            tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128 : (i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i_0 * 128 : (i_0 + 1) * 128, i_1 * 128 : (i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output


def roll2_4x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 4x4 matmul after 2 rolls (nested loops)."""
    output = np.empty((512, 512), dtype=np.float32)
    for i_0 in range(4):
        for i_1 in range(4):
            tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128 : (i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i_0 * 128 : (i_0 + 1) * 128, i_1 * 128 : (i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output


def roll1_2x2_red2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 2x2 red2 matmul after 1 roll (outer loop, inner unrolled)."""
    output = np.empty((256, 256), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        tensor_3 = a[128:256, i_0 * 128 : (i_0 + 1) * 128]
        tensor_4 = b[128:256, 0:128]
        tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
        output[i_0 * 128 : (i_0 + 1) * 128, 0:128] = tensor_5[0:128, 0:128]
        tensor_6 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
        tensor_7 = b[0:128, 128:256]
        tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
        tensor_9 = a[128:256, i_0 * 128 : (i_0 + 1) * 128]
        tensor_10 = b[128:256, 128:256]
        tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
        output[i_0 * 128 : (i_0 + 1) * 128, 128:256] = tensor_11[0:128, 0:128]
    return output


def roll2_2x2_red2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 2x2 red2 matmul after 2 rolls (nested loops)."""
    output = np.empty((256, 256), dtype=np.float32)
    for i_0 in range(2):
        for i_1 in range(2):
            tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128 : (i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i_0 * 128 : (i_0 + 1) * 128]
            tensor_4 = b[128:256, i_1 * 128 : (i_1 + 1) * 128]
            tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
            output[i_0 * 128 : (i_0 + 1) * 128, i_1 * 128 : (i_1 + 1) * 128] = tensor_5[0:128, 0:128]
    return output


def roll2_3x5_red2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 3x5 red2 matmul after 2 rolls (nested loops)."""
    output = np.empty((640, 384), dtype=np.float32)
    for i_0 in range(5):
        for i_1 in range(3):
            tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128 : (i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i_0 * 128 : (i_0 + 1) * 128]
            tensor_4 = b[128:256, i_1 * 128 : (i_1 + 1) * 128]
            tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
            output[i_0 * 128 : (i_0 + 1) * 128, i_1 * 128 : (i_1 + 1) * 128] = tensor_5[0:128, 0:128]
    return output


def roll2_2x3_red3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rolled 2x3 red3 matmul after 2 rolls (nested loops)."""
    output = np.empty((256, 384), dtype=np.float32)
    for i_0 in range(2):
        for i_1 in range(3):
            tensor_0 = a[0:128, i_0 * 128 : (i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128 : (i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i_0 * 128 : (i_0 + 1) * 128]
            tensor_4 = b[128:256, i_1 * 128 : (i_1 + 1) * 128]
            tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
            tensor_6 = a[256:384, i_0 * 128 : (i_0 + 1) * 128]
            tensor_7 = b[256:384, i_1 * 128 : (i_1 + 1) * 128]
            tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
            output[i_0 * 128 : (i_0 + 1) * 128, i_1 * 128 : (i_1 + 1) * 128] = tensor_8[0:128, 0:128]
    return output
