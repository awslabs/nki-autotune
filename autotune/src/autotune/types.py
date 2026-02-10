# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import numpy as np

INPUT_TENSORS_DTYPE = dict[str, np.ndarray]
OUTPUT_TENSORS_DTYPE = tuple[np.ndarray, ...]
INPUT_TENSOR_SHAPES_DTYPE = dict[str, tuple[int, ...]]
METRICS_DTYPE = dict[str, float]
KERNEL_DTYPE = tuple[str, str]
KERNEL_KWARGS_DTYPE = dict[str, Any]
PREPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE], None]
GOLDEN_FN_DTYPE = Callable[..., np.ndarray]
CORRECTNESS_CHECK_DTYPE = tuple[GOLDEN_FN_DTYPE, float, float] | None
