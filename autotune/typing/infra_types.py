from typing import Any, Callable

import numpy as np

INPUT_TENSORS_DTYPE = dict[str, np.ndarray]
OUTPUT_TENSORS_DTYPE = dict[str, np.ndarray]
INPUT_TENSOR_SHAPES_DTYPE = dict[str, tuple[int, ...]]
METRICS_DTYPE = dict[str, float]
KERNEL_DTYPE = tuple[str, str]
KERNEL_KWARGS_DTYPE = dict[str, Any]
PREPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE], None]
POSTPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE], None]
