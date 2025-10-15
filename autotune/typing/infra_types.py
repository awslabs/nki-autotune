from typing import Any, Callable, Dict, Tuple

import numpy as np

INPUT_TENSORS_DTYPE = Dict[str, np.ndarray]
OUTPUT_TENSORS_DTYPE = Dict[str, np.ndarray]
INPUT_TENSOR_SHAPES_DTYPE = Dict[str, Tuple[int, ...]]
METRICS_DTYPE = Dict[str, float]
KERNEL_DTYPE = Tuple[str, str]  # file path, kernel function name
KERNEL_KWARGS_DTYPE = Dict[str, Any]
PREPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE], None]
POSTPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE], None]
