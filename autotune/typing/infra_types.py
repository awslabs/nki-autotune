from typing import Any, Callable, Dict, Tuple

import numpy as np

INPUT_TENSORS_DTYPE = Tuple[np.ndarray, ...]
OUTPUT_TENSORS_DTYPE = Tuple[np.ndarray, ...]
METRICS_DTYPE = Dict[str, float]
KERNEL_KWARGS_DTYPE = Dict[str, Any]
PREPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE], None]
POSTPROCESSING_DTYPE = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE], None]
KERNEL_DTYPE = Tuple[str, str]  # file path, kernel function name
