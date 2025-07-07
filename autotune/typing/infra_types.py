from typing import Any, Callable, Dict, Tuple, TypeAlias

import numpy as np

INPUT_TENSORS_DTYPE: TypeAlias = Tuple[np.ndarray, ...]
OUTPUT_TENSORS_DTYPE: TypeAlias = Tuple[np.ndarray, ...]
METRICS_DTYPE: TypeAlias = Dict[str, float]
KERNEL_KWARGS_DTYPE: TypeAlias = Dict[str, Any]
PREPROCESSING_DTYPE: TypeAlias = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE], None]
POSTPROCESSING_DTYPE: TypeAlias = Callable[[INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE], None]
KERNEL_DTYPE: TypeAlias = Tuple[str, str]
