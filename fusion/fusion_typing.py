from typing import Callable, Tuple

import numpy as np

PREV_OUTPUT_DTYPE = np.ndarray
STEP_FUNC_DTYPE = Callable[[PREV_OUTPUT_DTYPE, Tuple[np.ndarray, ...], int], np.ndarray]
