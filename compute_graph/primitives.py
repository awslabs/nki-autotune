from typing import Dict, Tuple

INPUT_TENSOR_SHAPE = Tuple[int, ...]
AXIS = Tuple[str, int, int]  # tensor name, axis index, tile size
AXIS_LOC = Tuple[int, int]  # tile index, tile size
AXES_LOC = Dict[int, AXIS_LOC]  # axis index, AXIS_LOC
