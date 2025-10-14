from typing import Dict, List

import neuronxcc.nki.language as nl
import numpy as np

from nki_fusion.axes import Axis


class FusionChain:
    def __init__(self) -> None:
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

    def execute(
        self,
        input_tensors: Dict[str, np.ndarray],
        parallel_axes: List[Axis],
        sequential_axes: List[Axis],
        verbose: bool = False,
    ):
        for parallel_axis in parallel_axes:
            print(parallel_axis)
