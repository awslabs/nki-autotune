from typing import Dict, List

import numpy as np

from nki_fusion.axes import ParallelAxis, SequentialAxis


class FusionChain:
    def __init__(self, parallel_axes_config: List[ParallelAxis], sequential_axis_config: SequentialAxis) -> None:
        self.parallel_axes_config = parallel_axes_config
        self.sequential_axis_config = sequential_axis_config

    def execute(self, input_tensors: Dict[str, np.ndarray], verbose: bool = False):
        print(self.parallel_axes_config)
        print(self.sequential_axis_config)
        print(input_tensors)
