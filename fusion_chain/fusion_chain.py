from typing import Dict, List

import neuronxcc.nki as nki
import numpy as np

from compute_graph.axes import Axis


class FusionChain:
    def __init__(self, parallel_axes_config: List[Axis], sequential_axis_config: Axis) -> None:
        self.parallel_axes_config = parallel_axes_config
        self.sequential_axis_config = sequential_axis_config

    def __call__(self, input_tensors: Dict[str, np.ndarray], verbose: bool = False):
        print(self.parallel_axes_config)
        print(self.sequential_axis_config)
        print(input_tensors)


@nki.jit
def fusion_chain_wrapper(
    *input_tensors, tensor_names: List[str], parallel_axes_config: List[Axis], sequential_axis_config: Axis
):
    """
    NKI wrapper for FusionChain that accepts individual tensors.

    Args:
        *input_tensors: Variable number of input tensors
        tensor_names: List of names corresponding to input_tensors (in order)
        parallel_axes_config: List of Axis configurations for parallel dimensions
        sequential_axis_config: Axis configuration for the sequential dimension

    Returns:
        Output tensor from the fusion chain
    """
    input_tensor_dict = {name: tensor for name, tensor in zip(tensor_names, input_tensors)}

    chain = FusionChain(parallel_axes_config=parallel_axes_config, sequential_axis_config=sequential_axis_config)
    return chain(input_tensors=input_tensor_dict)
