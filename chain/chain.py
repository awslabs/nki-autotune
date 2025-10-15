from typing import Dict, List, Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from chain.axes import Axis


class Chain:
    """Manages parallel axis sharding for standard fusion chains."""

    def __init__(self, parallel_axes: Tuple[Axis]) -> None:
        """Initialize chain with parallel axis configurations."""
        self.parallel_axes = parallel_axes
        num_parallel_blocks = 1
        for parallel_axis in parallel_axes:
            num_parallel_blocks *= parallel_axis.num_blocks
        self.num_parallel_blocks = num_parallel_blocks

    def __call__(self, input_tensors: Dict[str, tensor], verbose: bool = False):
        """Execute fusion chain with given input tensors."""
        for counter in nl.affine_range(self.num_parallel_blocks):
            stride = self.num_parallel_blocks
            for parallel_axis in self.parallel_axes:
                stride = stride // parallel_axis.num_blocks
                block_index = (counter // stride) % parallel_axis.num_blocks


@nki.jit
def chain_wrapper(input_tensors: Dict[str, tensor], parallel_axes: List[Axis]):
    """NKI-compiled wrapper for Chain execution.

    Args:
        input_tensors: Dictionary mapping tensor names to input tensors
        parallel_axes: List of parallel axis configurations

    Returns:
        Output tensor from chain
    """
    chain = Chain(parallel_axes=parallel_axes)
    output = chain(input_tensors=input_tensors)
    return output
