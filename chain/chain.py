from typing import Dict, List

import neuronxcc.nki as nki
from neuronxcc.nki.typing import tensor

from chain.axes import Axis


class Chain:
    """Manages parallel axis sharding for standard fusion chains."""

    def __init__(self, parallel_axes: List[Axis]) -> None:
        """Initialize chain with parallel axis configurations."""
        self.parallel_axes = parallel_axes

    def __call__(self, input_tensors: Dict[str, tensor], verbose: bool = False):
        """Execute fusion chain with given input tensors."""
        print(self.parallel_axes)
        print(input_tensors)


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
