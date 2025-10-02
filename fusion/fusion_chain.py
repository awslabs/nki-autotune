"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

import math
from typing import Dict, List

from fusion.operators import Operator
from fusion.tensors import Tensor


class FusionChain:
    """
    Implements generalized fusion for multiple operators following MegaFuse principles.

    This class supports arbitrary sequences of blocking and accumulation operations,
    enabling fusion of complex patterns.
    """

    def __init__(self, fx: Operator, gbs: List[Operator], hbs: List[Operator], **kwargs):
        """
        Initialize FusionChain with a sequence of operators.
        """
        self.fx = fx
        self.gbs = gbs
        self.hbs = hbs

    def execute(self, fusion_axis: str, fusion_block_size: int, input_tensors: List[Tensor]) -> Tensor:
        """
        Execute the fusion chain on the provided inputs.

        Args:
            fusion_block_size: granularity of the fusion axis each forward step
            input_tensors: input tensors

        Returns:
            Output tensor after applying all operators
        """
        fusion_size = 0
        for tensor in input_tensors:
            if fusion_axis in tensor.axes:
                tensor_fusion_size = tensor.get_axis_size(fusion_axis)
                if fusion_size == 0:
                    fusion_size = tensor_fusion_size
                else:
                    assert fusion_size == tensor_fusion_size, f"Fusion size mismatch in input tensors"
        assert fusion_size > 0, "Did not find fusion axis in the input tensors"
        self.all_tensors: Dict[str, Tensor] = {}
        for tensor in input_tensors:
            self.all_tensors[tensor.name] = tensor
        num_fusion_steps = math.ceil(fusion_size / fusion_block_size)
        fx_input_tensors = [self.all_tensors[tensor_name] for tensor_name in self.fx.input_tensors]
        prev_O1 = self.fx.initialize_output(fusion_axis, fx_input_tensors)
        curr_O1 = Tensor(name=prev_O1.name.replace("prev", "curr"), axes=prev_O1.axes, data=prev_O1.data)
        for fusion_step in range(num_fusion_steps):
            fusion_axis_start = fusion_step * fusion_block_size
            fx_forward_inputs = [prev_O1]
            for tensor in fx_input_tensors:
                fx_forward_inputs.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_block_size)
                )
            self.fx.forward(inputs=fx_forward_inputs, next_output=curr_O1)
            prev_O1.data = curr_O1.data
        return curr_O1
