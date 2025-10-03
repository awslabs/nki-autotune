"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

import math
from typing import List

from fusion.operators import Operator
from fusion.tensors import Tensor


class FusionChain:
    """
    Implements generalized fusion for multiple operators following MegaFuse principles.

    This class supports arbitrary sequences of blocking and accumulation operations,
    enabling fusion of complex patterns.
    """

    def __init__(self, fx: Operator, gbs: List[Operator], hbs: List[Operator]):
        """
        Initialize FusionChain with a sequence of operators.
        """
        self.fx = fx
        self.gbs = gbs
        self.hbs = hbs

    def get_tensors(self, tensor_names: List[str]) -> List[Tensor]:
        tensors = [self.all_tensors[tensor_name] for tensor_name in tensor_names]
        return tensors

    def execute(self, fusion_axis: str, fusion_step_size: int, input_tensors: List[Tensor]) -> Tensor:
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
        num_fusion_steps = math.ceil(fusion_size / fusion_step_size)
        self.all_tensors = {tensor.name: tensor for tensor in input_tensors}

        fx_input_tensors = self.get_tensors(self.fx.input_tensors)
        prev_O1 = self.fx.initialize_output(fx_input_tensors, fusion_axis=fusion_axis)
        curr_O1 = Tensor(name=prev_O1.name.replace("prev", "curr"), axes=prev_O1.axes, data=prev_O1.data)

        # NOTE: test with one operator fusion first
        gb_op = self.gbs[0]
        hb_op = self.hbs[0]
        prev_gb_out = gb_op.initialize_output(input_tensors=[curr_O1])
        curr_gb_out = Tensor(
            name=prev_gb_out.name.replace("prev", "curr"), axes=prev_gb_out.axes, data=prev_gb_out.data
        )

        hb_input_tensors = self.get_tensors(hb_op.input_tensors)
        hb_out = hb_op.initialize_output(hb_input_tensors, fusion_axis=fusion_axis)
        prev_O2 = Tensor(name="prev_O2", axes=hb_out.axes, data=hb_out.data)
        curr_O2 = Tensor(name="curr_O2", axes=prev_O2.axes, data=prev_O2.data)

        for fusion_step in range(num_fusion_steps):
            fusion_axis_start = fusion_step * fusion_step_size
            fx_forward_inputs = [prev_O1]
            for tensor in fx_input_tensors:
                fx_forward_inputs.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                )
            self.fx.forward(input_tensors=fx_forward_inputs, output_tensor=curr_O1)
            gb_op.forward(input_tensors=[curr_O1], output_tensor=curr_gb_out)
            hb_forward_inputs: List[Tensor] = []
            for tensor in hb_input_tensors:
                hb_forward_inputs.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                )
            hb_op.forward(input_tensors=hb_forward_inputs, output_tensor=hb_out)
            bias = curr_gb_out.data[:, None] * hb_out.data
            if fusion_step > 0:
                scale_factor = curr_gb_out.data / prev_gb_out.data
                curr_O2.data = scale_factor[:, None] * prev_O2.data + bias
            else:
                curr_O2.data = bias
            prev_gb_out.data = curr_gb_out.data
            prev_O1.data = curr_O1.data
            prev_O2.data = curr_O2.data
        return curr_O2
