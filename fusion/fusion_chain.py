"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

import math
import re
from typing import Dict, List, Optional

import numpy as np

from fusion.operators import FxOperator, GbOperator, HbOperator
from fusion.tensors import Tensor


def broadcast_multiply(tensor_a: Tensor, tensor_b: Tensor) -> np.ndarray:
    """
    Broadcast tensor_a to tensor_b's shape and perform element-wise multiplication.

    Args:
        tensor_a: First tensor (typically with fewer dimensions)
        tensor_b: Second tensor (target shape for broadcasting)

    Returns:
        New Tensor with the result of broadcasting tensor_a to tensor_b's shape
        and performing element-wise multiplication
    """
    dim_diff = len(tensor_b.data.shape) - len(tensor_a.data.shape)
    indexing = [slice(None)] * len(tensor_a.data.shape)
    for _ in range(dim_diff):
        indexing.append(None)
    broadcasted_a = tensor_a.data[tuple(indexing)]
    result_data = broadcasted_a * tensor_b.data
    return result_data


def find_largest_O_new_value(data: Dict[str, Optional[Tensor]]) -> Tensor:
    """
    Find the largest key with pattern "O_{int}_new" and return its value.

    Args:
        data: Dictionary with string keys and any values

    Returns:
        The value associated with the largest "O_{int}_new" key,
        or None if no matching keys are found
    """
    pattern = re.compile(r"^O_(\d+)_new$")
    max_num = -1
    for key in data.keys():
        match = pattern.match(key)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                max_key = key
    out = data[max_key]
    assert out
    return out


def update_intermediates(intermediates: Dict[str, Optional[Tensor]]):
    new_tensor_names: List[str] = []
    for tensor_name in intermediates:
        if "new" in tensor_name:
            new_tensor_names.append(tensor_name)
    for new_tensor_name in new_tensor_names:
        old_tensor_name = new_tensor_name.replace("new", "old")
        new_tensor = intermediates[new_tensor_name]
        assert new_tensor
        intermediates[old_tensor_name] = Tensor(name=old_tensor_name, axes=new_tensor.axes, data=new_tensor.data)


class FusionChain:
    """
    Implements generalized fusion for multiple operators following MegaFuse principles.

    This class supports arbitrary sequences of blocking and accumulation operations,
    enabling fusion of complex patterns.
    """

    def __init__(self, fx: FxOperator, gbs: List[GbOperator], hbs: List[HbOperator]):
        """
        Initialize FusionChain with a sequence of operators.
        """
        self.fx = fx
        self.gbs = gbs
        self.hbs = hbs
        assert len(gbs) == len(hbs), f"Number of gB and hB functions mismatch"

    def get_tensors(self, tensor_names: List[str]) -> List[Tensor]:
        tensors = [self.input_tensors[tensor_name] for tensor_name in tensor_names]
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
        self.input_tensors = {tensor.name: tensor for tensor in input_tensors}
        print(f"input_tensors = {self.input_tensors}")

        intermediates: Dict[str, Optional[Tensor]] = {"O_0_old": None}
        for fusion_step in range(num_fusion_steps):
            print("-" * 20, f"Fusion step {fusion_step}", "-" * 20)
            fusion_axis_start = fusion_step * fusion_step_size

            fx_input_tensors: List[Tensor] = []
            for tensor in self.get_tensors(self.fx.input_tensors):
                fx_input_tensors.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                )
            if fusion_step == 0:
                intermediates["O_0_new"] = self.fx.initialize_output(
                    fx_input_tensors, fusion_axis=fusion_axis, output_tensor_name="O_0_new"
                )
            assert intermediates["O_0_new"]
            self.fx.forward(
                prev_output=intermediates["O_0_old"],
                input_tensors=fx_input_tensors,
                curr_output=intermediates["O_0_new"],
                reduction_axis=fusion_axis,
            )
            for operator_counter in range(1, len(self.gbs) + 1):
                print(f"Fusion operator {operator_counter}")
                gb_op = self.gbs[operator_counter - 1]
                if fusion_step == 0:
                    intermediates[f"g_{operator_counter}_new"] = gb_op.initialize_output(
                        dependent_output=intermediates[f"O_{operator_counter-1}_new"],
                        output_tensor_name=f"g_{operator_counter}_new",
                    )
                gb_op.forward(
                    dependent_output=intermediates[f"O_{operator_counter-1}_new"],
                    output_tensor=intermediates[f"g_{operator_counter}_new"],
                )

                hb_op = self.hbs[operator_counter - 1]
                hb_input_tensors: List[Tensor] = []
                for tensor in self.get_tensors(hb_op.input_tensors):
                    hb_input_tensors.append(
                        tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                    )
                if fusion_step == 0:
                    h_out = hb_op.initialize_output(
                        input_tensors=hb_input_tensors,
                        output_tensor_name=f"h_{operator_counter}",
                        fusion_axis=fusion_axis,
                    )
                hb_op.forward(input_tensors=hb_input_tensors, output_tensor=h_out, fusion_axis=fusion_axis)
                # FIXME: Should combine gB, hB as one operator to compute the bias
                bias = broadcast_multiply(intermediates[f"g_{operator_counter}_new"], h_out)

                if fusion_step == 0:
                    intermediates[f"O_{operator_counter}_new"] = Tensor(
                        name=f"O_{operator_counter}_new", axes=h_out.axes, data=bias
                    )
                else:
                    scale_factor = Tensor(
                        name="scale",
                        axes=intermediates[f"g_{operator_counter}_new"].axes,
                        data=(
                            intermediates[f"g_{operator_counter}_new"].data
                            / intermediates[f"g_{operator_counter}_old"].data
                        ),
                    )
                    intermediates[f"O_{operator_counter}_new"].data = (
                        broadcast_multiply(scale_factor, intermediates[f"O_{operator_counter}_old"]) + bias
                    )
            update_intermediates(intermediates)
        output = find_largest_O_new_value(intermediates)
        print(f"output = {output}")
        return output
