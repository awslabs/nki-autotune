"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

import math
import re
from typing import Dict, List

import numpy as np

from fusion.operators import BiasOperator, FxOperator, ScaleOperator
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


def find_largest_O_new_value(data: Dict[str, Tensor]) -> Tensor:
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


def update_intermediates(intermediates: Dict[str, Tensor]):
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

    def __init__(self, fx: FxOperator, bias_ops: List[BiasOperator], scale_ops: List[ScaleOperator]):
        """
        Initialize FusionChain with a sequence of operators.
        """
        self.fx = fx
        self.bias_ops = bias_ops
        self.scale_ops = scale_ops
        assert len(bias_ops) == len(scale_ops), f"Number of bias and scale operators mismatch"

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

        intermediates: Dict[str, Tensor] = {"O_0_old": None}
        for fusion_step in range(num_fusion_steps):
            fusion_axis_start = fusion_step * fusion_step_size

            fx_input_tensors: List[Tensor] = []
            for tensor in self.get_tensors(self.fx.input_tensors):
                fx_input_tensors.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                )
            if fusion_step == 0:
                intermediates["O_0_new"] = self.fx.initialize_output(
                    input_tensors=fx_input_tensors, output_tensor_name="O_0_new"
                )
            self.fx.forward(
                output_old=intermediates["O_0_old"], input_tensors=fx_input_tensors, output_new=intermediates["O_0_new"]
            )
            for operator_counter in range(1, len(self.bias_ops) + 1):
                print(
                    "-" * 20,
                    f"Fusion step {fusion_step}/{num_fusion_steps} Operator {operator_counter}/{len(self.bias_ops)}",
                    "-" * 20,
                )
                bias_op = self.bias_ops[operator_counter - 1]
                bias_input_tensors: List[Tensor] = []
                for tensor in self.get_tensors(bias_op.input_tensors):
                    bias_input_tensors.append(
                        tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                    )
                bias_tensor = bias_op.initialize_output(
                    outputs_new=[intermediates[f"O_{i}_new"] for i in range(operator_counter)],
                    input_tensors=bias_input_tensors,
                    output_tensor_name=f"bias_{operator_counter}",
                )
                bias_op.forward(
                    outputs_new=[intermediates[f"O_{i}_new"] for i in range(operator_counter)],
                    input_tensors=bias_input_tensors,
                    output_tensor=bias_tensor,
                )
                print(bias_tensor)

                if fusion_step == 0:
                    intermediates[f"O_{operator_counter}_new"] = Tensor(
                        name=f"O_{operator_counter}_new", axes=bias_tensor.axes, data=bias_tensor.data
                    )
                else:
                    scale_op = self.scale_ops[operator_counter - 1]
                    scale_tensor = scale_op.initialize_output(
                        outputs_old=[intermediates[f"O_{i}_old"] for i in range(operator_counter)],
                        outputs_new=[intermediates[f"O_{i}_new"] for i in range(operator_counter)],
                        output_tensor_name=f"scale_{operator_counter}",
                    )
                    scale_op.forward(
                        outputs_old=[intermediates[f"O_{i}_old"] for i in range(operator_counter)],
                        outputs_new=[intermediates[f"O_{i}_new"] for i in range(operator_counter)],
                        output_tensor=scale_tensor,
                    )
                    print(scale_tensor)
                    intermediates[f"O_{operator_counter}_new"].data = (
                        broadcast_multiply(scale_tensor, intermediates[f"O_{operator_counter}_old"]) + bias_tensor.data
                    )
                print(intermediates[f"O_{operator_counter}_new"])
            update_intermediates(intermediates)
        output = find_largest_O_new_value(intermediates)
        return output
