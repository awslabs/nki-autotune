"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

import math
import re
from typing import Dict, List

import numpy as np

from fusion.operators import Operator
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
        intermediates[old_tensor_name] = Tensor(axes=new_tensor.axes, data=new_tensor.data)


class FusionChain:
    """
    Implements generalized fusion for multiple operators following MegaFuse principles.

    This class supports arbitrary sequences of blocking and accumulation operations,
    enabling fusion of complex patterns.
    """

    def __init__(self, fx: Operator, bias_ops: List[Operator], scale_ops: List[Operator]):
        """
        Initialize FusionChain with a sequence of operators.
        """
        self.fx = fx
        self.bias_ops = bias_ops
        self.scale_ops = scale_ops
        assert len(bias_ops) == len(scale_ops), f"Number of bias and scale operators mismatch"
        self.fx.operator_index = 0
        for operator_index in range(len(self.bias_ops)):
            bias_op = self.bias_ops[operator_index]
            scale_op = self.scale_ops[operator_index]
            bias_op.operator_index = operator_index + 1
            scale_op.operator_index = operator_index + 1

    def get_input_tensors(self, tensor_names: List[str]) -> List[Tensor]:
        tensors = [self.input_tensors[tensor_name] for tensor_name in tensor_names]
        return tensors

    def execute(
        self, fusion_axis: str, fusion_step_size: int, input_tensors: Dict[str, Tensor], verbose: bool = False
    ) -> Tensor:
        """
        Execute the fusion chain on the provided inputs.

        Args:
            fusion_axis: The axis along which to perform fusion
            fusion_step_size: Granularity of the fusion axis each forward step
            input_tensors: Input tensors
            verbose: Whether to print iteration information

        Returns:
            Output tensor after applying all operators
        """
        fusion_size = None
        for tensor_name in input_tensors:
            tensor = input_tensors[tensor_name]
            if fusion_axis in tensor.axes:
                tensor_fusion_size = tensor.get_axis_size(fusion_axis)
                if not fusion_size:
                    fusion_size = tensor_fusion_size
                else:
                    assert fusion_size == tensor_fusion_size, f"Fusion size mismatch in input tensors"
        assert fusion_size, "Did not find fusion axis in the input tensors"
        num_fusion_steps = math.ceil(fusion_size / fusion_step_size)
        self.input_tensors = input_tensors

        # Collect iteration info for verbose output
        iteration_info = []

        intermediates: Dict[str, Tensor] = {}
        for fusion_step in range(num_fusion_steps):
            fusion_axis_start = fusion_step * fusion_step_size
            fusion_axis_end = min(fusion_axis_start + fusion_step_size, fusion_size)

            # Add fusion step header
            step_info = [
                f"Fusion Step {fusion_step + 1}/{num_fusion_steps}: {fusion_axis}[{fusion_axis_start}:{fusion_axis_end}]"
            ]

            fx_input_tensors: List[Tensor] = []
            for tensor in self.get_input_tensors(self.fx.input_tensors):
                fx_input_tensors.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                )
            if fusion_step == 0:
                intermediates.update(
                    self.fx.initialize_output(intermediate_tensors=intermediates, input_tensors=fx_input_tensors)
                )
            self.fx.forward(intermediate_tensors=intermediates, input_tensors=fx_input_tensors)
            step_info.append(f"  - fx: {self.fx.__class__.__name__}")

            for operator_counter in range(1, len(self.bias_ops) + 1):
                bias_op = self.bias_ops[operator_counter - 1]
                bias_input_tensors: List[Tensor] = []
                for tensor in self.get_input_tensors(bias_op.input_tensors):
                    bias_input_tensors.append(
                        tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                    )
                intermediates.update(
                    bias_op.initialize_output(intermediate_tensors=intermediates, input_tensors=bias_input_tensors)
                )
                bias_op.forward(intermediate_tensors=intermediates, input_tensors=bias_input_tensors)

                if fusion_step == 0:
                    step_info.append(f"  - bias_{operator_counter}: {bias_op.__class__.__name__}")
                    intermediates[f"O_{operator_counter}_new"] = Tensor(
                        axes=intermediates[f"bias_{operator_counter}"].axes,
                        data=intermediates[f"bias_{operator_counter}"].data,
                    )
                    step_info.append(f"  - O_{operator_counter}_new = bias_{operator_counter}")
                else:
                    scale_op = self.scale_ops[operator_counter - 1]
                    step_info.append(f"  - scale_{operator_counter}: {scale_op.__class__.__name__}")
                    step_info.append(f"  - bias_{operator_counter}: {bias_op.__class__.__name__}")
                    intermediates.update(
                        scale_op.initialize_output(intermediate_tensors=intermediates, input_tensors=[])
                    )
                    scale_op.forward(intermediate_tensors=intermediates, input_tensors=[])
                    intermediates[f"O_{operator_counter}_new"].data = (
                        broadcast_multiply(
                            intermediates[f"scale_{operator_counter}"], intermediates[f"O_{operator_counter}_old"]
                        )
                        + intermediates[f"bias_{operator_counter}"].data
                    )
                    step_info.append(
                        f"  - O_{operator_counter}_new = scale_{operator_counter} * O_{operator_counter}_old + bias_{operator_counter}"
                    )

            # Add step info to collection
            iteration_info.extend(step_info)
            if fusion_step < num_fusion_steps - 1:
                iteration_info.append("")  # Add blank line between steps

            update_intermediates(intermediates)

        # Print all iteration info at once if verbose
        if verbose:
            print("\n".join(iteration_info))
        output = find_largest_O_new_value(intermediates)
        return output
