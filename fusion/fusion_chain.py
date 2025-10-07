"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

import math
import re
from typing import Dict, List

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


def find_largest_curr_O_value(data: Dict[str, Tensor]) -> Tensor:
    """
    Find the largest key with pattern "curr_O_{int}" and return its value.

    Args:
        data: Dictionary with string keys and any values

    Returns:
        The value associated with the largest "curr_O_{int}" key,
        or None if no matching keys are found
    """
    pattern = re.compile(r"^curr_O_(\d+)$")
    max_num = -1
    for key in data.keys():
        match = pattern.match(key)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                max_key = key
    return data[max_key]


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

    def initialize_intermediates(self, fusion_axis: str, fusion_step_size: int):
        num_fused_operators = len(self.gbs)
        curr_intermediates: Dict[str, Tensor] = {}
        prev_intermediates: Dict[str, Tensor] = {}
        fx_input_tensors: List[Tensor] = []
        for tensor in self.get_tensors(self.fx.input_tensors):
            fx_input_tensors.append(tensor.get_axis_slice(fusion_axis, start=0, size=fusion_step_size))
        curr_intermediates["O_0"] = self.fx.initialize_output(
            fx_input_tensors, fusion_axis=fusion_axis, output_tensor_name="curr_O_0"
        )
        prev_intermediates["O_0"] = self.fx.initialize_output(
            fx_input_tensors, fusion_axis=fusion_axis, output_tensor_name="curr_O_0"
        )
        for operator_counter in range(1, num_fused_operators + 1):
            gb_op = self.gbs[operator_counter - 1]
            hb_op = self.hbs[operator_counter - 1]
            curr_intermediates[f"gB_O_{operator_counter-1}"] = gb_op.initialize_output(
                dependent_output=curr_intermediates[f"O_{operator_counter-1}"],
                output_tensor_name=f"gB_O_{operator_counter-1}",
            )
            prev_intermediates[f"gB_O_{operator_counter-1}"] = gb_op.initialize_output(
                dependent_output=prev_intermediates[f"O_{operator_counter-1}"],
                output_tensor_name=f"gB_O_{operator_counter-1}",
            )

            hb_input_tensors: List[Tensor] = []
            for tensor in self.get_tensors(hb_op.input_tensors):
                hb_input_tensors.append(tensor.get_axis_slice(fusion_axis, start=0, size=fusion_step_size))
            curr_intermediates[f"O_{operator_counter}"] = hb_op.initialize_output(
                input_tensors=hb_input_tensors, output_tensor_name=f"O_{operator_counter}", fusion_axis=fusion_axis
            )
            prev_intermediates[f"O_{operator_counter}"] = hb_op.initialize_output(
                input_tensors=hb_input_tensors, output_tensor_name=f"O_{operator_counter}", fusion_axis=fusion_axis
            )

        print(f"curr_intermediates = {curr_intermediates.keys()}")
        print(f"prev_intermediates = {prev_intermediates.keys()}")

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

        """
        curr: k
        prev: k-1
        """
        intermediates: Dict[str, Tensor] = {}
        for fusion_step in range(num_fusion_steps):
            print("-" * 20, f"Fusion step {fusion_step}", "-" * 20)
            fusion_axis_start = fusion_step * fusion_step_size

            fx_input_tensors: List[Tensor] = []
            for tensor in self.get_tensors(self.fx.input_tensors):
                fx_input_tensors.append(
                    tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                )
            if fusion_step == 0:
                intermediates["curr_O_0"] = self.fx.initialize_output(
                    fx_input_tensors, fusion_axis=fusion_axis, output_tensor_name="curr_O_0"
                )
                self.fx.forward(
                    prev_output=None,
                    input_tensors=fx_input_tensors,
                    curr_output=intermediates["curr_O_0"],
                    reduction_axis=fusion_axis,
                )
            else:
                self.fx.forward(
                    prev_output=intermediates["prev_O_0"],
                    input_tensors=fx_input_tensors,
                    curr_output=intermediates["curr_O_0"],
                    reduction_axis=fusion_axis,
                )
            for operator_counter in range(1, len(self.gbs) + 1):
                gb_op = self.gbs[operator_counter - 1]
                if fusion_step == 0:
                    intermediates[f"curr_gb_{operator_counter-1}"] = gb_op.initialize_output(
                        dependent_output=intermediates[f"curr_O_{operator_counter-1}"],
                        output_tensor_name=f"curr_gb_{operator_counter-1}",
                    )
                gb_op.forward(
                    dependent_output=intermediates[f"curr_O_{operator_counter-1}"],
                    output_tensor=intermediates[f"curr_gb_{operator_counter-1}"],
                )
                hb_op = self.hbs[operator_counter - 1]
                hb_input_tensors: List[Tensor] = []
                for tensor in self.get_tensors(hb_op.input_tensors):
                    hb_input_tensors.append(
                        tensor.get_axis_slice(fusion_axis, start=fusion_axis_start, size=fusion_step_size)
                    )

                if fusion_step == 0:
                    intermediates[f"hb_{operator_counter}"] = hb_op.initialize_output(
                        input_tensors=hb_input_tensors,
                        output_tensor_name=f"hb_{operator_counter}",
                        fusion_axis=fusion_axis,
                    )
                    hb_op.forward(input_tensors=hb_input_tensors, output_tensor=intermediates[f"hb_{operator_counter}"])
                    bias = broadcast_multiply(
                        intermediates[f"curr_gb_{operator_counter-1}"], intermediates[f"hb_{operator_counter}"]
                    )
                    intermediates[f"curr_O_{operator_counter}"] = Tensor(
                        name=f"curr_O_{operator_counter}", axes=intermediates[f"hb_{operator_counter}"].axes, data=bias
                    )
                else:
                    scale_factor = Tensor(
                        name="scale",
                        axes=intermediates[f"curr_gb_{operator_counter-1}"].axes,
                        data=(
                            intermediates[f"curr_gb_{operator_counter-1}"].data
                            / intermediates[f"prev_gb_{operator_counter-1}"].data
                        ),
                    )
                    bias = broadcast_multiply(
                        intermediates[f"curr_gb_{operator_counter-1}"], intermediates[f"hb_{operator_counter}"]
                    )
                    intermediates[f"curr_O_{operator_counter}"].data = (
                        broadcast_multiply(scale_factor, intermediates[f"prev_O_{operator_counter}"]) + bias
                    )
            print(intermediates.keys())

            """Update prev tensors"""
            curr_tensor_names = []
            for tensor_name in intermediates:
                if "curr" in tensor_name:
                    curr_tensor_names.append(tensor_name)
            for curr_tensor_name in curr_tensor_names:
                curr_tensor = intermediates[curr_tensor_name]
                prev_tensor_name = curr_tensor_name.replace("curr", "prev")
                intermediates[prev_tensor_name] = Tensor(
                    name=prev_tensor_name, axes=curr_tensor.axes, data=curr_tensor.data
                )
        output = find_largest_curr_O_value(intermediates)
        print(f"output = {output}")
        return output
