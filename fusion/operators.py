"""Operator definitions for the MegaFuse fusion framework."""

from abc import ABC, abstractmethod
from typing import List

from fusion.tensors import Tensor


class Operator(ABC):
    """Base class for all operators."""

    def __init__(self, input_tensors: List[str]) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, inputs: List[Tensor], next_output: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update next_output in place.
        """

    @abstractmethod
    def initialize_output(self, fusion_axis: str, inputs: List[Tensor]) -> Tensor:
        """Initialize the output tensor given inputs."""


class FxOperator(ABC):
    def __init__(self, input_tensors: List[str]) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, curr_O1: Tensor, prev_O1: Tensor, input_tensors: List[Tensor]) -> None:
        """
        Execute the operator forward pass.
        Update curr_O1 in place.

        Args:
            curr_O1 (Tensor): O1_k
            prev_O1 (Tensor): O1_k-1
            input_tensors (List[Tensor]): V1_k
        """

    @abstractmethod
    def initialize_output(self, fusion_axis: str, input_tensors: List[Tensor]) -> Tensor:
        """Initialize the output tensor given input_tensors."""
