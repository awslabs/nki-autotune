"""Operator definitions for the MegaFuse fusion framework."""

from abc import ABC, abstractmethod
from typing import List

from fusion.tensors import Tensor


class StatefulOperator(ABC):
    def __init__(self, input_tensors: List[str]) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, prev_output: Tensor, input_tensors: List[Tensor], curr_output: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update curr_output in place.
        """

    @abstractmethod
    def initialize_output(self, input_tensors: List[Tensor], fusion_axis: str, output_tensor_name: str) -> Tensor:
        """Initialize the output tensor given input_tensors."""


class HbOperator(ABC):
    def __init__(self, input_tensors: List[str]) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update output in place.
        """

    @abstractmethod
    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str, **kwargs) -> Tensor:
        """Initialize the output tensor given input_tensors."""


class GbOperator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update output in place.
        """

    @abstractmethod
    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str) -> Tensor:
        """Initialize the output tensor given input_tensors."""
