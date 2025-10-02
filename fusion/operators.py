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
