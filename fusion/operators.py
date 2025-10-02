"""Operator definitions for the MegaFuse fusion framework."""

from abc import ABC, abstractmethod
from typing import List, Optional

from fusion.tensors import Tensor


class Operator(ABC):
    """Base class for all operators."""

    def __init__(self, input_tensors: List[str], **kwargs) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update output_tensor in place.
        """

    @abstractmethod
    def initialize_output(self, input_tensors: List[Tensor], fusion_axis: Optional[str] = None) -> Tensor:
        """Initialize the output tensor given input_tensors."""
