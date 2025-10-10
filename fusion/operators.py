"""Operator definitions for the MegaFuse fusion framework."""

from abc import ABC, abstractmethod
from typing import Dict, List

from fusion.tensors import Tensor


class Operator(ABC):
    def __init__(self, input_tensors: List[str], **kwargs) -> None:
        """_summary_

        Args:
            index (int): _description_
            input_tensors (List[str]): set of operator input tensor names that are directly from the kernel input
        """
        super().__init__()
        self.input_tensors = input_tensors

    @property
    def operator_index(self) -> int:
        """Get the operator index."""
        return self.index

    @operator_index.setter
    def operator_index(self, value: int) -> None:
        """Set the operator index."""
        self.index = value

    @abstractmethod
    def forward(self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]) -> None:
        """
        Execute the operator forward pass.
        intermediate_tensors are forward input tensors that are not kernel inputs
        Update output_tensors in place.
        """

    @abstractmethod
    def initialize_output(
        self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]
    ) -> Dict[str, Tensor]:
        """
        Initialize the operator output tensors.
        """
