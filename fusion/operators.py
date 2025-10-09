"""Operator definitions for the MegaFuse fusion framework."""

from abc import ABC, abstractmethod
from typing import List, Optional

from fusion.tensors import Tensor


class FxOperator(ABC):
    def __init__(self, input_tensors: List[str], **kwargs) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, output_old: Optional[Tensor], input_tensors: List[Tensor], output_new: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update output_new in place.
        """

    @abstractmethod
    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str) -> Tensor:
        """Initialize the output tensor given input_tensors."""


class BiasOperator(ABC):
    def __init__(self, input_tensors: List[str], **kwargs) -> None:
        super().__init__()
        self.input_tensors = input_tensors

    @abstractmethod
    def forward(self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update output_tensor in place.
        """

    @abstractmethod
    def initialize_output(
        self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        """Initialize the output tensor."""


class ScaleOperator(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor: Tensor) -> None:
        """
        Execute the operator forward pass.
        Update output_tensor in place.
        """

    @abstractmethod
    def initialize_output(
        self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        """Initialize the output tensor."""
