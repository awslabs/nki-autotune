"""Tensor wrapper for fusion operations."""

from typing import List

import numpy as np


class Tensor:
    """Numpy array wrapper with fusion-specific helpers."""

    def __init__(self, name: str, axes: List[str], data: np.ndarray):
        self.name = name
        self.axes = axes
        self.data = data

    def __repr__(self) -> str:
        return f"Tensor(name={self.name}, axes={self.axes}, shape={self.data.shape}, dtype={self.data.dtype})"
