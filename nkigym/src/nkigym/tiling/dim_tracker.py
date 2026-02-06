"""Dimension tracking infrastructure for tiling analysis.

This module provides _DimTracker, a union-find data structure for tracking
dimension equivalences during symbolic tracing.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TracedOp:
    """A traced operation during symbolic execution.

    Attributes:
        op_name: Name of the operation (e.g., "matmul").
        inputs: List of input tensor names.
        output: Output tensor name.
    """

    op_name: str
    inputs: list[str]
    output: str


class _DimTracker:
    """Tracks dimension IDs and equivalences using union-find.

    Dimensions are unified during ops based on op semantics (e.g., matmul
    contracts a.dims[1] with b.dims[0]). The union-find structure tracks
    which dimensions are equivalent.
    """

    def __init__(self) -> None:
        """Initialize an empty dimension tracker."""
        self.next_id: int = 0
        self.dim_sizes: dict[str, int] = {}
        self.dim_order: list[str] = []
        self._parent: dict[str, str] = {}
        self.ops: list[TracedOp] = []
        self._next_intermediate_id: int = 0

    def record_op(self, op_name: str, inputs: list[str], output: str) -> None:
        """Record an operation during tracing.

        Args:
            op_name: Name of the operation (e.g., "matmul").
            inputs: List of input tensor names.
            output: Output tensor name.
        """
        self.ops.append(TracedOp(op_name=op_name, inputs=inputs, output=output))

    def new_intermediate_name(self) -> str:
        """Generate a unique name for an intermediate tensor.

        Returns:
            Generated name like "tensor_N".
        """
        name = f"tensor_{self._next_intermediate_id}"
        self._next_intermediate_id += 1
        return name

    def new_dim(self, size: int) -> str:
        """Create a new dimension ID with the given size.

        Args:
            size: Size of the dimension.

        Returns:
            The new dimension ID (e.g., "d0", "d1").
        """
        dim_id = f"d{self.next_id}"
        self.next_id += 1
        self.dim_sizes[dim_id] = size
        self.dim_order.append(dim_id)
        self._parent[dim_id] = dim_id
        return dim_id

    def find(self, dim_id: str) -> str:
        """Find the canonical representative for a dimension (with path compression).

        Args:
            dim_id: Dimension ID to look up.

        Returns:
            The canonical representative dimension ID.
        """
        if self._parent[dim_id] != dim_id:
            self._parent[dim_id] = self.find(self._parent[dim_id])
        return self._parent[dim_id]

    def unify(self, dim_a: str, dim_b: str) -> None:
        """Unify two dimensions, making them equivalent.

        The earlier dimension (by creation order) becomes the canonical representative.

        Args:
            dim_a: First dimension ID.
            dim_b: Second dimension ID.
        """
        root_a = self.find(dim_a)
        root_b = self.find(dim_b)
        if root_a != root_b:
            idx_a = self.dim_order.index(root_a)
            idx_b = self.dim_order.index(root_b)
            if idx_a < idx_b:
                self._parent[root_b] = root_a
            else:
                self._parent[root_a] = root_b

    def get_canonical_dims(self, dims: list[str]) -> list[str]:
        """Resolve each dimension ID to its canonical representative.

        Args:
            dims: List of dimension IDs (e.g., ["d0", "d1", "d2"]).

        Returns:
            List of canonical IDs. Unified dimensions map to the same canonical ID.
            For example, if d1 was unified with d0, ["d0", "d1"] returns ["d0", "d0"].
        """
        return [self.find(d) for d in dims]

    def get_canonical_order(self) -> list[str]:
        """Get dimension order with only canonical (non-unified) dimensions.

        Returns:
            List of canonical dimension IDs in discovery order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for dim_id in self.dim_order:
            canonical = self.find(dim_id)
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def __repr__(self) -> str:
        """Return formatted string representation."""
        dims = []
        for dim_id in self.dim_order:
            size = self.dim_sizes[dim_id]
            canonical = self.find(dim_id)
            if canonical != dim_id:
                dims.append(f"{dim_id}={size} (={canonical})")
            else:
                dims.append(f"{dim_id}={size}")
        return f"_DimTracker([{', '.join(dims)}])"
