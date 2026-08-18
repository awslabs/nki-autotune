"""Folded batch/tile DMA store."""

import operator
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import torch
from torch.fx import GraphModule

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.grouped_tensor_copy import grouped_attention


class NKIFoldedStore(NKIOp):
    """Store independent two-dimensional tiles into one packed HBM tensor."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "T", "P", "F"), "dst": ("G", "T", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("G", "T", "P"), ("F",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "T": "tiles", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GTPF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "T": 1, "P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "tiles", "partitions"})
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def __init__(self, groups: int, tiles: int, partitions: int) -> None:
        """Configure packed batch, tile, and partition extents."""
        if min(groups, tiles, partitions) < 1:
            raise ValueError("folded store extents must be positive")
        super().__init__(groups=groups, tiles=tiles, partitions=partitions)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip source."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIFoldedStore(src=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return a copy of the packed source."""
        return np.array(kwargs["src"], copy=True)


def grouped_context_attention_graph(
    f_torch: object, input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> GraphModule | None:
    """Build grouped batched context attention without cross-batch contractions."""
    target, bound = getattr(f_torch, "function", f_torch), getattr(f_torch, "bound_kwargs", {})
    if getattr(target, "__name__", "") != "attention_cte_torch_ref" or bound.get("tp_out"):
        return None
    q_shape, k_shape, v_shape = (input_specs[name][0] for name in ("q", "k", "v"))
    groups, reduction, sequence = q_shape
    output_width = v_shape[-1]
    if not (
        groups > 1
        and k_shape == q_shape
        and v_shape[:2] == (groups, sequence)
        and max(reduction, output_width) <= 128
        and sequence % 512 == 0
    ):
        return None
    graph = torch.fx.Graph()
    inputs = {name: graph.placeholder(name) for name in input_specs}
    for node, (shape, _dtype) in zip(inputs.values(), input_specs.values(), strict=True):
        node.meta["example_value"] = SimpleNamespace(shape=shape)

    def call(operation: object, args: tuple[object, ...], shape: tuple[int, ...], **kwargs: object) -> torch.fx.Node:
        """Add one statically shaped function call."""
        node = graph.call_function(operation, args, kwargs)  # type: ignore[arg-type]
        node.meta["example_value"] = SimpleNamespace(shape=shape)
        return node

    dimensions = (groups, sequence // 128, sequence // 512, reduction, 128, 512, output_width)
    g, q, t, r, p, w, h = dimensions
    layouts = {
        "q": (("grouped_context", "q", *dimensions), (r, g * q * p)),
        "k": (("grouped_context", "k", *dimensions), (r, g * t * w)),
        "v": (("grouped_context", "v", *dimensions), (128, t * 4 * g * h)),
        "bound_min": (("grouped_context", "lower", *dimensions), (p, g * q)),
        "bound_max": (("grouped_context", "upper", *dimensions), (p, g * q)),
    }
    values = {
        name: call(operator.getitem, (inputs[name], transform), shape) for name, (transform, shape) in layouts.items()
    }
    output = call(
        grouped_attention,
        tuple(values[name] for name in ("q", "k", "v", "bound_min", "bound_max")),
        (g * q * p, h),
        groups=g,
        queries=q,
        tiles=t,
        reduction=r,
        partitions=p,
        width=w,
        output_width=h,
    )
    graph.output([output])
    return GraphModule(torch.nn.Module(), graph)


__all__ = ["NKIFoldedStore", "grouped_context_attention_graph"]
