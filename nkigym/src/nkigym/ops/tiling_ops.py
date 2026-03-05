"""GymOps for tiling infrastructure: allocation, loading, indexing, and storing."""

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymStatement
from nkigym.ops.base import GymOp, Tensor
from nkigym.program_to_nki.context import get_kwarg, value_to_nki


class AllocateOp(GymOp):
    """Allocate an uninitialized output tensor.

    kwargs: dtype (numpy dtype type).
    """

    op_name = "allocate"
    inputs = ()
    outputs = (Tensor("result", ("P", "F")),)

    @classmethod
    def simulate(cls, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Allocate an empty array using output_shape and dtype from kwargs.

        Args:
            *args: Unused (AllocateOp has no inputs).
            **kwargs: Must include ``output_shape`` and ``dtype``.

        Returns:
            Uninitialized numpy array of the given shape and dtype.
        """
        dtype = np.dtype(kwargs["dtype"])  # type: ignore[arg-type]
        shape: tuple[int, ...] = kwargs["output_shape"]  # type: ignore[assignment]
        return np.empty(shape, dtype=dtype)

    @classmethod
    def output_shape(cls, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for AllocateOp since shape comes from kwargs."""
        raise NotImplementedError("AllocateOp shape is determined by kwargs, not input shapes")

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: "_LoweringContext") -> list[str]:  # type: ignore[override]
        """Lower allocate to ``nl.ndarray(..., buffer=nl.shared_hbm)``.

        Args:
            stmt: The allocate statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        dtype = get_kwarg(stmt, "dtype")
        if dtype is None:
            raise ValueError(f"allocate for '{stmt.output.name}' missing dtype kwarg")
        dtype_str = value_to_nki(dtype)
        shape_str = repr(stmt.output.shape)
        name = stmt.output.name
        ctx.buffers[name] = "SHARED_HBM"
        return [f"{name} = nl.ndarray({shape_str}, dtype={dtype_str}, buffer=nl.shared_hbm)"]


class LoadOp(GymOp):
    """Load a contiguous tile from HBM into on-chip memory.

    kwargs: src (TensorRef with HBM source and slices).
    """

    op_name = "load"
    inputs = (Tensor("src", ("P", "F")),)
    outputs = (Tensor("result", ("P", "F")),)
    tile_limits = {"P": 128}

    @classmethod
    def simulate(cls, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Return the pre-sliced source array.

        The caller (GymProgram.__call__) resolves TensorRef slices before
        dispatching, so the source is already the correct view.

        Args:
            *args: Single source array (already sliced by the resolver).
            **kwargs: Ignored.

        Returns:
            The source array unchanged.
        """
        return args[0]

    @classmethod
    def output_shape(cls, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for LoadOp since shape comes from kwargs."""
        raise NotImplementedError("LoadOp shape is determined by kwargs, not input shapes")

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: "_LoweringContext") -> list[str]:  # type: ignore[override]
        """Lower load to SBUF allocation + nisa.dma_copy.

        Args:
            stmt: The load statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        if not isinstance(src_ref, TensorRef):
            raise ValueError(f"load for '{stmt.output.name}' has non-TensorRef src")

        out_name = stmt.output.name
        src_subscript = ctx.subscript(src_ref)
        shape_str = repr(stmt.output.shape)
        out_sub = ctx.subscript(stmt.output)
        ctx.buffers[out_name] = "SBUF"
        return [
            f"{out_name} = nl.ndarray({shape_str}, dtype={ctx.dtype}, buffer=nl.sbuf)",
            f"nisa.dma_copy(dst={out_sub}, src={src_subscript})",
        ]


class IndexingOp(GymOp):
    """Pure indexing into on-chip memory (SBUF or PSUM).

    kwargs: src (TensorRef with on-chip source and slices).
    No data movement --- just a view into an existing buffer.
    """

    op_name = "indexing"
    inputs = (Tensor("src", ("P", "F")),)
    outputs = (Tensor("result", ("P", "F")),)

    @classmethod
    def simulate(cls, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Return the pre-sliced source array.

        Args:
            *args: Single source array (already sliced by the resolver).
            **kwargs: Ignored.

        Returns:
            The source array unchanged.
        """
        return args[0]

    @classmethod
    def output_shape(cls, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for IndexingOp since shape comes from kwargs."""
        raise NotImplementedError("IndexingOp shape is determined by kwargs, not input shapes")

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: "_LoweringContext") -> list[str]:  # type: ignore[override]
        """Lower indexing to pure subscript (no data movement).

        Buffer location is inherited from the source.

        Args:
            stmt: The indexing statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        if not isinstance(src_ref, TensorRef):
            raise ValueError(f"indexing for '{stmt.output.name}' has non-TensorRef src")

        src_buffer = ctx.buffer_of(src_ref.name)
        out_name = stmt.output.name
        src_subscript = ctx.subscript(src_ref)
        ctx.buffers[out_name] = src_buffer
        return [f"{out_name} = {src_subscript}"]


class StoreOp(GymOp):
    """Write a source tile into a destination tensor at a slice position.

    kwargs: src (TensorRef), dst (TensorRef).
    The statement output should be the same variable as dst.
    """

    op_name = "store"
    inputs = (Tensor("src", ("P", "F")), Tensor("dst", ("P", "F")))
    outputs = (Tensor("result", ("P", "F")),)
    tile_limits = {"P": 128}

    @classmethod
    def simulate(cls, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Write src into dst_view via mutation, return None.

        Args:
            *args: ``(src_tile, dst_view)`` --- source data and destination view.
            **kwargs: Ignored.

        Returns:
            None to signal no env assignment needed.
        """
        args[1][:] = args[0]
        return None  # type: ignore[return-value]

    @classmethod
    def output_shape(cls, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for StoreOp since shape comes from kwargs."""
        raise NotImplementedError("StoreOp shape is determined by kwargs, not input shapes")

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: "_LoweringContext") -> list[str]:  # type: ignore[override]
        """Lower store to dma_copy, with PSUM staging if needed.

        Args:
            stmt: The store statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        dst_ref = get_kwarg(stmt, "dst")

        if not isinstance(src_ref, TensorRef) or not isinstance(dst_ref, TensorRef):
            raise ValueError("store missing src or dst operand")

        src_subscript = ctx.subscript(src_ref)
        dst_subscript = ctx.subscript(dst_ref)

        resolved_src = ctx.resolve(src_ref.name)
        src_buffer = ctx.buffers.get(resolved_src, "SBUF")

        lines: list[str] = [f"nisa.dma_copy(dst={dst_subscript}, src={src_subscript})"]
        if src_buffer == "PSUM":
            staging_name = f"tensor_{ctx.tensor_counter}"
            ctx.tensor_counter += 1
            shape_str = repr(src_ref.shape)
            parts = ", ".join(f"0:{s}" for s in src_ref.shape)
            staging_sub = f"{staging_name}[{parts}]"
            lines = [
                f"{staging_name} = nl.ndarray({shape_str}, dtype={ctx.dtype}, buffer=nl.sbuf)",
                f"nisa.tensor_copy(dst={staging_sub}, src={src_subscript})",
                f"nisa.dma_copy(dst={dst_subscript}, src={staging_sub})",
            ]
        return lines
