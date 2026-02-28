"""GymOps for tiling infrastructure: allocation, slicing, and storing."""

import numpy as np

from nkigym.codegen.context import _LoweringContext, get_kwarg, value_to_nki
from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymStatement
from nkigym.ops.base import GymOp, Tensor


class EmptyOp(GymOp):
    """Allocate an uninitialized output tensor via np.empty.

    kwargs: shape0, shape1, dtype (all string literals).
    """

    op_name = "np_empty"
    inputs = ()
    outputs = (Tensor("result", ("P", "F")),)

    @classmethod
    def simulate(cls, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Allocate an empty array using output_shape and dtype from kwargs.

        Args:
            *args: Unused (EmptyOp has no inputs).
            **kwargs: Must include ``output_shape`` and ``dtype``.

        Returns:
            Uninitialized numpy array of the given shape and dtype.
        """
        dtype = np.dtype(kwargs["dtype"])  # type: ignore[arg-type]
        shape: tuple[int, ...] = kwargs["output_shape"]  # type: ignore[assignment]
        return np.empty(shape, dtype=dtype)

    @classmethod
    def output_shape(cls, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for EmptyOp since shape comes from kwargs."""
        raise NotImplementedError("EmptyOp shape is determined by kwargs, not input shapes")

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: _LoweringContext) -> list[str]:  # type: ignore[override]
        """Lower np_empty to ``nl.ndarray(..., buffer=nl.shared_hbm)``.

        Args:
            stmt: The np_empty statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        dtype = get_kwarg(stmt, "dtype")
        if dtype is None:
            raise ValueError(f"np_empty for '{stmt.output.name}' missing dtype kwarg")
        dtype_str = value_to_nki(dtype)
        shape_str = repr(stmt.output.shape)
        name = stmt.output.name
        ctx.buffers[name] = "SHARED_HBM"
        return [f"{name} = nl.ndarray({shape_str}, dtype={dtype_str}, buffer=nl.shared_hbm)"]


class SliceOp(GymOp):
    """Extract a contiguous slice from a tensor.

    kwargs: src (variable ref), start0, stop0, start1, stop1 (int literals).
    """

    op_name = "np_slice"
    inputs = (Tensor("src", ("P", "F")),)
    outputs = (Tensor("result", ("P", "F")),)

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
        """Not used for SliceOp since shape comes from kwargs."""
        raise NotImplementedError("SliceOp shape is determined by kwargs, not input shapes")

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: _LoweringContext) -> list[str]:  # type: ignore[override]
        """Lower np_slice to indexing or DMA copy.

        SBUF and PSUM sources use pure indexing (buffer inherited from
        source).  HBM sources require explicit SBUF allocation and
        ``nisa.dma_copy`` because NKI does not support direct HBM indexing.

        Args:
            stmt: The np_slice statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        if not isinstance(src_ref, TensorRef):
            raise ValueError(f"np_slice for '{stmt.output.name}' has non-TensorRef src")

        src_buffer = ctx.buffer_of(src_ref.name)
        out_name = stmt.output.name
        src_subscript = ctx.subscript(src_ref)

        lines: list[str] = [f"{out_name} = {src_subscript}"]
        ctx.buffers[out_name] = src_buffer
        if src_buffer in ("HBM", "SHARED_HBM"):
            ctx.buffers[out_name] = "SBUF"
            shape_str = repr(stmt.output.shape)
            out_sub = ctx.subscript(stmt.output)
            lines = [
                f"{out_name} = nl.ndarray({shape_str}, dtype={ctx.dtype}, buffer=nl.sbuf)",
                f"nisa.dma_copy(dst={out_sub}, src={src_subscript})",
            ]
        return lines


class StoreOp(GymOp):
    """Write a source tile into a destination tensor at a slice position.

    kwargs: src (variable ref), dst (variable ref),
            start0, stop0, start1, stop1 (int literals).
    The statement output should be the same variable as dst.
    """

    op_name = "np_store"
    inputs = (Tensor("src", ("P", "F")), Tensor("dst", ("P", "F")))
    outputs = (Tensor("result", ("P", "F")),)

    @classmethod
    def simulate(cls, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Write src into dst_view via mutation, return None.

        The caller resolves both ``src`` and ``dst`` TensorRefs before
        dispatching.  ``args[1]`` is a pre-sliced numpy view of the
        destination; mutation through the view updates the original
        array in the environment.

        Args:
            *args: ``(src_tile, dst_view)`` — source data and destination view.
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
    def to_nki(cls, stmt: GymStatement, ctx: _LoweringContext) -> list[str]:  # type: ignore[override]
        """Lower np_store to dma_copy, with PSUM staging if needed.

        Args:
            stmt: The np_store statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        dst_ref = get_kwarg(stmt, "dst")

        if not isinstance(src_ref, TensorRef) or not isinstance(dst_ref, TensorRef):
            raise ValueError("np_store missing src or dst operand")

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
