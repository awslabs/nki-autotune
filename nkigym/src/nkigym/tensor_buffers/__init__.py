"""Tensor buffer allocation: on-chip SBUF and PSUM buffers."""

from nkigym.tensor_buffers.buffers import find_psum_tensors_needing_sbuf, producer_op_tiles, render_buffers

__all__ = ["find_psum_tensors_needing_sbuf", "producer_op_tiles", "render_buffers"]
