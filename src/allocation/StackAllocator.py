"""
Copyright (c) 2024, Amazon.com. All Rights Reserved

StackAllocator.py - Stack allocator for nki kernel

"""

from contextlib import contextmanager
from typing import List, Optional, cast

import numpy as np

from neuronxcc.starfish.penguin.common import align
from neuronxcc.starfish.penguin.targets.nki.metaclasses import sbuf
from neuronxcc.starfish.penguin.targets.nki.allocator import (
    Allocator,
    n_elts,
    AllocFunc,
)
from neuronxcc.starfish.penguin.targets.nki.stmts import StmtScope
from neuronxcc.starfish.penguin.targets.nki.stmts import (
    StmtScope,
    SyncProgramScope,
    KernelScope,
    FunctionScope,
)
from neuronxcc.starfish.penguin.targets.nki.tensors import KernelTensor, InstTile
from neuronxcc.starfish.penguin.targets.tonga.TongaTensor import (
    NeuronSBTensor,
    NeuronPSUMTensor,
    NeuronLocalTensor,
)
from neuronxcc.starfish.support.LogContext import print_info, print_debug


class StackFrame:
    def __init__(self, sbuf_base, psum_base, scope: StmtScope):
        self.sbuf_base = self.sbuf_next_ptr = sbuf_base
        self.psum_base = self.psum_next_ptr = psum_base
        self.scope = scope

    def allocate_sbuf(self, size_in_bytes, min_align):
        sbuf_next_ptr = self.sbuf_next_ptr = align(self.sbuf_next_ptr, min_align)
        self.sbuf_next_ptr += size_in_bytes
        return sbuf_next_ptr, self.sbuf_next_ptr

    def allocate_psum(self, num_banks, min_align):
        psum_next_ptr = self.psum_next_ptr = align(self.psum_next_ptr, min_align)
        self.psum_next_ptr += num_banks
        return psum_next_ptr, self.psum_next_ptr

    def __repr__(self):
        return (
            f"StackFrame({self.scope.name}:"
            f" sbuf[{self.sbuf_base}, {self.sbuf_next_ptr}]"
            f" psum[{self.psum_base}, {self.psum_next_ptr}])"
        )


class StackAllocator(Allocator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, logger_name="NKI Stack Allocator")
        self.stack: List[StackFrame] = []
        self.act_bias_tensor = None

    @property
    def top_frame(self):
        return self.stack[-1]

    @property
    def top_frame_or_none(self):
        try:
            return self.stack[-1]
        except IndexError:
            return None

    def log_exit_scope(self, scope):
        print_info(f"Exit scope: {scope.name} top_frame: {self.top_frame_or_none}")

    def log_enter_scope(self, scope: StmtScope):
        print_info(f"Enter new scope: {scope.name} top_frame: {self.top_frame_or_none}")

    def allocate_pending_tensors_in_scope(self, scope):
        with self.indent_filter.indent():
            print_info(f"DEBUG: Allocating pending tensors in scope: {scope.name}")
            for tensor in scope.pop_pending_tensors():
                if isinstance(tensor, KernelTensor):
                    self.allocate(tensor, tensor.allocation)
                    continue

                if isinstance(tensor, InstTile):
                    self.allocate_pending_inst_tile(tensor)
                    continue

                raise RuntimeError(f"Unexpected tensor type {type(tensor)}")

    def enter_scope(self, scope: StmtScope):
        try:
            # FIXME: Try to get data like interleave factor from the loop scope
            #  e.g. `for i in nl.affine_range(*args, interleave_factor=2)`
            parent = self.top_frame
        except IndexError:
            # First scope, there is no parent
            assert isinstance(
                scope, KernelScope
            ), f"Unexpected scope type {type(scope)} for first stack frame!"
            self.create_kernel_frame(scope)
            return

        self.allocate_pending_tensors_in_scope(scope.parent)

        if isinstance(scope, SyncProgramScope):
            return

        # FIXME: Also allocate bias when we enter new (none inline?)
        #  function scope

        # print(f'parent={parent}')

        self.stack.append(
            StackFrame(
                sbuf_base=parent.sbuf_next_ptr,
                psum_base=parent.psum_next_ptr,
                scope=scope,
            )
        )

    def exit_scope(self, scope: StmtScope):
        # print(f'exit {scope.name}')
        self.allocate_pending_tensors_in_scope(scope)

        if isinstance(scope, SyncProgramScope):
            return

        assert self.top_frame.scope == scope, "stack push/pop order mismatch!"
        self.stack.pop()

    @contextmanager
    def scope(self, scope: StmtScope):
        self.log_enter_scope(scope)
        with self.indent_filter.indent():
            self.enter_scope(scope)
            yield
            self.exit_scope(scope)
        self.log_exit_scope(scope)

    def create_kernel_frame(self, scope: KernelScope):
        self.stack.append(StackFrame(sbuf_base=0, psum_base=0, scope=scope))
        self.allocate_activation_bias_tensor(scope)

    def allocate_activation_bias_tensor(self, scope: FunctionScope):
        self.act_bias_tensor = scope.buffer(
            (128, 1),
            dtype=np.float32,
            buffer=sbuf,
            name="stack_allocator_bias_tensor",
            init_value=0.0,
        )

    def get_activation_bias_tensor(self):
        return self.act_bias_tensor

    def allow_pe_transpose(self):
        return False

    def allocate_sbuf_tensor(
        self, tensor: KernelTensor, allocation: Optional[AllocFunc]
    ):
        if allocation is None:
            self.allocate_sbuf_on_stack(tensor)
            return

        # FIXME: Modify the allocation for modulo allocation in the hidden dimensions
        super().allocate_sbuf_tensor(tensor, allocation=allocation)

    def allocate_psum_tensor(
        self, tensor: KernelTensor, allocation: Optional[AllocFunc]
    ):
        if allocation is None:
            self.allocate_psum_on_stack(tensor)
            return

        # FIXME: Modify the allocation for modulo allocation in the hidden dimensions
        super().allocate_psum_tensor(tensor, allocation=allocation)

    def allocate_sbuf_on_stack(self, tensor: KernelTensor):
        tensor_ir = cast(NeuronSBTensor, tensor.tensor)
        block_shape, allocated_block_shape = tensor.extract_allocated_block_shape()

        num_blocks = n_elts(allocated_block_shape)
        tensor_partition_size_in_bytes = tensor_ir.partition_size_in_bytes
        alloc_size_in_bytes = num_blocks * tensor_partition_size_in_bytes
        min_align = self.target.reqd_sb_align_in_bytes
        alloc_size_in_bytes = align(alloc_size_in_bytes, min_align)

        base_addr, next_ptr = self.top_frame.allocate_sbuf(
            alloc_size_in_bytes, min_align=min_align
        )

        print_info(
            f"Allocating sbuf tensor {tensor_ir.name}: base_addr={base_addr},"
            f" size={alloc_size_in_bytes} num_blocks={num_blocks}"
            f" allocated_block_shape={allocated_block_shape}"
            f" size_per_block={tensor_partition_size_in_bytes}"
            f" next_ptr={next_ptr}"
        )

        assert (
            self.top_frame.sbuf_next_ptr
            <= self.target.statebuf_usable_par_size_in_bytes
        ), "stack overflow on sbuf"

        block_rank = len(block_shape)
        self.assign_stack_allocation(
            tensor_ir,
            allocated_block_shape=allocated_block_shape,
            allocated_bank_shape=(1,) * block_rank,
            byte_addr=base_addr,
            bank_id=0,
            min_align=min_align,
        )

    def allocate_psum_on_stack(self, tensor: KernelTensor):
        tensor_ir = cast(NeuronPSUMTensor, tensor.tensor)
        block_shape, allocated_block_shape = tensor.extract_allocated_block_shape()

        num_blocks = n_elts(allocated_block_shape)
        assert (
            num_blocks <= self.target.psum_num_banks
        ), f"num_blocks {num_blocks} > psum_num_banks {self.target.psum_num_banks}"

        base_bank, next_bank = self.top_frame.allocate_psum(num_blocks, min_align=1)

        print_info(
            f"Allocating psum tensor {tensor_ir.name}: base_bank={base_bank},"
            f" num_blocks={num_blocks}"
            f" allocated_block_shape={allocated_block_shape}"
            f" size_per_block={self.target.psum_par_size_in_bytes}"
            f" next_bank={next_bank}"
        )

        min_align = self.target.reqd_psum_align_in_bytes

        block_rank = len(block_shape)

        self.assign_stack_allocation(
            tensor_ir,
            allocated_block_shape=(1,) * block_rank,
            allocated_bank_shape=allocated_block_shape,
            byte_addr=0,
            bank_id=base_bank,
            min_align=min_align,
        )

    def assign_stack_allocation(
        self,
        tensor_ir: NeuronLocalTensor,
        min_align,
        byte_addr,
        allocated_block_shape,
        bank_id,
        allocated_bank_shape,
    ):
        tensor_ir.allocation = tensor_ir.createAllocation(
            allocated_block_shape=allocated_block_shape,
            allocated_bank_shape=allocated_bank_shape,
            byte_addr=byte_addr,
            bank_id=bank_id,
            min_align=min_align,
        )
        self._any_allocation = True

    def allocate(self, tensor: KernelTensor, allocation: Optional[AllocFunc]):
        tensor_ir_class = tensor.tensor_ir_class

        if issubclass(tensor_ir_class, NeuronSBTensor):
            self.allocate_sbuf_tensor(tensor, allocation=allocation)
            return

        if issubclass(tensor_ir_class, NeuronPSUMTensor):
            self.allocate_psum_tensor(tensor, allocation=allocation)
            return
