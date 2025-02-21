import logging
from neuronxcc.starfish.penguin.targets.nki.decorators import update_allocator
from src.allocation.StackAllocator import StackAllocator


def enable_stack_allocator(func=None, log_level=logging.CRITICAL):
    """
    Use stack allocator to allocate the psum and sbuf tensors in the kernel.

    Must use together with skip_middle_end_transformations.

    .. code-block:: python

      from neuronxcc import nki

      @nki.compiler.enable_stack_allocator
      @nki.compiler.skip_middle_end_transformations
      @nki.jit
      def kernel(...):
        ...

    """
    decorating_function = update_allocator(StackAllocator, log_level=log_level)
    if func is None:
        return decorating_function

    return decorating_function(func)
