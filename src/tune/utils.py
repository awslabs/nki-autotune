import shutil
from typing import Dict, Tuple

import numpy as np
from neuronpy.core.compile import compile_to_neff
from neuronxcc.nki.compile import GenericKernel

from src.cache.directories import get_hash_name

traced_kernel = None


def set_kernel(kernel: GenericKernel):
    global traced_kernel
    traced_kernel = kernel


def create_and_compile_nki_kernel(kernel_args: Tuple[np.ndarray, ...], configs: Dict, output_dir: str):
    """Standalone function to create and compile a NKI kernel"""

    try:
        neff_name = get_hash_name(traced_kernel, kernel_args, configs)
        compile_dir = f"{output_dir}/{neff_name}"
        traced_kernel.specialize(*kernel_args, **configs)
        neff = compile_to_neff(
            trace_kernel=traced_kernel,
            output_dir=compile_dir,
            additional_compiler_args="--internal-tensorizer-opt-level=nki --target=trn1 --auto-cast=none",
        )
        ret_neff = f"{output_dir}/{neff_name}.neff"
        shutil.move(neff, ret_neff)
        shutil.rmtree(compile_dir)
        ret = ret_neff

    except Exception as e:
        ret = e
    return ret
