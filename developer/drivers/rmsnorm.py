"""Driver for transform development on fixed row-wise RMSNorm followed by matmul."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from developer import developer
from developer.drivers._console import configure_console_logging
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.search.types import InputSpecs

M = 2048
K = 2048
N = 2048
EPSILON = 1e-6
INPUT_SPECS: InputSpecs = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_nkigym(lhs, rhs):
    """Define ``rmsnorm(lhs) @ rhs`` as an nkigym operator graph."""
    sbuf_lhs = NKILoad()(src=lhs)
    sbuf_square_sum = NKIActivationReduce(op="square", reduce_op="add")(data=sbuf_lhs)
    sbuf_rms_inverse = NKIActivation(op="rsqrt", scale=1.0 / K, bias=EPSILON)(data=sbuf_square_sum)
    sbuf_normalized = NKITensorScalar(op0="multiply")(data=sbuf_lhs, operand0=sbuf_rms_inverse)
    sbuf_normalized_T = NKIDMATranspose()(src=sbuf_normalized)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_output = NKIMatmul()(stationary=sbuf_normalized_T, moving=sbuf_rhs)
    sbuf_output = NKITensorCopy()(src=psum_output)
    hbm_output = NKIStore()(src=sbuf_output)
    return hbm_output


def _parser() -> argparse.ArgumentParser:
    """Build the fixed-workload command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH destination for the Trn2 profile worker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run transform development and return a process exit code."""
    configure_console_logging()
    arguments = _parser().parse_args(argv)
    result = developer(f_nkigym, INPUT_SPECS, arguments.host)
    print(f"verdict: {result.verdict}")
    print(f"run: {result.run_directory}")
    print(f"worktree: {result.worktree}")
    return 0 if result.verdict != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
