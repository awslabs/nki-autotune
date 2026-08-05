"""Driver for transform development on fixed single-head noncausal attention."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from developer import developer
from developer.drivers._console import configure_console_logging
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.search.types import InputSpecs

HEAD_DIM = 128
SEQUENCE_LENGTH = 16384
INPUT_SPECS: InputSpecs = {
    "query": ((HEAD_DIM, SEQUENCE_LENGTH), "bfloat16"),
    "key": ((HEAD_DIM, SEQUENCE_LENGTH), "bfloat16"),
    "value": ((SEQUENCE_LENGTH, HEAD_DIM), "bfloat16"),
}


@nkigym_kernel
def f_nkigym(query, key, value):
    """Define materialized scaled dot-product attention."""
    sbuf_query = NKILoad()(src=query)
    sbuf_key = NKILoad()(src=key)
    psum_scores = NKIMatmul()(stationary=sbuf_query, moving=sbuf_key)
    sbuf_scores = NKITensorCopy()(src=psum_scores)
    sbuf_scaled_scores = NKITensorScalar(op0="multiply")(data=sbuf_scores, operand0=HEAD_DIM**-0.5)
    sbuf_row_max = NKITensorReduce(op="maximum", axis=1)(data=sbuf_scaled_scores)
    sbuf_centered = NKITensorScalar(op0="subtract")(data=sbuf_scaled_scores, operand0=sbuf_row_max)
    sbuf_exp = NKIActivation(op="exp")(data=sbuf_centered)
    sbuf_row_sum = NKITensorReduce(op="add", axis=1)(data=sbuf_exp)
    sbuf_inverse_sum = NKIActivation(op="reciprocal")(data=sbuf_row_sum)
    sbuf_probability = NKITensorScalar(op0="multiply")(data=sbuf_exp, operand0=sbuf_inverse_sum)
    sbuf_probability_T = NKIDMATranspose()(src=sbuf_probability)
    sbuf_value = NKILoad()(src=value)
    psum_output = NKIMatmul()(stationary=sbuf_probability_T, moving=sbuf_value)
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
