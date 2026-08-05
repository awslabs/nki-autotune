"""Driver for transform development on a fixed ``lhs_T.T @ rhs`` workload."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from developer import developer
from developer.drivers._console import configure_console_logging
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.search.types import InputSpecs

SIZE = 2048
INPUT_SPECS: InputSpecs = {"lhs_T": ((SIZE, SIZE), "bfloat16"), "rhs": ((SIZE, SIZE), "bfloat16")}


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """Define ``lhs_T.T @ rhs`` as an nkigym operator graph."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_product = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_product = NKITensorCopy()(src=psum_product)
    hbm_output = NKIStore()(src=sbuf_product)
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
