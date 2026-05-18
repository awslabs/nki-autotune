"""Shared kernel + input_specs fixture for environment tests.

Re-exports the matmul fixture from ``test.transforms._fixtures`` so the
environment test suite uses the same canonical kernel as the transform suite.
"""

from __future__ import annotations

from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, f_matmul

__all__ = ["INPUT_SPECS", "build_canonical_ir", "f_matmul"]
