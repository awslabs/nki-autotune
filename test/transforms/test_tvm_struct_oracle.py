"""Tests for the Layer-B TVM structural oracle harness.

Skipped entirely when TVM is unavailable (shipped ``nkigym/`` never depends on TVM).
"""

import pytest

tvm = pytest.importorskip("tvm")
from test.transforms._tvm_struct_oracle import tvm_split_loopnest


def test_tvm_split_loopnest_shape():
    """A 16-extent loop split (4,4) -> nested (4,4) loops with binding i0*4 + i1."""
    nest = tvm_split_loopnest(extent=16, factors=[4, 4])
    assert nest.extents == [4, 4]
    assert nest.binding == "i0*4 + i1"
