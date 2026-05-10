"""Tag assertion for the nkigym_kernel decorator wrapper."""

import numpy as np

from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore


def test_nkigym_kernel_sets_tag_on_wrapper():
    """Decorated wrappers expose __nkigym_kernel__ = True."""

    @nkigym_kernel
    def identity(x):
        sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        hbm = NKIAlloc(location="hbm", shape=(128, 128), dtype="bfloat16")()
        NKILoad()(src=x, dst=sbuf)
        NKIStore()(src=sbuf, dst=hbm)
        return hbm

    assert getattr(identity, "__nkigym_kernel__", False) is True


def test_plain_function_has_no_tag():
    """Undecorated callables lack the tag."""

    def plain(x: np.ndarray) -> np.ndarray:
        return x

    assert getattr(plain, "__nkigym_kernel__", False) is False
