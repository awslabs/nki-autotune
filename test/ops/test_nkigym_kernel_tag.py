"""Tag assertion for the nkigym_kernel decorator wrapper."""

import numpy as np
import pytest

from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore


def test_nkigym_kernel_sets_tag_on_wrapper():
    """Decorated wrappers expose __nkigym_kernel__ = True."""

    @nkigym_kernel
    def identity(x):
        sbuf = NKILoad()(src=x)
        hbm = NKIStore()(src=sbuf)
        return hbm

    assert getattr(identity, "__nkigym_kernel__", False) is True


def test_plain_function_has_no_tag():
    """Undecorated callables lack the tag."""

    def plain(x: np.ndarray) -> np.ndarray:
        return x

    assert getattr(plain, "__nkigym_kernel__", False) is False


def test_nkigym_kernel_rejects_non_stored_non_hbm_return():
    """The decorator rejects kernels returning an sbuf-roled array."""

    @nkigym_kernel
    def bad_return(x):
        return NKILoad()(src=x)

    with pytest.raises(TypeError, match="returned role='sbuf'"):
        bad_return(np.ones((16, 16), dtype=np.float32))
