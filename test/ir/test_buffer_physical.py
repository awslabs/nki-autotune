"""Tests for physical buffer allocation shapes and dtypes."""

import pytest

from nkigym.ir.tree import Buffer


def test_buffer_physical_shape_expands_sbuf_psum_and_passes_hbm_through():
    """physical_shape() folds the leading extent into a 128-partition tile count."""
    sbuf = Buffer(name="s", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    psum = Buffer(name="p", shape=(256, 512), dtype="float32", location="psum")
    hbm = Buffer(name="h", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    assert sbuf.physical_shape() == (128, 16, 2048)
    assert psum.physical_shape() == (128, 2, 512)
    assert hbm.physical_shape() == (2048, 2048)


def test_buffer_versions_default_unchanged():
    """versions defaults to 1 and leaves physical_shape unchanged."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    assert buf.versions == 1
    assert buf.physical_shape() == (128, 1, 2048)


def test_buffer_versions_grows_tile_dim():
    """versions=2 doubles the tile dimension for sbuf and psum."""
    psum = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum", versions=2)
    sbuf = Buffer(name="sbuf_x", shape=(256, 512), dtype="bfloat16", location="sbuf", versions=2)
    assert psum.physical_shape() == (128, 2, 2048)
    assert sbuf.physical_shape() == (128, 4, 512)


def test_buffer_versions_hbm_unaffected():
    """shared_hbm keeps its logical shape regardless of versions."""
    hbm = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm", versions=2)
    assert hbm.physical_shape() == (2048, 2048)


def test_buffer_physical_dtype_overrides_psum_to_fp32():
    """physical_dtype() returns fp32 for psum and the logical dtype otherwise."""
    psum = Buffer(name="p", shape=(256, 512), dtype="bfloat16", location="psum")
    sbuf = Buffer(name="s", shape=(256, 512), dtype="bfloat16", location="sbuf")
    hbm = Buffer(name="h", shape=(256, 512), dtype="bfloat16", location="shared_hbm")
    assert psum.physical_dtype() == "float32"
    assert sbuf.physical_dtype() == "bfloat16"
    assert hbm.physical_dtype() == "bfloat16"


def test_buffer_list_len_default_unchanged():
    """list_len defaults to 1 and leaves physical shapes unchanged."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    assert buf.list_len == 1
    assert buf.physical_shape() == (128, 16, 2048)
    assert buf.per_tile_physical_shape() == (128, 16, 2048)


def test_per_tile_physical_shape_splits_middle_dim():
    """list_len greater than one divides the tile dimension."""
    buf = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=16)
    assert buf.physical_shape() == (128, 16, 512)
    assert buf.per_tile_physical_shape() == (128, 1, 512)


def test_per_tile_physical_shape_identity_when_one():
    """list_len=1 leaves the physical shape unchanged."""
    buf = Buffer(name="p", shape=(2048, 2048), dtype="float32", location="psum")
    assert buf.per_tile_physical_shape() == buf.physical_shape()


def test_per_tile_physical_shape_rejects_hbm_split():
    """shared_hbm has no tile axis and cannot use list_len greater than one."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm", list_len=4)
    with pytest.raises(AssertionError):
        buf.per_tile_physical_shape()


def test_per_tile_physical_shape_rejects_versions_and_tiles_combo():
    """versions and list_len cannot both exceed one."""
    buf = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", versions=2, list_len=16)
    with pytest.raises(AssertionError):
        buf.per_tile_physical_shape()
