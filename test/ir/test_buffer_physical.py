"""Tests for physical buffer allocation shapes and dtypes."""

import pytest

from nkigym.ir.tree import Buffer


def test_physical_shapes_include_tiles_versions_and_hbm_rules() -> None:
    """Physical shapes expand on-chip tiles and versions but preserve HBM shapes."""
    sbuf = Buffer(name="s", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    psum = Buffer(name="p", shape=(256, 512), dtype="float32", location="psum")
    hbm = Buffer(name="h", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    assert sbuf.physical_shape() == (128, 16, 2048)
    assert psum.physical_shape() == (128, 2, 512)
    assert hbm.physical_shape() == (2048, 2048)

    default = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    assert default.versions == 1
    assert default.physical_shape() == (128, 1, 2048)
    versioned_psum = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum", versions=2)
    versioned_sbuf = Buffer(name="sbuf_x", shape=(256, 512), dtype="bfloat16", location="sbuf", versions=2)
    assert versioned_psum.physical_shape() == (128, 2, 2048)
    assert versioned_sbuf.physical_shape() == (128, 4, 512)
    versioned_hbm = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm", versions=2)
    assert versioned_hbm.physical_shape() == (2048, 2048)


def test_physical_dtype_honors_storage_override() -> None:
    """Physical dtype is logical by default and follows an explicit producer override."""
    transpose_psum = Buffer(name="t", shape=(256, 512), dtype="bfloat16", location="psum")
    matmul_psum = Buffer(name="p", shape=(256, 512), dtype="bfloat16", location="psum", storage_dtype="float32")
    sbuf = Buffer(name="s", shape=(256, 512), dtype="bfloat16", location="sbuf")
    hbm = Buffer(name="h", shape=(256, 512), dtype="bfloat16", location="shared_hbm")
    assert transpose_psum.physical_dtype() == "bfloat16"
    assert matmul_psum.physical_dtype() == "float32"
    assert sbuf.physical_dtype() == "bfloat16"
    assert hbm.physical_dtype() == "bfloat16"


def test_list_length_refactorizes_the_tile_dimension() -> None:
    """List length defaults to one and otherwise divides the on-chip tile dimension."""
    packed = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    assert packed.list_len == 1
    assert packed.physical_shape() == (128, 16, 2048)
    assert packed.per_tile_physical_shape() == (128, 16, 2048)

    listed = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=16)
    assert listed.physical_shape() == (128, 16, 512)
    assert listed.per_tile_physical_shape() == (128, 1, 512)
    identity = Buffer(name="p", shape=(2048, 2048), dtype="float32", location="psum")
    assert identity.per_tile_physical_shape() == identity.physical_shape()


def test_invalid_list_geometries_raise() -> None:
    """HBM buffers cannot split tiles and versions cannot combine with lists."""
    hbm = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm", list_len=4)
    with pytest.raises(AssertionError):
        hbm.per_tile_physical_shape()
    versioned_list = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", versions=2, list_len=16)
    with pytest.raises(AssertionError):
        versioned_list.per_tile_physical_shape()
