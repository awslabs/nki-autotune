"""Tests for first-class buffer infrastructure: NKIOp ClassVars, Tensor.location,
SBlock.reads_writes, and their interactions through the canonical builder."""

import numpy as np

from nkigym.ir.ir import SBlock, Tensor
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp


def test_nkiop_defaults_rmw_and_rfactor_and_input_operands():
    """Every NKIOp subclass should inherit safe defaults: no RMW, no rfactor recipe,
    no declared input operands (subclasses override per-op)."""
    assert NKIOp.RMW_OPERANDS == frozenset()
    assert NKIOp.RFACTOR_RECIPE is None
    assert NKIOp.INPUT_OPERANDS == frozenset()


def test_tensor_location_field_defaults_and_accepts_literal():
    """Tensor gains a location field; every tensor must declare hbm/sbuf/psum."""
    t = Tensor(
        name="psum_acc", dim_ids=("d0", "d1"), shape=(128, 512), dtype="float32", origin="intermediate", location="psum"
    )
    assert t.location == "psum"


def test_body_leaf_reads_writes_defaults_empty():
    """SBlock.reads_writes defaults to an empty dict.

    v2 analog of the v1 BodyLeaf test: reads_writes is now a
    ``dict[str, BufferAccess]`` keyed by slot name; an empty block has
    no RMW operands.
    """
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[])
    assert block.reads_writes == {}


def test_nkialloc_has_empty_operand_axes_and_no_rmw():
    """NKIAlloc is a declaration op: no operand axes, no reads, no RMW."""
    assert NKIAlloc.OPERAND_AXES == {}
    assert NKIAlloc.RMW_OPERANDS == frozenset()
    assert NKIAlloc.INPUT_OPERANDS == frozenset()
    assert NKIAlloc.RFACTOR_RECIPE is None


def test_nkialloc_cpu_sim_returns_uninitialised_array_with_location_role():
    """CPU simulation allocates an uninitialised ``np.empty`` of declared shape/dtype.

    The role on the returned _RoleArray matches ``location`` so downstream
    ops' ``_check_roles`` sees the correct residency tag when the kernel
    is called directly on numpy inputs.
    """
    alloc = NKIAlloc(location="sbuf", shape=(4, 8), dtype="float32")
    result = alloc()
    assert result.shape == (4, 8)
    assert str(result.dtype) == "float32"
    assert getattr(result, "role", None) == "sbuf"

    alloc_hbm = NKIAlloc(location="hbm", shape=(2, 3), dtype="float32")
    assert getattr(alloc_hbm(), "role", None) == "hbm"
    alloc_psum = NKIAlloc(location="psum", shape=(2, 3), dtype="float32")
    assert getattr(alloc_psum(), "role", None) == "psum"


def test_nkimemset_writes_dst_with_value():
    """Memset writes a constant value into its dst; no reads."""
    from nkigym.ops.base import _RoleArray
    from nkigym.ops.memset import NKIMemset

    assert NKIMemset.INPUT_OPERANDS == frozenset()
    assert NKIMemset.OPERAND_AXES == {"dst": ("P", "F")}
    dst = _RoleArray(np.zeros((4, 8), dtype=np.float32), "sbuf")
    NKIMemset(value=1.5)(dst=dst)
    assert (dst == 1.5).all()


def test_nkitensor_copy_src_to_dst():
    """tensor_copy reads src, writes dst."""
    from nkigym.ops.tensor_copy import NKITensorCopy

    assert NKITensorCopy.INPUT_OPERANDS == frozenset({"src"})
    assert NKITensorCopy.OPERAND_AXES == {"src": ("P", "F"), "dst": ("P", "F")}


def test_nkitensor_reduce_reads_data_writes_dst():
    """tensor_reduce reads data, writes dst, accepts axis + op kwargs."""
    from nkigym.ops.tensor_reduce import NKITensorReduce

    assert NKITensorReduce.INPUT_OPERANDS == frozenset({"data"})
    assert "data" in NKITensorReduce.OPERAND_AXES
    assert "dst" in NKITensorReduce.OPERAND_AXES


from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_matmul_dst_is_rmw():
    assert NKIMatmul.RMW_OPERANDS == frozenset({"dst"})
    assert NKIMatmul.RFACTOR_RECIPE == "rmw"
    assert NKIMatmul.INPUT_OPERANDS == frozenset({"stationary", "moving"})
    assert "dst" in NKIMatmul.OPERAND_AXES
    assert NKIMatmul.OPERAND_AXES["dst"] == ("M", "N")


def test_activation_reduce_has_both_dst_and_reduce_res_as_writes():
    assert NKIActivationReduce.RFACTOR_RECIPE == "slot"
    assert NKIActivationReduce.INPUT_OPERANDS == frozenset({"data"})
    assert "dst" in NKIActivationReduce.OPERAND_AXES
    assert "reduce_res" in NKIActivationReduce.OPERAND_AXES
    assert NKIActivationReduce.OPERAND_AXES["dst"] == ("P", "F")
    assert NKIActivationReduce.OPERAND_AXES["reduce_res"] == ("P",)


def test_every_existing_op_declares_dst_in_operand_axes():
    """Every op that writes a tensor now declares its dst slot explicitly."""
    write_ops = [NKILoad, NKIStore, NKIActivation, NKITensorScalar, NKITranspose, NKIDMATranspose]
    for op_cls in write_ops:
        assert "dst" in op_cls.OPERAND_AXES, f"{op_cls.__name__} missing dst in OPERAND_AXES"


def test_every_existing_op_declares_input_operands():
    """Every op subclass declares its read-only slots. Covers the discriminator
    the canonical builder uses to split operands into reads vs writes vs reads_writes."""
    cases = [
        (NKILoad, frozenset({"src"})),
        (NKIStore, frozenset({"src"})),
        (NKIActivation, frozenset({"data"})),
        (NKITensorScalar, frozenset({"data", "operand0"})),
        (NKITranspose, frozenset({"src"})),
        (NKIDMATranspose, frozenset({"src"})),
    ]
    for op_cls, expected in cases:
        assert op_cls.INPUT_OPERANDS == expected, f"{op_cls.__name__}: {op_cls.INPUT_OPERANDS} != {expected}"


def test_canonical_parses_nkialloc_into_module_tensors():
    """An NKIAlloc call in f_nkigym registers a Tensor in module.tensors with
    declared location/shape/dtype — no inference, no OP_LOCAL_BUFFERS, no phase."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def f(lhs):
        lhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=lhs, dst=lhs_sbuf)
        NKIStore()(src=lhs_sbuf, dst=hbm_out)
        return hbm_out

    input_specs = {"lhs": {"shape": (128, 512), "dtype": "bfloat16"}}
    module = build_initial_ir(f, input_specs)

    assert "lhs_sbuf" in module.tensors
    assert module.tensors["lhs_sbuf"].location == "sbuf"
    assert module.tensors["lhs_sbuf"].shape == (128, 512)
    assert module.tensors["lhs_sbuf"].dtype == "bfloat16"
    assert module.tensors["hbm_out"].location == "hbm"


def test_canonical_matmul_leaf_has_dst_in_reads_writes():
    """After canonical build, NKIMatmul's SBlock carries dst in reads_writes
    (not in reads or writes)."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import blocks_under
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def f(lhs_T, rhs):
        lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        psum_acc = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMemset(value=0.0)(dst=psum_acc)
        NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
        NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    input_specs = {
        "lhs_T": {"shape": (128, 128), "dtype": "bfloat16"},
        "rhs": {"shape": (128, 512), "dtype": "bfloat16"},
    }
    module = build_initial_ir(f, input_specs)

    matmul_blocks = [
        block
        for root in module.body
        for block in blocks_under(root)
        if block.body and block.body[0].op_cls.__name__ == "NKIMatmul"
    ]
    assert len(matmul_blocks) == 1
    block = matmul_blocks[0]
    rmw_names = {a.tensor_name for a in block.reads_writes.values()}
    read_names = {a.tensor_name for a in block.reads.values()}
    write_names = {a.tensor_name for a in block.writes.values()}
    assert "psum_acc" in rmw_names
    assert "psum_acc" not in read_names
    assert "psum_acc" not in write_names
    assert {k: v.tensor_name for k, v in block.reads.items()} == {"stationary": "lhs_T_sbuf", "moving": "rhs_sbuf"}


def test_render_emits_alloc_inline_at_tree_position():
    """Rendered kernel declares each tensor at the alloc leaf's tree
    position, not at a global function top."""
    from nkigym.codegen.render import render
    from nkigym.ir.build import build_initial_ir
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def f(lhs_T, rhs):
        lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        psum_acc = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMemset(value=0.0)(dst=psum_acc)
        NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
        NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    input_specs = {
        "lhs_T": {"shape": (128, 128), "dtype": "bfloat16"},
        "rhs": {"shape": (128, 512), "dtype": "bfloat16"},
    }
    module = build_initial_ir(f, input_specs)
    src = render(module)
    assert "psum_acc = nl.ndarray" in src
    assert "buffer=nl.psum" in src
    assert "hbm_out = nl.ndarray" in src
    assert "buffer=nl.shared_hbm" in src
    assert "nisa.memset(psum_acc" in src
    assert "nisa.nc_matmul" in src
    assert "dst=psum_acc" in src
    assert "nisa.tensor_copy" in src


def test_render_emits_3d_sbuf_and_2d_hbm_shapes():
    """Regression: SBUF/PSUM tensors must be 3D (P_tile, num_slots, F_total);
    HBM tensors keep the declared 2D shape."""
    from nkigym.codegen.render import render
    from nkigym.ir.build import build_initial_ir
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def f(lhs):
        sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
        NKILoad()(src=lhs, dst=sbuf)
        NKIStore()(src=sbuf, dst=hbm_out)
        return hbm_out

    input_specs = {"lhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_initial_ir(f, input_specs)
    src = render(module)
    """SBUF tensor: 3D (128, 16, 2048) — P_tile=128, num_p_tiles=2048/128=16, full F=2048."""
    assert "sbuf = nl.ndarray((128, 16, 2048)" in src, f"sbuf 3D shape missing; got: {src}"
    """HBM tensor: 2D (2048, 2048) — declared shape."""
    assert "hbm_out = nl.ndarray((2048, 2048)" in src, f"hbm_out 2D shape missing; got: {src}"
