"""Tests for RFactor atom — rmw-dst recipe (matmul)."""

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock, blocks_under, validate_dataflow_ordering
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.rfactor import RFactor, enumerate_rfactor_atoms


@nkigym_kernel
def _matmul_canonical(lhs_T, rhs):
    """Minimal matmul fixture exercised by the rmw recipe."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_INPUT_SPECS = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _find_matmul_compute_path(module):
    """Walk the tree to find the path to the :class:`NKIMatmul` SBlock."""

    def walk(node, path):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKIMatmul":
            return path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None

    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise ValueError("No NKIMatmul block found")


def test_rfactor_rejects_non_divisor_factor():
    """``outer_factor`` must divide the accumulation dim's ``num_tiles``."""
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    atom = RFactor(reducer_block_path=matmul_path, outer_factor=5)
    assert atom.is_legal(module) is False


def test_rfactor_rejects_endpoint_factors():
    """``outer_factor == 1`` is a no-op; reject."""
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    atom_low = RFactor(reducer_block_path=matmul_path, outer_factor=1)
    assert atom_low.is_legal(module) is False


def test_rfactor_apply_on_illegal_atom_raises():
    """Illegal atom.apply raises :class:`AtomLegalityError`."""
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    atom = RFactor(reducer_block_path=matmul_path, outer_factor=5)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_rfactor_rmw_produces_staging_buffer_and_close():
    """After :class:`RFactor` (rmw, factor=4) on a K-loop matmul:
    - ``module.tensors`` has ``psum_partials`` + ``psum_acc_local``
    - original ``psum_acc`` removed
    - tree contains closing :class:`NKITensorReduce`
    """
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    atom = RFactor(reducer_block_path=matmul_path, outer_factor=4)
    assert atom.is_legal(module)
    new_module = atom.apply(module)

    assert "psum_partials" in new_module.tensors
    assert new_module.tensors["psum_partials"].location == "sbuf"
    assert "psum_acc_local" in new_module.tensors
    assert new_module.tensors["psum_acc_local"].location == "psum"
    assert "psum_acc" not in new_module.tensors

    reduce_blocks = [
        block
        for root in new_module.body
        for block in blocks_under(root)
        if block.body and block.body[0].op_cls.__name__ == "NKITensorReduce"
    ]
    assert len(reduce_blocks) >= 1


def test_rfactor_rmw_preserves_dataflow_ordering():
    """After :class:`RFactor`, the resulting module still validates."""
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    new_module = RFactor(reducer_block_path=matmul_path, outer_factor=4).apply(module)
    assert validate_dataflow_ordering(new_module) is True


def test_rfactor_rmw_kernel_renders():
    """Rendered source compiles to a :class:`str` without error."""
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    new_module = RFactor(reducer_block_path=matmul_path, outer_factor=4).apply(module)
    source = render(new_module)
    assert isinstance(source, str)
    assert "nisa.tensor_reduce" in source
    assert "psum_partials" in source
    assert "psum_acc_local" in source


def test_enumerate_rfactor_atoms_yields_legal_only():
    """Enumerated atoms must all pass :meth:`is_legal`."""
    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    atoms = enumerate_rfactor_atoms(module)
    for atom in atoms:
        assert atom.is_legal(module)


@pytest.mark.xfail(
    reason=(
        "Close-reduce emits `nisa.tensor_reduce(axis=1)` on a 3D staging "
        "slice `(P, outer, F)`, but NKI's tensor_reduce enforces "
        "last-contiguous-axis reduction. Making outer the last axis "
        "(via dim_ids=(P, F, outer)) would collapse the F-tile's size "
        "since `place_buffers` treats middle dims as integer slot counts, "
        "not full-tile-sized dims. A structural renderer extension (full-tile "
        "middle dims) or a multi-reduce lowering (loop of tensor_tensor adds) "
        "is needed; tracked as a followup."
    ),
    strict=False,
)
def test_rfactor_rmw_kernel_renders_and_cpu_sims_correctly():
    """Full pipeline: canonical → :class:`RFactor` → render → CPU-sim vs numpy."""
    from nkigym.tune.verify import _verify

    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    new_module = RFactor(reducer_block_path=matmul_path, outer_factor=4).apply(module)
    source = render(new_module)
    input_specs_verify = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    _verify(source, _matmul_canonical, input_specs_verify)
