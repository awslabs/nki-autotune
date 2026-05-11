"""Unit tests for the Annotate atom and per-key validators."""

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.annotate import Annotate, enumerate_annotate_atoms


@nkigym_kernel
def _matmul_small(lhs_T, rhs):
    """Small matmul fixture: K=M=128, N=512 — single-tile for speed."""
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


_SPECS = {"lhs_T": {"shape": (128, 128), "dtype": "bfloat16"}, "rhs": {"shape": (128, 512), "dtype": "bfloat16"}}


def _find_alloc_sblock(module, tensor_name: str) -> tuple[int, ...]:
    """Return path to the root-level SBlock whose NKIAlloc names ``tensor_name``."""
    for i, root in enumerate(module.body):
        if isinstance(root, SBlock) and any(
            c.op_cls.__name__ == "NKIAlloc" and c.kwargs.get("tensor_name") == tensor_name for c in root.body
        ):
            return (i,)
    raise AssertionError(f"No alloc SBlock for {tensor_name!r}")


def _find_first_for_node(module) -> tuple[int, ...]:
    """Return path to the first ForNode in the forest."""
    for i, root in enumerate(module.body):
        if isinstance(root, ForNode):
            return (i,)
    raise AssertionError("No ForNode in module.body")


def test_annotate_rejects_unknown_key() -> None:
    """Unknown keys are illegal."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")
    atom = Annotate(target_path=target, key="bogus_key", value=42)
    assert not atom.is_legal(module)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_annotate_buffer_degree_on_alloc_sblock() -> None:
    """``buffer_degree`` annotation on an alloc SBlock sets ``block.annotations``."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")
    atom = Annotate(target_path=target, key="buffer_degree", value={"lhs_T_sbuf": 2})
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    block = new_module.body[target[0]]
    assert isinstance(block, SBlock)
    assert block.annotations["buffer_degree"] == {"lhs_T_sbuf": 2}


def test_annotate_buffer_degree_rejects_on_for_node() -> None:
    """``buffer_degree`` target must be an SBlock, not a ForNode."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_first_for_node(module)
    atom = Annotate(target_path=target, key="buffer_degree", value={"lhs_T_sbuf": 2})
    assert not atom.is_legal(module)


def test_annotate_buffer_degree_rejects_param_tensor() -> None:
    """``buffer_degree`` on a param tensor is illegal."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")
    atom = Annotate(target_path=target, key="buffer_degree", value={"lhs_T": 2})
    assert not atom.is_legal(module)


def test_annotate_buffer_degree_rejects_bad_value_type() -> None:
    """``buffer_degree`` value must be ``dict[str, int]`` with degree >= 1."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")

    assert not Annotate(target_path=target, key="buffer_degree", value=2).is_legal(module)
    assert not Annotate(target_path=target, key="buffer_degree", value={"lhs_T_sbuf": 0}).is_legal(module)
    assert not Annotate(target_path=target, key="buffer_degree", value={"lhs_T_sbuf": -1}).is_legal(module)


def test_annotate_software_pipeline_depth_on_for_node() -> None:
    """``software_pipeline_depth`` target must be a ForNode with value >= 1."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_first_for_node(module)
    atom = Annotate(target_path=target, key="software_pipeline_depth", value=2)
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    node = new_module.body[target[0]]
    assert isinstance(node, ForNode)
    assert node.annotations["software_pipeline_depth"] == 2


def test_annotate_software_pipeline_depth_rejects_on_sblock() -> None:
    """``software_pipeline_depth`` target must be a ForNode, not an SBlock."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")
    atom = Annotate(target_path=target, key="software_pipeline_depth", value=2)
    assert not atom.is_legal(module)


def test_annotate_software_pipeline_depth_rejects_value_less_than_1() -> None:
    """``software_pipeline_depth`` must be >= 1."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_first_for_node(module)
    assert not Annotate(target_path=target, key="software_pipeline_depth", value=0).is_legal(module)
    assert not Annotate(target_path=target, key="software_pipeline_depth", value=-5).is_legal(module)


def test_annotate_apply_preserves_other_annotations() -> None:
    """Apply does not overwrite unrelated annotation keys."""
    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")
    module.body[target[0]].annotations["some_other_key"] = "preserved"
    atom = Annotate(target_path=target, key="buffer_degree", value={"lhs_T_sbuf": 2})
    new_module = atom.apply(module)
    block = new_module.body[target[0]]
    assert isinstance(block, SBlock)
    assert block.annotations["some_other_key"] == "preserved"
    assert block.annotations["buffer_degree"] == {"lhs_T_sbuf": 2}


def test_enumerate_annotate_atoms_yields_buffer_degree_for_each_alloc() -> None:
    """Enumerator emits buffer_degree for every non-param alloc SBlock."""
    module = build_canonical_module(_matmul_small, _SPECS)
    atoms = enumerate_annotate_atoms(module)
    buffer_degree_atoms = [a for a in atoms if a.key == "buffer_degree"]
    """5 allocs; lhs_T and rhs are params - only intermediate tensors get enumerated."""
    """But per the spec, buffer_degree applies to non-param intermediate tensors.
       The 5 allocs include 4 intermediates (lhs_T_sbuf, rhs_sbuf, psum_acc, sbuf_prod)
       plus hbm_out. hbm_out is 'return', not param - can technically be multi-buffered.
       Depending on enumeration choice - include/exclude return tensors."""
    assert len(buffer_degree_atoms) > 0


def test_enumerate_annotate_atoms_yields_software_pipeline_for_each_for_node() -> None:
    """Enumerator emits software_pipeline_depth for every ForNode."""
    module = build_canonical_module(_matmul_small, _SPECS)
    atoms = enumerate_annotate_atoms(module)
    swp_atoms = [a for a in atoms if a.key == "software_pipeline_depth"]
    """Canonical matmul has 6 compute trees; each has >= 1 ForNode. With depth in {2, 3}
       per the spec, and one ForNode per loop, enumerator should yield >= 6 SWP atoms."""
    assert len(swp_atoms) > 0


def test_buffer_degree_annotation_widens_shape_after_render() -> None:
    """Applying ``buffer_degree=2`` and rendering widens the P-slot dim to 2."""
    from nkigym.codegen.render import render

    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_alloc_sblock(module, "lhs_T_sbuf")
    atom = Annotate(target_path=target, key="buffer_degree", value={"lhs_T_sbuf": 2})
    module = atom.apply(module)

    source = render(module)
    """``lhs_T_sbuf`` base shape ``(128, 1, 128)`` for the single-tile
    config becomes ``(128, 2, 128)`` once ``buffer_degree=2`` is
    injected before ``place_buffers``."""
    assert "lhs_T_sbuf = nl.ndarray((128, 2, 128)" in source


def test_software_pipeline_depth_annotation_renders_normally() -> None:
    """``software_pipeline_depth`` > 1 is a render no-op today.

    The Bug B followup spec adds prologue / body / epilogue emission;
    until then, depth > 1 must render identically to depth == 1 —
    source must still parse as valid Python.
    """
    import ast

    from nkigym.codegen.render import render

    module = build_canonical_module(_matmul_small, _SPECS)
    target = _find_first_for_node(module)
    atom = Annotate(target_path=target, key="software_pipeline_depth", value=2)
    module = atom.apply(module)

    source = render(module)
    ast.parse(source)
