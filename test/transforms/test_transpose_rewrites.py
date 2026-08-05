"""Direct contract tests for primitive transpose rewrites."""

from __future__ import annotations

from dataclasses import replace
from test.transforms._fixtures import f_lhs_matmul

import pytest

from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.transforms import (
    CancelTransposePair,
    CancelTransposePairOption,
    InsertTransposePair,
    InsertTransposePairOption,
    TransformLegalityError,
    TransposeThroughLoad,
    TransposeThroughLoadOption,
    TransposeThroughMatmul,
    TransposeThroughMatmulOption,
    TransposeThroughTensorCopy,
    TransposeThroughTensorCopyOption,
)
from nkigym.transforms._canonical_rewrite import replace_buffer

PAIR_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs_T": ((128, 512), "bfloat16"),
    "rhs": ((128, 1024), "bfloat16"),
}
LHS_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs": ((128, 128), "bfloat16"), "rhs": ((128, 512), "bfloat16")}


@nkigym_kernel
def _matmul(lhs_T, rhs):
    """Return a canonical rectangular matmul."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


@nkigym_kernel
def _aliased_matmul(value):
    """Use the same loaded tensor as both matmul operands."""
    sbuf_value = NKILoad()(src=value)
    psum_prod = NKIMatmul()(stationary=sbuf_value, moving=sbuf_value)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def _leaves(ir: KernelIR, op_name: str) -> list[tuple[int, ISANode]]:
    """Return ISA leaves whose class name equals ``op_name``."""
    leaves: list[tuple[int, ISANode]] = []
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ISANode) and node.op_cls.__name__ == op_name:
            leaves.append((nid, node))
    return leaves


def _op_names(ir: KernelIR) -> list[str]:
    """Return ISA class names in execution order."""
    return [ir.tree.isa(nid).op_cls.__name__ for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]


def _store_pair_option(ir: KernelIR) -> InsertTransposePairOption:
    """Return the insertion option targeting the output store."""
    store_nid = _leaves(ir, "NKIStore")[0][0]
    option = InsertTransposePairOption(consumer_nid=store_nid, operand="src", source="sbuf_prod")
    assert option in InsertTransposePair().analyze(ir)
    return option


def _insert_pair(ir: KernelIR) -> KernelIR:
    """Insert one transpose pair on the matmul output edge."""
    return InsertTransposePair().apply(ir, _store_pair_option(ir))


def _commute_matmul(ir: KernelIR) -> KernelIR:
    """Commute the sole eligible transpose through a matmul."""
    transform = TransposeThroughMatmul()
    options = transform.analyze(ir)
    assert len(options) == 1
    return transform.apply(ir, options[0])


def test_insert_and_cancel_transpose_pair() -> None:
    """Insertion rewires one edge and cancellation restores the original."""
    ir = build_initial_ir(_matmul, PAIR_SPECS)
    original = render(ir)
    inserted = _insert_pair(ir)
    assert render(ir) == original
    assert _op_names(inserted) == [
        "NKILoad",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKITranspose",
        "NKITensorCopy",
        "NKITranspose",
        "NKITensorCopy",
        "NKIStore",
    ]

    transposes = _leaves(inserted, "NKITranspose")
    drains = _leaves(inserted, "NKITensorCopy")
    store = _leaves(inserted, "NKIStore")[0][1]
    first_output = drains[1][1].operand_bindings["dst"].tensor
    second_output = drains[2][1].operand_bindings["dst"].tensor
    assert transposes[0][1].operand_bindings["data"].tensor == "sbuf_prod"
    assert transposes[1][1].operand_bindings["data"].tensor == first_output
    assert store.operand_bindings["src"].tensor == second_output
    assert inserted.buffer(first_output).shape == (1024, 512)
    assert inserted.buffer(second_output).shape == (512, 1024)

    cancel = CancelTransposePair()
    options = cancel.analyze(inserted)
    assert len(options) == 1
    restored = cancel.apply(inserted, options[0])
    assert render(restored) == original
    assert InsertTransposePair().analyze(ir)


def test_cancel_preserves_source_allocation_owned_by_removed_block() -> None:
    """Cancellation retains a live source declared on the first pair block."""
    ir = build_initial_ir(_matmul, PAIR_SPECS)
    original = render(ir)
    inserted = _insert_pair(ir)
    option = CancelTransposePair().analyze(inserted)[0]
    source = inserted.buffer("sbuf_prod")
    root = inserted.tree.block(inserted.tree.root)
    first = inserted.tree.block(option.first_transpose_nid)
    inserted.tree.graph.nodes[inserted.tree.root]["data"] = replace(
        root, alloc_buffers=tuple(buffer for buffer in root.alloc_buffers if buffer.name != source.name)
    )
    inserted.tree.graph.nodes[option.first_transpose_nid]["data"] = replace(
        first, alloc_buffers=(*first.alloc_buffers, source)
    )

    restored = CancelTransposePair().apply(inserted, option)
    assert restored.buffer(source.name) == source
    assert render(restored) == original


def test_pair_transforms_reject_stale_and_unknown_options() -> None:
    """Pair transforms fail loudly when their selected graph edge is absent."""
    ir = build_initial_ir(_matmul, PAIR_SPECS)
    insert = InsertTransposePair()
    insert_option = _store_pair_option(ir)
    inserted = insert.apply(ir, insert_option)
    with pytest.raises(TransformLegalityError, match="not an eligible canonical SBUF edge"):
        insert.apply(inserted, insert_option)
    with pytest.raises(TransformLegalityError, match="not an eligible canonical SBUF edge"):
        insert.apply(ir, InsertTransposePairOption(consumer_nid=ir.tree.root, operand="src", source="sbuf_prod"))

    cancel = CancelTransposePair()
    cancel_option = cancel.analyze(inserted)[0]
    restored = cancel.apply(inserted, cancel_option)
    with pytest.raises(TransformLegalityError, match="not a cancellable adjacent pair"):
        cancel.apply(restored, cancel_option)
    with pytest.raises(TransformLegalityError, match="not a cancellable adjacent pair"):
        cancel.apply(ir, CancelTransposePairOption(first_transpose_nid=ir.tree.root))


def test_insert_rejects_mismatched_physical_dtype() -> None:
    """Pair insertion must not materialize an invalid transpose dtype."""
    ir = build_initial_ir(_matmul, PAIR_SPECS)
    replace_buffer(ir, replace(ir.buffer("sbuf_prod"), storage_dtype="float32"))
    store_nid = _leaves(ir, "NKIStore")[0][0]
    assert all(option.consumer_nid != store_nid for option in InsertTransposePair().analyze(ir))


def test_insert_rebinds_only_one_aliased_operand() -> None:
    """An eligible edge remains selectable when another operand aliases it."""
    ir = build_initial_ir(_aliased_matmul, {"value": ((128, 512), "bfloat16")})
    insert = InsertTransposePair()
    matmul_nid = _leaves(ir, "NKIMatmul")[0][0]
    options = [option for option in insert.analyze(ir) if option.consumer_nid == matmul_nid]
    assert {option.operand for option in options} == {"moving", "stationary"}
    transformed = insert.apply(ir, next(option for option in options if option.operand == "moving"))
    matmul = _leaves(transformed, "NKIMatmul")[0][1]
    assert matmul.operand_bindings["stationary"].tensor == "sbuf_value"
    assert matmul.operand_bindings["moving"].tensor == "sbuf_value_tt"


def test_insert_rejects_edges_of_an_existing_identity_pair() -> None:
    """Pair insertion does not recursively nest an intact identity pair."""
    inserted = _insert_pair(build_initial_ir(_matmul, PAIR_SPECS))
    drains = _leaves(inserted, "NKITensorCopy")
    pair_tensors = {
        "sbuf_prod",
        drains[-2][1].operand_bindings["dst"].tensor,
        drains[-1][1].operand_bindings["dst"].tensor,
    }
    insert = InsertTransposePair()
    tensor_copy = TransposeThroughTensorCopy()
    states = [inserted]
    first_materialized = tensor_copy.apply(inserted, tensor_copy.analyze(inserted)[0])
    states.append(first_materialized)
    second_materialized = tensor_copy.apply(first_materialized, tensor_copy.analyze(first_materialized)[0])
    states.append(second_materialized)

    for state in states:
        assert all(option.source not in pair_tensors for option in insert.analyze(state))
        store_nid, store = _leaves(state, "NKIStore")[0]
        source = store.operand_bindings["src"].tensor
        option = InsertTransposePairOption(consumer_nid=store_nid, operand="src", source=source)
        assert option not in insert.analyze(state)
        with pytest.raises(TransformLegalityError, match="not an eligible canonical SBUF edge"):
            insert.apply(state, option)


def test_transpose_through_load_replaces_the_canonical_chain() -> None:
    """A load, transpose, and drain become one HBM-to-SBUF DMA transpose."""
    ir = build_initial_ir(f_lhs_matmul, LHS_SPECS)
    transform = TransposeThroughLoad()
    options = transform.analyze(ir)
    assert len(options) == 1
    transformed = transform.apply(ir, options[0])
    assert _op_names(transformed) == [
        "NKIDMATranspose",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKIStore",
    ]
    assert "sbuf_lhs" not in transformed.all_buffers()
    assert "psum_lhs_T" not in transformed.all_buffers()
    dma = _leaves(transformed, "NKIDMATranspose")[0][1]
    assert dma.operand_bindings["src"].tensor == "lhs"
    assert dma.operand_bindings["dst"].tensor == "sbuf_lhs_T"


def test_transpose_through_load_rechecks_its_contract() -> None:
    """The load commute is pure and rejects stale, unknown, or malformed inputs."""
    ir = build_initial_ir(f_lhs_matmul, LHS_SPECS)
    original = render(ir)
    transform = TransposeThroughLoad()
    option = transform.analyze(ir)[0]
    transformed = transform.apply(ir, option)
    assert render(ir) == original
    assert transform.analyze(transformed) == []
    with pytest.raises(TransformLegalityError, match="not an eligible canonical"):
        transform.apply(transformed, option)
    with pytest.raises(TransformLegalityError, match="not an eligible canonical"):
        transform.apply(ir, TransposeThroughLoadOption(target_nid=ir.tree.root))

    malformed = build_initial_ir(f_lhs_matmul, LHS_SPECS)
    replace_buffer(malformed, replace(malformed.buffer("psum_lhs_T"), storage_dtype="float32"))
    assert transform.analyze(malformed) == []


def test_transpose_through_matmul_swaps_operands_and_rebuilds_dependencies() -> None:
    """The commute consumes one transpose and rebuilds the swapped matmul chain."""
    ir = build_initial_ir(_matmul, PAIR_SPECS)
    transform = TransposeThroughMatmul()
    assert transform.analyze(ir) == []
    inserted = _insert_pair(ir)
    assert len(transform.analyze(inserted)) == 1
    transformed = _commute_matmul(inserted)
    assert _op_names(transformed) == [
        "NKILoad",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKITranspose",
        "NKITensorCopy",
        "NKIStore",
    ]

    matmul_nid, matmul = _leaves(transformed, "NKIMatmul")[0]
    first_drain_nid, first_drain = _leaves(transformed, "NKITensorCopy")[0]
    transpose_nid, transpose = _leaves(transformed, "NKITranspose")[0]
    final_drain_nid, final_drain = _leaves(transformed, "NKITensorCopy")[1]
    assert matmul.operand_bindings["stationary"].tensor == "sbuf_rhs"
    assert matmul.operand_bindings["moving"].tensor == "sbuf_lhs_T"
    assert transpose.operand_bindings["data"].tensor == first_drain.operand_bindings["dst"].tensor
    assert transformed.dependency.direct_consumers(matmul_nid) == [first_drain_nid]
    assert transformed.dependency.direct_consumers(first_drain_nid) == [transpose_nid]
    assert transformed.dependency.direct_consumers(transpose_nid) == [final_drain_nid]

    swapped_psum = matmul.operand_bindings["dst"].tensor
    swapped_sbuf = first_drain.operand_bindings["dst"].tensor
    restored_sbuf = final_drain.operand_bindings["dst"].tensor
    assert transformed.buffer(swapped_psum).shape == (1024, 512)
    assert transformed.buffer(swapped_psum).physical_dtype() == "float32"
    assert transformed.buffer(swapped_sbuf).shape == (1024, 512)
    assert transformed.buffer(restored_sbuf).shape == (512, 1024)


def test_transpose_through_matmul_rechecks_its_contract() -> None:
    """The matmul commute is pure and rejects stale or unknown targets."""
    inserted = _insert_pair(build_initial_ir(_matmul, PAIR_SPECS))
    original = render(inserted)
    transform = TransposeThroughMatmul()
    option = transform.analyze(inserted)[0]
    transformed = transform.apply(inserted, option)
    assert render(inserted) == original
    with pytest.raises(TransformLegalityError, match="not adjacent to an eligible canonical matmul"):
        transform.apply(transformed, option)
    with pytest.raises(TransformLegalityError, match="not adjacent to an eligible canonical matmul"):
        transform.apply(
            build_initial_ir(_matmul, PAIR_SPECS), TransposeThroughMatmulOption(transpose_nid=inserted.tree.root)
        )


def test_transpose_through_matmul_rejects_malformed_storage() -> None:
    """The commute rejects invalid accumulator and input physical dtypes."""
    transform = TransposeThroughMatmul()
    for target in ("accumulator", "stationary", "moving"):
        inserted = _insert_pair(build_initial_ir(_matmul, PAIR_SPECS))
        matmul = _leaves(inserted, "NKIMatmul")[0][1]
        if target == "accumulator":
            tensor = matmul.operand_bindings["dst"].tensor
            storage_dtype = None
        else:
            tensor = matmul.operand_bindings[target].tensor
            storage_dtype = "float32"
        replace_buffer(inserted, replace(inserted.buffer(tensor), storage_dtype=storage_dtype))
        assert transform.analyze(inserted) == [], target


def test_transpose_through_matmul_accepts_a_moving_extent_below_512() -> None:
    """The commute follows ISA limits instead of requiring full-width tiles."""
    specs = {"lhs_T": ((128, 256), "bfloat16"), "rhs": ((128, 128), "bfloat16")}
    transformed = _commute_matmul(_insert_pair(build_initial_ir(_matmul, specs)))
    matmul = _leaves(transformed, "NKIMatmul")[0][1]
    assert matmul.operand_bindings["moving"].ranges[1][1] == Const(value=256)


def test_transpose_through_tensor_copy_materializes_a_dma_transpose() -> None:
    """A logical transpose and drain become one SBUF-to-SBUF DMA transpose."""
    ir = build_initial_ir(f_lhs_matmul, LHS_SPECS)
    transform = TransposeThroughTensorCopy()
    options = transform.analyze(ir)
    assert len(options) == 1
    transformed = transform.apply(ir, options[0])
    assert _op_names(transformed) == [
        "NKILoad",
        "NKIDMATranspose",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKIStore",
    ]
    assert "psum_lhs_T" not in transformed.all_buffers()
    assert "sbuf_lhs_T" in transformed.all_buffers()
    dma = _leaves(transformed, "NKIDMATranspose")[0][1]
    assert dma.operand_bindings["src"].tensor == "sbuf_lhs"


def test_transpose_through_tensor_copy_rechecks_its_contract() -> None:
    """The tensor-copy commute is pure and rejects invalid targets or storage."""
    ir = build_initial_ir(f_lhs_matmul, LHS_SPECS)
    original = render(ir)
    transform = TransposeThroughTensorCopy()
    option = transform.analyze(ir)[0]
    transformed = transform.apply(ir, option)
    assert render(ir) == original
    with pytest.raises(TransformLegalityError, match="not an eligible logical transpose"):
        transform.apply(transformed, option)
    with pytest.raises(TransformLegalityError, match="not an eligible logical transpose"):
        transform.apply(ir, TransposeThroughTensorCopyOption(transpose_nid=ir.tree.root))

    malformed = build_initial_ir(f_lhs_matmul, LHS_SPECS)
    replace_buffer(malformed, replace(malformed.buffer("psum_lhs_T"), storage_dtype="float32"))
    assert transform.analyze(malformed) == []
