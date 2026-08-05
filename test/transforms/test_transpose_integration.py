"""Rendered and end-to-end checks for primitive transpose rewrites."""

from __future__ import annotations

import math
from pathlib import Path
from test._simulation import _load_source
from test.transforms._fixtures import f_lhs_matmul, f_matmul

import numpy as np

from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    CancelTransposePair,
    InsertTransposePair,
    InsertTransposePairOption,
    Split,
    SplitOption,
    TransposeThroughLoad,
    TransposeThroughMatmul,
    TransposeThroughTensorCopy,
)

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
INPUT_SPECS: InputSpecs = {"lhs_T": ((3584, 4096), "bfloat16"), "rhs": ((3584, 128), "bfloat16")}
K, M = INPUT_SPECS["lhs_T"][0]
_, N = INPUT_SPECS["rhs"][0]
SMALL_INPUT_SPECS: InputSpecs = {"lhs_T": ((128, 512), "bfloat16"), "rhs": ((128, 128), "bfloat16")}
PAIR_NUMERIC_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs_T": ((128, 512), "bfloat16"),
    "rhs": ((128, 1024), "bfloat16"),
}
LARGE_LHS_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs": ((2048, 2048), "bfloat16"),
    "rhs": ((2048, 512), "bfloat16"),
}


def _leaves(ir: KernelIR, op_name: str) -> list[int]:
    """Return every ISA leaf with class name ``op_name``."""
    return [
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == op_name
    ]


def _layout_ladder(input_specs: InputSpecs) -> list[tuple[str, KernelIR]]:
    """Build the deterministic primitive rewrite chain used by these tests."""
    ir = build_initial_ir(f_matmul, input_specs)
    ladder = [("canonical", ir)]
    store_nid = _leaves(ir, "NKIStore")[0]
    insert = InsertTransposePair()
    insert_option = next(
        option
        for option in insert.analyze(ir)
        if option.consumer_nid == store_nid and option.operand == "src" and option.source == "sbuf_prod"
    )
    ir = insert.apply(ir, insert_option)
    ladder.append(("insert_pair", ir))
    matmul = TransposeThroughMatmul()
    ir = matmul.apply(ir, matmul.analyze(ir)[0])
    ladder.append(("commute_matmul", ir))
    tensor_copy = TransposeThroughTensorCopy()
    ir = tensor_copy.apply(ir, tensor_copy.analyze(ir)[0])
    ladder.append(("commute_tensor_copy", ir))
    return ladder


def _instances(ir: KernelIR, leaf_nid: int) -> int:
    """Return the static number of executions of one canonical ISA leaf."""
    return math.prod(
        ir.tree.loop(ancestor).extent
        for ancestor in ir.tree.ancestors(leaf_nid)
        if isinstance(ir.tree.data(ancestor), ForNode)
    )


def _simulate(
    ir: KernelIR,
    inputs: dict[str, np.ndarray],
    expected: np.ndarray,
    tmp_path: Path,
    module_name: str,
    function_name: str,
) -> None:
    """Render one IR state and assert its fp32 simulation matches NumPy."""
    module = _load_source(render(ir), tmp_path, module_name)
    actual = np.asarray(simulate_fp32(getattr(module, function_name))(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3, err_msg=module_name)


def test_layout_ladder_exposes_primitive_search_actions() -> None:
    """The MDP inserts, cancels, and commutes one concrete action at a time."""
    ladder = _layout_ladder(INPUT_SPECS)
    assert [name for name, _ir in ladder] == ["canonical", "insert_pair", "commute_matmul", "commute_tensor_copy"]
    assert [len(_leaves(ir, "NKITranspose")) for _name, ir in ladder] == [0, 2, 1, 0]
    assert [len(_leaves(ir, "NKIDMATranspose")) for _name, ir in ladder] == [0, 0, 0, 1]
    assert [len(_leaves(ir, "NKIMatmul")) for _name, ir in ladder] == [1, 1, 1, 1]

    environment = KernelMDP(
        f_matmul,
        SMALL_INPUT_SPECS,
        transforms=[
            InsertTransposePair(),
            CancelTransposePair(),
            TransposeThroughMatmul(),
            TransposeThroughTensorCopy(),
        ],
    )
    initial = environment.reset()
    insert_action = next(
        action
        for action in environment.legal_actions(initial)
        if isinstance(action[0], InsertTransposePair)
        and isinstance(action[1], InsertTransposePairOption)
        and action[1].source == "sbuf_prod"
    )
    inserted = environment.step(initial, insert_action)
    inserted_actions = environment.legal_actions(inserted)
    assert any(isinstance(transform, CancelTransposePair) for transform, _option in inserted_actions)
    commute_action = next(action for action in inserted_actions if isinstance(action[0], TransposeThroughMatmul))
    commuted = environment.step(inserted, commute_action)
    materialize_action = next(
        action for action in environment.legal_actions(commuted) if isinstance(action[0], TransposeThroughTensorCopy)
    )
    materialized = environment.step(commuted, materialize_action)
    assert _leaves(materialized, "NKITranspose") == []
    assert len(_leaves(materialized, "NKIDMATranspose")) == 1

    cancel_action = next(action for action in inserted_actions if isinstance(action[0], CancelTransposePair))
    assert render(environment.step(inserted, cancel_action)) == render(initial)


def test_layout_ladder_changes_the_matmul_orientation() -> None:
    """The commute replaces narrow matmuls with one quarter as many wide matmuls."""
    canonical, pair_inserted, commuted, _dma_commuted = [ir for _name, ir in _layout_ladder(INPUT_SPECS)]
    canonical_matmul = _leaves(canonical, "NKIMatmul")[0]
    pair_matmul = _leaves(pair_inserted, "NKIMatmul")[0]
    commuted_matmul = _leaves(commuted, "NKIMatmul")[0]
    canonical_moving_width = canonical.tree.isa(canonical_matmul).operand_bindings["moving"].ranges[1][1]
    commuted_moving_width = commuted.tree.isa(commuted_matmul).operand_bindings["moving"].ranges[1][1]
    assert canonical_moving_width == Const(value=128)
    assert commuted_moving_width == Const(value=512)
    assert _instances(canonical, canonical_matmul) == (K // 128) * (M // 128)
    assert _instances(pair_inserted, pair_matmul) == (K // 128) * (M // 128)
    assert _instances(commuted, commuted_matmul) == (K // 128) * (N // 128) * (M // 512)
    assert sum(_instances(commuted, leaf) for leaf in _leaves(commuted, "NKITranspose")) == M // 128

    canonical_source = render(canonical)
    commuted_source = render(commuted)
    assert "lhs_T.shape == (3584, 4096)" in canonical_source
    assert "rhs.shape == (3584, 128)" in canonical_source
    assert "moving=sbuf_rhs[0][0:128, i_d0_0, 0:0 + 128]" in canonical_source
    assert "stationary=sbuf_rhs[0][0:128, i_d0_0, 0:0 + 128]" in commuted_source
    assert "moving=sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 512:i_d1_0 * 512 + 512]" in commuted_source


def test_layout_ladder_states_match_numpy(tmp_path: Path) -> None:
    """Every primitive layout state preserves the rectangular matmul."""
    rng = np.random.default_rng(0)
    inputs = {
        "lhs_T": rng.standard_normal((128, 512)).astype(np.float32),
        "rhs": rng.standard_normal((128, 128)).astype(np.float32),
    }
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    for name, ir in _layout_ladder(SMALL_INPUT_SPECS):
        _simulate(ir, inputs, expected, tmp_path, f"layout_{name}", "nki_f_matmul")


def test_pair_and_matmul_commute_handle_multiple_output_tiles(tmp_path: Path) -> None:
    """Pair insertion and matmul commute preserve a multi-tile output."""
    rng = np.random.default_rng(2)
    inputs = {
        "lhs_T": rng.standard_normal((128, 512)).astype(np.float32),
        "rhs": rng.standard_normal((128, 1024)).astype(np.float32),
    }
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    for name, ir in _layout_ladder(PAIR_NUMERIC_SPECS)[1:3]:
        _simulate(ir, inputs, expected, tmp_path, f"multitile_{name}", "nki_f_matmul")


def test_dma_rewrite_paths_handle_rectangular_matmuls(tmp_path: Path) -> None:
    """Both DMA materialization paths preserve three rectangular orientations."""
    shapes = ((128, 128, 512), (256, 128, 256), (128, 256, 128))
    for m, k, n in shapes:
        specs = {"lhs": ((m, k), "bfloat16"), "rhs": ((k, n), "bfloat16")}
        rng = np.random.default_rng(m + k + n)
        inputs = {
            "lhs": rng.standard_normal((m, k)).astype(np.float32),
            "rhs": rng.standard_normal((k, n)).astype(np.float32),
        }
        expected = inputs["lhs"] @ inputs["rhs"]
        initial = build_initial_ir(f_lhs_matmul, specs)
        load = TransposeThroughLoad()
        load_options = load.analyze(initial)
        assert len(load_options) == 1
        tensor_copy = TransposeThroughTensorCopy()
        tensor_copy_options = tensor_copy.analyze(initial)
        assert len(tensor_copy_options) == 1
        transformed_states = (
            ("TransposeThroughLoad", load.apply(initial, load_options[0])),
            ("TransposeThroughTensorCopy", tensor_copy.apply(initial, tensor_copy_options[0])),
        )
        for transform_name, transformed in transformed_states:
            name = f"{transform_name}_{m}_{k}_{n}"
            _simulate(transformed, inputs, expected, tmp_path, name, "nki_f_lhs_matmul")


def test_load_commute_preserves_orientation_after_free_axis_split() -> None:
    """Normalizing a split DMA block retains the original HBM orientation."""
    ir = build_initial_ir(f_lhs_matmul, LARGE_LHS_SPECS)
    transform = TransposeThroughLoad()
    transformed = transform.apply(ir, transform.analyze(ir)[0])
    dma_leaf = _leaves(transformed, "NKIDMATranspose")[0]
    free_loop = next(
        nid
        for nid in transformed.tree.ancestors(dma_leaf)
        if isinstance(transformed.tree.data(nid), ForNode) and transformed.tree.loop(nid).loop_var == "i_d1_0"
    )
    split = Split().apply(transformed, SplitOption(free_loop, (2, 8), None))
    source = render(split)
    assert (
        "src=lhs[i_d0_0 * 512:i_d0_0 * 512 + 512, " "i_d1_0 * 1024 + i_d1_1 * 128:i_d1_0 * 1024 + i_d1_1 * 128 + 128]"
    ) in source
