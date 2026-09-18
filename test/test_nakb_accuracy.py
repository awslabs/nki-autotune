"""Regression coverage for the accuracy contracts copied from NAKB."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from benchmark import NAKB_WORKLOADS, accuracy_validation, validate_nakb_outputs


def test_topk_accepts_different_ties_and_checks_selected_values() -> None:
    """Accept alternative tied indices but reject wrong and out-of-range selections."""
    criteria = accuracy_validation(NAKB_WORKLOADS["rotational_topk"][5])[1]
    inputs = {"inp": np.array([[5.0, 5.0, 2.0, 1.0]], dtype=np.float32)}
    values = np.array([[5.0, 5.0]], dtype=np.float32)
    expected = values, np.array([[0, 1]], dtype=np.uint32)
    validate_nakb_outputs((values, np.array([[1, 0]], dtype=np.uint32)), expected, inputs, criteria)
    for indices in (np.array([[1, 2]], dtype=np.uint32), np.array([[1, 4]], dtype=np.uint32)):
        with pytest.raises(AssertionError):
            validate_nakb_outputs((values, indices), expected, inputs, criteria)


def test_topk_unfolds_padded_input_before_checking_indices() -> None:
    """Validate rotational indices against the original unpadded input row."""
    criteria = accuracy_validation(NAKB_WORKLOADS["rotational_topk"][10])[1]
    original = np.array([[5.0, 5.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]], dtype=np.float32)
    folded = np.full((1, 2, 8), -np.inf, dtype=np.float32)
    folded[:, :, :4] = original.reshape(1, 2, 4)
    inputs = {"inp": folded.reshape(1, 16), "rotation_topk_2": np.eye(2, dtype=np.float32)}
    values = np.array([[5.0, 5.0, 2.0, 1.0]], dtype=np.float32)
    expected = values, np.array([[0, 1, 2, 3]], dtype=np.uint32)
    validate_nakb_outputs((values, np.array([[1, 0, 2, 3]], dtype=np.int32)), expected, inputs, criteria)
    with pytest.raises(AssertionError):
        validate_nakb_outputs((values, np.array([[1, 0, 2, 9]], dtype=np.int32)), expected, inputs, criteria)


def test_global_max_tolerance_and_output_selection() -> None:
    """Apply MoE's original global error rule only to its validated output."""
    criteria = accuracy_validation(NAKB_WORKLOADS["moe_block_tkg"][0])[1]
    expected = (np.array([[100.0, 0.0]], dtype=np.float32), np.ones((1, 2), dtype=np.float32))
    actual = (np.array([[100.0, 0.9]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32))
    validate_nakb_outputs(actual, expected, {}, criteria)
    with pytest.raises(AssertionError):
        validate_nakb_outputs((actual[0] + 2.0, actual[1]), expected, {}, criteria)


@pytest.mark.parametrize(("index", "rows"), [(0, 256), (1, 2), (4, 2), (8, 2), (9, 2)])
def test_rmsnorm_scale_uses_its_own_tolerance(index: int, rows: int) -> None:
    """Keep the original 0.7% scale check distinct from the 7.2% FP8 check."""
    criteria = accuracy_validation(NAKB_WORKLOADS["rmsnorm_quant"][index])[1]
    expected = (np.ones((rows, 16384), dtype=np.float32), np.ones((rows,), dtype=np.float32))
    validate_nakb_outputs((expected[0] * 1.06, expected[1] * 1.006), expected, {}, criteria)
    with pytest.raises(AssertionError):
        validate_nakb_outputs((expected[0], expected[1] * 1.008), expected, {}, criteria)


def test_serialized_validator_preserves_local_decisions() -> None:
    """Exercise the exact NumPy-only callback transmitted to CPU workers."""
    source, criteria = accuracy_validation(NAKB_WORKLOADS["hf_ffn"][0])
    namespace: dict[str, object] = {"np": np}
    exec(compile(source, "<nakb-test-validator>", "exec"), namespace)
    validator = cast(Callable[..., None], namespace["validate_nakb_outputs"])
    expected = np.array([[1.0, 100.0]], dtype=np.float32)
    validator(expected + 1.0, expected, {}, criteria)
    with pytest.raises(AssertionError):
        validator(expected + 6.0, expected, {}, criteria)
