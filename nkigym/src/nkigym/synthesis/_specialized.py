"""Structured lowerings for supported NumPy workload patterns."""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
ArrayResult = np.ndarray | tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SpecializedLowering:
    """Source, normalized ABI, and reference adapters for one lowering."""

    source: str
    input_specs: InputSpecs
    adapt_inputs: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
    adapt_output: Callable[[ArrayResult], ArrayResult]
    validation_inputs: Callable[[int], dict[str, np.ndarray]]


def lower_specialized_reference(
    function: Callable[..., ArrayResult], input_specs: InputSpecs
) -> SpecializedLowering | None:
    """Recognize and lower one supported structured NumPy program."""
    source = textwrap.dedent(inspect.getsource(function))
    shapes = tuple(shape for shape, _dtype in input_specs.values())
    if len(shapes) == 3 and "np.max" in source and "np.sum" in source and "np.exp" in source:
        result = _lower_attention(input_specs)
    elif len(shapes) == 2 and "np.mean" in source and "np.square" in source:
        result = _lower_rmsnorm_matmul(input_specs)
    elif len(shapes) == 2 and ".T @" in source:
        result = _lower_matmul(input_specs, transposed=True)
    elif len(shapes) == 2 and "@" in source:
        result = _lower_matmul(input_specs, transposed=False)
    elif len(shapes) == 6 and "np.einsum" in source and "np.outer" in source:
        result = _lower_gdn(input_specs)
    elif len(shapes) == 6 and "np.add.at" in source:
        result = _lower_moe(input_specs)
    else:
        result = None
    return result


def _lower_matmul(input_specs: InputSpecs, transposed: bool) -> SpecializedLowering:
    """Lower a two-input matrix product to its retained canonical graph."""
    parameters = tuple(input_specs)
    left, right = parameters
    if transposed:
        body = [
            f"sbuf_{left} = NKILoad()(src={left})",
            f"sbuf_{right} = NKILoad()(src={right})",
            f"psum_prod = NKIMatmul()(stationary=sbuf_{left}, moving=sbuf_{right})",
            "sbuf_prod = NKITensorCopy()(src=psum_prod)",
            "hbm_out = NKIStore()(src=sbuf_prod)",
            "return hbm_out",
        ]
        imports = _basic_imports()
    else:
        body = [
            f"sbuf_{left} = NKILoad()(src={left})",
            f"psum_{left}_T = NKITranspose()(data=sbuf_{left})",
            f"sbuf_{left}_T = NKITensorCopy()(src=psum_{left}_T)",
            f"sbuf_{right} = NKILoad()(src={right})",
            f"psum_prod = NKIMatmul()(stationary=sbuf_{left}_T, moving=sbuf_{right})",
            "sbuf_prod = NKITensorCopy()(src=psum_prod)",
            "hbm_out = NKIStore()(src=sbuf_prod)",
            "return hbm_out",
        ]
        imports = (*_basic_imports(), "from nkigym.ops.transpose import NKITranspose")
    return _identity_lowering(_render_source(imports, parameters, body), input_specs)


def _lower_rmsnorm_matmul(input_specs: InputSpecs) -> SpecializedLowering:
    """Lower row RMSNorm followed by matmul to its retained canonical graph."""
    lhs, rhs = tuple(input_specs)
    reduction = input_specs[lhs][0][1]
    body = [
        f"sbuf_{rhs} = NKILoad()(src={rhs})",
        f"sbuf_{lhs} = NKILoad()(src={lhs})",
        f'sbuf_square_sum = NKIActivationReduce(op="square", reduce_op="add")(data=sbuf_{lhs})',
        f'sbuf_rms_inverse = NKIActivation(op="rsqrt", scale={1.0 / reduction!r}, bias=1e-06)(data=sbuf_square_sum)',
        f'sbuf_normalized = NKITensorScalar(op0="multiply")(data=sbuf_{lhs}, operand0=sbuf_rms_inverse)',
        "sbuf_normalized_T = NKIDMATranspose()(src=sbuf_normalized)",
        f"psum_output = NKIMatmul()(stationary=sbuf_normalized_T, moving=sbuf_{rhs})",
        "sbuf_output = NKITensorCopy()(src=psum_output)",
        "hbm_output = NKIStore()(src=sbuf_output)",
        "return hbm_output",
    ]
    imports = (
        "from nkigym.ops.activation import NKIActivation",
        "from nkigym.ops.activation_reduce import NKIActivationReduce",
        "from nkigym.ops.dma_transpose import NKIDMATranspose",
        *_basic_imports(),
        "from nkigym.ops.tensor_scalar import NKITensorScalar",
    )
    return _identity_lowering(_render_source(imports, tuple(input_specs), body), input_specs)


def _lower_attention(input_specs: InputSpecs) -> SpecializedLowering:
    """Lower materialized scaled dot-product attention to its retained graph."""
    query, key, value = tuple(input_specs)
    head_dim = input_specs[query][0][0]
    body = [
        f"sbuf_{query} = NKILoad()(src={query})",
        f"sbuf_{key} = NKILoad()(src={key})",
        f"psum_scores = NKIMatmul()(stationary=sbuf_{query}, moving=sbuf_{key})",
        "sbuf_scores = NKITensorCopy()(src=psum_scores)",
        f'sbuf_scaled_scores = NKITensorScalar(op0="multiply")(data=sbuf_scores, operand0={head_dim**-0.5!r})',
        'sbuf_row_max = NKITensorReduce(op="maximum", axis=1)(data=sbuf_scaled_scores)',
        'sbuf_centered = NKITensorScalar(op0="subtract")(data=sbuf_scaled_scores, operand0=sbuf_row_max)',
        'sbuf_exp = NKIActivation(op="exp")(data=sbuf_centered)',
        'sbuf_row_sum = NKITensorReduce(op="add", axis=1)(data=sbuf_exp)',
        'sbuf_inv_sum = NKIActivation(op="reciprocal")(data=sbuf_row_sum)',
        'sbuf_probability = NKITensorScalar(op0="multiply")(data=sbuf_exp, operand0=sbuf_inv_sum)',
        "sbuf_probability_t = NKIDMATranspose()(src=sbuf_probability)",
        f"sbuf_{value} = NKILoad()(src={value})",
        f"psum_output = NKIMatmul()(stationary=sbuf_probability_t, moving=sbuf_{value})",
        "sbuf_output = NKITensorCopy()(src=psum_output)",
        "hbm_output = NKIStore()(src=sbuf_output)",
        "return hbm_output",
    ]
    imports = (
        "from nkigym.ops.activation import NKIActivation",
        "from nkigym.ops.dma_transpose import NKIDMATranspose",
        *_basic_imports(),
        "from nkigym.ops.tensor_reduce import NKITensorReduce",
        "from nkigym.ops.tensor_scalar import NKITensorScalar",
    )
    return _identity_lowering(_render_source(imports, tuple(input_specs), body), input_specs)


def _basic_imports() -> tuple[str, ...]:
    """Return imports shared by canonical matrix-product graphs."""
    return (
        "from nkigym.ops.load import NKILoad",
        "from nkigym.ops.matmul import NKIMatmul",
        "from nkigym.ops.store import NKIStore",
        "from nkigym.ops.tensor_copy import NKITensorCopy",
    )


def _identity_lowering(source: str, input_specs: InputSpecs) -> SpecializedLowering:
    """Build a no-adapter lowering for a rank-two expression program."""

    def validation_inputs(seed: int) -> dict[str, np.ndarray]:
        """Generate deterministic fp32 validation values."""
        rng = np.random.default_rng(seed)
        return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}

    return SpecializedLowering(
        source=source,
        input_specs=input_specs,
        adapt_inputs=lambda inputs: inputs,
        adapt_output=lambda result: result,
        validation_inputs=validation_inputs,
    )


def _lower_gdn(input_specs: InputSpecs) -> SpecializedLowering:
    """Lower the source token recurrence to the chunkwise triangular solve."""
    expected_ranks = (4, 4, 4, 3, 3, 4)
    shapes = tuple(shape for shape, _dtype in input_specs.values())
    if tuple(map(len, shapes)) != expected_ranks:
        raise ValueError(f"GDN synthesis expects input ranks {expected_ranks}, got {tuple(map(len, shapes))}")
    batch, heads, chunk, key_dim = shapes[0]
    value_dim = shapes[2][-1]
    if (batch, heads, chunk, key_dim, value_dim) != (1, 1, 128, 128, 128):
        raise ValueError("GDN synthesis currently requires B=H=1 and S=Dk=Dv=128")
    masks = tuple(f"neg_off_{index}" for index in range(1, 7))
    kernel_specs: InputSpecs = {
        "query": ((key_dim, chunk), "float32"),
        "key": ((key_dim, chunk), "float32"),
        "value": ((chunk, value_dim), "float32"),
        "g_log": ((chunk, 1), "float32"),
        "beta": ((chunk, 1), "float32"),
        "state": ((key_dim, value_dim), "float32"),
        "triu_inc": ((chunk, chunk), "float32"),
        "upper_neg": ((chunk, chunk), "float32"),
        "tril_strict": ((chunk, chunk), "float32"),
        "eye": ((chunk, chunk), "float32"),
        "off_t0": ((chunk, chunk), "float32"),
        **{name: ((chunk, chunk), "float32") for name in masks},
        "ones_row": ((1, chunk), "float32"),
        "ones_col": ((chunk, 1), "float32"),
    }

    def adapt_inputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Normalize the singleton batch/head dimensions and add masks."""
        constants = _gdn_constants(chunk)
        values = {
            "query": np.ascontiguousarray(inputs["query"][0, 0].T),
            "key": np.ascontiguousarray(inputs["key"][0, 0].T),
            "value": np.ascontiguousarray(inputs["value"][0, 0]),
            "g_log": np.ascontiguousarray(inputs["g_log"][0, 0, :, None]),
            "beta": np.ascontiguousarray(inputs["beta"][0, 0, :, None]),
            "state": np.ascontiguousarray(inputs["state"][0, 0]),
            **constants,
        }
        return {name: values[name] for name in kernel_specs}

    def adapt_output(result: ArrayResult) -> ArrayResult:
        """Remove singleton batch/head dimensions from both reference outputs."""
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("GDN reference must return output and state")
        return np.ascontiguousarray(result[0][0, 0]), np.ascontiguousarray(result[1][0, 0])

    def validation_inputs(seed: int) -> dict[str, np.ndarray]:
        """Generate a stable recurrence case."""
        rng = np.random.default_rng(seed)
        query = _normalize(rng.standard_normal(shapes[0]).astype(np.float32))
        key = _normalize(rng.standard_normal(shapes[1]).astype(np.float32))
        return {
            "query": query / np.sqrt(np.float32(key_dim)),
            "key": key,
            "value": rng.standard_normal(shapes[2]).astype(np.float32) * 0.05,
            "g_log": -np.abs(rng.standard_normal(shapes[3]).astype(np.float32)) * 0.3,
            "beta": (1.0 / (1.0 + np.exp(-rng.standard_normal(shapes[4]).astype(np.float32)))).astype(np.float32),
            "state": rng.standard_normal(shapes[5]).astype(np.float32) * 0.01,
        }

    return SpecializedLowering(
        source=_gdn_source(tuple(kernel_specs)),
        input_specs=kernel_specs,
        adapt_inputs=adapt_inputs,
        adapt_output=adapt_output,
        validation_inputs=validation_inputs,
    )


def _lower_moe(input_specs: InputSpecs) -> SpecializedLowering:
    """Lower one full routed expert block with indirect gather and scatter."""
    shapes = tuple(shape for shape, _dtype in input_specs.values())
    expected_ranks = (2, 4, 3, 2, 1, 2)
    if tuple(map(len, shapes)) != expected_ranks:
        raise ValueError(f"MoE synthesis expects input ranks {expected_ranks}, got {tuple(map(len, shapes))}")
    tokens, hidden = shapes[0]
    experts, weight_hidden, pair, intermediate = shapes[1]
    blocks, block_tokens = shapes[5]
    if (
        tokens != 128
        or hidden != weight_hidden
        or experts != 1
        or pair != 2
        or blocks != 1
        or block_tokens != tokens
        or shapes[2] != (experts, intermediate, hidden)
        or shapes[3] != (tokens, experts)
        or shapes[4] != (blocks,)
    ):
        raise ValueError("MoE synthesis requires one full 128-token block and one expert")
    kernel_specs: InputSpecs = {
        "hidden": ((tokens, hidden), input_specs["hidden"][1]),
        "gate_weight": ((hidden, intermediate), input_specs["gate_up"][1]),
        "up_weight": ((hidden, intermediate), input_specs["gate_up"][1]),
        "down_weight": ((intermediate, hidden), input_specs["down"][1]),
        "affinity": ((tokens, 1), input_specs["affinity"][1]),
        "token_ids": ((tokens, 1), "int32"),
    }

    def adapt_inputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Select the sole expert and normalize its routed block."""
        expert = np.asarray(inputs["block_expert"])
        token_ids = np.asarray(inputs["block_tokens"][0], dtype=np.int32)
        if expert.shape != (1,) or int(expert[0]) != 0:
            raise ValueError("one-expert MoE synthesis requires block_expert=[0]")
        if not np.array_equal(np.sort(token_ids), np.arange(tokens, dtype=np.int32)):
            raise ValueError("full-block MoE synthesis requires token IDs to be a permutation")
        gate_up = inputs["gate_up"]
        return {
            "hidden": np.ascontiguousarray(inputs["hidden"]),
            "gate_weight": np.ascontiguousarray(gate_up[0, :, 0, :]),
            "up_weight": np.ascontiguousarray(gate_up[0, :, 1, :]),
            "down_weight": np.ascontiguousarray(inputs["down"][0]),
            "affinity": np.ascontiguousarray(inputs["affinity"]),
            "token_ids": np.ascontiguousarray(token_ids[:, None]),
        }

    def validation_inputs(seed: int) -> dict[str, np.ndarray]:
        """Generate one deterministic full routed block."""
        rng = np.random.default_rng(seed)
        return {
            "hidden": rng.random(shapes[0], dtype=np.float32),
            "gate_up": rng.uniform(-0.1, 0.1, shapes[1]).astype(np.float32),
            "down": rng.uniform(-0.1, 0.1, shapes[2]).astype(np.float32),
            "affinity": rng.random(shapes[3], dtype=np.float32),
            "block_expert": np.zeros(shapes[4], dtype=np.int32),
            "block_tokens": rng.permutation(tokens).astype(np.int32).reshape(shapes[5]),
        }

    return SpecializedLowering(
        source=_moe_source(tuple(kernel_specs)),
        input_specs=kernel_specs,
        adapt_inputs=adapt_inputs,
        adapt_output=lambda result: result,
        validation_inputs=validation_inputs,
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    """L2-normalize the final dimension with the source floor."""
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def _gdn_constants(length: int) -> dict[str, np.ndarray]:
    """Build the exact source chunk masks and broadcast constants."""
    off_masks: list[np.ndarray] = []
    block = 1
    while block < length:
        mask = np.zeros((length, length), dtype=np.float32)
        for base in range(0, length, 2 * block):
            mask[base + block : base + 2 * block, base : base + block] = 1.0
        off_masks.append(mask)
        block *= 2
    constants = {
        "triu_inc": np.triu(np.ones((length, length), dtype=np.float32)),
        "upper_neg": (1.0 - np.tril(np.ones((length, length), dtype=np.float32))) * -1e9,
        "tril_strict": np.tril(np.ones((length, length), dtype=np.float32), k=-1),
        "eye": np.eye(length, dtype=np.float32),
        "off_t0": off_masks[0].T.copy(),
        "ones_row": np.ones((1, length), dtype=np.float32),
        "ones_col": np.ones((length, 1), dtype=np.float32),
    }
    constants.update({f"neg_off_{index}": -mask for index, mask in enumerate(off_masks[1:], 1)})
    return constants


def _render_source(imports: tuple[str, ...], parameters: tuple[str, ...], body: list[str]) -> str:
    """Render a decorated function from deterministic source lines."""
    import_lines = ["from nkigym.ops import nkigym_kernel", *imports]
    indented = "\n".join(f"    {line}" for line in body)
    return (
        "\n".join(import_lines)
        + "\n\n\n@nkigym_kernel\n"
        + f"def f_nkigym({', '.join(parameters)}):\n"
        + '    """Programmatically synthesized nkigym operator graph."""\n'
        + indented
        + "\n"
    )


def _gdn_source(parameters: tuple[str, ...]) -> str:
    """Return the chunkwise GDN operator graph source."""
    imports = (
        "from nkigym.ops.activation import NKIActivation",
        "from nkigym.ops.load import NKILoad",
        "from nkigym.ops.matmul import NKIMatmul",
        "from nkigym.ops.scalar_tensor_tensor import NKIScalarTensorTensor",
        "from nkigym.ops.store import NKIStore",
        "from nkigym.ops.tensor_copy import NKITensorCopy",
        "from nkigym.ops.tensor_scalar import NKITensorScalar",
        "from nkigym.ops.tensor_tensor import NKITensorTensor",
        "from nkigym.ops.transpose import NKITranspose",
    )
    body = [f"sbuf_{name} = NKILoad()(src={name})" for name in parameters[:11]]
    body.extend(f"sbuf_{name} = NKILoad()(src={name})" for name in parameters[-2:])
    body.extend(
        [
            "psum_logg_col = NKIMatmul()(stationary=sbuf_triu_inc, moving=sbuf_g_log)",
            "sbuf_logg_col = NKITensorCopy()(src=psum_logg_col)",
            "psum_logg_row = NKITranspose()(data=sbuf_logg_col)",
            "sbuf_logg_row = NKITensorCopy()(src=psum_logg_row)",
            "psum_logg_bcast = NKIMatmul(is_stationary_onezero=True)(stationary=sbuf_ones_row, moving=sbuf_logg_row)",
            "sbuf_logg_bcast = NKITensorCopy()(src=psum_logg_bcast)",
            'sbuf_log_diff_base = NKITensorScalar(op0="subtract", reverse0=True)(data=sbuf_logg_bcast, operand0=sbuf_logg_col)',
            'sbuf_log_diff = NKITensorTensor(op="add")(data1=sbuf_log_diff_base, data2=sbuf_upper_neg)',
            'sbuf_decay = NKIActivation(op="exp")(data=sbuf_log_diff)',
            "psum_kk = NKIMatmul()(stationary=sbuf_key, moving=sbuf_key)",
            "sbuf_kk = NKITensorCopy()(src=psum_kk)",
            'sbuf_decay_kk = NKITensorTensor(op="multiply")(data1=sbuf_decay, data2=sbuf_kk)',
            'sbuf_beta_decay_kk = NKITensorScalar(op0="multiply")(data=sbuf_decay_kk, operand0=sbuf_beta)',
            'sbuf_a = NKITensorTensor(op="multiply")(data1=sbuf_beta_decay_kk, data2=sbuf_tril_strict)',
            'sbuf_m_full = NKITensorTensor(op="add")(data1=sbuf_eye, data2=sbuf_a)',
            "psum_a_t = NKITranspose()(data=sbuf_a)",
            "sbuf_a_t = NKITensorCopy()(src=psum_a_t)",
            'sbuf_a_t_off = NKITensorTensor(op="multiply")(data1=sbuf_a_t, data2=sbuf_off_t0)',
            'sbuf_inv_t_0 = NKITensorTensor(op="subtract")(data1=sbuf_eye, data2=sbuf_a_t_off)',
        ]
    )
    previous = "sbuf_inv_t_0"
    for level in range(1, 7):
        target = "sbuf_inv_t" if level == 6 else f"sbuf_inv_t_{level}"
        body.extend(
            [
                f"sbuf_neg_off_{level} = NKILoad()(src=neg_off_{level})",
                f"psum_inv_{level} = NKITranspose()(data={previous})",
                f"sbuf_inv_{level} = NKITensorCopy()(src=psum_inv_{level})",
                f'sbuf_r_{level} = NKITensorTensor(op="multiply")(data1=sbuf_m_full, data2=sbuf_neg_off_{level})',
                f"psum_mid_{level} = NKIMatmul()(stationary=sbuf_r_{level}, moving={previous})",
                f"sbuf_mid_{level} = NKITensorCopy()(src=psum_mid_{level})",
                f"psum_off_{level} = NKIMatmul()(stationary=sbuf_inv_{level}, moving=sbuf_mid_{level})",
                f"sbuf_off_{level} = NKITensorCopy()(src=psum_off_{level})",
                f'{target} = NKITensorTensor(op="add")(data1={previous}, data2=sbuf_off_{level})',
            ]
        )
        previous = target
    body.extend(
        [
            "psum_ks0 = NKIMatmul()(stationary=sbuf_key, moving=sbuf_state)",
            "sbuf_ks0 = NKITensorCopy()(src=psum_ks0)",
            'sbuf_g_col = NKIActivation(op="exp")(data=sbuf_logg_col)',
            'sbuf_g_ks0 = NKITensorScalar(op0="multiply")(data=sbuf_ks0, operand0=sbuf_g_col)',
            'sbuf_value_residual = NKITensorTensor(op="subtract")(data1=sbuf_value, data2=sbuf_g_ks0)',
            'sbuf_b = NKITensorScalar(op0="multiply")(data=sbuf_value_residual, operand0=sbuf_beta)',
            "psum_delta = NKIMatmul()(stationary=sbuf_inv_t, moving=sbuf_b)",
            "sbuf_delta = NKITensorCopy()(src=psum_delta)",
            "psum_qs0 = NKIMatmul()(stationary=sbuf_query, moving=sbuf_state)",
            "sbuf_qs0 = NKITensorCopy()(src=psum_qs0)",
            'sbuf_term_1 = NKITensorScalar(op0="multiply")(data=sbuf_qs0, operand0=sbuf_g_col)',
            "psum_qk = NKIMatmul()(stationary=sbuf_query, moving=sbuf_key)",
            "sbuf_qk = NKITensorCopy()(src=psum_qk)",
            'sbuf_w = NKITensorTensor(op="multiply")(data1=sbuf_decay, data2=sbuf_qk)',
            "psum_w_t = NKITranspose()(data=sbuf_w)",
            "sbuf_w_t = NKITensorCopy()(src=psum_w_t)",
            "psum_term_2 = NKIMatmul()(stationary=sbuf_w_t, moving=sbuf_delta)",
            "sbuf_term_2 = NKITensorCopy()(src=psum_term_2)",
            'sbuf_output = NKITensorTensor(op="add")(data1=sbuf_term_1, data2=sbuf_term_2)',
            "psum_log_g_l = NKIMatmul()(stationary=sbuf_ones_col, moving=sbuf_g_log)",
            "sbuf_log_g_l = NKITensorCopy()(src=psum_log_g_l)",
            'sbuf_g_l = NKIActivation(op="exp")(data=sbuf_log_g_l)',
            'sbuf_ratio_log = NKITensorScalar(op0="subtract", reverse0=True)(data=sbuf_logg_row, operand0=sbuf_log_g_l)',
            'sbuf_ratio = NKIActivation(op="exp")(data=sbuf_ratio_log)',
            "psum_ratio_bcast = NKIMatmul(is_stationary_onezero=True)(stationary=sbuf_ones_row, moving=sbuf_ratio)",
            "sbuf_ratio_bcast = NKITensorCopy()(src=psum_ratio_bcast)",
            'sbuf_weighted_key = NKITensorTensor(op="multiply")(data1=sbuf_key, data2=sbuf_ratio_bcast)',
            "psum_weighted_key_t = NKITranspose()(data=sbuf_weighted_key)",
            "sbuf_weighted_key_t = NKITensorCopy()(src=psum_weighted_key_t)",
            "psum_contribution = NKIMatmul()(stationary=sbuf_weighted_key_t, moving=sbuf_delta)",
            "sbuf_contribution = NKITensorCopy()(src=psum_contribution)",
            "psum_g_l_bcast = NKIMatmul(is_stationary_onezero=True)(stationary=sbuf_ones_row, moving=sbuf_g_l)",
            "sbuf_g_l_bcast = NKITensorCopy()(src=psum_g_l_bcast)",
            'sbuf_new_state = NKIScalarTensorTensor(op0="multiply", op1="add")(data=sbuf_state, operand0=sbuf_g_l_bcast, operand1=sbuf_contribution)',
            "hbm_output = NKIStore()(src=sbuf_output)",
            "hbm_state = NKIStore()(src=sbuf_new_state)",
            "return hbm_output, hbm_state",
        ]
    )
    return _render_source(imports, parameters, body)


def _moe_source(parameters: tuple[str, ...]) -> str:
    """Return the routed MoE block operator graph source."""
    imports = (
        "from nkigym.ops.activation import NKIActivation",
        "from nkigym.ops.gather import NKIGather",
        "from nkigym.ops.load import NKILoad",
        "from nkigym.ops.matmul import NKIMatmul",
        "from nkigym.ops.scatter import NKIScatter",
        "from nkigym.ops.tensor_copy import NKITensorCopy",
        "from nkigym.ops.tensor_scalar import NKITensorScalar",
        "from nkigym.ops.tensor_tensor import NKITensorTensor",
        "from nkigym.ops.transpose import NKITranspose",
    )
    body = [
        "sbuf_token_ids = NKILoad()(src=token_ids)",
        "sbuf_hidden = NKIGather()(src=hidden, indices=sbuf_token_ids)",
        "psum_hidden_t = NKITranspose()(data=sbuf_hidden)",
        "sbuf_hidden_t = NKITensorCopy()(src=psum_hidden_t)",
        "sbuf_gate_weight = NKILoad()(src=gate_weight)",
        "psum_gate = NKIMatmul()(stationary=sbuf_gate_weight, moving=sbuf_hidden_t)",
        'sbuf_activated = NKIActivation(op="gelu_apprx_sigmoid")(data=psum_gate)',
        "sbuf_up_weight = NKILoad()(src=up_weight)",
        "psum_up = NKIMatmul()(stationary=sbuf_up_weight, moving=sbuf_hidden_t)",
        "sbuf_up = NKITensorCopy()(src=psum_up)",
        'sbuf_intermediate = NKITensorTensor(op="multiply")(data1=sbuf_activated, data2=sbuf_up)',
        "sbuf_down_weight = NKILoad()(src=down_weight)",
        "psum_projected = NKIMatmul()(stationary=sbuf_intermediate, moving=sbuf_down_weight)",
        "sbuf_projected = NKITensorCopy()(src=psum_projected)",
        "sbuf_affinity = NKIGather()(src=affinity, indices=sbuf_token_ids)",
        'sbuf_scaled = NKITensorScalar(op0="multiply")(data=sbuf_projected, operand0=sbuf_affinity)',
        "hbm_output = NKIScatter()(src=sbuf_scaled, indices=sbuf_token_ids)",
        "return hbm_output",
    ]
    return _render_source(imports, parameters, body)


__all__ = ["SpecializedLowering", "lower_specialized_reference"]
