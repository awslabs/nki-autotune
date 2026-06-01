"""Numpy → nkigym source translation via the Claude Agent SDK.

Single public entry point: :func:`compile_numpy_to_nkigym`. The
orchestrator runs a multi-turn conversation with ``ClaudeSDKClient``:
the agent proposes a candidate ``f_nkigym``, Python executes it
against the numpy reference at fp32, and the orchestrator feeds the
structured validation result back as the next user turn. The loop
continues until the candidate passes or a retry budget is exhausted.

All prompt scaffolding, reply parsing, and numerical validation are
private implementation details.
"""

import ast
import asyncio
import inspect
import re
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock


def _log(message: str) -> None:
    """Write a prefixed status line to stderr, flushed immediately.

    stderr keeps the log stream separate from anything the caller
    prints to stdout (e.g. the returned source).
    """
    print(f"[synthesis] {message}", file=sys.stderr, flush=True)


_SYSTEM_PROMPT = r"""\
You translate numpy reference functions into nkigym math functions for a NKI-kernel compiler.

# Task

Given a numpy function `f_numpy` and an `INPUT_SPECS` dict, produce an `f_nkigym` expressing the same math as a DAG of `NKIOp` subclasses. The orchestrator runs your candidate against the numpy reference after each turn and replies with the validator result.

# Output contract

- Decorate the function with `@nkigym_kernel`. The decorator enforces that every HBM input flows through `NKILoad` before any compute op touches it, and that the kernel returns the output of an `NKIStore`.
- Positional parameters match `INPUT_SPECS` keys and order (same names, same order).
- Body consists ONLY of `var = NKIOp()(...)` assignment statements and the final `return` — no `np.*`, no Python arithmetic, no `if` / `for`, no helper functions.
- Each op ALLOCATES AND RETURNS its output. Bind that result to a named local — there is NO `NKIAlloc`, NO explicit buffer declaration, NO `dst=` operand anywhere.
- Returns exactly one tensor: the HBM output produced by `NKIStore`.
- Matmul uses `stationary.T @ moving`. For raw `A @ B`, transpose `A` first: `psum_A_T = NKITranspose()(src=A)`, drain to SBUF, then `NKIMatmul()(stationary=sbuf_A_T, moving=B)`.
- Plain stateless DAG — the vanilla decomposition of the numpy math. NEVER emit an online/single-pass reformulation (flash attention, running-softmax, fused running-mean, etc.). Online fusion is a separate downstream rewrite that operates on your output.

# Imports

Each NKIOp class lives in its own submodule under `nkigym.ops.<module>`. `nkigym_kernel` is the only name exported from `nkigym.ops` directly. Emit only the import lines for the ops you actually use, drawn from this table:

```python
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
```

Do NOT import NKIOp classes from `nkigym.ops` (flat namespace) — that package only exports `nkigym_kernel` and the abstract `NKIOp` base.

# Op cheat sheet

Each op allocates and returns its output. Bind the return value to a named local; there is no `dst=` operand and no `NKIAlloc`.

| Op | Import | Reads → returns | Call |
|---|---|---|---|
| `NKILoad` | `nkigym.ops.load` | src:(P,F) → sbuf:(P,F) | `sbuf_x = NKILoad()(src=param)` — HBM → SBUF |
| `NKIStore` | `nkigym.ops.store` | src:(P,F) → hbm:(P,F) | `hbm_out = NKIStore()(src=sbuf)` — SBUF → HBM; return this |
| `NKITensorCopy` | `nkigym.ops.tensor_copy` | src:(P,F) → sbuf:(P,F) | `sbuf_prod = NKITensorCopy()(src=psum)` — typically PSUM → SBUF drain |
| `NKITensorReduce` | `nkigym.ops.tensor_reduce` | data:(P,F) → reduced:(P,) | `reduced = NKITensorReduce(axis=1, op="add\|max")(data=X)` — reduce along axis |
| `NKIMatmul` | `nkigym.ops.matmul` | stationary:(K,M), moving:(K,N) → psum:(M,N) | `psum = NKIMatmul()(stationary=A_T, moving=B)` — returns PSUM result; PSUM is zeroed automatically |
| `NKITranspose` | `nkigym.ops.transpose` | src:(P,F) → psum:(F,P) | `psum_T = NKITranspose()(src=sbuf)` — TE transpose, ≤128×128; returns a PSUM tensor |
| `NKIDMATranspose` | `nkigym.ops.dma_transpose` | src:(P,F) → sbuf:(F,P) | `sbuf_T = NKIDMATranspose()(src=sbuf_a)` — DMA transpose, frees TE |
| `NKIActivationReduce` | `nkigym.ops.activation_reduce` | data:(P,F) → reduced:(P,) | `reduced = NKIActivationReduce(op=..., reduce_op=...)(data=X)` — returns the per-row reduction vector |
| `NKIActivation` | `nkigym.ops.activation` | data:(P,F) or (P,) → same shape | `Y = NKIActivation(op=..., scale=?, bias=?)(data=X)` — elementwise |
| `NKITensorScalar` | `nkigym.ops.tensor_scalar` | data:(P,F), operand0:(P,) → out:(P,F) | `Y = NKITensorScalar(op=...)(data=X, operand0=v)` — per-row vector broadcast along F |

Op-arg vocabulary: `op` ∈ `{square, exp, copy, reciprocal, tanh, rsqrt, sqrt}`; `reduce_op` ∈ `{add, max}`; `NKITensorScalar.op` ∈ `{multiply, add, subtract}`; `NKIActivation.scale` / `NKIActivation.bias` apply per-element pre-activation.

# Translation procedure

1. Load every parameter from `INPUT_SPECS` at the top: `sbuf_param = NKILoad()(src=param)`. The load allocates and returns the SBUF tensor.
2. List every tensor-level step in `f_numpy`, stripping `.astype(...)` and `keepdims=True` (numpy bookkeeping, not primitives).
3. Map each step to one or more `NKIOp` calls, binding each result to a named local. Key patterns:
   - Matmul: `psum = NKIMatmul()(stationary=A_T, moving=B)` returns the PSUM result. The accumulator is zeroed automatically — do NOT author an `NKIMemset` before a matmul. Then drain the PSUM result to SBUF before storing (PSUM can't store directly): `sbuf_prod = NKITensorCopy()(src=psum)`. The drain copy stays explicit.
   - Transpose: `psum_T = NKITranspose()(src=sbuf_input)` returns a PSUM tensor; drain it with `sbuf_T = NKITensorCopy()(src=psum_T)`.
   - `NKIActivationReduce` returns its per-row reduction vector `(P,)` directly: `reduced = NKIActivationReduce(op=..., reduce_op=...)(data=X)`. There is no scratch/`reduce_res` operand to author.
   - Fused reduce-then-activation (e.g. rmsnorm's `rsqrt(sum(x²)/F + eps)`): split into two DSL calls. Emit `raw_reduced = NKIActivationReduce(op=<act>, reduce_op=<red>)(data=X)` to get the raw reduction; then feed that into `post_reduced = NKIActivation(op=<post>, scale=<scalar>, bias=<scalar>)(data=raw_reduced)` to apply the post-reduction activation with its affine scale/bias. `NKIActivation` applies `op(data * scale + bias)` per-element on its input; for `rsqrt(reduced/F + eps)`, use `scale=1/F` and `bias=eps`. The post-reduction scale/bias is a SEPARATE `NKIActivation` call.
   - `X * v[:, None]` with `v` shape `(P,)`: `Y = NKITensorScalar(op="multiply")(data=X, operand0=v)`. Broadcasts along F.
4. Fix matmul operand shapes. `NKIMatmul` computes `stationary.T @ moving`. For `A @ B`, transpose `A` first using the transpose-then-drain pattern above.
5. Bind every intermediate to a named local — no chained calls.
6. Final step: `hbm_out = NKIStore()(src=sbuf_final)` and `return hbm_out`.

# Example: matmul

```python
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out
```

# Conversation protocol

- Every assistant turn emits exactly ONE ```python fenced code block: imports + any constants + `def f_nkigym(...):`. Nothing else — no prose before or after.
- The orchestrator replies with a validator result. If `passed` is true the conversation ends. If false, read `error` / `max_abs_diff` / `max_rel_diff` and emit a revised candidate in the same format.
- Common failure modes: wrong matmul transpose orientation, post-reduction activation squeezed into `NKIActivationReduce` (it has no `post_op` / `scale` / `bias` — emit a separate `NKIActivation` instead), wrong axis in a reduction, missing step in the DAG, output-shape mismatch (usually a missing or extra transpose), forgetting the PSUM-to-SBUF drain after matmul or transpose.
"""


def _build_initial_user_prompt(
    f_numpy: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> str:
    """Format the first user turn — the translation request."""
    src = inspect.getsource(f_numpy)
    specs_lines = [f'    "{name}": ({tuple(shape)}, "{dt}"),' for name, (shape, dt) in input_specs.items()]
    specs_block = "INPUT_SPECS = {\n" + "\n".join(specs_lines) + "\n}"
    return f"""\
Translate this numpy function into an nkigym math function.

## `f_numpy`

```python
{src}```

## `INPUT_SPECS`

```python
{specs_block}
```

Emit one ```python fenced block with imports, any needed constants, and the `f_nkigym` definition. Nothing else. I will reply with the validator result.
"""


def _format_validator_feedback(result: dict[str, Any]) -> str:
    """Format a validator result as a follow-up user turn."""
    lines = [
        "Validator result (fp32, numpy reference vs your candidate):",
        f"  passed: {result['passed']}",
        f"  error: {result['error']}",
        f"  max_abs_diff: {result['max_abs_diff']}",
        f"  max_rel_diff: {result['max_rel_diff']}",
        "",
        "Revise and emit a new ```python fenced block (imports + constants + `def f_nkigym(...):`). Nothing else.",
    ]
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _parse_nkigym_source(reply: str) -> str:
    """Extract the first python fenced code block from a single assistant turn."""
    match = _FENCE_RE.search(reply)
    if match is None:
        raise ValueError("no ```python fenced block in assistant turn")
    source = match.group(1)
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"fenced block isn't valid Python: {e}") from e
    return source


_VALIDATION_ATOL = 1e-5
_VALIDATION_RTOL = 1e-5
"""fp32 tolerance. Validation always runs in fp32 regardless of the
user-declared ``INPUT_SPECS`` dtypes — the input_specs dtypes describe
the eventual kernel ABI, not the correctness check, and bf16/fp16
rounding would dominate any real math-mismatch signal at this stage."""


def _exec_nkigym_source(source: str) -> Callable[..., np.ndarray]:
    """Execute the translated source in a fresh namespace and return ``f_nkigym``.

    Enforces that ``f_nkigym`` is decorated with ``@nkigym_kernel`` —
    without the decorator the runtime load/store lineage check is
    silently bypassed and a storeless kernel can pass numerics.
    """
    namespace: dict[str, Any] = {"__name__": "__nkigym_generated__"}
    exec(source, namespace)  # noqa: S102
    func = namespace.get("f_nkigym")
    if not callable(func):
        raise ValueError("generated source does not define `f_nkigym`")
    _require_kernel_decorator(source)
    return func


def _require_kernel_decorator(source: str) -> None:
    """Raise ``ValueError`` unless ``f_nkigym`` is decorated with ``nkigym_kernel``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"re-parse for decorator check failed: {e}") from e
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "f_nkigym":
            if any(_is_nkigym_kernel_decorator(dec) for dec in node.decorator_list):
                return
            raise ValueError("f_nkigym must be decorated with @nkigym_kernel")
    raise ValueError("no `f_nkigym` definition found")


def _is_nkigym_kernel_decorator(dec: ast.expr) -> bool:
    """Return True if ``dec`` is ``@nkigym_kernel`` or ``@<module>.nkigym_kernel``."""
    if isinstance(dec, ast.Name):
        return dec.id == "nkigym_kernel"
    if isinstance(dec, ast.Attribute):
        return dec.attr == "nkigym_kernel"
    return False


def _generate_fp32_inputs(input_specs: dict[str, tuple[tuple[int, ...], str]], seed: int) -> dict[str, np.ndarray]:
    """Draw a reproducible random fp32 input bundle from ``input_specs``.

    Ignores the dtype field in ``input_specs`` — validation runs in fp32
    to isolate math correctness from low-precision rounding.
    """
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in input_specs.items()}


def _run_validation(
    source: str, f_numpy: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], seed: int
) -> dict[str, Any]:
    """Run validation and return structured feedback (never raises).

    Returns ``{"passed", "error", "max_abs_diff", "max_rel_diff"}``.
    """
    try:
        f_nkigym = _exec_nkigym_source(source)
    except Exception as e:
        return {"passed": False, "error": f"exec failed: {e!r}", "max_abs_diff": None, "max_rel_diff": None}

    nkigym_params = list(inspect.signature(f_nkigym).parameters)
    spec_params = list(input_specs)
    if nkigym_params != spec_params:
        return {
            "passed": False,
            "error": f"f_nkigym params {nkigym_params} != INPUT_SPECS keys {spec_params}",
            "max_abs_diff": None,
            "max_rel_diff": None,
        }

    try:
        inputs = _generate_fp32_inputs(input_specs, seed)
        expected = f_numpy(**inputs)
        actual = f_nkigym(**inputs)
    except Exception as e:
        return {"passed": False, "error": f"call failed: {e!r}", "max_abs_diff": None, "max_rel_diff": None}

    if actual.shape != expected.shape:
        return {
            "passed": False,
            "error": f"output shape mismatch: actual {actual.shape} vs expected {expected.shape}",
            "max_abs_diff": None,
            "max_rel_diff": None,
        }

    abs_diff = np.abs(actual - expected)
    max_abs = float(abs_diff.max())
    max_rel = float((abs_diff / (np.abs(expected) + _VALIDATION_ATOL)).max())
    tolerance_ok = np.allclose(actual, expected, atol=_VALIDATION_ATOL, rtol=_VALIDATION_RTOL)
    return {
        "passed": bool(tolerance_ok),
        "error": None if tolerance_ok else f"fp32 mismatch at atol={_VALIDATION_ATOL} rtol={_VALIDATION_RTOL}",
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


async def _drain_assistant_turn(client: ClaudeSDKClient) -> str:
    """Collect the assistant's text response to the most recent user turn."""
    chunks: list[str] = []
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks)


async def _run_feedback_loop(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    seed: int,
    max_attempts: int,
) -> str:
    """Run the propose → validate → feedback loop until pass or budget exhausted."""
    options = ClaudeAgentOptions(system_prompt=_SYSTEM_PROMPT, allowed_tools=[])
    last_result: dict[str, Any] | None = None
    specs_summary = ", ".join(f"{name}:{tuple(shape)}:{dt}" for name, (shape, dt) in input_specs.items())
    _log(f"compiling {f_numpy.__name__} — inputs=[{specs_summary}] seed={seed} max_attempts={max_attempts}")
    async with ClaudeSDKClient(options=options) as client:
        await client.query(_build_initial_user_prompt(f_numpy, input_specs))
        for attempt in range(1, max_attempts + 1):
            _log(f"attempt {attempt}/{max_attempts}: waiting for agent")
            reply = await _drain_assistant_turn(client)
            source = _parse_nkigym_source(reply)
            _log(
                f"attempt {attempt}/{max_attempts}: agent proposed {source.count(chr(10)) + 1}-line candidate; validating"
            )
            result = _run_validation(source, f_numpy, input_specs, seed)
            if result["passed"]:
                _log(f"attempt {attempt}/{max_attempts}: PASSED — returning validated source")
                return source
            last_result = result
            _log(
                f"attempt {attempt}/{max_attempts}: FAILED — {result['error']} "
                f"(max_abs_diff={result['max_abs_diff']} max_rel_diff={result['max_rel_diff']})"
            )
            await client.query(_format_validator_feedback(result))
    _log(f"budget exhausted ({max_attempts} attempts); last result: {last_result}")
    raise RuntimeError(f"compile_numpy_to_nkigym exhausted max_attempts={max_attempts}; last result: {last_result}")


def compile_numpy_to_nkigym(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    seed: int = 0,
    max_attempts: int = 5,
) -> str:
    """Translate a numpy reference into an ``f_nkigym`` source string.

    Drives a multi-turn Claude Agent SDK conversation: the agent proposes
    a candidate, Python executes it against ``f_numpy`` at fp32, and the
    validator result is fed back as the next user turn. The loop
    terminates when the candidate passes or ``max_attempts`` is reached.

    Args:
        f_numpy: Plain-numpy reference function. Its positional parameters
            and order must match the keys of ``input_specs``.
        input_specs: ``{param_name: (shape, dtype_str)}``. Same contract
            as the rest of the autotune stack; validation ignores the
            dtype field and always runs at fp32.
        seed: Seed for the random input draw used during numerical
            validation. Held fixed across all attempts so the agent can
            reason about repeated diffs.
        max_attempts: Maximum candidates the agent may emit before
            ``RuntimeError`` is raised.

    Returns:
        Python source containing ``from nkigym.ops.* import ...`` imports,
        any module-level constants the function references, and
        ``def f_nkigym(...):``.

    Raises:
        ValueError: An assistant turn lacked a python fenced block or the
            block didn't parse.
        RuntimeError: ``max_attempts`` candidates were tried and none
            passed fp32 validation.
    """
    return asyncio.run(_run_feedback_loop(f_numpy, input_specs, seed, max_attempts))
