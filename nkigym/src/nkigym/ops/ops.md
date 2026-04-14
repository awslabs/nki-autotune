## 1. Logical Computation
**Running example**: causal single-head attention `softmax(mask(scale * Q @ K^T)) @ V`. Inputs use standard ML layout `(seq, hidden)`:

- `q: float16[4096, 128]` — `(seq_q, d_k)`
- `k: float16[4096, 128]` — `(seq_k, d_k)`
- `v: float16[4096, 128]` — `(seq_k, d_v)`
- output: `float16[4096, 128]` — `(seq_q, d_v)`

The math function defines what to compute — no loads/stores, no tiling, no memory hierarchy. Each `GymXXX` call maps 1:1 to a real `nisa.*` ISA instruction, but the math function only specifies the logical data flow.

```python
def attention(Q, K, V, scale):
    Q_t = GymTranspose()(Q)
    K_t = GymTranspose()(K)
    S = GymMatmul()(Q_t, K_t)
    masked_S = GymAffineSelect()(
        pattern=[[-1, S.shape[1]]], channel_multiplier=1,
        on_true_tile=S, on_false_value=-np.inf,
        cmp_op="greater_equal")
    scaled_S = GymTensorScalar()(masked_S, op0="multiply",
                                     operand0=scale)
    neg_max_S = GymTensorReduce()(op="max", data=scaled_S,
                                      axis=1, negate=True)
    exp_S, sum_exp = GymActivationReduce()(
        op="exp", data=scaled_S, reduce_op="add",
        bias=neg_max_S)
    inv_sum = GymActivation()(op="reciprocal", data=sum_exp)
    exp_S_t = nkigym.nc_transpose()(exp_S)
    attn = GymMatmul()(exp_S_t, V)
    output = GymTensorScalar()(attn, op0="multiply",operand0=inv_sum)
    return output
```

---

## 2. NKIGym Operators

Every `GymXXX` op mirrors a real `nisa.*` or `nl.*` ISA operation — no ops are invented. At the logical level, an op only declares its axis semantics and a CPU simulation. No hardware constraints, tiling, or memory placement.

```python
class GymOp:
    """Logical NKIGym operator — axis semantics and CPU simulation only."""
    NAME: str
    OPERAND_AXES: dict[str, tuple[str, ...]]
    OUTPUT_AXES: dict[str, tuple[str, str, ...]]

    @abstractmethod
    def __call__(self, **kwargs):
        """CPU simulation using Numpy in default float64.
        Takes input arrays + config, returns output array(s).
        """
```

### 2.1 Operator Subclasses

```python
class GymMatmul(GymOp):
    """stationary(K, M).T @ moving(K, N) → output(M, N).
    K is the contraction (accumulation) axis.
    """

    NAME = "nc_matmul"
    OPERAND_AXES = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES = {"output": ("M", "N")}

    def __call__(self, stationary, moving):
        return stationary.T @ moving


class GymTranspose(GymOp):
    """data(P, F) → output(F, P). Real hardware op, not a free view."""

    NAME = "nc_transpose"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("F", "P")}

    def __call__(self, data):
        return data.T


class GymTensorScalar(GymOp):
    """(data <op0> operand0) <op1> operand1 → output(P, F).
    operand0/operand1: scalar constant or (P,) column vector,
    broadcast across the free axis.
    reverse0/reverse1 swap operand order for non-commutative ops.
    """

    NAME = "tensor_scalar"
    OPERAND_AXES = {"data": ("P", "F"), "operand0": ("P",), "operand1": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F")}

    def __call__(self, data, op0, operand0, reverse0=False,
                 op1=None, operand1=None, reverse1=False):
        ops = {"multiply": np.multiply, "subtract": np.subtract, "add": np.add}
        b = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0
        result = ops[op0](b, data) if reverse0 else ops[op0](data, b)
        if op1 is not None:
            c = operand1[..., np.newaxis] if isinstance(operand1, np.ndarray) else operand1
            result = ops[op1](c, result) if reverse1 else ops[op1](result, c)
        return result


class GymAffineSelect(GymOp):
    """Position-predicated element select.
    affine_value = offset + p * channel_multiplier + Σ(idx_i * step_i).
    Compares affine_value to 0; selects on_true_tile or on_false_value.
    pattern: list of [step, count] pairs describing free-axis layout.
    """

    NAME = "affine_select"
    OPERAND_AXES = {"on_true_tile": ("P", "F")}
    OUTPUT_AXES = {"output": ("P", "F")}

    def __call__(self, pattern, channel_multiplier, on_true_tile,
                 on_false_value, cmp_op="equal", offset=0):
        P = on_true_tile.shape[0]
        F = int(np.prod([n for _, n in pattern]))
        p_idx = np.arange(P)[:, np.newaxis]
        f_vals = np.array([0])
        for step, count in pattern:
            f_vals = (f_vals[:, np.newaxis] + np.arange(count) * step).ravel()
        affine = offset + p_idx * channel_multiplier + f_vals[np.newaxis, :]
        cmps = {"greater_equal": np.greater_equal, "equal": np.equal}
        mask = cmps[cmp_op](affine, 0)
        return np.where(mask, on_true_tile.reshape(P, F), on_false_value)


class GymTensorReduce(GymOp):
    """Reduce along specified axis with optional negation.
    data(P, F) → output(P,).
    axis must be trailing free dimension(s); partition axis cannot be reduced.
    """

    NAME = "tensor_reduce"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("P",)}

    def __call__(self, op, data, axis, negate=False, keepdims=False):
        reduce = {"max": np.max, "add": np.sum}
        result = reduce[op](data, axis=axis, keepdims=keepdims)
        if negate:
            result = -result
        return result


class GymActivationReduce(GymOp):
    """op(data * scale + bias) → output(P, F), and simultaneously
    reduce_op(output) along free axis → reduce_res(P,).
    reduce_op only supports "add".
    """

    NAME = "activation_reduce"
    OPERAND_AXES = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F"), "reduce_res": ("P",)}

    def __call__(self, op, data, reduce_op, bias=None, scale=1.0):
        fns = {
            "exp": np.exp, "tanh": np.tanh, "square": np.square,
            "reciprocal": lambda x: 1.0 / x,
        }
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        elem = fns[op](data * s + b)
        red = {"add": np.sum}[reduce_op](elem, axis=1)
        return elem, red


class GymActivation(GymOp):
    """output = op(data * scale + bias).
    Applies unary activation element-wise.
    """

    NAME = "activation"
    OPERAND_AXES = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F")}

    def __call__(self, op, data, bias=None, scale=1.0):
        fns = {
            "exp": np.exp, "tanh": np.tanh, "square": np.square,
            "reciprocal": lambda x: 1.0 / x,
            "rsqrt": lambda x: 1.0 / np.sqrt(x),
        }
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        return fns[op](data * s + b)
```

### 2.2 CPU Simulation

The math function is plain Python — each `GymXXX` call dispatches to `GymOp.__call__()` which executes the op with numpy at float64 precision. No parsing or IR needed; just call the function directly:

```python
q = np.random.randn(4096, 128)
k = np.random.randn(4096, 128)
v = np.random.randn(4096, 128)
output = attention(q, k, v, scale=1.0 / np.sqrt(128))
```

The result is the **reference output** — a float64 array that any correctly rendered and compiled NKI kernel must match (within hardware precision tolerance).
