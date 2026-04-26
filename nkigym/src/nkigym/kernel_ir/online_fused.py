"""Hand-built online-fused KernelIRs.

Normal ``build_ir(func)`` parses an nkigym math function — a stateless
DAG of ``NKIOp()(...)`` calls. Online fusion needs an inter-iteration
read-before-write on a running accumulator (``sbuf_m_state`` in
rmsnorm+matmul), which the math-function abstraction can't express.

These constructors bypass parsing and assemble a :class:`KernelIR`
directly. The output is still a plain ``KernelIR`` that plugs into
:func:`nkigym.codegen.render_ir`, :func:`nkigym.kernel_ir.sample.sample`,
:func:`nkigym.kernel_ir.validate.validity_report`, and the rest of the
tuning infrastructure.
"""

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo


def build_rmsnorm_matmul_online_ir(input_specs: dict[str, tuple[tuple[int, ...], str]], eps: float = 1e-6) -> KernelIR:
    """Construct the online-fused rmsnorm+matmul :class:`KernelIR` from scratch.

    Args:
        input_specs: ``{"lhs": (shape, dtype), "rhs": (shape, dtype)}``.
            ``lhs`` shape is ``(M, K)``, ``rhs`` shape is ``(K, N)``.
            Both dtypes must match — the TE matmul drains through an
            fp32 PSUM and the final output inherits the input dtype.
        eps: rmsnorm stabiliser added inside ``rsqrt(mean + eps)``.

    Returns:
        A canonical-knobs ``KernelIR`` laying out the online recurrence:

            load lhs, rhs
            for k in K:
                rms_old  = rsqrt(m_state / K + eps)          # reads m_{k-1}
                m_state += sum_f(lhs^2)                      # writes m_k
                rms_new  = rsqrt(m_state / K + eps)          # reads m_k
                rms_inv  = reciprocal(rms_old)
                scale    = rms_new * rms_inv
                lhs_sc   = lhs * rms_new
                lhs_T    = dma_transpose(lhs_sc)
                output   = output * scale + lhs_T @ rhs

        Op 2 reads ``sbuf_m_state`` before op 3 writes it in the ops
        list — renderer walks ops in list order, so at runtime op 2
        sees the previous iteration's m_state. k=1 boundary is
        numerically stable: ``m_state=0`` ⇒ ``rms_old = rsqrt(eps)``
        (large finite) ⇒ ``s_1`` is small ⇒ ``0 * s_1 + psum_1 =
        psum_1``.

    Raises:
        ValueError: When ``lhs`` / ``rhs`` shapes or dtypes don't match
            the online-fusion contract.
    """
    _validate_input_specs(input_specs)
    lhs_shape, lhs_dtype = input_specs["lhs"]
    rhs_shape, rhs_dtype = input_specs["rhs"]
    M, K = lhs_shape
    _, N = rhs_shape

    """Tile caps match the per-op ``TILE_LIMITS`` declarations of the
    ops we emit — P=128 (activation/transpose/dma_transpose/load/store),
    K=128 (matmul), N=512 (matmul). Collapse to the actual dim size
    when the workload is smaller than the HW cap."""
    p_tile = min(128, M)
    k_tile = min(128, K)
    n_tile = min(512, N)
    dimensions: dict[str, DimInfo] = {
        "d0": DimInfo(dim_size=M, logical_tile_size=p_tile, physical_tile_size=p_tile, role=DimRole.PARALLEL),
        "d1": DimInfo(dim_size=K, logical_tile_size=k_tile, physical_tile_size=k_tile, role=DimRole.ACCUMULATION),
        "d2": DimInfo(dim_size=N, logical_tile_size=n_tile, physical_tile_size=n_tile, role=DimRole.PARALLEL),
    }

    logical_tensors: dict[str, TensorInfo] = {
        "lhs": TensorInfo(dim_ids=("d0", "d1"), shape=lhs_shape, dtype=lhs_dtype),
        "rhs": TensorInfo(dim_ids=("d1", "d2"), shape=rhs_shape, dtype=rhs_dtype),
        "output": TensorInfo(dim_ids=("d0", "d2"), shape=(M, N), dtype=lhs_dtype),
    }

    physical_buffers: dict[str, PhysicalBuffer] = {
        "sbuf_lhs": _buf((p_tile, k_tile), ("d0", "d1"), lhs_dtype, "d0", "d1"),
        "sbuf_rhs": _buf((k_tile, n_tile), ("d1", "d2"), rhs_dtype, "d1", "d2"),
        "sbuf_lhs_scaled": _buf((p_tile, k_tile), ("d0", "d1"), lhs_dtype, "d0", "d1"),
        "sbuf_lhs_T": _buf((k_tile, p_tile), ("d1", "d0"), lhs_dtype, "d1", "d0"),
        "sbuf_m_state": _buf((p_tile, 1), ("d0",), "float32", "d0", None),
        "sbuf_rms_old": _buf((p_tile, 1), ("d0",), "float32", "d0", None),
        "sbuf_rms_new": _buf((p_tile, 1), ("d0",), "float32", "d0", None),
        "sbuf_rms_inv": _buf((p_tile, 1), ("d0",), "float32", "d0", None),
        "sbuf_scale": _buf((p_tile, 1), ("d0",), "float32", "d0", None),
        "sbuf_output": _buf((p_tile, n_tile), ("d0", "d2"), lhs_dtype, "d0", "d2"),
        "hbm_output": _buf((M, N), ("d0", "d2"), lhs_dtype, "d0", "d2"),
    }

    inv_K = 1.0 / K
    ops: list[Op] = [
        Op(kind="NKILoad", inputs={"data": "lhs"}, outputs=["sbuf_lhs"], axis_map={"P": "d0", "F": "d1"}),
        Op(kind="NKILoad", inputs={"data": "rhs"}, outputs=["sbuf_rhs"], axis_map={"P": "d1", "F": "d2"}),
        Op(
            kind="NKIActivation",
            inputs={"data": "sbuf_m_state"},
            outputs=["sbuf_rms_old"],
            axis_map={"P": "d0"},
            blocking_dims={"d1"},
            kwargs={"op": "rsqrt", "scale": inv_K, "bias": eps},
        ),
        Op(
            kind="NKIActivationReduce",
            inputs={"data": "sbuf_lhs"},
            outputs=["sbuf_m_state"],
            axis_map={"P": "d0", "F": "d1"},
            blocking_dims={"d1"},
            kwargs={"op": "square", "reduce_op": "add"},
        ),
        Op(
            kind="NKIActivation",
            inputs={"data": "sbuf_m_state"},
            outputs=["sbuf_rms_new"],
            axis_map={"P": "d0"},
            blocking_dims={"d1"},
            kwargs={"op": "rsqrt", "scale": inv_K, "bias": eps},
        ),
        Op(
            kind="NKIActivation",
            inputs={"data": "sbuf_rms_old"},
            outputs=["sbuf_rms_inv"],
            axis_map={"P": "d0"},
            blocking_dims={"d1"},
            kwargs={"op": "reciprocal"},
        ),
        Op(
            kind="NKITensorScalar",
            inputs={"data": "sbuf_rms_new", "operand0": "sbuf_rms_inv"},
            outputs=["sbuf_scale"],
            axis_map={"P": "d0"},
            blocking_dims={"d1"},
            kwargs={"op": "multiply"},
        ),
        Op(
            kind="NKITensorScalar",
            inputs={"data": "sbuf_lhs", "operand0": "sbuf_rms_new"},
            outputs=["sbuf_lhs_scaled"],
            axis_map={"P": "d0", "F": "d1"},
            blocking_dims={"d1"},
            kwargs={"op": "multiply"},
        ),
        Op(
            kind="NKIDMATranspose",
            inputs={"data": "sbuf_lhs_scaled"},
            outputs=["sbuf_lhs_T"],
            axis_map={"P": "d0", "F": "d1"},
            blocking_dims={"d1"},
        ),
        Op(
            kind="NKIOnlineMatmul",
            inputs={"stationary": "sbuf_lhs_T", "moving": "sbuf_rhs", "scale": "sbuf_scale"},
            outputs=["sbuf_output"],
            axis_map={"K": "d1", "M": "d0", "N": "d2"},
            blocking_dims={"d1"},
        ),
        Op(kind="NKIStore", inputs={"data": "sbuf_output"}, outputs=["hbm_output"], axis_map={"P": "d0", "F": "d2"}),
    ]

    edges = _derive_edges(ops)
    sbuf_names = [name for name in physical_buffers if not name.startswith("hbm_")]
    return KernelIR(
        func_name="rmsnorm_matmul_online",
        param_names=["lhs", "rhs"],
        return_name="output",
        dimensions=dimensions,
        logical_tensors=logical_tensors,
        physical_buffers=physical_buffers,
        ops=ops,
        edges=edges,
        dim_order=["d0", "d1", "d2"],
        ltiles_per_block={"d0": 1, "d1": 1, "d2": 1},
        buffer_scopes={name: _default_scope(name) for name in sbuf_names},
        num_buffers={name: NumBuffers() for name in sbuf_names},
        emission_depth={name: 0 for name in sbuf_names},
    )


def _default_scope(buf_name: str) -> BufferScope:
    """Per-buffer default scope.

    ``sbuf_output`` — MIDDLE so the K-reducer accumulates into a
    buffer that spans the full N axis (d2 full-extent) while d0 stays
    per-block. Under ``dim_order=[d0, d1, d2]``, store fires at depth
    1 (right after d1 closes, outside both d1 and d2) — the
    accumulator's slot must already cover d2 at that point.

    Everything else — INNER. Per-block tile sizing on every axis.
    """
    if buf_name == "sbuf_output":
        return BufferScope.MIDDLE
    return BufferScope.INNER


def _buf(
    tile: tuple[int, int], dim_ids: tuple[str, ...], dtype: str, p_axis: str, f_axis: str | None
) -> PhysicalBuffer:
    """Compact ``PhysicalBuffer`` constructor for the hand-built IR."""
    return PhysicalBuffer(tile=tile, dim_ids=dim_ids, dtype=dtype, p_axis=p_axis, f_axis=f_axis)


def _derive_edges(ops: list[Op]) -> list[tuple[int, int]]:
    """Forward-only producer→consumer edges.

    Cross-iteration reads (op 2 reading ``sbuf_m_state`` before op 3
    writes it) emit no edge — the producer hasn't been seen yet when
    the consumer is visited. That leaves the edge list a DAG, which
    downstream IR consumers rely on.
    """
    producer_of: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        for tname in op.inputs.values():
            src = producer_of.get(tname)
            if src is not None and src != i:
                edges.append((src, i))
        for out in op.outputs:
            producer_of[out] = i
    return edges


def _validate_input_specs(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """Enforce the online-fusion input contract.

    The kernel assumes ``lhs: (M, K)`` and ``rhs: (K, N)`` with matching
    dtypes and ``K`` dimensions. Mismatches would produce a renderable
    but numerically wrong IR; fail loudly here instead.
    """
    if "lhs" not in input_specs or "rhs" not in input_specs:
        raise ValueError(f"input_specs must contain 'lhs' and 'rhs' keys; got {sorted(input_specs)}")
    lhs_shape, lhs_dtype = input_specs["lhs"]
    rhs_shape, rhs_dtype = input_specs["rhs"]
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise ValueError(f"lhs and rhs must be 2D; got shapes {lhs_shape}, {rhs_shape}")
    if lhs_shape[1] != rhs_shape[0]:
        raise ValueError(f"K-dim mismatch: lhs.shape[1]={lhs_shape[1]} != rhs.shape[0]={rhs_shape[0]}")
    if lhs_dtype != rhs_dtype:
        raise ValueError(f"lhs/rhs dtypes must match; got {lhs_dtype} vs {rhs_dtype}")
