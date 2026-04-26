"""Online-fused :class:`KernelIR` constructors.

Each constructor defines the vanilla (stateless-DAG) math function for
the workload, runs it through :func:`nkigym.kernel_ir.build.build_ir`,
then applies the recipe-based :class:`OnlineFusion` rewrite. The
rewrite swaps the vanilla op chain for the online-fusion recurrence
(with cross-iteration read-before-write on an accumulator buffer)
using the per-workload recipe registered in
:mod:`nkigym.kernel_ir.rewrites.online_fusion`.

The output is still a plain ``KernelIR`` that plugs into
:func:`nkigym.codegen.render_ir`, :func:`nkigym.kernel_ir.sample.sample`,
:func:`nkigym.kernel_ir.validate.validity_report`, and the tuning
infrastructure.
"""

from nkigym.kernel_ir.build import build_ir
from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.rewrites.online_fusion import OnlineFusion
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def build_rmsnorm_matmul_online_ir(input_specs: dict[str, tuple[tuple[int, ...], str]], eps: float = 1e-6) -> KernelIR:
    """Build the online-fused rmsnorm+matmul :class:`KernelIR`.

    Args:
        input_specs: ``{"lhs": (shape, dtype), "rhs": (shape, dtype)}``.
            ``lhs`` shape is ``(M, K)``, ``rhs`` shape is ``(K, N)``.
            Dtypes must match — the TE matmul drains through an fp32
            PSUM and the final output inherits the input dtype.
        eps: rmsnorm stabiliser added inside ``rsqrt(mean + eps)``.

    Returns:
        A canonical-knobs ``KernelIR`` realising the online recurrence:

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

        Constructed by parsing the vanilla
        ``NKIActivationReduce → NKITensorScalar → NKITranspose → NKIMatmul``
        chain via :func:`build_ir` and applying :class:`OnlineFusion`.
        The recipe mints the fp32 scratch buffers (``sbuf_*_m_state``,
        ``sbuf_*_rms_{old,new,inv}``, ``sbuf_*_scale``,
        ``sbuf_*_V0_scaled``, ``sbuf_*_V0_T``) and rewires ops into
        the cross-iteration structure.

    Raises:
        RuntimeError: When the rewrite finds no matching pattern.
    """
    _validate_input_specs(input_specs)

    """``_rmsnorm_matmul`` captures ``eps`` by closure and references
    it as ``bias=eps``. The parser's Name-resolution pass picks up
    ``eps`` from the function's globals (populated via ``build_ir``'s
    ``func.__globals__``) — the generated IR carries ``bias`` as a
    literal. ``scale`` is a placeholder; the recipe overrides it with
    ``1/K`` at rewrite time."""
    _eps = eps

    def rmsnorm_matmul_online(lhs, rhs):
        """Vanilla stateless-DAG math function for rmsnorm(lhs) @ rhs.

        The name survives as ``ir.func_name`` through the rewrite and
        ends up as the generated kernel's ``@nki.jit`` function name —
        call sites key the kernel off the IR's ``func_name`` field.
        """
        rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1.0, bias=_eps)(data=lhs)
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        output = NKIMatmul()(stationary=lhs_T, moving=rhs)
        return output

    vanilla = build_ir(rmsnorm_matmul_online, input_specs)
    rewrite = OnlineFusion()
    matches = rewrite.analyze(vanilla)
    if not matches:
        raise RuntimeError("OnlineFusion recipe did not match the vanilla rmsnorm+matmul IR")
    return rewrite.apply(vanilla, matches)


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
