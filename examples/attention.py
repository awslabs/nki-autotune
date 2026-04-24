"""Causal attention: ``build_ir`` + atomic OnlineFusion to flash attention.

Defines the naive attention math as an nkigym function, uses
``build_ir`` to synthesize the baseline ``KernelIR`` from that
function + input shapes, then applies :class:`OnlineFusion` to
fixpoint. The atomic rewrite reaches flash attention in two
applications that share the same running-max state:

1. ``(R = tensor_reduce(max), A = activation_reduce(exp+sum))`` on the
   seq_k dim — direct bias link. Allocates ``sbuf_running_*`` and
   ``sbuf_scale_*``, inserts ``update_running`` + ``compute_scale``
   ops, rescales the exp+sum accumulator.
2. ``(R = tensor_reduce(max), A = matmul(exp_S_t, V))`` — transitive
   link through ``transpose → activation_reduce(bias=running)``.
   Reuses the existing running/scale buffers; only inserts a second
   ``rescale`` on the P@V accumulator.

Math: ``output = softmax(scale * Q_T @ K) @ V`` with Q given
already-transposed (the matmul op contract is ``stationary.T @ moving``
so ``stationary = Q_T`` yields ``Q @ K^T`` when ``moving = K``). The
causal mask is ignored — it is the job of :class:`ComputeSkipping`.
"""

import numpy as np

from nkigym.kernel_ir import KernelIR, build_ir
from nkigym.kernel_ir.rewrites import OnlineFusion
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.reciprocal import NKIReciprocal
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

SEQ_LEN = 2048
D_K = 128
D_V = 128
SCALE = 1.0 / (D_K**0.5)


def attention_nkigym(Q_T: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Naive attention math — one nkigym op per pass.

    ``stationary.T @ moving`` contract: we pass ``Q_T (d_k, seq_q)``
    and ``K (d_k, seq_k)``, yielding ``S (seq_q, seq_k)``.
    """
    S = NKIMatmul()(stationary=Q_T, moving=K)
    scaled_S = NKITensorScalar()(data=S, op0="multiply", operand0=SCALE)
    neg_max = NKITensorReduce()(data=scaled_S, op="maximum", axis=1, negate=True)
    exp_S, sum_exp = NKIActivationReduce()(data=scaled_S, op="exp", bias=neg_max, reduce_op="add")
    inv_sum = NKIReciprocal()(data=sum_exp)
    exp_S_t = NKITranspose()(data=exp_S)
    O_unnorm = NKIMatmul()(stationary=exp_S_t, moving=V)
    output = NKITensorScalar()(data=O_unnorm, op0="multiply", operand0=inv_sum)
    return output


def build_naive_attention_ir() -> KernelIR:
    """``build_ir`` from the math function — naive baseline with canonical defaults."""
    input_specs = {
        "Q_T": ((D_K, SEQ_LEN), "bfloat16"),
        "K": ((D_K, SEQ_LEN), "bfloat16"),
        "V": ((SEQ_LEN, D_V), "bfloat16"),
    }
    return build_ir(attention_nkigym, input_specs)


def print_ops(ir: KernelIR, title: str) -> None:
    """Pretty-print the op list for quick diffing."""
    print(f"\n{title} — {len(ir.ops)} ops, {len(ir.physical_buffers)} physical buffers")
    print("─" * 88)
    for i, op in enumerate(ir.ops):
        inputs = ", ".join(f"{r}={n}" for r, n in op.inputs.items())
        outputs = ", ".join(op.outputs)
        block = f"  blocking={sorted(op.blocking_dims)}" if op.blocking_dims else ""
        role = op.attrs.get("online_fusion_role")
        role_str = f"  role={role}" if role else ""
        fused = "  [X-fused]" if op.attrs.get("online_fused") else ""
        cfused = "  [A-fused]" if op.attrs.get("online_fused_consumer") else ""
        print(f"  #{i:2d} {op.kind:24s} ({inputs}) -> [{outputs}]{block}{role_str}{fused}{cfused}")


def describe_changes(naive: KernelIR, fused: KernelIR) -> None:
    """Print the key IR-field diffs that OnlineFusion produced."""
    print("\n" + "=" * 88)
    print("OnlineFusion changes")
    print("=" * 88)

    added_buffers = sorted(set(fused.physical_buffers) - set(naive.physical_buffers))
    print(f"\nNew physical buffers: {added_buffers}")
    for name in added_buffers:
        pb = fused.physical_buffers[name]
        print(f"  {name}: tile={pb.tile} dims={pb.dim_ids} dtype={pb.dtype}")

    inserted_roles = [op.attrs.get("online_fusion_role") for op in fused.ops if op.attrs.get("online_fusion_role")]
    print(f"\nInserted correction ops: {inserted_roles}")

    print("\nBlocking_dims before → after:")
    for i, op in enumerate(naive.ops):
        if not op.blocking_dims:
            continue
        matched = [j for j, fop in enumerate(fused.ops) if fop.kind == op.kind and fop.outputs == op.outputs]
        if not matched:
            continue
        j = matched[0]
        after = fused.ops[j].blocking_dims
        marker = " ★" if after != op.blocking_dims else ""
        print(f"  {op.kind:24s} #{i}→#{j}: {sorted(op.blocking_dims)} -> {sorted(after)}{marker}")


if __name__ == "__main__":
    naive = build_naive_attention_ir()
    print_ops(naive, "Naive attention IR (from build_ir)")

    of = OnlineFusion()
    current = naive
    step = 0
    while True:
        matches = of.match(current)
        if not matches:
            break
        step += 1
        m = matches[0]
        r = current.ops[m.reducer_index]
        c = current.ops[m.consumer_index]
        shared = "(shared X — reuse running/scale)" if m.shared_x else "(first X — allocate running/scale)"
        print(
            f"\nStep {step}: R#{m.reducer_index} {r.kind}({list(r.outputs)})  →  "
            f"C#{m.consumer_index} {c.kind}({list(c.outputs)})  on {m.blocking_dim} via role={m.bias_role} {shared}"
        )
        current = of.apply(current, m)

    flash = current
    print(f"\nTotal OnlineFusion applications: {step}")
    print_ops(flash, "Flash attention IR (after OnlineFusion)")
    describe_changes(naive, flash)
