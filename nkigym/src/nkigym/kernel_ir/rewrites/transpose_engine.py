"""``TransposeEngine``: flip :class:`NKITranspose` backend mode.

Sets ``Op.attrs["mode"]`` on each matched ``NKITranspose`` between
``"dma_transpose"`` and ``"nc_transpose"``. Each mode maps to a
different NKI ISA instruction:

* ``dma_transpose`` — SBUF→SBUF via ``nisa.dma_transpose`` (DMA).
* ``nc_transpose`` — SBUF→PSUM via Tensor Engine; PSUM→SBUF via
  ``tensor_copy``.

The rewrite is stochastic and toggles state, so it is not intended
for :func:`apply_rewrites_until_fixpoint`. Samplers draw one
``(op_index, target_mode)`` match at a time.
"""

from dataclasses import dataclass, replace

from nkigym.kernel_ir.ir import KernelIR, Op

_VALID_MODES: tuple[str, ...] = ("dma_transpose", "nc_transpose")


@dataclass(frozen=True)
class TransposeEngineMatch:
    """One candidate application of ``TransposeEngine``.

    Attributes:
        op_index: Index into ``ir.ops`` of the target ``NKITranspose``.
        target_mode: Mode to write; must differ from the op's current mode.
    """

    op_index: int
    target_mode: str


class TransposeEngine:
    """Flip the ``mode`` attribute on one ``NKITranspose`` op."""

    name = "transpose_engine"

    def match(self, ir: KernelIR) -> list[TransposeEngineMatch]:
        """Enumerate every ``(transpose_op, alternate_mode)`` pair.

        Each transpose op contributes one match — the mode that is
        NOT its current one.
        """
        matches: list[TransposeEngineMatch] = []
        for i, op in enumerate(ir.ops):
            if op.kind != "NKITranspose":
                continue
            current = op.attrs.get("mode", "dma_transpose")
            if current not in _VALID_MODES:
                raise ValueError(f"NKITranspose at index {i} has invalid mode {current!r}")
            alternate = "nc_transpose" if current == "dma_transpose" else "dma_transpose"
            matches.append(TransposeEngineMatch(op_index=i, target_mode=alternate))
        return matches

    def apply(self, ir: KernelIR, instance: TransposeEngineMatch) -> KernelIR:
        """Return a new IR with the target op's mode set to ``instance.target_mode``."""
        if instance.target_mode not in _VALID_MODES:
            raise ValueError(f"invalid target_mode {instance.target_mode!r}")
        old_op = ir.ops[instance.op_index]
        if old_op.kind != "NKITranspose":
            raise ValueError(
                f"TransposeEngine.apply at index {instance.op_index}: " f"expected NKITranspose, got {old_op.kind!r}"
            )
        new_attrs = dict(old_op.attrs)
        new_attrs["mode"] = instance.target_mode
        new_op = replace(old_op, attrs=new_attrs)
        new_ops: list[Op] = list(ir.ops)
        new_ops[instance.op_index] = new_op
        return replace(ir, ops=new_ops)
