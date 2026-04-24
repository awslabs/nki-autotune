"""Generic pattern-rewrite driver for ``KernelIR``."""

from typing import Protocol, runtime_checkable

from nkigym.kernel_ir.ir import KernelIR


@runtime_checkable
class MatchInstance(Protocol):
    """Marker type for pattern match instances."""


@runtime_checkable
class PatternRewrite(Protocol):
    """Protocol for one kind of ir rewrite.

    Patterns operate on a ``KernelIR`` and return a fresh
    ``KernelIR`` on ``apply``. The driver threads the IR through
    ``match`` / ``apply`` without mutation between calls.
    """

    name: str

    def match(self, ir: KernelIR) -> list[MatchInstance]:
        """Return zero or more ``MatchInstance`` describing applications of this pattern."""
        ...

    def apply(self, ir: KernelIR, instance: MatchInstance) -> KernelIR:
        """Rewrite ``ir`` for one match; return the mutated IR."""
        ...


def apply_rewrites_until_fixpoint(ir: KernelIR, patterns: list[PatternRewrite], max_iterations: int = 64) -> KernelIR:
    """Run every pattern until a full pass yields zero matches.

    Raises:
        RuntimeError: if the driver does not converge within
            ``max_iterations`` passes.
    """
    current = ir
    for _ in range(max_iterations):
        any_applied = False
        for pattern in patterns:
            while True:
                instances = pattern.match(current)
                if not instances:
                    break
                current = pattern.apply(current, instances[0])
                any_applied = True
        if not any_applied:
            return current
    raise RuntimeError(
        f"pattern-rewrite driver did not reach fixpoint after {max_iterations} passes — "
        "check for oscillating patterns"
    )
