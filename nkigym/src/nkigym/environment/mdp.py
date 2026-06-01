"""Markov Decision Process wrapper over Split + Fuse transforms.

State = ``KernelIR``. Action = ``(Transform, TransformOption)``. The env is
pure-functional: every method takes the state explicitly. See
``docs/superpowers/specs/2026-05-17-mdp-environment-design.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.transforms import Transform, TransformOption

Action = tuple[Transform, TransformOption]


class KernelMDP:
    """Pure-functional MDP over ``KernelIR`` with a caller-supplied transform list.

    Attributes:
        kernel_func: ``@nkigym_kernel``-decorated callable used by ``reset``.
        input_specs: ``{param_name: (shape, dtype)}`` passed to ``build_initial_ir``.
        transforms: Action namespace - each entry contributes its
            ``analyze`` options to ``legal_actions``.
    """

    def __init__(
        self,
        kernel_func: Callable[..., Any],
        input_specs: dict[str, tuple[tuple[int, ...], str]],
        transforms: list[Transform],
    ) -> None:
        """Store the kernel + transforms. No IR is built until ``reset``."""
        self.kernel_func = kernel_func
        self.input_specs = input_specs
        self.transforms = transforms

    def reset(self) -> KernelIR:
        """Build and return the canonical IR from ``kernel_func`` + ``input_specs``."""
        return build_initial_ir(self.kernel_func, self.input_specs)

    def legal_actions(self, state: KernelIR) -> list[Action]:
        """Flat list of ``(transform, option)`` pairs across every transform's analyze.

        Order: transforms in construction order; within each transform, the
        order produced by its ``analyze`` method.
        """
        actions: list[Action] = []
        for transform in self.transforms:
            for option in transform.analyze(state):
                actions.append((transform, option))
        return actions

    def step(self, state: KernelIR, action: Action) -> KernelIR:
        """Unpack ``(transform, option)`` and apply.

        Returns a new ``KernelIR``; does not mutate ``state``
        (``Transform.apply`` already deep-copies). Raises
        ``TransformLegalityError`` (loud failure) if ``option`` is illegal
        for ``state``.
        """
        transform, option = action
        return transform.apply(state, option)

    def is_terminal(self, state: KernelIR) -> bool:
        """Return ``True`` iff ``legal_actions(state)`` is empty."""
        return len(self.legal_actions(state)) == 0


__all__ = ["Action", "KernelMDP"]
