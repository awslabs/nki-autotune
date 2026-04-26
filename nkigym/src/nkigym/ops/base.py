"""NKIOp base class for math-level kernel descriptions.

Each ``NKIOp`` subclass maps 1:1 to a real ``nisa.*`` ISA instruction.
Subclasses implement ``__call__`` for CPU simulation (numpy) and declare
axis semantics and hardware limits via class attributes.
"""

from abc import abstractmethod
from typing import Any, ClassVar


class NKIOp:
    """Base for all NKI operator definitions.

    Supports two call-site syntaxes:

    * ``Op()(data=x)`` — single call, all kwargs on ``__call__``.
    * ``Op(op='square', ...)(data=x)`` — split: configuration literals
      on the constructor, tensor operands on the invocation. Constructor
      kwargs are stashed and merged into the final ``__call__`` kwargs
      at CPU-sim time.

    Attributes:
        NAME: ISA call name (e.g. ``"nc_matmul"``).
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
        OUTPUT_DTYPES: Optional per-output dtype override. Default empty
            means outputs inherit the first operand's dtype (typical for
            elementwise/math ops). ``NKIActivationReduce`` overrides its
            reduce output to ``float32`` so the Scalar Engine can
            accumulate without narrowing.
        BLOCKING_AXES: Accumulation axes (e.g. ``{"K"}`` for matmul).
        TILE_LIMITS: Hardware tile size per abstract axis.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    def __init__(self, **kwargs: Any) -> None:
        """Stash constructor kwargs for merging into ``__call__`` kwargs."""
        self._init_kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Subclass-specific numpy simulation. Gets the merged kwargs."""

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation — merge init kwargs with call kwargs and dispatch to ``_run``."""
        merged = {**getattr(self, "_init_kwargs", {}), **kwargs}
        return self._run(**merged)
