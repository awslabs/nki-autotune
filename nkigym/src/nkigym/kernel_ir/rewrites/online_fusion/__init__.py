"""Online-fusion rewrite package.

Layout:

* :mod:`core` — rewrite class, recipe registry, shared dataclasses,
  pattern-match utilities. Workload-agnostic.
* :mod:`rmsnorm_matmul` — pattern + ``build_after`` for
  ``ActRed(rsqrt) → TensorScalar(multiply) → Transpose → Matmul``.
* :mod:`flash_attention` — pattern + ``build_after`` for the 9-op
  ``Matmul → ActRed(max) → TensorScalar(sub) → ActRed(exp+sum) →
  Activation(exp) → Activation(recip) → TensorScalar(mul) →
  Transpose → Matmul`` chain.

Each recipe module registers itself on import via
:func:`core.register_recipe`. Importing this package triggers every
recipe's registration, so callers that want the rewrite ready-to-fire
simply do ``from nkigym.kernel_ir.rewrites.online_fusion import OnlineFusion``.
"""

from nkigym.kernel_ir.rewrites.online_fusion import flash_attention, rmsnorm_matmul  # noqa: F401  (registers)
from nkigym.kernel_ir.rewrites.online_fusion.core import OnlineFusion, OnlineFusionMatch, Recipe, RewriteOutput

__all__ = ["OnlineFusion", "OnlineFusionMatch", "Recipe", "RewriteOutput"]
