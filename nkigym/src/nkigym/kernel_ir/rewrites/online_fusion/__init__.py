"""Online-fusion rewrite package.

Layout:

* :mod:`core` — rewrite class, recipe registry, shared dataclasses,
  pattern-match utilities. Workload-agnostic.
* :mod:`rmsnorm_matmul` — pattern + ``build_after`` for
  ``ActRed(rsqrt) → TensorScalar(multiply) → Transpose → Matmul``.
* Future: :mod:`flash_attention`, etc.

Each recipe module registers itself on import via
:func:`core.register_recipe`. Importing this package triggers every
recipe's registration, so callers that want the rewrite ready-to-fire
simply do ``from nkigym.kernel_ir.rewrites.online_fusion import OnlineFusion``.
"""

from nkigym.kernel_ir.rewrites.online_fusion import rmsnorm_matmul  # noqa: F401  (registers recipe)
from nkigym.kernel_ir.rewrites.online_fusion.core import OnlineFusion, OnlineFusionMatch, Recipe, RewriteOutput

__all__ = ["OnlineFusion", "OnlineFusionMatch", "Recipe", "RewriteOutput"]
