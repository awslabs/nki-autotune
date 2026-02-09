"""Transform passes for NKI Gym.

Transforms are atomic rewrites of the tiled IR. Each transform is
independent and self-contained: it inspects the IR, finds optimization
opportunities, and applies them. One transform may expose opportunities
for the next, but they have no internal coupling. The autotuner searches
over combinations to find the best kernel.

Every transform follows the analyze/transform protocol defined by the
``Transform`` base class:

1. ``analyze(func)`` — inspect the IR and return a list of opportunities.
2. ``transform(func, option)`` — apply a single opportunity, returning a new callable.

Available transforms:

- ``DataReuseTransform`` — deduplicates identical tensor loads.
"""

from nkigym.transforms.base import Transform
from nkigym.transforms.data_reuse import (
    DataReuseTransform,
    analyze_data_reuse,
    merge_reusable_tensors,
    normalize_reuse_groups,
)

__all__ = ["Transform", "DataReuseTransform", "analyze_data_reuse", "merge_reusable_tensors", "normalize_reuse_groups"]
