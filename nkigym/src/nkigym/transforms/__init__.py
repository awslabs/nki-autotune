"""Transform passes for NKI Gym.

Transforms operate on the tiled IR produced by the tiling pass, optimizing
the compute graph before lowering to NKI. All transforms subclass the
Transform base class, which provides the analyze/transform interface.

Available transforms:
- DataReuseTransform: Identifies and merges redundant tensor loads.

To add a new transform:
1. Subclass Transform from nkigym.transforms.base
2. Implement analyze() and transform()
3. Export from this __init__.py
"""

from nkigym.transforms.base import Transform
from nkigym.transforms.data_reuse import (
    DataReuseTransform,
    analyze_data_reuse,
    merge_reusable_tensors,
    normalize_reuse_groups,
)

__all__ = ["Transform", "DataReuseTransform", "analyze_data_reuse", "merge_reusable_tensors", "normalize_reuse_groups"]
