"""Search: generate unique sampled KernelContext variants and profile them remotely."""

from nkigym.search.api import remote_search
from nkigym.search.sampler import sample_variants

__all__ = ["remote_search", "sample_variants"]
