"""Transform graph search for NKI Gym.

Explores the space of all possible transform application sequences.
Transforms follow an analyze/transform protocol where each ``transform()``
call produces a new callable that can feed into the next.  Different orderings
of independent transforms can converge to the same state.  This module
provides systematic exploration and sampling of that search space.
"""

from nkigym.search.search import search

__all__ = ["search"]
