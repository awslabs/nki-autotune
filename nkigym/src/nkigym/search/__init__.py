"""Schedule search for NKI Gym.

Provides two search strategies:
- ``search``: exhaustive enumeration of loop orders x placements x blocking.
- ``graph_search``: random transform sampling via a variant graph.
"""

from nkigym.search.compile import SearchResults
from nkigym.search.search import graph_search, search

__all__ = ["SearchResults", "graph_search", "search"]
