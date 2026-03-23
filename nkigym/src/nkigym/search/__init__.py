"""Combinatorial schedule search for NKI Gym.

Enumerates all valid schedules from loop orders x op placements x blocking.
Each unique schedule is rendered to NKI source, compiled, and benchmarked
on hardware.
"""

from nkigym.search.compile import SearchResults
from nkigym.search.search import search

__all__ = ["SearchResults", "search"]
