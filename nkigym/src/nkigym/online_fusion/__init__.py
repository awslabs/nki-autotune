"""Online fusion: greedy math-level preprocessing.

Detects X + Accumulation patterns and eliminates blocking barriers
by introducing running state and scale corrections. Applied once
before programmatic transforms — not part of the search space.
"""
