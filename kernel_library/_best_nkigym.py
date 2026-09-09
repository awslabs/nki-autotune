"""Recorded NKIGym kernels and transform ladders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class BestNKIGymLadderStep(TypedDict):
    """One replayable public-transform application."""

    transform: str
    option: dict[str, object]


@dataclass(frozen=True)
class BestNKIGymArtifact:
    """Compatibility source and transform ladder paired with one recorded latency."""

    kernel: str
    ladder: tuple[BestNKIGymLadderStep, ...]


BEST_NKIGYM_ARTIFACTS: dict[str, BestNKIGymArtifact] = {}


__all__ = ["BEST_NKIGYM_ARTIFACTS", "BestNKIGymArtifact", "BestNKIGymLadderStep"]
