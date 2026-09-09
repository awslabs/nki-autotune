"""Recorded best NKIGym transform ladders."""

from __future__ import annotations

from typing import Any

from nkigym.transforms import Transform, TransformOption

BEST_NKIGYM_LADDERS: dict[str, tuple[tuple[Transform[Any], TransformOption], ...]] = {}


__all__ = ["BEST_NKIGYM_LADDERS"]
