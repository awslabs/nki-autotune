"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

``nkigym_compile`` is exposed as a top-level attribute via lazy
``__getattr__`` so that importing submodules (e.g. ``nkigym.ops``)
does not eagerly pull in the synthesis + compile chain — that chain
depends on ``claude_agent_sdk``, which is absent on remote gym
workers that only need the math-layer ``nkigym.ops``.
"""

from typing import Any

__all__ = ["nkigym_compile"]


def __getattr__(name: str) -> Any:
    """Lazy-load ``nkigym_compile`` on first access."""
    if name == "nkigym_compile":
        from nkigym.compile import nkigym_compile

        return nkigym_compile
    raise AttributeError(f"module 'nkigym' has no attribute {name!r}")
