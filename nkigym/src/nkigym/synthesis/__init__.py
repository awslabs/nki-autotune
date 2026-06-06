"""Synthesis: translate numpy references into nkigym source via LLM calls.

``simulate_fp32`` is eagerly re-exported — it's a pure-numpy simulator with
no heavy deps. ``compile_numpy_to_nkigym`` is exposed lazily via ``__getattr__``
because its module hard-imports ``claude_agent_sdk`` at load time; an eager
re-export would drag that dependency into every consumer of this package
(e.g. the examples, which only want ``simulate_fp32``).
"""

from typing import Any

from nkigym.synthesis.simulate_nki import simulate_fp32

__all__ = ["compile_numpy_to_nkigym", "simulate_fp32"]


def __getattr__(name: str) -> Any:
    """Lazily resolve ``compile_numpy_to_nkigym`` to defer its heavy import."""
    if name == "compile_numpy_to_nkigym":
        from nkigym.synthesis.numpy_to_nkigym import compile_numpy_to_nkigym

        return compile_numpy_to_nkigym
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
