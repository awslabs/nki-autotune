"""Smoke tests for the transforms base classes."""

from dataclasses import dataclass

from nkigym.transforms import TransformLegalityError, TransformOption


def test_transform_option_is_frozen_dataclass():
    """``TransformOption`` instances must be hashable (frozen dataclass)."""

    @dataclass(frozen=True)
    class _Opt(TransformOption):
        x: int = 0

    a = _Opt(x=1)
    b = _Opt(x=1)
    assert a == b
    assert hash(a) == hash(b)


def test_transform_legality_error_is_value_error():
    """``TransformLegalityError`` must be a ``ValueError`` subclass."""
    assert issubclass(TransformLegalityError, ValueError)
