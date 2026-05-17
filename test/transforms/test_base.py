"""Smoke tests for the transforms base classes."""

from dataclasses import dataclass

import pytest

from nkigym.transforms import Transform, TransformLegalityError, TransformOption


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


def test_transform_base_methods_raise_not_implemented():
    """``Transform.analyze`` and ``Transform.apply`` must raise ``NotImplementedError``."""
    t = Transform()
    with pytest.raises(NotImplementedError):
        t.analyze(ir=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        t.apply(ir=None, option=TransformOption())  # type: ignore[arg-type]
