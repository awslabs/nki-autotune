"""Weak interning for immutable value objects."""

from __future__ import annotations

from typing import Self, cast
from weakref import WeakValueDictionary

_InternKey = tuple[type["InternedValue"], tuple[object, ...], tuple[tuple[str, object], ...]]
_INTERNED_VALUES: WeakValueDictionary[_InternKey, "InternedValue"] = WeakValueDictionary()


class InternedValue:
    """Reuse structurally equal immutable values while they remain live."""

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        """Return the live instance for one constructor argument set."""
        if not args and not kwargs:
            return cast(Self, object.__new__(cls))
        key: _InternKey = (cls, args, tuple(sorted(kwargs.items())))
        instance = _INTERNED_VALUES.get(key)
        if instance is None:
            instance = object.__new__(cls)
            _INTERNED_VALUES[key] = instance
        return cast(Self, instance)
