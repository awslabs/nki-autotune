"""Load one SBUF scalar through ``nisa.register_load``."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import _CONTROL_ITERATORS, _REPEAT_TRACE, NKIOp, _operand_role


class ControlEmitter(TorchArithmetic):
    """Emit bounded repetitions and explicit register-controlled regions."""

    def iota(self, width: int) -> str:
        """Emit exact floating positions for one contiguous row."""
        return self.emit("NKIIota", "", f"partitions=1, width={width}, pattern=[[1, {width}]], channel_multiplier=0")

    def line(self, statement: str) -> None:
        """Append a statement at the current lexical depth."""
        self.body.append("    " * self.depth + statement)

    def write(self, destination: str, value: str, predicate: str) -> None:
        """Update a scalar state tensor under an explicit predicate."""
        condition = self.cast("NKIUInt32Cast", predicate)
        self.imports.add("NKITensorCopyPredicated")
        self.line(f"{destination} = NKITensorCopyPredicated()(src={value}, predicate={condition}, dst={destination})")

    def copy_windows(self, destinations: tuple[str, ...], sources: tuple[str, ...], offset: str) -> None:
        """Copy matching windows into existing state arrays at one unsigned offset."""
        self.imports.add("NKIDynamicSliceCopy")
        for destination, value in zip(destinations, sources, strict=True):
            self.line(f"{destination} = NKIDynamicSliceCopy()(src={value}, offset={offset}, dst={destination})")

    @contextmanager
    def repeat(self, count: int) -> Iterator[None]:
        """Emit a fixed repetition without introducing a unit-trip IR loop."""
        if count > 1:
            self.imports.add("nkigym_repeat")
            self.line(f"for _ in nkigym_repeat({count}):")
            self.depth += 1
        try:
            yield
        finally:
            if count > 1:
                self.depth -= 1

    @contextmanager
    def guard(self, predicate: str) -> Iterator[None]:
        """Execute a region only when its Boolean register is true."""
        self.imports.add("nkigym_when")
        condition = self.cast("NKIUInt32Cast", predicate)
        register = self.emit("NKIRegisterLoad", f"src={condition}", "index=0")
        self.line(f"for _ in nkigym_when({register}):")
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1


def nkigym_when(predicate: Any, *, zero: bool = False) -> Iterator[None]:
    """Execute for a Boolean register's true value, or its zero value when requested."""
    if type(zero) is not bool:
        raise TypeError("nkigym_when zero must be a static Boolean")
    context = _REPEAT_TRACE.get()
    if context is None:
        value = np.asarray(predicate)
        if value.shape != (1, 1) or value.dtype != np.uint32 or int(value[0, 0]) not in {0, 1}:
            raise ValueError("nkigym_when requires one uint32 Boolean register")
        if bool(value[0, 0]) != zero:
            yield None
    else:
        name = getattr(predicate, "source_name", None)
        if getattr(predicate, "location", None) != "register" or not isinstance(name, str):
            raise TypeError("nkigym_when requires an explicit scalar register producer")
        start = len(context[0])
        yield None
        stop = len(context[0])
        if stop == start:
            raise ValueError("nkigym_when requires a nonempty primitive body")
        context[1].append((start, stop, (name, zero)))


_CONTROL_ITERATORS.add(nkigym_when)


class NKIRegisterLoad(NKIOp):
    """Read one uint32 selector into a virtual scalar register."""

    NAME: ClassVar[str] = "register_load"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("I", "J"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"src": "uint32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 1, "F": 1}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"I", "J", "P", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"I": 1, "J": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"I": 1, "J": None, "P": 1, "F": 1}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"src": ((1, "index", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"index", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "register"
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    def __init__(self, index: int) -> None:
        """Select one element from a single-row SBUF tensor."""
        super().__init__(index=index, width=1)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one uint32 SBUF row and an in-range scalar selection."""
        source = np.asarray(kwargs["src"])
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKIRegisterLoad(src=<role={role}>) expects SBUF")
        if source.ndim != 2 or source.shape[0] != 1 or source.dtype != np.uint32:
            raise ValueError("NKIRegisterLoad requires a single uint32 row")
        if not 0 <= int(kwargs["index"]) < source.shape[1]:
            raise ValueError("NKIRegisterLoad index exceeds the source width")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the selected scalar without changing its bits."""
        index = int(kwargs["index"])
        return np.asarray(kwargs["src"])[:, index : index + 1].copy()


__all__ = ["NKIRegisterLoad", "nkigym_when"]
