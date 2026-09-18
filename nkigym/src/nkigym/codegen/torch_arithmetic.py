"""Emit explicit native operations for exact binary32 arithmetic."""

from dataclasses import dataclass, field


@dataclass
class TorchArithmetic:
    """Append named primitive operations to one frontend source body."""

    stem: str
    body: list[str]
    imports: set[str]
    depth: int = field(default=0, kw_only=True)

    def emit(self, operation: str, operands: str, configuration: str = "") -> str:
        """Emit one native-operation instance and return its SSA name."""
        name = f"{self.stem}_{len(self.body)}"
        self.imports.add(operation)
        self.body.append(f"{'    ' * self.depth}{name} = {operation}({configuration})({operands})")
        return name

    def cast(self, operation: str, source: str) -> str:
        """Emit one explicit numeric cast or bit reinterpretation."""
        operand = "data" if operation == "NKIFloat32Cast" else "src"
        return self.emit(operation, f"{operand}={source}")

    def scalar(self, operation: str, data: str, operand: str | float, reverse: bool = False) -> str:
        """Apply a scalar or per-partition operand to a tensor."""
        value = operand if isinstance(operand, str) else repr(operand)
        return self.emit(
            "NKITensorScalar",
            f"data={data}, operand0={value}",
            f"op0={operation!r}" + (", reverse0=True" if reverse else ""),
        )

    def binary(self, operation: str, left: str, right: str | int | float) -> str:
        """Choose the native scalar or tensor encoding of a binary operation."""
        bitwise = operation.startswith("bitwise_") or operation in {"left_shift", "right_shift"}
        prefix = "NKIBitwise" if bitwise else "NKITensor"
        tensor = isinstance(right, str)
        kind = "Tensor" if tensor else "Scalar"
        operands = f"data1={left}, data2={right}" if tensor else f"data={left}, operand0={right!r}"
        configuration = f"{'op' if tensor else 'op0'}={operation!r}"
        return self.emit(f"{prefix}{kind}", operands, configuration)

    def select(self, predicate: str, on_true: str, on_false: str) -> str:
        """Emit copy then native predicated copy, preserving RMW dependencies."""
        destination = self.emit("NKITensorCopy", f"src={on_false}", "engine='vector'")
        condition = self.cast("NKIUInt32Cast", predicate)
        return self.emit("NKITensorCopyPredicated", f"dst={destination}, src={on_true}, predicate={condition}")

    def integer(self, operation: str, left: str, right: str) -> str:
        """Emit exact word arithmetic without tensor-scalar floating conversion."""
        return self.emit("NKIUInt32Tensor", f"data1={left}, data2={right}", f"op={operation!r}")

    def inverse(self, predicate: str) -> str:
        """Invert a zero-or-one predicate."""
        return self.emit("NKITensorScalar", f"data={predicate}, operand0=1.0", "op0='subtract', reverse0=True")

    def slice(self, source: str, start: int, width: int = 1) -> str:
        """Read a fixed contiguous free-axis interval."""
        return self.emit("NKITensorSlice", f"src={source}", f"start={start}, width={width}")

    def prefix(self, source: str) -> str:
        """Compute an inclusive count of zero-or-one entries."""
        return self.emit("NKITensorScalarCumulative", f"src={source}", "op0='add', op1='add', imm0=0.0")

    def copy(self, value: str) -> str:
        """Allocate a separate tensor containing the same value."""
        return self.emit("NKITensorCopy", f"src={value}", "engine='vector'")

    def before(self, left: str, right: str) -> str:
        """Compare descending values with NaNs ordered first."""
        finite_left, finite_right = self.binary("equal", left, left), self.binary("equal", right, right)
        missing = self.inverse(finite_left)
        return self.binary(
            "maximum", self.binary("greater", left, right), self.binary("multiply", missing, finite_right)
        )

    def clamp(self, data: str, lower: str | float, upper: str | float) -> str:
        """Clamp finite scalar or vector values with one native instruction."""
        return self.emit(
            "NKITensorScalarSequence",
            f"data={data}, operand0={lower}, operand1={upper}",
            "op0='maximum', op1='minimum', engine='vector'",
        )

    def half(self, value: str) -> str:
        """Floor a nonnegative integer using an unsigned shift."""
        integer = self.cast("NKIUInt32Cast", self.scalar("maximum", value, 0.0))
        shifted = self.emit("NKIBitwiseScalar", f"data={integer}", "op0='right_shift', operand0=1")
        return self.cast("NKIFloat32Cast", shifted)
