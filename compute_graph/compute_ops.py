from typing import Any

from compute_graph.operators import Operator


class TensorScalar(Operator):
    """Element-wise operations on data tiles with scalar/vector operands.

    Supports chaining up to two operations with broadcasting along partition axis.
    Data and destination have shape (P, F); operands are scalars or (P, 1) vectors.
    """

    def __init__(
        self,
        dest: str,
        data: str,
        op0: Any,
        operand0: float | str,
        op1: Any = None,
        operand1: float | str | None = None,
    ) -> None:
        read_args = ("data",)
        write_args = ("dest",)
        axis_semantics = {"data": ("P", "F"), "dest": ("P", "F")}
        arg_to_var = {"data": data, "dest": dest}

        if isinstance(operand0, str):
            read_args += ("operand0",)
            axis_semantics["operand0"] = ("P", "1")
            arg_to_var["operand0"] = operand0

        if isinstance(operand1, str):
            read_args += ("operand1",)
            axis_semantics["operand1"] = ("P", "1")
            arg_to_var["operand1"] = operand1

        super().__init__(
            read_args=read_args, write_args=write_args, arg_to_var=arg_to_var, axis_semantics=axis_semantics
        )

        self.op0 = op0
        self.operand0 = operand0
        self.op1 = op1
        self.operand1 = operand1

    def codegen(self) -> str:
        """Generate NKI code for tensor_scalar operation."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]
        operand0 = self.arg_to_var["operand0"]

        args = [f"data={data}", f"op0={self.op0}", f"operand0={operand0}"]

        if self.op1 is not None:
            operand1 = self.arg_to_var["operand1"]
            args.append(f"op1={self.op1}")
            args.append(f"operand1={operand1}")

        args_str = ", ".join(args)
        return f"nisa.tensor_scalar({dest}, {args_str})"

    def __repr__(self) -> str:
        args = [f"data={self._format_tensor('data')}"]
        args.append(f"op0={self.op0}")

        if isinstance(self.arg_to_var, str):
            args.append(f"operand0={self._format_tensor('operand0')}")
        else:
            args.append(f"operand0={self.operand0}")

        if self.op1 is not None:
            args.append(f"op1={self.op1}")

            if isinstance(self.operand1, str):
                args.append(f"operand1={self._format_tensor('operand1')}")
            else:
                args.append(f"operand1={self.operand1}")

        args_str = ", ".join(args)
        return f"{self._format_tensor('dest')} = TensorScalar({args_str})"


class Activation(Operator):
    """Apply activation functions element-wise to input tiles.

    Optionally reduces along the free axis to shape (P, 1).
    Input and output shapes: (P, F) where P is partition, F is free axis.
    """

    def __init__(self, dest: str, op: Any, data: str, reduce_op: Any = None, reduce_res: str | None = None) -> None:
        read_args = ("data",)
        write_args = ("dest",)
        axis_semantics = {"data": ("P", "F"), "dest": ("P", "F")}
        arg_to_var = {"dest": dest, "data": data}
        if reduce_res:
            write_args += ("reduce_res",)
            axis_semantics["reduce_res"] = ("P", "1")
            arg_to_var["reduce_res"] = reduce_res
        super().__init__(
            read_args=read_args, write_args=write_args, arg_to_var=arg_to_var, axis_semantics=axis_semantics
        )

        self.op = op
        self.reduce_op = reduce_op

    def codegen(self) -> str:
        """Generate NKI code for activation operation."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]

        args = [f"op={self.op}", f"data={data}"]

        if self.reduce_op is not None and "reduce_res" in self.arg_to_var:
            reduce_res = self.arg_to_var["reduce_res"]
            args.append(f"reduce_op={self.reduce_op}")
            args.append(f"reduce_res={reduce_res}")

        args_str = ", ".join(args)
        return f"nisa.activation({dest}, {args_str})"

    def __repr__(self) -> str:
        data_str = self._format_tensor("data")
        args = [f"op={self.op}", f"data={data_str}"]
        result = self._format_tensor("dest")
        if "reduce_res" in self.arg_to_var and self.arg_to_var["reduce_res"]:
            reduce_res_str = self._format_tensor("reduce_res")
            args.append(f"reduce_op={self.reduce_op}")
            args.append(f"reduce_res={reduce_res_str}")
        args_str = ", ".join(args)
        return f"{result} = Activation({args_str})"


class Transpose(Operator):
    """2D transpose swapping partition and free axes.

    Transforms input (P, F) to output (F, P).
    """

    def __init__(self, dest: str, data: str) -> None:
        read_args = ("data",)
        write_args = ("dest",)
        axis_semantics = {"data": ("P", "F"), "dest": ("F", "P")}
        arg_to_var = {"data": data, "dest": dest}

        super().__init__(
            read_args=read_args, write_args=write_args, arg_to_var=arg_to_var, axis_semantics=axis_semantics
        )

    def codegen(self) -> str:
        """Generate NKI code for nc_transpose operation."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]
        return f"nisa.nc_transpose({dest}, {data})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = nisa.nc_transpose(data={self._format_tensor('data')})"


class TileTranspose(Operator):
    """In-tile transpose maintaining (P, F) shape.

    Rearranges element layout within the tile without changing axes,
    unlike nc_transpose which swaps partition and free dimensions.
    """

    def __init__(self, dest: str, data: str) -> None:
        read_args = ("data",)
        write_args = ("dest",)
        axis_semantics = {"data": ("P", "F"), "dest": ("P", "F")}
        arg_to_var = {"data": data, "dest": dest}

        super().__init__(
            read_args=read_args, write_args=write_args, arg_to_var=arg_to_var, axis_semantics=axis_semantics
        )

    def codegen(self) -> str:
        """Generate NKI code for in-tile transpose using nc_transpose."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]
        return f"nisa.nc_transpose({dest}, {data})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = TileTranspose(data={self._format_tensor('data')})"


class Matmul(Operator):
    """Matrix multiplication: lhs @ rhs with optional lhs transpose.

    Computes (M, K) @ (K, N) → (M, N), or (K, M).T @ (K, N) → (M, N).
    M, K, N represent rows, contraction, and columns axes respectively.
    """

    def __init__(self, dest: str, lhs: str, rhs: str, lhs_transposed: bool) -> None:
        read_args = ("lhs", "rhs")
        write_args = ("dest",)

        if lhs_transposed:
            axis_semantics = {"lhs": ("K", "M"), "rhs": ("K", "N"), "dest": ("M", "N")}
        else:
            axis_semantics = {"lhs": ("M", "K"), "rhs": ("K", "N"), "dest": ("M", "N")}

        arg_to_var = {"lhs": lhs, "rhs": rhs, "dest": dest}

        super().__init__(
            read_args=read_args, write_args=write_args, arg_to_var=arg_to_var, axis_semantics=axis_semantics
        )

        self.lhs_transposed = lhs_transposed

    def codegen(self) -> str:
        """Generate NKI code for nc_matmul operation.

        nc_matmul computes: dst = stationary.T @ moving
        For lhs @ rhs where lhs is (M, K) and rhs is (K, N):
        - stationary = lhs (will be transposed internally)
        - moving = rhs
        """
        dest = self.arg_to_var["dest"]
        lhs = self.arg_to_var["lhs"]
        rhs = self.arg_to_var["rhs"]
        return f"{dest} = nisa.nc_matmul({lhs}, {rhs})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = Matmul(lhs={self._format_tensor('lhs')}, rhs={self._format_tensor('rhs')})"
