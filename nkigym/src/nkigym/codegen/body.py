import math

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops.alloc import NKIAlloc

_INDENT = "    "
_PARTITION_DIM = 128


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body as a sequence of top-level subtrees.

    Each direct child of the tree root is rendered independently via
    :func:`_emit_block` with a fresh ``enclosing_loops`` map, so
    cardinal suffixes on ``i_<dim>_<cardinal>`` reset across siblings
    and only accumulate within a single loopnest path. The recursion
    is generic: any combination of :class:`ForNode`, :class:`NKIAlloc`
    leaves, and compute :class:`ISANode` leaves at any depth is
    supported. Both alloc and compute leaves enforce the axis-coverage
    invariant ``Π enclosing trips × tensorize_sizes[axis] ==
    dim_sizes[dim]`` for every axis in ``axis_map``.
    """
    block_nids = ir.tree.children(ir.tree.root)
    if not block_nids:
        raise ValueError("emit_body: tree root has no children")
    codes: list[str] = []
    for block_nid in block_nids:
        block_code = _emit_block(ir, block_nid, depth=1, enclosing_loops={}, block_code=[])
        codes += block_code
    return "\n".join(codes) + "\n"


def _emit_block(
    ir: KernelIR, nid: int, depth: int, enclosing_loops: dict[str, list[int]], block_code: list[str]
) -> list[str]:
    """Render the subtree rooted at ``nid`` as a multi-line string.

    ``enclosing_loops[d]`` is the list of trip counts for every
    enclosing :class:`ForNode` ancestor over dim ``d`` on the current
    path within this block, outermost-first. The cardinal for a fresh
    :class:`ForNode` over dim ``d`` is ``len(enclosing_loops[d])``.
    An ISA reference to dim ``d`` resolves to the linear combination
    ``Σ i_d_k * Π_{j>k} enclosing_loops[d][j]`` (a flat tile coordinate
    over the original axis range). The trip count appended on entry is
    popped on return so siblings under a shared parent see independent
    enclosing-loop chains.
    """
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    pushed_dim: str | None = None
    if isinstance(node, ForNode):
        enclosing_loops.setdefault(node.dim, []).append(node.trip)
        pushed_dim = node.dim
        block_code.append(indent + emit_loop(node, enclosing_loops))
        depth += 1
    elif isinstance(node, ISANode):
        _check_axis_coverage(node, ir, enclosing_loops)
        if node.op_cls is NKIAlloc:
            block_code.append(indent + emit_alloc(node))
        else:
            block_code.append(indent + emit_isa_call(node, ir, enclosing_loops))
    else:
        raise TypeError(f"unexpected node type {node}")

    for child_node in ir.tree.children(nid):
        _emit_block(ir, child_node, depth, enclosing_loops, block_code)

    if pushed_dim is not None:
        enclosing_loops[pushed_dim].pop()
        if not enclosing_loops[pushed_dim]:
            del enclosing_loops[pushed_dim]
    return block_code


def emit_loop(node: ForNode, enclosing_loops: dict[str, list[int]]) -> str:
    cardinal = len(enclosing_loops[node.dim]) - 1
    loop_str = f"for i_{node.dim}_{cardinal} in range({node.trip}):"
    return loop_str


def emit_alloc(node: ISANode) -> str:
    """Emit ``<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<location>)`` for one alloc.

    Shape follows the buffer's partition layout:

    * ``shared_hbm`` — 2D ``(P, F)`` (HBM is partition-flat).
    * ``sbuf`` / ``psum`` — 3D ``(128, P // 128, F)`` (SBUF/PSUM expose
      128 partitions explicitly; ``P`` must be a multiple of 128).

    The per-iteration footprint is ``tensorize_sizes`` (P, F); the
    axis-coverage invariant ``Π enclosing trips × tensorize_size ==
    dim_extent`` is enforced by :func:`_check_axis_coverage` before
    this emitter runs.
    """
    if len(node.writes) != 1:
        raise AssertionError(f"{NKIAlloc.__name__} expects exactly one write; got {len(node.writes)}: {node.writes}")
    P_size = node.tensorize_sizes["P"]
    F_size = node.tensorize_sizes["F"]
    if node.location == "shared_hbm":
        shape = f"({P_size}, {F_size})"
    else:
        if P_size % _PARTITION_DIM != 0:
            raise AssertionError(
                f"{NKIAlloc.__name__} on {node.location}: P={P_size} must be a multiple of {_PARTITION_DIM}"
            )
        shape = f"({_PARTITION_DIM}, {P_size // _PARTITION_DIM}, {F_size})"
    return f"{node.writes[0]} = nl.ndarray({shape}, dtype=nl.{node.dtype}, buffer=nl.{node.location})"


def emit_isa_call(node: ISANode, ir: KernelIR, enclosing_loops: dict[str, list[int]]) -> str:
    """Emit ``nisa.<NAME>(slot=<sliced_tensor>, ..., kwarg=value, ...)`` for one ISA node.

    Recovers each operand's slot name by re-walking ``op_cls.OPERAND_AXES``
    (slots are partitioned into reads / rmw / writes in declaration order),
    then builds a slice expression per slot from the operand tensor's
    location and the node's ``axis_map`` / ``tensorize_sizes``. SBUF/PSUM
    operands use a 3D access matching their 3D alloc ``(128, P//128, F)``:
    ``[0:P_tile, P_coord, F_coord*F_tile:(F_coord+1)*F_tile]``. HBM
    operands use a 2D access matching the flat ``(P, F)`` alloc:
    ``[(P_coord)*P_tile:(P_coord+1)*P_tile, (F_coord)*F_tile:(F_coord+1)*F_tile]``.
    When a dim has multiple ancestors on the path (post-Split same-dim
    nest), the coord is the flat linear combination
    ``i_d_0 * Π_{j>0} t_j + ... + i_d_{n-1}``. Non-operand
    ``ISANode.kwargs`` (e.g. ``value=0.0`` for memset) trail the operand
    kwargs in declaration order.
    """
    kwarg_to_tensor = _kwarg_to_tensor(node)
    operand_args: list[str] = []
    for kwarg in kwarg_to_tensor:
        tensor_name = kwarg_to_tensor[kwarg]
        tensor_slice_str = _render_tensor_slice(node, ir, kwarg, tensor_name, enclosing_loops)
        operand_args.append(f"{kwarg}={tensor_slice_str}")
    extra_args = [f"{k}={v!r}" for k, v in node.kwargs.items()]
    args = ", ".join(operand_args + extra_args)
    return f"nisa.{node.op_cls.NAME}({args})"


def _kwarg_to_tensor(node: ISANode) -> dict[str, str]:
    """Re-pair ``OPERAND_AXES`` kwargs with the tensor names on ``node``.

    Validates that ``node.reads`` / ``node.writes`` / ``node.rmw``
    each have exactly as many entries as the corresponding partition
    of ``OPERAND_AXES`` declares — extra or missing tensors are
    malformed IR and must fail loudly. Walks ``OPERAND_AXES`` in
    declaration order so the rendered ISA call matches the op's
    canonical kwarg ordering.
    """
    op_cls = node.op_cls
    expected = {"reads": 0, "rmw": 0, "writes": 0}
    for kwarg in op_cls.OPERAND_AXES:
        if kwarg in op_cls.INPUT_OPERANDS:
            expected["reads"] += 1
        elif kwarg in op_cls.RMW_OPERANDS:
            expected["rmw"] += 1
        else:
            expected["writes"] += 1
    actual = {"reads": len(node.reads), "rmw": len(node.rmw), "writes": len(node.writes)}
    if actual != expected:
        raise AssertionError(
            f"{op_cls.__name__}: operand-count mismatch. Expected {expected} from OPERAND_AXES, got {actual}"
        )
    read_iter = iter(node.reads)
    rmw_iter = iter(node.rmw)
    write_iter = iter(node.writes)
    kwarg_to_tensor: dict[str, str] = {}
    for kwarg in op_cls.OPERAND_AXES:
        if kwarg in op_cls.INPUT_OPERANDS:
            kwarg_to_tensor[kwarg] = next(read_iter)
        elif kwarg in op_cls.RMW_OPERANDS:
            kwarg_to_tensor[kwarg] = next(rmw_iter)
        else:
            kwarg_to_tensor[kwarg] = next(write_iter)
    return kwarg_to_tensor


def _check_axis_coverage(node: ISANode, ir: KernelIR, enclosing_loops: dict[str, list[int]]) -> None:
    """Assert ``Π enclosing trips × tensorize_size == dim_extent`` for every axis in ``axis_map``.

    The coverage rule is universal: any combination of enclosing-loop
    trip product and ``tensorize_sizes[axis]`` whose product equals the
    axis extent is legal. Zero enclosing loops are allowed when the
    leaf's ``tensorize_size`` already covers the full extent (empty
    product = 1). Names the leaf's op class and the failing concrete
    dim so error messages point at the offending node.
    """
    for abstract_axis, concrete_axis in node.axis_map.items():
        enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
        trip_product = math.prod(enclosing_axis_loops)
        tensorize_size = node.tensorize_sizes[abstract_axis]
        axis_extent = ir.dim_sizes[concrete_axis]
        assert trip_product * tensorize_size == axis_extent, (
            f"{node.op_cls.__name__} on axis {concrete_axis}: trip product {trip_product} "
            f"* tensorize size {tensorize_size} != axis extent {axis_extent} "
            f"(enclosing trips {enclosing_axis_loops})"
        )


def _render_tensor_slice(
    node: ISANode, ir: KernelIR, kwarg: str, tensor_name: str, enclosing_loops: dict[str, list[int]]
) -> str:
    """Render a single operand as ``<name>[<slice>]`` per its location and axes."""
    axes = node.op_cls.OPERAND_AXES[kwarg]
    location = ir.tensors[tensor_name].location
    slice_strs: list[str] = []
    for counter, abstract_axis in enumerate(axes):
        concrete_axis = node.axis_map[abstract_axis]
        enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
        tensorize_size = node.tensorize_sizes[abstract_axis]
        terms = [
            f"i_{concrete_axis}_{cardinal}*{math.prod(enclosing_axis_loops[cardinal + 1 :])}"
            for cardinal in range(len(enclosing_axis_loops))
        ]
        coord = " + ".join(terms) if terms else "0"
        if counter == 0 and location != "shared_hbm":
            slice_strs.append(f"0:{tensorize_size}")
            slice_strs.append(f"{coord}")
        else:
            slice_strs.append(f"({coord})*{tensorize_size}:({coord}+1)*{tensorize_size}")
    return f"{tensor_name}[{', '.join(slice_strs)}]"
