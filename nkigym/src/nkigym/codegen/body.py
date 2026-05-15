import math

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops.alloc import NKIAlloc

_INDENT = "    "
_P_TILE_SIZE = 128


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body as a sequence of top-level subtrees.

    Each direct child of the tree root is rendered independently via
    :func:`_emit_block` with a fresh ``dim_trips`` map, so cardinal
    suffixes on ``i_<dim>_<cardinal>`` reset across siblings and only
    accumulate within a single loopnest path. The recursion is
    generic: any combination of :class:`ForNode`, :class:`NKIAlloc`
    leaves, and compute :class:`ISANode` leaves at any depth is
    supported. Both alloc and compute leaves enforce the axis-coverage
    invariant ``Π enclosing trips × tensorize_sizes[axis] ==
    dim_sizes[dim]`` for every axis in ``axis_map``.
    """
    codes: list[str] = []
    for block_nid in ir.tree.children(ir.tree.root):
        block_code = _emit_block(ir, block_nid, depth=1, enclosing_loops={}, block_code=[])
        codes += block_code
    return "\n".join(codes) + "\n" if codes else ""


def _emit_block(
    ir: KernelIR, nid: int, depth: int, enclosing_loops: dict[str, list[int]], block_code: list[str]
) -> list[str]:
    """Render the subtree rooted at ``nid`` as a multi-line string.

    ``dim_trips[d]`` is the list of trip counts for every enclosing
    :class:`ForNode` ancestor over dim ``d`` on the current path within
    this block, outermost-first. The cardinal for a fresh
    :class:`ForNode` over dim ``d`` is ``len(dim_trips[d])``. An ISA
    reference to dim ``d`` resolves to the linear combination
    ``Σ i_d_k * Π_{j>k} dim_trips[d][j]`` (a flat tile coordinate over
    the original axis range).
    """
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        if node.dim not in enclosing_loops:
            enclosing_loops[node.dim] = []
        enclosing_loops[node.dim].append(node.trip)
        depth += 1
        block_code.append(indent + emit_loop(node, enclosing_loops))
    elif isinstance(node, ISANode) and node.op_cls is NKIAlloc:
        emit_alloc(node, ir, enclosing_loops)
    elif isinstance(node, ISANode):
        block_code.append(indent + emit_isa_call(node, ir, enclosing_loops))
    else:
        raise TypeError(f"unexpected node type {node}")

    for child_node in ir.tree.children(nid):
        _emit_block(ir, child_node, depth, enclosing_loops, block_code)
    return block_code


def emit_loop(node: ForNode, enclosing_loops: dict[str, list[int]]) -> str:
    cardinal = len(enclosing_loops[node.dim]) - 1
    loop_str = f"for i_{node.dim}_{cardinal} in range({node.trip}):"
    return loop_str


def emit_alloc(node: ISANode, ir: KernelIR, enclosing_loops: dict[str, list[int]]) -> str:
    """Emit ``<name> = nl.ndarray(<shape>, dtype=..., buffer=...)`` for one alloc.

    The buffer's per-iteration footprint is ``tensorize_sizes`` (P, F).
    When the alloc sits inside a loop nest, only that footprint is
    materialised per iteration, so the same axis-coverage invariant
    used by ISA calls applies: ``accumulated_dims[dim] ×
    tensorize_sizes[axis] == dim_sizes[dim]`` for every axis in
    ``axis_map``. A mismatch means the loop nest does not fully cover
    the declared tensor extent (or covers it more than once).
    """
    assert node.op_cls is NKIAlloc, f"emit_alloc expects {NKIAlloc} node, received {node.op_cls}"
    assert len(node.writes) == 1, f"{NKIAlloc} expects one write, received {node.writes}"
    P_concrete_dim = node.axis_map["P"]
    F_concrete_dim = node.axis_map["F"]
    P_size = ir.dim_sizes[P_concrete_dim]
    F_size = ir.dim_sizes[F_concrete_dim]
    # print(f"node {node} enclosing_loops {enclosing_loops} P_size {P_size} F_size {F_size}")
    # P_size = node.tensorize_sizes["P"]
    # F_size = node.tensorize_sizes["F"]
    # if node.location == "hbm":
    #     alloc_str = f"{node.writes[0]} = nl.ndarray(({P_size}, {F_size}), dtype=nl.{node.dtype}, buffer=nl.shared_hbm)"
    # else:
    #     num_P_tiles = P_size // _P_TILE_SIZE
    #     assert (
    #         P_size % _P_TILE_SIZE == 0
    #     ), f"only supports multiples of P dimension tile size of {_P_TILE_SIZE}, received {P_size}"
    #     alloc_str = f"{node.writes[0]} = nl.ndarray(({_P_TILE_SIZE}, {num_P_tiles}, {F_size}), dtype=nl.{node.dtype}, buffer=nl.{node.location})"
    # return alloc_str


def emit_isa_call(node: ISANode, ir: KernelIR, enclosing_loops: dict[str, list[int]]) -> str:
    """Emit ``nisa.<NAME>(slot=<sliced_tensor>, ..., kwarg=value, ...)`` for one ISA node.

    Recovers each operand's slot name by re-walking ``op_cls.OPERAND_AXES``
    (slots are partitioned into reads / rmw / writes in declaration order),
    then builds a slice expression per slot from the operand tensor's
    location and the node's ``axis_map`` / ``tensorize_sizes``. HBM operands
    use a 2D ``[P_coord*P_tile:P_coord*P_tile+P_tile, F_coord*F_tile:F_coord*F_tile+F_tile]``
    slice; SBUF/PSUM operands use the 3D layout matching :func:`emit_alloc`:
    ``[0:128, P_coord, F_coord*F_tile:F_coord*F_tile+F_tile]``. When a
    dim has multiple ancestors on the path (post-Split same-dim nest),
    the coord is the flat linear combination
    ``i_d_0 * Π_{j>0} t_j + ... + i_d_{n-1}``. Non-operand
    ``ISANode.kwargs`` (e.g. ``value=0.0`` for memset) trail the operand
    kwargs in declaration order.
    """
    kwarg_to_tensor = _kwarg_to_tensor(node)
    operand_args: list[str] = []
    for kwarg in kwarg_to_tensor:
        tensor_name = kwarg_to_tensor[kwarg]
        operand_args.append(_render_tensor_slice(node, ir, kwarg, tensor_name, enclosing_loops))
    extra_args = [f"{k}={v!r}" for k, v in node.kwargs.items()]
    args = ", ".join(operand_args + extra_args)
    return f"nisa.{node.op_cls.NAME}({args})"


def _kwarg_to_tensor(node: ISANode) -> dict[str, str]:
    """Re-pair ``OPERAND_AXES`` kwargs with the tensor names on ``node``."""
    op_cls = node.op_cls
    read_iter = iter(node.reads)
    write_iter = iter(node.writes)
    rmw_iter = iter(node.rmw)
    kwarg_to_tensor: dict[str, str] = {}
    for kwarg in op_cls.OPERAND_AXES:
        if kwarg in op_cls.INPUT_OPERANDS:
            kwarg_to_tensor[kwarg] = next(read_iter)
        elif kwarg in op_cls.RMW_OPERANDS:
            kwarg_to_tensor[kwarg] = next(rmw_iter)
        else:
            kwarg_to_tensor[kwarg] = next(write_iter)
    return kwarg_to_tensor


def _render_tensor_slice(
    node: ISANode, ir: KernelIR, kwarg: str, tensor_name: str, enclosing_loops: dict[str, list[int]]
) -> str:
    """Render a single operand as ``<name>[<slice>]`` per its location and axes."""
    axes = node.op_cls.OPERAND_AXES[kwarg]
    location = ir.tensors[tensor_name].location
    slice_strs: list[str] = []
    for counter, abstract_axis in enumerate(axes):
        concrete_axis = node.axis_map[abstract_axis]
        enclosing_axis_loops = enclosing_loops[concrete_axis]
        total_trip_count = math.prod(enclosing_axis_loops)
        tensorize_size = node.tensorize_sizes[abstract_axis]
        axis_extent = ir.dim_sizes[concrete_axis]
        assert (
            total_trip_count * tensorize_size == axis_extent
        ), f"Loop trip counts {enclosing_axis_loops} * ISA tensorize size {tensorize_size} mismatch axis_extent {axis_extent}"
        terms: list[str] = []
        for cardinal in range(len(enclosing_axis_loops)):
            stride = math.prod(enclosing_axis_loops[cardinal + 1 :])
            term = f"i_{concrete_axis}_{cardinal}*{stride}"
            terms.append(term)
        coord = " + ".join(terms)
        if counter == 0 and location == "hbm":
            slice_strs.append(f"0:{tensorize_size}")
            slice_strs.append(f"{coord}")
        else:
            slice_strs.append(f"({coord})*{tensorize_size}:({coord}+1)*{tensorize_size}")
    tensor_slice = ", ".join(slice_strs)
    tensor_slice = f"{tensor_name}[{tensor_slice}]"
    return tensor_slice
