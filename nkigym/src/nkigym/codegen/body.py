from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode, KernelTree
from nkigym.ops.alloc import NKIAlloc

_INDENT = "    "
_P_TILE_SIZE = 128


def emit_body(ir: KernelIR) -> str:
    body: list[str] = []
    for nid in range(1, ir.tree.num_nodes):
        node_data = ir.tree.data(nid)
        depth = _loop_depth(ir.tree, nid)
        indent = _INDENT * (depth + 1)
        if type(node_data) == ISANode:
            if node_data.op_cls == NKIAlloc:
                alloc_str = emit_alloc(node_data)
                body.append(indent + alloc_str)
            else:
                isa_str = emit_isa_call(node_data, ir)
                body.append(indent + isa_str)
        elif type(node_data) == ForNode:
            loop_str = emit_loop(node_data)
            body.append(indent + loop_str)
    body_str = "\n".join(body) + "\n" if body else ""
    return body_str


def _loop_depth(tree: KernelTree, nid: int) -> int:
    """Count enclosing :class:`ForNode` ancestors of ``nid`` (root excluded)."""
    return sum(1 for aid in tree.ancestors(nid) if isinstance(tree.data(aid), ForNode))


def emit_alloc(node: ISANode) -> str:
    assert node.op_cls == NKIAlloc, f"emit_alloc expects {NKIAlloc} node, received {node.op_cls}"
    assert len(node.writes) == 1, f"{NKIAlloc} expects one write, received {node.writes}"
    alloc_buffer_var = node.writes[0]
    P_size = node.tensorize_sizes["P"]
    F_size = node.tensorize_sizes["F"]
    if node.location == "hbm":
        alloc_str = (
            f"{alloc_buffer_var} = nl.ndarray(({P_size}, {F_size}), dtype=nl.{node.dtype}, buffer=nl.shared_hbm)"
        )
    else:
        P_tile_size = 128
        num_P_tiles = P_size // P_tile_size
        assert P_size % P_tile_size == 0, f"only supports multiples of P dimension tile size of 128, received {P_size}"
        alloc_str = f"{alloc_buffer_var} = nl.ndarray(({P_tile_size}, {num_P_tiles}, {F_size}), dtype=nl.{node.dtype}, buffer=nl.{node.location})"
    return alloc_str


def emit_loop(node: ForNode) -> str:
    loop_str = f"for i_{node.dim} in range({node.trip}):"
    return loop_str


def emit_isa_call(node: ISANode, ir: KernelIR) -> str:
    """Emit ``nisa.<NAME>(slot=<sliced_tensor>, ..., kwarg=value, ...)`` for one ISA node.

    Recovers each operand's slot name by re-walking ``op_cls.OPERAND_AXES``
    (slots are partitioned into reads / rmw / writes in declaration order),
    then builds a slice expression per slot from the operand tensor's
    location and the node's ``axis_map`` / ``tensorize_sizes``. HBM operands
    use a 2D ``[P_var*P_tile:P_var*P_tile+P_tile, F_var*F_tile:F_var*F_tile+F_tile]``
    slice; SBUF/PSUM operands use the 3D layout matching :func:`emit_alloc`:
    ``[0:128, P_var, F_var*F_tile:F_var*F_tile+F_tile]``. Non-operand
    ``ISANode.kwargs`` (e.g. ``value=0.0`` for memset) trail the operand
    kwargs in declaration order.
    """
    slot_to_tensor = _slot_to_tensor(node)
    operand_args = [f"{slot}={_render_slot_slice(node, ir, slot, tensor)}" for slot, tensor in slot_to_tensor.items()]
    extra_args = [f"{k}={v!r}" for k, v in node.kwargs.items()]
    args = ", ".join(operand_args + extra_args)
    return f"nisa.{node.op_cls.NAME}({args})"


def _slot_to_tensor(node: ISANode) -> dict[str, str]:
    """Re-pair ``OPERAND_AXES`` slots with the tensor names on ``node``."""
    op_cls = node.op_cls
    read_iter = iter(node.reads)
    write_iter = iter(node.writes)
    rmw_iter = iter(node.rmw)
    slot_to_tensor: dict[str, str] = {}
    for slot in op_cls.OPERAND_AXES:
        if slot in op_cls.INPUT_OPERANDS:
            slot_to_tensor[slot] = next(read_iter)
        elif slot in op_cls.RMW_OPERANDS:
            slot_to_tensor[slot] = next(rmw_iter)
        else:
            slot_to_tensor[slot] = next(write_iter)
    return slot_to_tensor


def _render_slot_slice(node: ISANode, ir: KernelIR, slot: str, tensor_name: str) -> str:
    """Render a single operand as ``<name>[<slice>]`` per its location and axes."""
    axes = node.op_cls.OPERAND_AXES[slot]
    location = ir.tensors[tensor_name].location
    if len(axes) == 2:
        p_label, f_label = axes
        p_var = f"i_{node.axis_map[p_label]}"
        f_var = f"i_{node.axis_map[f_label]}"
        p_tile = node.tensorize_sizes[p_label]
        f_tile = node.tensorize_sizes[f_label]
        if location == "hbm":
            slice_str = f"{p_var}*{p_tile}:{p_var}*{p_tile}+{p_tile}, " f"{f_var}*{f_tile}:{f_var}*{f_tile}+{f_tile}"
        else:
            slice_str = f"0:{_P_TILE_SIZE}, {p_var}, {f_var}*{f_tile}:{f_var}*{f_tile}+{f_tile}"
    elif len(axes) == 1:
        (p_label,) = axes
        p_var = f"i_{node.axis_map[p_label]}"
        slice_str = f"0:{_P_TILE_SIZE}, {p_var}"
    else:
        raise ValueError(f"Unsupported operand-axes arity for {node.op_cls.__name__}.{slot}: {axes}")
    return f"{tensor_name}[{slice_str}]"
