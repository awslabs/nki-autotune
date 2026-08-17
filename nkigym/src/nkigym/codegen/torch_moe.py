"""Source emission for tiled Torch MoE expert computation."""

from nkigym.codegen.torch_values import TorchValue, emit_slice


def emit_moe(
    inputs: tuple[TorchValue, TorchValue, TorchValue, TorchValue, TorchValue],
    experts: int,
    intermediate: int,
    stem: str,
    body: list[str],
    imports: set[str],
) -> TorchValue:
    """Emit the selected expert MLPs as native tiled matmuls."""
    hidden, gate_up_weights, down_weights, affinities, indices = inputs
    hidden_width = hidden.shape[1]
    if (
        hidden.shape[0] != 1
        or gate_up_weights.shape != (experts, hidden_width * 2 * intermediate)
        or down_weights.shape != (experts, intermediate * hidden_width)
        or affinities.shape != indices.shape
        or affinities.shape != (1, 8)
        or intermediate % 128
    ):
        raise ValueError("Torch MoE expert tensors have incompatible normalized shapes")
    imports.update(
        "NKIActivation NKIActivationReduce NKIDMATranspose NKIHBMScalarRowSlice "
        "NKIMatmul NKITensorCopy NKITensorScalar NKITensorTensor".split()
    )
    hidden_stationary = hidden
    if not hidden.transposed:
        hidden_stationary = TorchValue(f"sbuf_{stem}_hidden_stationary", hidden.shape, True)
        body.append(f"{hidden_stationary.name} = NKIDMATranspose()(src={hidden.name})")
    total: TorchValue | None = None
    for route in range(8):
        affinity = emit_slice(affinities, route, 1, f"sbuf_{stem}_affinity_{route}", "", body, imports)
        index = emit_slice(indices, route, 1, f"sbuf_{stem}_index_{route}", "", body, imports)
        affinity_vector = TorchValue(f"{affinity.name}_vector", (1,))
        body.append(f'{affinity_vector.name} = NKIActivationReduce(op="copy", reduce_op="add")(data={affinity.name})')
        gate_up = TorchValue(f"sbuf_{stem}_gate_up_weight_{route}", (hidden_width, 2 * intermediate))
        body.append(
            f"{gate_up.name} = NKIHBMScalarRowSlice(rows={hidden_width}, width={2 * intermediate})"
            f"(src={gate_up_weights.name}, indices={index.name})"
        )
        gate_up_psum = f"psum_{stem}_gate_up_{route}"
        projected = TorchValue(f"sbuf_{stem}_gate_up_{route}", (1, 2 * intermediate))
        body.extend(
            (
                f"{gate_up_psum} = NKIMatmul()(stationary={hidden_stationary.name}, moving={gate_up.name})",
                f"{projected.name} = NKITensorCopy()(src={gate_up_psum})",
            )
        )
        gate = emit_slice(projected, 0, intermediate, f"sbuf_{stem}_gate_{route}", "", body, imports)
        up = emit_slice(projected, intermediate, intermediate, f"sbuf_{stem}_up_{route}", "", body, imports)
        sigmoid = TorchValue(f"sbuf_{stem}_sigmoid_{route}", gate.shape)
        activated = TorchValue(f"sbuf_{stem}_activated_{route}", gate.shape)
        intermediate_value = TorchValue(f"sbuf_{stem}_intermediate_{route}", gate.shape)
        body.extend(
            (
                f'{sigmoid.name} = NKIActivation(op="sigmoid", scale=1.702)(data={gate.name})',
                f'{activated.name} = NKITensorTensor(op="multiply")(data1={gate.name}, data2={sigmoid.name})',
                f'{intermediate_value.name} = NKITensorTensor(op="multiply")'
                f"(data1={activated.name}, data2={up.name})",
            )
        )
        stationary = TorchValue(f"sbuf_{stem}_down_stationary_{route}", intermediate_value.shape, True)
        down = TorchValue(f"sbuf_{stem}_down_weight_{route}", (intermediate, hidden_width))
        body.extend(
            (
                f"{stationary.name} = NKIDMATranspose()(src={intermediate_value.name})",
                f"{down.name} = NKIHBMScalarRowSlice(rows={intermediate}, width={hidden_width})"
                f"(src={down_weights.name}, indices={index.name})",
            )
        )
        down_psum = f"psum_{stem}_down_{route}"
        partial = TorchValue(f"sbuf_{stem}_partial_{route}", hidden.shape)
        scaled = TorchValue(f"sbuf_{stem}_scaled_{route}", hidden.shape)
        body.extend(
            (
                f"{down_psum} = NKIMatmul()(stationary={stationary.name}, moving={down.name})",
                f"{partial.name} = NKITensorCopy()(src={down_psum})",
                f'{scaled.name} = NKITensorScalar(op0="multiply")'
                f"(data={partial.name}, operand0={affinity_vector.name})",
            )
        )
        if total is None:
            total = scaled
        else:
            combined = TorchValue(f"sbuf_{stem}_total_{route}", hidden.shape)
            body.append(f'{combined.name} = NKITensorTensor(op="add")(data1={total.name}, data2={scaled.name})')
            total = combined
    if total is None:
        raise ValueError("Torch MoE requires at least one expert")
    return total


__all__ = ["emit_moe"]
