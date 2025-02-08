def update_base_addr(base_addr: int, tensor, advance: bool) -> int:
    if tensor.ndim == 2:
        # pardim, fdim
        buf_size = tensor.shape[1] * tensor.itemsize
    elif tensor.ndim == 3:
        # block_dim, pardim, fdim
        buf_size = tensor.shape[0] * tensor.shape[2] * tensor.itemsize
    else:
        raise NotImplementedError(
            f"Buffer size for tensor shape {tensor.shape} is unknown"
        )
    if advance:
        next_base_addr = base_addr + buf_size
        # print(f"Allocate buf @ {base_addr} -> {next_base_addr}")
    else:
        next_base_addr = base_addr - buf_size
        # print(f"Restore buf @ {base_addr} -> {next_base_addr}")
    return next_base_addr
