import neuronxcc.nki.language as nl


def copy_block(src_block, dest_block):
    block_size, par_size, free_size = src_block.shape
    _block_size, _par_size, _free_size = dest_block.shape
    assert block_size == _block_size and par_size == _par_size and free_size == _free_size

    for block_id in nl.affine_range(block_size):
        dest_block[block_id] = nl.copy(src_block[block_id], dtype=src_block.dtype)
