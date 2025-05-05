import numpy as np
import pytest
from neuronxcc.nki import baremetal

from kernel_library.multi_buffer import stack_allocated_fused_self_attn_for_SD_small_head_size


@pytest.mark.parametrize("batch, seqlen, dim, eps", [(1, 1024, 4096, 1e-6)])
def test_stack_allocated_fused_self_attn(batch, seqlen, dim, eps):
    dtype = np.float32
    bs, seqlen, d = 1, seqlen, 128
    q = np.random.random_sample((bs, d, seqlen))
    k = np.random.random_sample((bs, d, seqlen))
    v = np.random.random_sample((bs, seqlen, d))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)

    numeric_func = baremetal(stack_allocated_fused_self_attn_for_SD_small_head_size)
    nki_out = numeric_func[bs](q, k, v)
    print(nki_out.shape)
    # nki_out, metrics = run_kernel("stack_allocated_fused_self_attn_for_SD_small_head_size", (q,k,v))
    # print(metrics)
