import os
import shutil

import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
from neuronxcc.starfish.support import dtype as dt
from neuronxcc.test.unit.TestUtils import remote_trace

from src.cache.directories import NKI_CACHE_DIR
from src.tune.benchmark import profile_kernel


def profile():
    cache_dir = f"{NKI_CACHE_DIR}/{rmsnorm_quant_isa_kernel.func_name}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    quant_dtype = dt.float8_e4m3
    dtype_size_scale = dt.sizeinbytes(np.float32) // dt.sizeinbytes(quant_dtype)
    batch, seqlen, hidden = 1, 2048, 1024
    data_type = nl.bfloat16
    input1 = np.random.random_sample((batch, seqlen, hidden)).astype(data_type)
    gamma = np.random.random_sample((hidden)).astype(data_type)
    rmsnorm_output = np.zeros((batch, seqlen, hidden + dtype_size_scale)).astype(data_type)

    rmsnorm_kernel = remote_trace(func=rmsnorm_quant_isa_kernel, test_config=test_config)
    rmsnorm_kernel(input1, gamma, 0.0, rmsnorm_output, "RMSNormQuant")


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile()
