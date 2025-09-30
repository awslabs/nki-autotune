from dataclasses import dataclass
from itertools import product
from math import ceil


class trn1:
    name = "trn1"
    pe_freq = 2.8
    act_freq = 1.4
    dve_freq = 1.12
    gpsimd_freq = 1.4
    hbm_bandwidth = 260  # GB/s
    fp8_speedup_ratio = 1


class trn2:
    name = "trn2"
    pe_freq = 2.4
    act_freq = 1.2
    dve_freq = 0.96
    gpsimd_freq = 1.2
    hbm_bandwidth = 320  # GB/s
    fp8_speedup_ratio = 2


class trn2e:
    name = "trn2e"
    pe_freq = 2.4  # ghz: 1e9 cycles/second
    act_freq = 1.2
    dve_freq = 0.96
    gpsimd_freq = 1.2
    hbm_bandwidth = 410  # GB/s
    fp8_speedup_ratio = 2


class trn3:
    name = "trn3"
    pe_freq = 2.4  # ghz: 1e9 cycles/second
    act_freq = 1.2
    dve_freq = 1.2
    gpsimd_freq = 1.2
    hbm_bandwidth = 471  # GB/s
    fp8_speedup_ratio = 4


def print_hardware_specs(specs):
    g_flops = 128 * 128 * 2 * specs.pe_freq
    t_flops = g_flops // 1000
    intensity = g_flops / specs.hbm_bandwidth  # FLOPs/byte

    act_copy_g_flops = 128 * specs.act_freq
    act_intensity = act_copy_g_flops / specs.hbm_bandwidth

    fp8_g_flops = g_flops * specs.fp8_speedup_ratio
    fp8_t_flops = fp8_g_flops // 1000
    fp8_intensity = fp8_g_flops / specs.hbm_bandwidth

    print(f"{specs.name}: TFLOPs {t_flops}, roofline intensity {intensity:.1f}")
    print(f"{specs.name}: act roofline intensity {act_intensity:.1f}")
    print(f"{specs.name}: fp8 TFLOPs {fp8_t_flops}, fp8 roofline intensity {fp8_intensity:.1f}")


@dataclass
class matmul_dtype:
    act: int  # size_in_bytes
    weight: int  # size_in_bytes
    psum_res: int = 4  # size_in_bytes
    sbuf_res: int = 2  # size_in_bytes


def mm_arithmetic_intensity(N, M, K):
    # streaming [K, N] @ stationary [K, M] = result [M, N]
    # for LLMs, typically N = num_tokens; K = d_model; M is part of weight size

    m_block_size = 512  # in elements
    n_block_size = 512

    num_m_blocks = int(ceil(M / m_block_size))
    num_n_blocks = int(ceil(N / n_block_size))
    total_blocks = num_m_blocks * num_n_blocks
    act_reload_blocks = total_blocks - (num_m_blocks - 1)
    weight_reload_blocks = num_m_blocks

    dtype_configs = {}
    dtype_configs["bf16"] = matmul_dtype(act=2, weight=2)
    dtype_configs["fp8_weights"] = matmul_dtype(act=2, weight=1)
    dtype_configs["fp8"] = matmul_dtype(act=1, weight=1)
    print(f"\nN: {N}, M: {M}, K: {K}")

    for dtype_config, sizes in dtype_configs.items():
        dma_traffic_per_block = 0

        act_bytes_per_block = K * n_block_size * sizes.act  # TODO: can cache a little of this for some n_blocks
        weight_bytes_per_block = K * m_block_size * sizes.weight
        res_bytes_per_block = m_block_size * n_block_size * sizes.sbuf_res

        act_traffic = act_bytes_per_block * act_reload_blocks
        weight_traffic = weight_bytes_per_block * weight_reload_blocks
        res_traffic = res_bytes_per_block * total_blocks
        dma_traffic = act_traffic + weight_traffic + res_traffic

        flops = 2 * K * M * N

        intensity = flops / dma_traffic
        print(f"act_traffic {act_traffic}: act_bytes_per_block {act_bytes_per_block}, act_blocks {act_reload_blocks}")
        print(
            f"weight_traffic {weight_traffic}: weight_bytes_per_block {weight_bytes_per_block}, weight_blocks {weight_reload_blocks}"
        )
        print(f"res_traffic {res_traffic}: res_bytes_per_block {res_bytes_per_block}, res_blocks {total_blocks}")
        print(f"{dtype_config}: {intensity:.1f}")

        act_flops = M * N  # from psum copying
        act_intensity = act_flops / dma_traffic
        print(f"act intensity: {act_intensity:.1f}")


print("intensity measured in FLOPs/byte\n")

spec_list = [trn1(), trn2(), trn2e(), trn3()]
for specs in spec_list:
    print_hardware_specs(specs)

N = [512, 1024, 2048]
M = [512, 1024, 2048]
K = [4096]
for n, m, k in product(N, M, K):
    mm_arithmetic_intensity(n, m, k)
