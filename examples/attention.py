from typing import Tuple

from autotune.core.metrics import calculate_mfu


def compute_attention_mfu(
    q_shape: Tuple[int, int], k_shape: Tuple[int, int], v_shape: Tuple[int, int], latency_ms: float
):
    mac_count = q_shape[0] * q_shape[1] * k_shape[1] + q_shape[0] * k_shape[1] * v_shape[1]
    mfu = calculate_mfu(mac_count=mac_count, time_ms=latency_ms, target_instance_family="trn1")
    return mfu


if __name__ == "__main__":
    mfu = compute_attention_mfu(q_shape=(512, 128), k_shape=(128, 10240), v_shape=(10240, 128), latency_ms=0.356)
    print(mfu)
