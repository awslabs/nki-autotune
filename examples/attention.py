from autotune.core.metrics import calculate_mfu


def compute_attention_mfu(seq: int, hidden: int, latency_ms: float):
    mac_count = seq * hidden * seq + seq * seq * hidden
    mfu = calculate_mfu(mac_count=mac_count, time_ms=latency_ms, target_instance_family="trn2")
    return mfu


if __name__ == "__main__":
    mfu = compute_attention_mfu(seq=10 * 1024, hidden=128, latency_ms=2)
    print(mfu)
