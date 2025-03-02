import os

home_dir = os.environ["HOME"]
CACHE_ROOT_DIR = f"{home_dir}/autotune-cache"
TORCH_CACHE_DIR = f"{CACHE_ROOT_DIR}/torch"
NKI_CACHE_DIR = f"{CACHE_ROOT_DIR}/nki"
TUNED_NKI_CACHE_DIR = f"{CACHE_ROOT_DIR}/tuned-nki"
