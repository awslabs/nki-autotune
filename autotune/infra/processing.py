import pickle

from autotune.typing.infra_types import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, POSTPROCESSING_DTYPE


def postprocessing_fun_wrapper(
    processing_fun: POSTPROCESSING_DTYPE,
    input_tensors: INPUT_TENSORS_DTYPE,
    kernel_kwargs: KERNEL_KWARGS_DTYPE,
    cache_dir: str,
):
    kernel_outputs = pickle.load(open(f"{cache_dir}/kernel_outputs.pkl", "rb"))
    processing_fun(input_tensors, kernel_kwargs, kernel_outputs)
