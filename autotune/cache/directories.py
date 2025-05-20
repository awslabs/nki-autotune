import hashlib
import os
import re
import time
from typing import List, Tuple

from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE

home_dir = os.environ["HOME"]
CACHE_ROOT_DIR = f"{home_dir}/autotune-cache"
TORCH_CACHE_DIR = f"{CACHE_ROOT_DIR}/torch"


def get_hash_name(kernel_name: str, input_tensors: INPUT_TENSORS_DTYPE, configs: KERNEL_KWARGS_DTYPE):
    input_tensors_str = parse_tensor_shapes([str(arg.shape) for arg in input_tensors])
    configs_str = dict_to_string(configs)
    timestamp = str(time.time())
    combined_str = f"{kernel_name}_{input_tensors_str}_{configs_str}_{timestamp}"
    hash_value = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
    hash_name = f"{kernel_name}-{hash_value}"
    return hash_name


def dict_to_string(configs: KERNEL_KWARGS_DTYPE) -> str:
    """
    Convert a dictionary to a string by concatenating keys and values with hyphens.

    Args:
        configs (Dict): The dictionary to convert.

    Returns:
        str: A string representation of the dictionary where each key and value
             are converted to strings and joined with hyphens.

    Example:
        >>> dict_to_string({'a': 1, 'b': 2})
        'a-1-b-2'
    """
    result = []
    for key, val in configs.items():
        result.append(str(key))
        result.append(str(val))
    return "-".join(result)


def split_file_info(filepath: str) -> Tuple[str, str, str]:
    """
    Split a file path into its directory, filename, and file type components.

    Args:
        filepath (str): The file path to split.

    Returns:
        tuple: A tuple containing:
            - directory (str): The directory path.
            - filename (str): The filename without extension.
            - file_type (str): The file type/extension (without the dot).

    Example:
        >>> split_file_info('/path/to/file.txt')
        ('/path/to', 'file', 'txt')
    """
    directory = os.path.dirname(filepath)
    full_filename = os.path.basename(filepath)
    filename, file_type = os.path.splitext(full_filename)
    file_type = file_type.lstrip(".")
    return directory, filename, file_type


def parse_tensor_shapes(tensor_shapes: List[str]) -> str:
    """
    Convert a list of tensor shape strings from bracket notation to hyphen-separated dimensions
    connected by underscores.

    Parameters:
    -----------
    tensor_shapes : list of str
        A list of tensor shapes in the format "[dim1, dim2, ...]".
        Example: ["[1, 8192, 4096]", "[4096, 8192]"]

    Returns:
    --------
    str
        A single string containing all tensor shapes converted to the format
        "dim1-dim2-..._dim1-dim2-...". Dimensions within a shape are separated by hyphens,
        and different shapes are separated by underscores.
        Example: "1-8192-4096_4096-8192"

    Examples:
    ---------
    >>> parse_tensor_shapes(["[1, 8192, 4096]", "[4096, 8192]"])
    '1-8192-4096_4096-8192'

    >>> parse_tensor_shapes(["[1, 768]", "[768, 3072]", "[3072, 768]"])
    '1-768_768-3072_3072-768'

    Notes:
    ------
    - The function removes all whitespace in the dimension values
    - Empty input list will return an empty string
    - Input strings must be properly formatted with square brackets
    """
    processed_shapes = []

    for shape in tensor_shapes:
        # Remove square brackets
        shape = shape.strip("[]")

        # Split by comma and remove spaces
        dimensions = [dim.strip() for dim in shape.split(",")]

        # Join dimensions with hyphens
        processed_shape = "-".join(dimensions)

        processed_shapes.append(processed_shape)

    # Join all processed shapes with underscores
    result = "_".join(processed_shapes)

    return result


def get_cache_dir(workload_name: str, cache_type: str, **kwargs):
    """
    Generate a cache directory path based on workload name, cache type, and additional parameters.

    This function creates a cache directory path by combining the root directory,
    workload name, cache type, and key-value pairs from the provided keyword arguments.
    The key-value pairs are appended to the directory path in the format "key1value1-key2value2".

    Args:
        workload_name (str): Name of the workload to create a cache directory for
        cache_type (str): Type of cache directory; must be one of "baseline", "tuned", or "plots"
        **kwargs: Arbitrary keyword arguments that will be used to create subdirectory names
                 in the format "keyvalue-"

    Returns:
        str: The full cache directory path

    Raises:
        ValueError: If cache_type is not one of "baseline", "tuned", or "plots"

    Example:
        >>> get_cache_dir("sentiment_analysis", "tuned", model="bert", batch=32)
        "CACHE_ROOT_DIR/sentiment_analysis/tuned/modelbert-batch32"
    """
    if cache_type not in ["baseline", "tuned", "plots"]:
        raise ValueError(f"{cache_type} cache directory is not supported. Expecting (baseline, tuned, plots).")

    cache_dir = f"{CACHE_ROOT_DIR}/{workload_name}/{cache_type}"
    # Join all key-value pairs
    key_val_string = ""
    for key, value in kwargs.items():
        key_val_string += f"{key}{value}-"

    # Remove the last hyphen if any kwargs were provided
    if key_val_string:
        key_val_string = key_val_string[:-1]
        cache_dir = f"{cache_dir}/{key_val_string}"

    return cache_dir


def get_save_path(plots_dir: str, plot_type, m=None, n=None, k=None):
    """
    Determine the save path based on plot type and dimensions.

    Parameters:
    -----------
    plots_dir : str
        Base directory for plots
    plot_type : str
        Type of the plot
    m, n, k : int, optional
        Matrix dimensions if applicable
    run_type : str, optional
        'tuned' or 'baseline'

    Returns:
    --------
    tuple
        (save_path, filename)
    """
    # MFU, HFU plots go directly in the plots folder
    if plot_type in ["Model_Flops_Utilization", "Hardware_Flops_Utilization"]:
        return plots_dir, f"{plot_type}_M{m}_N{n}.png"
    else:
        raise NotImplementedError(f"Plot type {plot_type} is not supported.")


def extract_mnk_from_dirname(dirname):
    """
    Extract M, N, K values from a directory name.

    Args:
        dirname: Directory name in format M{m}-N{n}-K{k}

    Returns:
        Tuple of (m, n, k) as integers, or None for any missing value
    """
    match = re.match(r"M(\d+)-N(\d+)-K(\d+)", dirname)
    if match:
        return tuple(map(int, match.groups()))
    return None, None, None
