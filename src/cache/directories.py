import os

home_dir = os.environ["HOME"]
CACHE_ROOT_DIR = f"{home_dir}/autotune-cache"
TORCH_CACHE_DIR = f"{CACHE_ROOT_DIR}/torch"
NKI_CACHE_DIR = f"{CACHE_ROOT_DIR}/nki"
TUNED_NKI_CACHE_DIR = f"{CACHE_ROOT_DIR}/tuned-nki"


def split_file_info(filepath):
    directory = os.path.dirname(filepath)
    full_filename = os.path.basename(filepath)
    filename, file_type = os.path.splitext(full_filename)
    file_type = file_type.lstrip(".")
    return directory, filename, file_type


def convert_tensor_shapes(tensor_shapes):
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
    >>> convert_tensor_shapes(["[1, 8192, 4096]", "[4096, 8192]"])
    '1-8192-4096_4096-8192'

    >>> convert_tensor_shapes(["[1, 768]", "[768, 3072]", "[3072, 768]"])
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
