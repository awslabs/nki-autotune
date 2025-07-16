import os
from typing import List

import dspy


def read_python_file(file_path: str) -> str:
    """
    Read a Python file and pass its contents to the LLM.

    Args:
        file_path (str): Path to the Python file

    Returns:
        The file content as a str
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the file content
    with open(file_path, "r") as file:
        file_content = file.read()
    return file_content


def read_python_files(file_paths: List[str]) -> str:
    all_codes = []
    for file_path in file_paths:
        codes = read_python_file(file_path)
        all_codes.append(file_path)
        all_codes.append(codes)
    return "\n".join(all_codes)


if __name__ == "__main__":
    lm = dspy.LM(
        "bedrock/converse/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        thinking={"type": "enabled", "budget_tokens": 32000},
        max_tokens=128000,
        temperature=1,  # Keep 1
        cache=False,  # Keep False
        num_retries=50,  # Keep 20
    )
    prompt = "create_gemm_kernel tries to generate a NKI GEMM kernel. The output format has indentation issues. refer generated_lhsT_rhs_gemm_0.py. Help me debug."
    codes = read_python_files(
        [
            "autotune/generation/loop_nest.py",
            "autotune/modules/lhsT_rhs_generator.py",
            "generated_kernels/generated_lhsT_rhs_gemm_0.py",
        ]
    )
    response = lm("\n".join([prompt, codes]))
    response = "\n".join(response)
    with open("llm_response.txt", "w") as f:
        f.write(response)
