def process_compiler_flags(compiler_flags: str):
    """
    Process compiler flags string to extract target instance family and clean remaining flags.

    This function extracts the target instance family (trn1 or trn2) from compiler flags,
    removes that target flag from the string, and normalizes whitespace in the remaining flags.

    Args:
        compiler_flags: A string containing compiler flags, which must include either
                       '--target=trn1' or '--target=trn2'

    Returns:
        tuple: A tuple containing:
            - target_instance_family (str): The extracted target instance family ('trn1' or 'trn2')
            - compiler_flags (str): The remaining compiler flags with normalized whitespace

    Raises:
        NotImplementedError: If the compiler flags do not contain either '--target=trn1'
                             or '--target=trn2'
    """
    if "--target=trn1" in compiler_flags:
        target_instance_family = "trn1"
        compiler_flags = compiler_flags.replace("--target=trn1", "")
    elif "--target=trn2" in compiler_flags:
        target_instance_family = "trn2"
        compiler_flags = compiler_flags.replace("--target=trn2", "")
    else:
        raise NotImplementedError(
            f"Only support --target=trn1 or --target=trn2 in compiler flags. Received {compiler_flags}."
        )
    compiler_flags = " ".join(compiler_flags.split())
    return target_instance_family, compiler_flags
