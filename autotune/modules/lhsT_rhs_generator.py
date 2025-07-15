import os
from pathlib import Path
from typing import Dict, List


class CodeInliner:
    def __init__(self, loop_order: str, tensor_positions: Dict[str, int]):
        self.loop_order = loop_order
        self.tensor_positions = tensor_positions

        # Analyze which operations happen at which positions
        self.op_positions = self._analyze_operations()

    def _analyze_operations(self) -> Dict[int, List[str]]:
        """Analyze which operations happen at each position"""
        ops = {-1: [], 0: [], 1: [], 2: []}

        for tensor_name, position in self.tensor_positions.items():
            if tensor_name in ["lhsT_block", "rhs_block", "result_block"]:
                ops[position].append(f"load {tensor_name}")

        # Add matmul operation
        matmul_pos = max(self.tensor_positions["lhsT_block"], self.tensor_positions["rhs_block"])
        self.tensor_positions["matmul"] = matmul_pos
        ops[matmul_pos].append("matmul")

        # Add save operations (mirror of init)
        ops[self.tensor_positions["result_block"]].append("save_result")
        return ops

    def generate_inlined_kernel(self) -> str:
        """Generate the complete inlined kernel function"""

        docstring = self._generate_docstring()

        # Generate loop structure with inlined operations
        # loop_body = self._generate_loop_structure()

        kernel_code = f"""
@nki.jit
def lhsT_rhs_gemm_inlined_{self.loop_order}(
    lhsT: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int
):
    {docstring}
    
    # Setup mm compatibility object
    input_tensors=(lhsT, rhs)
    kernel_kwargs = {{
        "NUM_BLOCK_M": NUM_BLOCK_M,
        "NUM_BLOCK_N": NUM_BLOCK_N,
        "NUM_BLOCK_K": NUM_BLOCK_K,
    }}
    preprocessing(input_tensors, kernel_kwargs)
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(input_tensors, kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    
    return result
"""

        """
        # Inlined operations at position 0 (before all loops)
    {self._generate_position_operations(0, "    ")}
        
    {loop_body}
        
        # Final save operations if needed
    {self._generate_final_save_operations()}
        """
        return kernel_code

    def _generate_docstring(self) -> str:
        loop_structure = ["Loop structure:"]
        indentation = ""
        for pos in self.op_positions:
            if pos > -1:
                loop_structure.append(f"{indentation}loop_{pos}: {self.loop_order[pos]}")
                indentation += "    "
            if self.op_positions[pos]:
                loop_structure.append(f"{indentation}{', '.join(self.op_positions[pos])}")

        loop_structure_str = "\n".join(loop_structure)
        docstring = f'''
    """
Auto-generated inlined lhsT (K, M) @ rhs (K, N) GEMM kernel that computes result = lhsT^T @ rhs.

{loop_structure_str}

This kernel uses block-based computation with a specific loop ordering.
    """
        '''
        return docstring

    def _generate_loop_structure(self) -> str:
        """Generate the nested loop structure with inlined operations"""
        loop_vars = [f"block_id_{dim}" for dim in self.loop_order]
        loop_ranges = [f"mm.NUM_BLOCK_{dim}" for dim in self.loop_order]

        # Generate nested loops
        loops = []
        indent = "    "

        for i, (var, range_expr) in enumerate(zip(loop_vars, loop_ranges)):
            loops.append(f"{indent}for {var} in nl.affine_range({range_expr}):")
            indent += "    "

            # Add operations for this position
            pos_ops = self._generate_position_operations(i + 1, indent)
            if pos_ops.strip():
                loops.append(pos_ops)

        # Add innermost operations (position 3)
        innermost_ops = self._generate_position_operations(3, indent)
        if innermost_ops.strip():
            loops.append(innermost_ops)

        # Add save operations in reverse order
        for i in range(2, -1, -1):
            indent = "    " + "    " * (i + 1)
            save_ops = self._generate_save_operations(i + 1, indent, loop_vars[: i + 1])
            if save_ops.strip():
                loops.append(save_ops)

        return "\n".join(loops)

    def _generate_position_operations(self, position: int, indent: str) -> str:
        """Generate operations for a specific position"""
        if position not in self.position_operations:
            return ""

        ops = []

        for op in self.position_operations[position]:
            if op == "init_lhsT_block":
                ops.extend(self._generate_lhsT_block_init(position, indent))
            elif op == "init_rhs_block":
                ops.extend(self._generate_rhs_block_init(position, indent))
            elif op == "init_result_block":
                ops.extend(self._generate_result_block_init(position, indent))
            elif op == "matmul":
                ops.extend(self._generate_matmul_operation(position, indent))
        ops.append(f"{indent}# Operations at position {position}")
        ops.append(f"{indent}1+1")
        return "\n".join(ops)

    def _generate_lhsT_block_init(self, position: int, indent: str) -> List[str]:
        """Generate lhsT block initialization code"""
        ops = []

        # Calculate block shape and offset based on position
        shape_calc = self._get_block_shape_calculation("lhsT", ("K", "M"), position)
        offset_calc = self._get_block_offset_calculation(("K", "M"), position)

        ops.append(f"{indent}# Load lhsT block")
        ops.append(f"{indent}lhsT_block_shape = {shape_calc}")
        ops.append(f"{indent}lhsT_ofs = {offset_calc}")
        ops.append(f"{indent}lhsT_block = load_tensor_block(")
        ops.append(f"{indent}    input_tensor=lhsT,")
        ops.append(f"{indent}    ofs=lhsT_ofs,")
        ops.append(f"{indent}    load_shape=lhsT_block_shape")
        ops.append(f"{indent})")

        return ops

    def _generate_rhs_block_init(self, position: int, indent: str) -> List[str]:
        """Generate rhs block initialization code"""
        ops = []

        # Calculate block shape and offset based on position
        shape_calc = self._get_block_shape_calculation("rhs", ("K", "N"), position)
        offset_calc = self._get_block_offset_calculation(("K", "N"), position)

        ops.append(f"{indent}# Load rhs block")
        ops.append(f"{indent}rhs_block_shape = {shape_calc}")
        ops.append(f"{indent}rhs_ofs = {offset_calc}")
        ops.append(f"{indent}rhs_block = load_tensor_block(")
        ops.append(f"{indent}    input_tensor=rhs,")
        ops.append(f"{indent}    ofs=rhs_ofs,")
        ops.append(f"{indent}    load_shape=rhs_block_shape")
        ops.append(f"{indent})")

        return ops

    def _generate_result_block_init(self, position: int, indent: str) -> List[str]:
        """Generate result block initialization code"""
        ops = []

        shape_calc = self._get_block_shape_calculation("result", ("M", "N"), position)

        ops.append(f"{indent}# Initialize result block")
        ops.append(f"{indent}result_block_shape = {shape_calc}")
        ops.append(f"{indent}result_block = nl.zeros(result_block_shape, dtype=lhsT.dtype, buffer=nl.sbuf)")

        return ops

    def _generate_matmul_operation(self, position: int, indent: str) -> List[str]:
        """Generate matmul operation code"""
        ops = []

        offset_calc = self._get_result_offset_calculation(position)

        ops.append(f"{indent}# Perform matrix multiplication")
        ops.append(f"{indent}result_ofs = {offset_calc}")
        ops.append(f"{indent}matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)")

        return ops

    def _generate_save_operations(self, position: int, indent: str, current_block_ids: List[str]) -> str:
        """Generate save operations for a specific position"""
        result_pos = self.tensor_positions.get("result_block_position", 2)
        if result_pos != position:
            return ""

        ops = []
        ops.append(f"{indent}# Save result block at position {position}")

        # Calculate tile index offset based on current block IDs
        offset_calc = self._get_save_offset_calculation(position, current_block_ids)

        ops.append(f"{indent}tile_index_ofs = {offset_calc}")
        ops.append(f"{indent}save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)")

        return "\n".join(ops)

    def _get_block_shape_calculation(self, tensor_type: str, dims: tuple, position: int) -> str:
        """Generate block shape calculation code"""
        dim1, dim2 = dims

        # Determine number of blocks for each dimension
        num_blocks = []
        for dim in dims:
            dim_pos = self.loop_order.index(dim)
            if dim_pos < position:
                num_blocks.append("1")
            else:
                num_blocks.append(f"mm.NUM_BLOCK_{dim}")

        return (
            f"(mm.TILE_{dim1}, {num_blocks[0]} * mm.TILES_IN_BLOCK_{dim1}, "
            f"{num_blocks[1]} * mm.TILES_IN_BLOCK_{dim2}, mm.TILE_{dim2})"
        )

    def _get_block_offset_calculation(self, dims: tuple, position: int) -> str:
        """Generate block offset calculation code"""
        dim1, dim2 = dims

        offsets = []
        for dim in dims:
            dim_pos = self.loop_order.index(dim)
            if dim_pos < position:
                block_var = f"block_id_{dim}"
                offsets.append(f"{block_var} * mm.BLOCK_{dim}")
            else:
                offsets.append("0")

        return f"({offsets[0]}, {offsets[1]})"

    def _get_result_offset_calculation(self, position: int) -> str:
        """Generate result offset calculation for matmul"""
        return "(0, 0)"  # Simplified for now

    def _get_save_offset_calculation(self, position: int, current_block_ids: List[str]) -> str:
        """Generate save offset calculation"""
        if position == 0:
            return "(0, 0)"
        elif position == 1:
            M_pos = self.loop_order.index("M")
            N_pos = self.loop_order.index("N")
            if M_pos == 0:
                return f"({current_block_ids[0]} * mm.TILES_IN_BLOCK_M, 0)"
            elif N_pos == 0:
                return f"(0, {current_block_ids[0]} * mm.TILES_IN_BLOCK_N)"
        elif position == 2:
            M_pos = self.loop_order.index("M")
            N_pos = self.loop_order.index("N")
            if M_pos < N_pos:
                return f"({current_block_ids[M_pos]} * mm.TILES_IN_BLOCK_M, {current_block_ids[N_pos]} * mm.TILES_IN_BLOCK_N)"
            else:
                return f"({current_block_ids[N_pos]} * mm.TILES_IN_BLOCK_M, {current_block_ids[M_pos]} * mm.TILES_IN_BLOCK_N)"

        return "(0, 0)"

    def _generate_final_save_operations(self) -> str:
        """Generate any final save operations needed"""
        return "    # Final save operations completed in loops"


def create_inlined_kernel(loop_order: str, tensor_positions: Dict[str, int], **kwargs) -> str:
    """Create an inlined kernel based on configuration"""

    inliner = CodeInliner(loop_order, tensor_positions)
    kernel_code = inliner.generate_inlined_kernel()
    return kernel_code


def save_kernel_to_file(kernel_code: str, filename: str, output_dir: str = "generated_kernels") -> str:
    """
    Save generated kernel code to a Python file.

    Args:
        kernel_code: The generated kernel code string
        filename: Name for the output file (without .py extension)
        output_dir: Directory to save the file in

    Returns:
        str: Full path to the saved file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure filename has .py extension
    if not filename.endswith(".py"):
        filename += ".py"

    # Full file path
    filepath = os.path.join(output_dir, filename)

    # Create file header with imports
    file_content = """# Auto-generated NKI GEMM kernel
# Generated by CodeInliner
# Do not modify directly, instead, modify the generator source code

from typing import Dict, Tuple
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block, save_result_dma
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE

def preprocessing(input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE):
    '''
    Constraints:
        1. result_block initialization must occur before K loop and matmul operation
        2. matmul operation must occur after lhsT_block and rhs_block loads
        3. lhsT_block and rhs_block loads must be on the same side of K loop
        4. Loop order must contain exactly the characters 'M', 'N', and 'K'
    '''
        
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(input_tensors=input_tensors, kernel_kwargs=kernel_kwargs)
    loop_order = kernel_kwargs["loop_order"]
    tensor_positions = kernel_kwargs["tensor_positions"]
    if len(loop_order) != 3 or sorted(loop_order) != sorted("MNK"):
        raise ValueError(f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K.")
    M_position = loop_order.index("M")
    N_position = loop_order.index("N")
    K_position = loop_order.index("K")
    lhsT_block_position = tensor_positions["lhsT_block_position"]
    rhs_block_position = tensor_positions["rhs_block_position"]
    result_block_position = tensor_positions["result_block_position"]
    matmul_position = max(lhsT_block_position, rhs_block_position)
    assert (
        result_block_position <= K_position and result_block_position <= matmul_position
    ), f"result_block init must be before K loop and matmul. Received result_block_position {result_block_position}, K_position {K_position}, matmul_position {matmul_position}."
    assert (
        matmul_position <= lhsT_block_position and matmul_position <= rhs_block_position
    ), f"matmul must be after lhsT_block, rhs_block loads. Received matmul_position {matmul_position}, lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}."
    assert (lhsT_block_position <= K_position and rhs_block_position <= K_position) or (
        lhsT_block_position > K_position and rhs_block_position > K_position
    ), f"lhsT_block and rhs_block must be on the same side of K loop. Received lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}, K_position {K_position}."
"""

    # Add the generated kernel code
    file_content += kernel_code

    # Write to file
    with open(filepath, "w") as f:
        f.write(file_content)

    print(f"Kernel saved to: {filepath}")
    return filepath
