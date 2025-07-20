import ast
import copy
import inspect
from typing import Dict


class ConstantFolder(ast.NodeTransformer):
    """Fold constant expressions like [False, True][0] into just False"""

    def visit_Subscript(self, node):
        # First visit children
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        # Try to evaluate constant subscripts
        if isinstance(node.value, ast.Constant) and isinstance(node.slice, ast.Constant):
            try:
                # For lists, tuples, etc.
                value = node.value.value[node.slice.value]
                return ast.Constant(value=value)
            except (TypeError, IndexError, KeyError):
                pass

        # Handle list/dict literals with constant indexes
        if isinstance(node.value, ast.List) and isinstance(node.slice, ast.Constant):
            try:
                index = node.slice.value
                if 0 <= index < len(node.value.elts) and isinstance(node.value.elts[index], ast.Constant):
                    return node.value.elts[index]
            except (TypeError, IndexError):
                pass

        return node


class ConstantPropagator(ast.NodeTransformer):
    """Replace variables with known compile-time constants"""

    def __init__(self, constants):
        self.constants = constants

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            return ast.Constant(value=self.constants[node.id])
        return node

    def visit_Subscript(self, node):
        # Process children first
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        # Handle dictionary access with constant keys
        if isinstance(node.value, ast.Name) and node.value.id in self.constants:
            container = self.constants[node.value.id]
            if isinstance(container, dict) and isinstance(node.slice, ast.Constant):
                try:
                    value = container[node.slice.value]
                    return ast.Constant(value=value)
                except (KeyError, TypeError):
                    pass

        # Then let the constant folder try to simplify
        return ConstantFolder().visit(node)


class FunctionInliner(ast.NodeTransformer):
    """Inline function calls with their body"""

    def __init__(self, functions_dict):
        self.functions_dict = functions_dict

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            if func_name in self.functions_dict:
                # Get the function definition
                func_def = self.functions_dict[func_name]

                # Map parameters to arguments
                param_map = {}

                # Process positional arguments
                for i, param in enumerate(func_def.args.args):
                    if i < len(node.value.args):
                        # For constants, store the value directly
                        if isinstance(node.value.args[i], ast.Constant):
                            param_map[param.arg] = node.value.args[i].value
                        # For other expressions, we need to fold them if possible
                        else:
                            folded_arg = ConstantFolder().visit(node.value.args[i])
                            if isinstance(folded_arg, ast.Constant):
                                param_map[param.arg] = folded_arg.value

                # Process keyword arguments
                for kw in node.value.keywords:
                    if isinstance(kw.value, ast.Constant):
                        param_map[kw.arg] = kw.value.value
                    else:
                        folded_arg = ConstantFolder().visit(kw.value)
                        if isinstance(folded_arg, ast.Constant):
                            param_map[kw.arg] = folded_arg.value

                # Clone the function body
                inlined_body = copy.deepcopy(func_def.body)

                # Propagate constants into the body
                constant_prop = ConstantPropagator(param_map)
                inlined_body = [constant_prop.visit(stmt) for stmt in inlined_body]

                # Fold any constant expressions
                folder = ConstantFolder()
                inlined_body = [folder.visit(stmt) for stmt in inlined_body]

                # Apply dead code elimination to the inlined body
                eliminator = DeadCodeEliminator()
                processed_body = []
                for stmt in inlined_body:
                    result = eliminator.visit(stmt)
                    if isinstance(result, list):
                        processed_body.extend(result)
                    elif result is not None:
                        processed_body.append(result)

                # Return the processed body
                return processed_body if processed_body else None

        # Not a function call we can inline
        return self.generic_visit(node)


class DeadCodeEliminator(ast.NodeTransformer):
    """Eliminate dead code branches and empty loops"""

    def visit_If(self, node):
        # First fold any constant expressions in the condition
        node.test = ConstantFolder().visit(node.test)

        # Check if condition is a constant
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                # Condition is True, keep the if block
                result = []
                for stmt in node.body:
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        result.extend(visited)
                    elif visited is not None:
                        result.append(visited)
                return result
            else:
                # Condition is False, keep the else block
                result = []
                for stmt in node.orelse:
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        result.extend(visited)
                    elif visited is not None:
                        result.append(visited)
                return result

        # Not a constant condition, process both branches
        node.body = [self.visit(stmt) for stmt in node.body if self.visit(stmt) is not None]
        node.orelse = [self.visit(stmt) for stmt in node.orelse if self.visit(stmt) is not None]
        return node

    def visit_For(self, node):
        # Process the loop body first
        new_body = []
        for stmt in node.body:
            visited = self.visit(stmt)
            if isinstance(visited, list):
                new_body.extend(visited)
            elif visited is not None:
                new_body.append(visited)

        node.body = new_body

        # If the loop body is empty, remove it
        if not node.body:
            return None

        return node


def pre_compile_kernel(func, **kwargs):
    """Precompile a function with compile-time constants"""
    # Get the source code
    source = inspect.getsource(func)

    # Parse the AST
    tree = ast.parse(source)
    function_node = tree.body[0]

    # Get module for other function definitions
    module = inspect.getmodule(func)
    module_source = inspect.getsource(module)
    module_tree = ast.parse(module_source)

    # Extract function definitions
    functions = {}
    for node in module_tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node

    # Process lists passed as kwargs to handle array-like access
    processed_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, list):
            # Convert list to dictionary with integer keys
            processed_kwargs[key] = {i: v for i, v in enumerate(value)}
        else:
            processed_kwargs[key] = value

    # Apply optimizations with multiple passes

    # Step 1: Propagate constants from kwargs
    propagated = copy.deepcopy(function_node)
    propagator = ConstantPropagator(processed_kwargs)
    propagated = propagator.visit(propagated)

    # Apply constant folding
    folder = ConstantFolder()
    propagated = folder.visit(propagated)

    ast.fix_missing_locations(propagated)

    # Step 2: Inline function calls
    inliner = FunctionInliner(functions)

    # Process each statement separately to handle list returns
    new_body = []
    for stmt in propagated.body:
        result = inliner.visit(stmt)
        if isinstance(result, list):
            new_body.extend(result)
        elif result is not None:
            new_body.append(result)

    propagated.body = new_body
    ast.fix_missing_locations(propagated)

    # Apply constant folding again after inlining
    propagated = folder.visit(propagated)
    ast.fix_missing_locations(propagated)

    # Step 3: Eliminate dead code
    eliminator = DeadCodeEliminator()
    optimized = eliminator.visit(propagated)
    ast.fix_missing_locations(optimized)

    # Return the optimized code
    return ast.unparse(optimized)


import numpy as np


def top_level_general(inits: Dict[int, bool]):
    maybe_init(init=inits[0])
    for i in range(10):
        maybe_init(inits[1])


def maybe_init(init: bool):
    if init:
        lhsT = np.random.normal(size=(1024, 4096))


def compiled_false_true():
    for i in range(10):
        lhsT = np.random.normal(size=(1024, 4096))


def compiled_true_false():
    lhsT = np.random.normal(size=(1024, 4096))


def compiled_true_true():
    lhsT = np.random.normal(size=(1024, 4096))
    for i in range(10):
        lhsT = np.random.normal(size=(1024, 4096))


def save_code_to_file(filepath: str, kernel_code: str):
    with open(filepath, "w") as f:
        f.write(kernel_code)


from autotune.modules.lhsT_rhs import lhsT_rhs_gemm_general

if __name__ == "__main__":
    # Should match compiled_true_true
    kernel_code = pre_compile_kernel(top_level_general, inits=[True, True])
    print("True, True:")
    print(kernel_code)
    print()

    # Should match compiled_true_false
    kernel_code = pre_compile_kernel(top_level_general, inits=[True, False])
    print("True, False:")
    print(kernel_code)
    print()

    # Should match compiled_false_true
    kernel_code = pre_compile_kernel(top_level_general, inits=[False, True])
    print("False, True:")
    print(kernel_code)
    print()

    lhsT = np.random.normal(size=(1024, 2048))
    rhs = np.random.normal(size=(1024, 4096))
    config = {
        "lhsT": lhsT,
        "rhs": rhs,
        "NUM_BLOCK_M": 2,
        "NUM_BLOCK_N": 1,
        "NUM_BLOCK_K": 4,
        "loop_order": "MKN",
        "tensor_positions": {"result_block": -1, "rhs_block": 1, "lhsT_block": 2},
    }
    kernel_code = pre_compile_kernel(lhsT_rhs_gemm_general, **config)
    save_code_to_file("generated_kernels/generated_lhsT_rhs.py", kernel_code)
