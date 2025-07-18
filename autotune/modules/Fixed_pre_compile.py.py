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

        # Handle dictionary/list access with constant keys
        if isinstance(node.value, ast.Name) and node.value.id in self.constants:
            container = self.constants[node.value.id]
            if isinstance(node.slice, ast.Constant):
                try:
                    value = container[node.slice.value]
                    return ast.Constant(value=value)
                except (KeyError, IndexError, TypeError):
                    pass

        # Then let the constant folder try to simplify
        return ConstantFolder().visit(node)


class FunctionInliner(ast.NodeTransformer):
    """Inline function calls with their body"""

    def __init__(self, functions_dict):
        self.functions_dict = functions_dict
        self.replacements = []  # Store replacements to be made at statement level

    def visit_Call(self, node):
        # First visit children
        node = self.generic_visit(node)

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.functions_dict:
                # Get the function definition
                func_def = self.functions_dict[func_name]

                # Map parameters to arguments
                param_map = {}

                # Process positional arguments
                for i, param in enumerate(func_def.args.args):
                    if i < len(node.args):
                        # For constants, store the value directly
                        if isinstance(node.args[i], ast.Constant):
                            param_map[param.arg] = node.args[i].value
                        # For other expressions, we need to fold them if possible
                        else:
                            folded_arg = ConstantFolder().visit(node.args[i])
                            if isinstance(folded_arg, ast.Constant):
                                param_map[param.arg] = folded_arg.value

                # Process keyword arguments
                for kw in node.keywords:
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

                # Store the replacement for later processing
                self.replacements.append((node, inlined_body))

                # Return a placeholder that we'll replace later
                return ast.Constant(value=f"__INLINE_PLACEHOLDER_{len(self.replacements)-1}__")

        return node

    def visit_Expr(self, node):
        # Visit the expression
        original_replacements_len = len(self.replacements)
        node.value = self.visit(node.value)

        # Check if this expression got a replacement
        if (
            len(self.replacements) > original_replacements_len
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.value.value.startswith("__INLINE_PLACEHOLDER_")
        ):

            # Extract the replacement index
            placeholder_id = int(node.value.value.split("_")[-2])
            _, replacement_body = self.replacements[placeholder_id]

            # Return the inlined body
            return replacement_body

        return node


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
                    elif visited is not None:  # Skip None (eliminated statements)
                        result.append(visited)
                return result if result else None
            else:
                # Condition is False, keep the else block
                result = []
                for stmt in node.orelse:
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        result.extend(visited)
                    elif visited is not None:
                        result.append(visited)
                return result if result else None

        # Not a constant condition, process both branches
        node.body = [self.visit(stmt) for stmt in node.body if self.visit(stmt) is not None]
        node.orelse = [self.visit(stmt) for stmt in node.orelse if self.visit(stmt) is not None]
        return node

    def visit_For(self, node):
        # Visit the body first
        new_body = []
        for stmt in node.body:
            visited = self.visit(stmt)
            if isinstance(visited, list):
                new_body.extend(visited)
            elif visited is not None:
                new_body.append(visited)

        # If the body is empty after optimization, remove the entire loop
        if not new_body:
            return None

        node.body = new_body
        return node

    def visit_While(self, node):
        # Similar logic for while loops
        new_body = []
        for stmt in node.body:
            visited = self.visit(stmt)
            if isinstance(visited, list):
                new_body.extend(visited)
            elif visited is not None:
                new_body.append(visited)

        # If the body is empty after optimization, remove the entire loop
        if not new_body:
            return None

        node.body = new_body
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

    # Apply optimizations with multiple passes
    optimized = copy.deepcopy(function_node)

    # Step 1: Propagate constants from kwargs
    propagator = ConstantPropagator(kwargs)
    optimized = propagator.visit(optimized)

    # Apply constant folding
    folder = ConstantFolder()
    optimized = folder.visit(optimized)
    ast.fix_missing_locations(optimized)

    # Step 2: Inline function calls
    inliner = FunctionInliner(functions)

    # Process the body statements
    new_body = []
    for stmt in optimized.body:
        result = inliner.visit(stmt)
        if isinstance(result, list):
            new_body.extend(result)
        elif result is not None:
            new_body.append(result)

    optimized.body = new_body
    ast.fix_missing_locations(optimized)

    # Apply constant folding again after inlining
    optimized = folder.visit(optimized)
    ast.fix_missing_locations(optimized)

    # Step 3: Eliminate dead code
    eliminator = DeadCodeEliminator()

    # Process the body statements
    new_body = []
    for stmt in optimized.body:
        result = eliminator.visit(stmt)
        if isinstance(result, list):
            new_body.extend(result)
        elif result is not None:
            new_body.append(result)

    optimized.body = new_body
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
