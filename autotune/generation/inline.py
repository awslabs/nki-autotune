import ast
import copy
import inspect
import textwrap
from typing import Dict, List, Union

import astor


def inline_helper_functions(main_function, helper_names: List[str]) -> str:
    """
    Inline helper functions into a main function.

    Args:
        main_function: The main function object to inline helpers into
        helper_names: List of helper function names to inline

    Returns:
        String containing the modified function code with helpers inlined
    """
    # Get the module where the main function is defined
    module = inspect.getmodule(main_function)

    # Get source code of the main function
    main_source = inspect.getsource(main_function)
    main_source = textwrap.dedent(main_source)

    # Parse the source code
    tree = ast.parse(main_source)
    main_func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == main_function.__name__:
            main_func_node = node
            break

    if main_func_node is None:
        raise ValueError(f"Main function '{main_function.__name__}' not found")

    # Get helper function definitions
    helper_funcs = {}
    for helper_name in helper_names:
        if hasattr(module, helper_name):
            helper_func = getattr(module, helper_name)
            helper_source = inspect.getsource(helper_func)
            helper_source = textwrap.dedent(helper_source)
            helper_tree = ast.parse(helper_source)
            for node in helper_tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == helper_name:
                    helper_funcs[helper_name] = node
                    break
        else:
            raise ValueError(f"Helper function '{helper_name}' not found in module")

    # Create inliner
    inliner = FunctionInliner(helper_funcs)

    # Inline helpers in the main function
    inlined_main = inliner.visit(main_func_node)

    # Convert back to source code
    try:
        return astor.to_source(inlined_main).strip()
    except ImportError:
        # Fallback if astor is not available
        return ast.unparse(inlined_main)


class VariableSubstituter(ast.NodeTransformer):
    def __init__(self, arg_map: Dict[str, ast.AST]):
        self.arg_map = arg_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.arg_map and isinstance(node.ctx, ast.Load):
            # Return a deep copy of the argument expression
            return copy.deepcopy(self.arg_map[node.id])
        return node


class FunctionInliner(ast.NodeTransformer):
    def __init__(self, helper_funcs: Dict[str, ast.FunctionDef]):
        self.helper_funcs = helper_funcs

    def visit_Expr(self, node: ast.Expr) -> Union[ast.Expr, List[ast.stmt]]:
        """Handle expression statements that might contain function calls to inline"""
        if isinstance(node.value, ast.Call):
            result = self.visit_Call(node.value)
            if isinstance(result, list):
                return result
            node.value = result
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Union[ast.Call, List[ast.stmt]]:
        """Handle function calls that might be helper functions"""
        if isinstance(node.func, ast.Name) and node.func.id in self.helper_funcs:
            helper_func = self.helper_funcs[node.func.id]
            return self._inline_function_call(node, helper_func)

        # Visit the arguments of non-helper function calls
        for i, arg in enumerate(node.args):
            node.args[i] = self.visit(arg)
        for kw in node.keywords:
            kw.value = self.visit(kw.value)
        return node

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, List[ast.stmt]]:
        """Handle assignments where the value might be a function call to inline"""
        if isinstance(node.value, ast.Call):
            result = self.visit_Call(node.value)
            if isinstance(result, list):
                # Function was inlined - need to handle return value
                if result and isinstance(result[-1], ast.Return):
                    # Replace the return with an assignment
                    return_stmt = result[-1]
                    assignment = ast.Assign(targets=node.targets, value=return_stmt.value)
                    return result[:-1] + [assignment]
                # No return value, just execute the statements
                return result
            node.value = result
        return self.generic_visit(node)

    def _inline_function_call(self, call_node: ast.Call, func_def: ast.FunctionDef) -> List[ast.stmt]:
        """Inline a function call by substituting arguments and returning the function body"""
        # Create argument mapping
        arg_map = {}
        params = [arg.arg for arg in func_def.args.args]

        # Map positional arguments
        for i, arg_value in enumerate(call_node.args):
            if i < len(params):
                arg_map[params[i]] = arg_value

        # Map keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg in params:
                arg_map[keyword.arg] = keyword.value

        # Clone and substitute the function body
        substituted_body = []
        for stmt in func_def.body:
            # Deep copy the statement and substitute variables
            stmt_copy = copy.deepcopy(stmt)
            substituter = VariableSubstituter(arg_map)
            new_stmt = substituter.visit(stmt_copy)
            substituted_body.append(new_stmt)

        return substituted_body
