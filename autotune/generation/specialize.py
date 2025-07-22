import ast
import copy
import inspect
import textwrap
from typing import Any, Dict, List, Union

import astor


class GlobalConstantPropagator(ast.NodeTransformer):
    """Replace variables with known compile-time constants throughout a function"""

    def __init__(self, constants: Dict[str, Any]):
        self.constants = constants

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Replace variable names with constant values if available"""
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            value = self.constants[node.id]
            if isinstance(value, (int, float, str, bool, type(None))):
                return ast.Constant(value=value)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Handle dictionary subscripting like dict_var["key"]"""
        # Visit children first to handle nested expressions
        node = self.generic_visit(node)

        # Check if this is accessing a dictionary constant
        if isinstance(node.value, ast.Name) and node.value.id in self.constants:
            dict_var = self.constants[node.value.id]

            if isinstance(dict_var, dict) and isinstance(node.slice, ast.Constant):
                key = node.slice.value
                if key in dict_var:
                    # Replace with the constant value from the dictionary
                    return ast.Constant(value=dict_var[key])

        return node


class ConstantExpressionEvaluator(ast.NodeTransformer):
    """Evaluate expressions involving constants"""

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """Evaluate comparison operations with constant operands"""
        # Visit all parts of the comparison first
        node.left = self.visit(node.left)
        node.comparators = [self.visit(comp) for comp in node.comparators]

        # Check if all parts are constants
        if isinstance(node.left, ast.Constant) and all(isinstance(comp, ast.Constant) for comp in node.comparators):
            # For simple comparisons with one operator
            if len(node.ops) == 1 and len(node.comparators) == 1:
                try:
                    left_val = node.left.value
                    right_val = node.comparators[0].value

                    # Evaluate based on operator type
                    if isinstance(node.ops[0], ast.Eq):
                        result = left_val == right_val
                    elif isinstance(node.ops[0], ast.NotEq):
                        result = left_val != right_val
                    elif isinstance(node.ops[0], ast.Lt):
                        result = left_val < right_val
                    elif isinstance(node.ops[0], ast.LtE):
                        result = left_val <= right_val
                    elif isinstance(node.ops[0], ast.Gt):
                        result = left_val > right_val
                    elif isinstance(node.ops[0], ast.GtE):
                        result = left_val >= right_val
                    else:
                        return node

                    return ast.Constant(value=result)
                except Exception:
                    pass

        return node


class ConditionalEvaluator(ast.NodeTransformer):
    """Evaluate if-else statements with constant conditions"""

    def visit_If(self, node: ast.If) -> Union[ast.If, List[ast.stmt]]:
        # Apply constant evaluation to the condition
        node.test = self.visit(node.test)
        expr_evaluator = ConstantExpressionEvaluator()
        node.test = expr_evaluator.visit(node.test)

        # If condition is now a constant, determine which branch to keep
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                # Process and return the body (True branch)
                return [self.visit(stmt) for stmt in node.body]
            else:
                # Process and return the orelse (False branch)
                return [self.visit(stmt) for stmt in node.orelse]

        # If condition is not a constant, process both branches
        node.body = [self.visit(stmt) for stmt in node.body]
        node.orelse = [self.visit(stmt) for stmt in node.orelse]

        return node


class DeadCodeEliminator(ast.NodeTransformer):
    """Remove loops with empty bodies and other dead code"""

    def visit_For(self, node: ast.For) -> Union[ast.For, List[ast.stmt]]:
        # Visit the body first to process any nested transformations
        node.body = [self.visit(stmt) for stmt in node.body if stmt]

        # If the body is empty after transformations, remove the loop
        if not node.body:
            return []
        return node

    def visit_While(self, node: ast.While) -> Union[ast.While, List[ast.stmt]]:
        node.body = [self.visit(stmt) for stmt in node.body if stmt]
        if not node.body:
            return []
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Handle function definitions, inlining any helper calls in the body"""
        node.body = self._visit_and_flatten_body(node.body)
        return node

    def _visit_and_flatten_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        """Visit statements in a body and flatten any lists returned by transformations"""
        result = []
        for stmt in body:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                # If a statement was transformed into multiple statements, add them all
                result.extend(transformed)
            elif transformed is not None:
                # If a statement was transformed into a single statement, add it
                result.append(transformed)
        return result

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
            if isinstance(new_stmt, list):
                substituted_body.extend(new_stmt)
            elif new_stmt is not None:
                substituted_body.append(new_stmt)

        return substituted_body


class VariableSubstituter(ast.NodeTransformer):
    def __init__(self, arg_map: Dict[str, ast.AST]):
        self.arg_map = arg_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.arg_map and isinstance(node.ctx, ast.Load):
            # Return a deep copy of the argument expression
            return copy.deepcopy(self.arg_map[node.id])
        return node


def specialize_kernel(main_function, helper_names: List[str], **kwargs) -> str:
    """
    Inline helper functions into a main function and apply optimizations:
    1. Propagate constants to the specified helper functions
    2. Evaluate conditionals in helper functions
    3. Inline helper functions into the main function
    4. Perform dead code elimination
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
                    # Apply transformations to helper function
                    if kwargs:
                        # Step 1: Propagate constants
                        constant_propagator = GlobalConstantPropagator(kwargs)
                        node = constant_propagator.visit(node)

                        # Step 2: Evaluate expressions
                        expr_evaluator = ConstantExpressionEvaluator()
                        node = expr_evaluator.visit(node)

                        # Step 3: Evaluate conditionals
                        conditional_evaluator = ConditionalEvaluator()
                        node = conditional_evaluator.visit(node)

                        # Step 4: Remove dead code
                        dead_code_eliminator = DeadCodeEliminator()
                        node = dead_code_eliminator.visit(node)

                    helper_funcs[helper_name] = node
                    break
        else:
            raise ValueError(f"Helper function '{helper_name}' not found in module")

    # Create inliner with the processed helper functions
    inliner = FunctionInliner(helper_funcs)

    # Inline helpers in the main function
    inlined_main = inliner.visit(main_func_node)

    # Apply final transformations to the main function
    if kwargs:
        # Propagate any remaining constants in the main function
        constant_propagator = GlobalConstantPropagator(kwargs)
        inlined_main = constant_propagator.visit(inlined_main)

        # Evaluate any remaining expressions
        expr_evaluator = ConstantExpressionEvaluator()
        inlined_main = expr_evaluator.visit(inlined_main)

        # Evaluate any remaining conditionals
        conditional_evaluator = ConditionalEvaluator()
        inlined_main = conditional_evaluator.visit(inlined_main)

    # Perform final dead code elimination
    dead_code_eliminator = DeadCodeEliminator()
    inlined_main = dead_code_eliminator.visit(inlined_main)

    # Convert back to source code
    try:
        return astor.to_source(inlined_main).strip()
    except ImportError:
        # Fallback if astor is not available
        return ast.unparse(inlined_main)
