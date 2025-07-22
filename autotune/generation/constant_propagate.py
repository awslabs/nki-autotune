import ast
from typing import Any, Dict, List


class ConstantPropagator(ast.NodeTransformer):
    """Replace variables with known compile-time constants only in kwargs to target functions"""

    def __init__(self, constants: Dict[str, Any], targets: List[str]):
        print(f"Replace {constants} in {targets}.")
        self.constants = constants
        self.targets = targets or []
        self.in_target_call_kwargs = False

    def visit_Call(self, node):
        """Handle function calls"""
        # First visit the function normally
        node.func = self.visit(node.func)

        # Check if this is a call to a target function
        is_target_call = isinstance(node.func, ast.Name) and node.func.id in self.targets

        # For keyword arguments, enable constant replacement only if in target function
        old_in_target = self.in_target_call_kwargs
        if is_target_call:
            self.in_target_call_kwargs = True

        # Process arguments with potential constant replacement
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(kw) for kw in node.keywords]

        # Restore previous state
        self.in_target_call_kwargs = old_in_target

        return node

    def visit_Name(self, node):
        """Replace names with constants only in kwargs to target functions"""
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            # Only replace if we're in keyword arguments to a target function
            print(node.id, self.in_target_call_kwargs)
            if self.in_target_call_kwargs:
                return ast.Constant(value=self.constants[node.id])
        return node
