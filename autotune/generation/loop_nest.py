from typing import List


class LoopContent:
    def __init__(
        self, comment: str, loop_var: str, loop_range: str, opening_ops: List[str], closing_ops: List[str]
    ) -> None:
        self.comment = comment
        self.loop_var = loop_var
        self.loop_range = loop_range
        self.opening_ops = opening_ops
        self.closing_ops = closing_ops
        self.is_loop = loop_var and loop_range
        self.indent_size = 4

    def _get_indent(self, level):
        """Get indentation string for given nesting level."""
        return " " * (self.indent_size * level)

    def generate_opening_code(self):
        """Generate opening operations as a list of strings."""
        openings = []
        if self.comment:
            openings.append(f'"""{self.comment}"""')
        for op in self.opening_ops:
            openings.append(op)
        return openings

    def generate_closing_code(self):
        """Generate closing operations as a list of strings."""
        return self.closing_ops


class LoopNestGenerator:
    def __init__(self, function_header: str, loops: List[LoopContent]):
        self.function_header = function_header
        self.loops = loops
        self.num_loops = self._determine_loop_count()

    def _determine_loop_count(self) -> int:
        """Determine the number of loops based on operation positions."""
        num_loops = 0
        for loop in self.loops:
            if loop.is_loop:
                num_loops += 1
        return num_loops

    def _get_indent(self, level):
        """Get indentation string for given nesting level."""
        return " " * (4 * level)

    def _insert_operations(self, ops, position, indent_level):
        """Insert operations at given position with proper indentation."""
        if position in ops and ops[position]:
            lines = []
            for op in ops[position]:
                lines.append(f"{self._get_indent(indent_level)}{op}")
            return lines
        return []

    def generate_loopnest(self, indent_offset: int, openings: List[List[str]], closings: List[List[str]]) -> List[str]:
        """Generate the nested loop structure with opening and closing operations."""
        lines = []
        indent_level = indent_offset

        for i, loop in enumerate(self.loops):
            if loop.is_loop:
                # Add loop declaration
                lines.append(f"{self._get_indent(indent_level)}for {loop.loop_var} in {loop.loop_range}:")
                # Add opening operations for this loop
                indent_level += 1
                for op in openings[i]:
                    lines.append(f"{self._get_indent(indent_level)}{op}")
            else:
                # Handle non-loop content (like outer operations)
                for op in openings[i]:
                    lines.append(f"{self._get_indent(indent_level)}{op}")

        # Add closing operations in reverse order
        for i in range(len(self.loops) - 1, -1, -1):
            loop = self.loops[i]
            for op in closings[i]:
                lines.append(f"{self._get_indent(indent_level)}{op}")
            if loop.is_loop:
                indent_level -= 1

        return lines

    def generate_code(self):
        """Generate the complete code as a single string."""
        openings = []
        closings = []
        for loop in self.loops:
            loop_opening = loop.generate_opening_code()
            loop_closing = loop.generate_closing_code()
            openings.append(loop_opening)
            closings.append(loop_closing)
        lines = [self.function_header]
        kernel_lines = self.generate_loopnest(1, openings, closings)
        lines.extend(kernel_lines)
        kernel_code = "\n".join(lines)
        return kernel_code


# Example usage and testing
if __name__ == "__main__":
    loop_contents = [
        LoopContent(
            comment=f"This is outer ops",
            loop_var="",
            loop_range="",
            opening_ops=[f"outer_init()"],
            closing_ops=[f"outer_clean()"],
        )
    ]
    for loop_var in "MKN":
        loop_content = LoopContent(
            comment=f"This is loop {loop_var}",
            loop_var=f"block_id_{loop_var}",
            loop_range=f"nl.affine_range({loop_var})",
            opening_ops=[f"initialization_{loop_var}()"],
            closing_ops=[f"clean_loop_{loop_var}()"],
        )
        loop_contents.append(loop_content)
    generator = LoopNestGenerator(function_header="", loops=loop_contents)
    kernel_code = generator.generate_code()
    print(kernel_code)
