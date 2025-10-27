class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op: str, src: list[str], dest: str, params: dict = {}) -> None:
        self.op = op
        self.src = src
        self.dest = dest
        self.params = params

    def __repr__(self) -> str:
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op}': {self.src} -> {self.dest}{params_str})"
