class SBUFTensor:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:
        return f"SBUFTensor({self.name}, shape={self.shape})"


class PSUMTensor:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:
        return f"PSUMTensor({self.name}, shape={self.shape})"
