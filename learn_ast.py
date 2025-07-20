import ast
import inspect


def test_func(x: int, y, test: bool = True) -> int:
    z = x + y
    return z


if __name__ == "__main__":
    # Parse some simple code
    source = inspect.getsource(test_func)
    tree = ast.parse(source)
    for content in tree.body:
        for field in content._fields:
            print(field, getattr(content, field))
    print(ast.dump(tree, indent=2))
