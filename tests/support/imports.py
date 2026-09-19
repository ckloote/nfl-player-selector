"""What a module imports, read from its source rather than by importing it."""

import ast
from pathlib import Path


def top_level_imports(module) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found |= {a.name for a in node.names}
    return found
