#!/usr/bin/env python3
"""Find stub implementations with only pass statements."""

import ast
import os
from pathlib import Path


class StubFinder(ast.NodeVisitor):
    """Find methods and functions that only contain pass statements."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.stubs = []

    def visit_FunctionDef(self, node):
        """Check if function/method only contains pass statement."""
        # Skip empty functions (just pass and docstring)
        non_pass_statements = [
            stmt for stmt in node.body
            if not isinstance(stmt, (ast.Pass, ast.Expr)) or
            (isinstance(stmt, ast.Expr) and not isinstance(stmt.value, ast.Constant))
        ]

        # If only pass statements (and possibly docstrings)
        if not non_pass_statements:
            has_pass = any(isinstance(stmt, ast.Pass) for stmt in node.body)
            if has_pass:
                self.stubs.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                    'filepath': self.filepath,
                    'class': getattr(self, '_current_class', None)
                })

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Track current class for method context."""
        old_class = getattr(self, '_current_class', None)
        self._current_class = node.name

        # Check if entire class is empty (only __init__ with pass)
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if methods:
            all_stub_methods = True
            for method in methods:
                non_pass_statements = [
                    stmt for stmt in method.body
                    if not isinstance(stmt, (ast.Pass, ast.Expr)) or
                    (isinstance(stmt, ast.Expr) and not isinstance(stmt.value, ast.Constant))
                ]
                if non_pass_statements:
                    all_stub_methods = False
                    break

            if all_stub_methods:
                self.stubs.append({
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                    'filepath': self.filepath,
                    'methods': [m.name for m in methods]
                })

        self.generic_visit(node)
        self._current_class = old_class


def find_stubs_in_file(filepath):
    """Find stub implementations in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        finder = StubFinder(filepath)
        finder.visit(tree)
        return finder.stubs
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def main():
    """Find all stub implementations in the project."""
    project_root = Path("/Users/jchadwick/code/billiards-trainer")
    all_stubs = []

    # Find all Python files in backend
    for py_file in project_root.rglob("backend/**/*.py"):
        if "/.venv/" in str(py_file) or "/venv/" in str(py_file):
            continue

        stubs = find_stubs_in_file(py_file)
        all_stubs.extend(stubs)

    # Group by file
    by_file = {}
    for stub in all_stubs:
        filepath = stub['filepath']
        if filepath not in by_file:
            by_file[filepath] = []
        by_file[filepath].append(stub)

    # Print results
    print("=== STUB IMPLEMENTATIONS FOUND ===\n")

    for filepath, stubs in sorted(by_file.items()):
        rel_path = str(Path(filepath).relative_to(project_root))
        print(f"📄 {rel_path}")

        for stub in stubs:
            if stub['type'] == 'class':
                print(f"  🏗️  Class: {stub['name']} (line {stub['line']}) - All methods are stubs")
                for method in stub['methods']:
                    print(f"     • {method}()")
            else:
                class_info = f" in {stub['class']}" if stub['class'] else ""
                print(f"  🔧 Function: {stub['name']}(){class_info} (line {stub['line']})")
        print()

    print(f"Total stub files: {len(by_file)}")
    print(f"Total stub implementations: {len(all_stubs)}")


if __name__ == "__main__":
    main()
