"""The rules that keep one library able to host many papers (``papers/README.md``)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = sorted(p.name for p in (ROOT / "papers").iterdir() if p.is_dir() and (p / "__init__.py").exists())


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_there_are_papers_to_check():
    assert {"decomposition", "stealth_execution"} <= set(PAPERS)


def test_the_library_and_its_tools_never_import_a_paper():
    for folder in ("order_book", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            bad = {m for m in imported_modules(path) if m.split(".")[0] == "papers"}
            assert not bad, f"{path} imports {bad}: anything two papers need belongs in the library"


def test_the_library_never_imports_its_scripts():
    for path in (ROOT / "order_book").rglob("*.py"):
        assert not {m for m in imported_modules(path) if m.split(".")[0] == "scripts"}, path


def test_papers_never_import_each_other():
    for paper in PAPERS:
        for path in (ROOT / "papers" / paper).rglob("*.py"):
            for module in imported_modules(path):
                parts = module.split(".")
                if parts[0] == "papers":
                    assert parts[1] == paper, f"{path} imports {module}: move what they share into the library"
