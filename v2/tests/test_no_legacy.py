"""Guard: v2 is written from the ground up.

v2 code may import the standard library, ``sn120_v2`` itself, and the
third-party names listed in ``third_party_imports.txt``. It must never import
a legacy v1 or prototype module, and it must never use ``sys.path`` to reach
one.

Two workstreams may reuse v1, as the spec says: the benchmark service
(``v2/bench/``) and the website (``v2/site/``). Only those two paths are
exempt. The guard is strict everywhere else.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).resolve().parents[1]
EXEMPT_DIRS = (V2_ROOT / "bench", V2_ROOT / "site")
LEGACY_ROOTS = frozenset(
    "bench caps configs dashboard docs examples miner operator_profiles ops priors prospective "
    "prototype research sn120v2 subnet systemd tools vendor".split()
)
DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def approved_third_party() -> frozenset[str]:
    lines = (Path(__file__).parent / "third_party_imports.txt").read_text().splitlines()
    return frozenset(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))


def imported_roots(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name.split(".")[0]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module.split(".")[0]))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if called in DYNAMIC_IMPORTERS and isinstance(node.args[0].value, str):
                found.append((node.lineno, node.args[0].value.split(".")[0]))
    return found


def sys_path_uses(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    ]


def violations(source: str, path: Path, approved: frozenset[str]) -> list[str]:
    tree = ast.parse(source, filename=str(path))
    problems = []
    for line, root in imported_roots(tree):
        if root == "sn120_v2" or root in sys.stdlib_module_names or root in approved:
            continue
        kind = "legacy module" if root in LEGACY_ROOTS else "module that is not approved"
        problems.append(f"{path}:{line}: imports {kind} '{root}'")
    problems += [f"{path}:{line}: uses sys.path" for line in sys_path_uses(tree)]
    return problems


def checked_files() -> list[Path]:
    return [
        path
        for path in sorted(V2_ROOT.rglob("*.py"))
        if ".venv" not in path.parts and not any(path.is_relative_to(exempt) for exempt in EXEMPT_DIRS)
    ]


def test_v2_code_imports_no_legacy_module():
    approved = approved_third_party()
    files = checked_files()
    assert files, "the guard found no v2 files to check"
    problems = [p for path in files for p in violations(path.read_text(), path, approved)]
    assert not problems, "\n".join(problems)


def test_a_legacy_name_can_never_be_approved():
    assert not approved_third_party() & LEGACY_ROOTS


def test_only_the_benchmark_and_website_paths_are_exempt():
    assert sorted(path.relative_to(V2_ROOT).as_posix() for path in EXEMPT_DIRS) == ["bench", "site"]


@pytest.mark.parametrize(
    "source",
    [
        "import subnet.chain",
        "from priors.baseline_v1 import estimate",
        "from sn120v2.judge import z_crit",
        "import importlib\nimportlib.import_module('ops.hourly_burn')",
        "__import__('miner')",
        "import sys\nsys.path.insert(0, '..')",
        "import ff_format",
        "import torch",
    ],
)
def test_the_guard_rejects(source, tmp_path):
    assert violations(source, tmp_path / "module.py", frozenset({"pytest"}))


@pytest.mark.parametrize(
    "source",
    [
        "import json",
        "import pytest",
        "from sn120_v2.canonical import digest",
        "from . import errors",
        "from .errors import V2Error",
    ],
)
def test_the_guard_accepts(source, tmp_path):
    assert not violations(source, tmp_path / "module.py", frozenset({"pytest"}))
