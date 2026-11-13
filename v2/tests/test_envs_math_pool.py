"""The math task pool and the prompt of a task."""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import pytest

from sn120_v2.envs import math_pool
from sn120_v2.envs.math_pool import MANDATE, POOL_FILE, POOL_SHA256, Task, load_pool
from sn120_v2.errors import ContractError, VerifyError


@pytest.fixture(scope="module")
def pool():
    return load_pool()


def test_the_pool_is_public_pinned_and_the_same_for_everyone(pool):
    assert POOL_FILE.parent == Path(math_pool.__file__).parent and POOL_FILE.is_file(), "the pool ships inside the package"
    assert pool.sha256 == POOL_SHA256
    assert len(pool) == len(pool.task_ids) == len(set(pool.task_ids)) == 21574
    assert Counter(pool.task(task_id).slice for task_id in pool.task_ids) == {"affine_math": 7492, "dapo": 14082}
    assert pool.task_ids[0] == "math:0" and load_pool().task_ids == pool.task_ids, "ids keep the order of the file"


def test_a_pool_with_other_bytes_is_refused(tmp_path):
    changed = tmp_path / "pool.jsonl"
    changed.write_bytes(POOL_FILE.read_bytes().replace(b"1", b"2", 1))
    with pytest.raises(VerifyError):
        load_pool(changed)
    with pytest.raises(VerifyError):
        load_pool(tmp_path / "missing.jsonl")
    with pytest.raises(VerifyError):
        math_pool.MathPool((Task("a", "s", "same problem", "1"), Task("b", "s", "same problem", "2")), "x")


def test_an_unknown_task_id_is_a_contract_error(pool):
    with pytest.raises(ContractError):
        pool.task("math:999999999")


def test_the_scoring_prompt_is_the_mandate_then_the_problem(pool):
    task = pool.task(pool.task_ids[7])
    assert task.messages() == [{"role": "system", "content": MANDATE}, {"role": "user", "content": task.problem}]
    assert task.messages() == pool.task(task.task_id).messages(), "the prompt depends on the task only"
    assert "\\boxed{}" in MANDATE and task.answer and task.problem


def test_a_batch_prompt_finds_its_task(pool):
    task = pool.task("dapo:5")
    assert task.batch_prompt() == MANDATE + "\n\n" + task.problem
    assert pool.task_for_prompt(task.batch_prompt()) == task
    assert pool.task_for_prompt(task.problem) == task
    assert pool.task_for_prompt(task.problem + " ") is None
    assert pool.task_for_prompt("a problem that is not in the pool") is None


def test_the_pool_module_needs_the_standard_library_only():
    tree = ast.parse(POOL_FILE.with_name("math_pool.py").read_text())
    absolute = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom) and not getattr(node, "level", 0)]
    roots = {(node.module if isinstance(node, ast.ImportFrom) else node.names[0].name).split(".")[0] for node in absolute}
    assert roots <= sys.stdlib_module_names
