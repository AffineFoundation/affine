"""The math task pool and the prompt of a task."""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from sn120_v2.envs import math_pool
from sn120_v2.envs.math_pool import (
    MANDATE,
    POOL_FILE,
    POOL_SHA256,
    POOL_V2_0_FILE,
    POOL_V2_0_SHA256,
    Task,
    load_pool,
    pool_for_draw,
)
from sn120_v2.errors import ContractError, VerifyError


@pytest.fixture(scope="module")
def pool():
    return load_pool()


def test_the_pool_is_public_pinned_and_the_same_for_everyone(pool):
    assert POOL_FILE.parent == Path(math_pool.__file__).parent and POOL_FILE.is_file(), "the pool ships inside the package"
    assert pool.sha256 == POOL_SHA256
    assert len(pool) == len(pool.task_ids) == len(set(pool.task_ids)) == 20120
    assert Counter(pool.task(task_id).slice for task_id in pool.task_ids) == {"affine_math": 7465, "dapo": 12655}
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


def test_the_new_pool_is_the_old_bytes_minus_the_published_dropped_list(pool):
    old = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    assert len(old) == 21574 and old.task_ids[:4] == pool.task_ids[:4] == tuple(f"math:{i}" for i in range(4))
    path = POOL_FILE.with_suffix(".dropped.json")
    report = json.loads(path.read_bytes())
    dropped = [row["id"] for row in report["dropped"]]
    assert report["from_sha256"] == POOL_V2_0_SHA256 and report["to_sha256"] == POOL_SHA256
    assert report["source"] == "catoneone/sn120-k1-fixes@427295a"
    assert len(dropped) == len(set(dropped)) == 1454 and len(report["kept_key_format"]) == 156
    assert dropped == sorted(dropped, key=lambda task_id: (task_id.split(":")[0], int(task_id.split(":")[1])))
    removed = set(dropped)
    assert not removed & set(pool.task_ids) and removed <= set(old.task_ids)
    old_lines = POOL_V2_0_FILE.read_bytes().splitlines(keepends=True)
    expected = b"".join(line for line in old_lines if json.loads(line)["id"] not in removed)
    assert POOL_FILE.read_bytes() == expected and expected.endswith(b"\n")
    kept_by_id = {json.loads(line)["id"]: line for line in POOL_FILE.read_bytes().splitlines(keepends=True)}
    assert all(kept_by_id[json.loads(line)["id"]] == line for line in old_lines if json.loads(line)["id"] not in removed)


def test_an_old_round_draw_selects_the_archived_pool_and_a_dropped_prompt_is_not_a_pool_task(pool):
    old = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    dropped = next(task_id for task_id in old.task_ids if task_id not in pool.task_ids)
    assert pool_for_draw((dropped,)).sha256 == POOL_V2_0_SHA256
    assert pool_for_draw((pool.task_ids[0],)).sha256 == POOL_SHA256
    assert pool_for_draw((dropped,), "v2.0").task(dropped) == old.task(dropped)
    assert pool.task_for_prompt(old.task(dropped).batch_prompt()) is None
    with pytest.raises(ContractError, match="not in the current math pool"):
        pool_for_draw((dropped,), "current")


def test_the_pool_builder_reproduces_both_files():
    pytest.importorskip("math_verify")
    from sn120_v2.envs.tools import build_clean_pool

    pool, dropped = build_clean_pool.build()
    assert pool == POOL_FILE.read_bytes()
    assert dropped == POOL_FILE.with_suffix(".dropped.json").read_bytes()


def test_the_pool_module_needs_the_standard_library_only():
    tree = ast.parse(POOL_FILE.with_name("math_pool.py").read_text())
    absolute = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom) and not getattr(node, "level", 0)]
    roots = {(node.module if isinstance(node, ast.ImportFrom) else node.names[0].name).split(".")[0] for node in absolute}
    assert roots <= sys.stdlib_module_names
