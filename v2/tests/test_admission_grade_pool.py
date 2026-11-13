"""Admission over the real math env and its ``GradePool``: a hard time limit, and one bad file does not stop the others.

These tests need the ``envs`` extra.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from sn120_v2 import layout
from sn120_v2.admission import FAULT, TIME_LIMIT, Invalid, StoreAdmission, check_batch, pool_grader
from sn120_v2.canonical import sha256_hex
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.errors import InfraError

math_grader = pytest.importorskip("sn120_v2.envs.math_grader", reason="install the envs extra: uv pip install -e '.[dev,envs]'")


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


class Store(dict):
    put = dict.__setitem__
    get = dict.__getitem__

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


def row(answer, reward, think: str) -> dict:
    return {"output": f"{think}\\boxed{{{answer}}}", "reward": reward}


def file(groups, hotkey: str) -> bytes:
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": groups}).encode()


def scripted_job(task, output: str, capped: bool):
    """A grade job for the real pool. The output text says what the worker process does."""
    if "die" in output:
        os._exit(1)
    if "hang" in output:
        time.sleep(600)
    if "raise" in output:
        raise ZeroDivisionError("the text of this row breaks the grader")
    return math_grader.grade(task, output, capped)


def test_real_grade_pool_kills_a_hung_row_and_skips_only_that_file(build):
    pool = load_pool()
    tasks = [pool.task(task_id) for task_id in pool.task_ids[:6]]
    think = "<think>\n" + "step " * 40 + "\n</think>\n\n"

    def groups(poison: str = "") -> list[dict]:
        return [{"prompt": task.batch_prompt(), "rollouts": [row(task.answer, 1, think), row(task.answer + "+1", 0, think + (poison + " " if n == 2 else ""))]}
                for n, task in enumerate(tasks)]

    grade_pool = math_grader.GradePool(workers=2, timeout_s=3.0, job=scripted_job)
    try:
        official = pool_grader(pool, grade_pool)
        official.probe()  # the first pool tasks pass with their own answers
        adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=official)
        files = {"good": groups(), "hangs": groups("hang"), "dies": groups("die"), "raises": groups("raise"), "also_good": groups()}
        for hotkey, content in files.items():
            adm.store.put(layout.miner_object(hotkey), file(content, hotkey))
        started = time.monotonic()
        got = adm.verdicts(list(files))
        assert time.monotonic() - started < 60
        assert {hotkey: verdict.rule for hotkey, verdict in got.items()} == {"good": "", "hangs": TIME_LIMIT, "dies": FAULT, "raises": FAULT, "also_good": ""}
        assert adm.freeze(build.reg(1, hotkey="hangs"), "r000001") is None and adm.freeze(build.reg(2, hotkey="also_good"), "r000001") is not None
    finally:
        grade_pool.close()
    broken = math_grader.GradePool(workers=2, timeout_s=3.0, job=os._exit)  # every job kills its worker: the grader is broken for all
    try:
        with pytest.raises(InfraError, match="known-good"):
            check_batch(file(groups(), "good"), "good", build.params(), tokens=WordTokens(), grader=pool_grader(pool, broken))
    finally:
        broken.close()
