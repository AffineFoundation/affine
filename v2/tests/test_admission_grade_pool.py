"""Admission over the real math env and its ``GradePool``: a hard time limit, and one bad file does not stop the others.

These tests need the ``envs`` extra.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import CHECK_FAULT, FAULT, FAULT_HOLD_S, FAULT_TRIES, TIME_LIMIT, Invalid, StoreAdmission, check_batch, env_grader, pool_grader
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


def hang(task, output: str, capped: bool):
    """A grade job that never ends: a pool that is too slow for every row."""
    time.sleep(600)


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
        now = [1000.0]
        adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=official, clock=lambda: now[0])
        files = {"good": groups(), "hangs": groups("hang"), "dies": groups("die"), "raises": groups("raise"), "also_good": groups()}
        for hotkey, content in files.items():
            adm.store.put(layout.miner_object(hotkey), file(content, hotkey))
        started = time.monotonic()
        got = adm.verdicts(list(files))
        assert time.monotonic() - started < 60
        assert {hotkey: verdict.rule for hotkey, verdict in got.items()} == {"good": "", "hangs": TIME_LIMIT, "dies": FAULT, "raises": FAULT, "also_good": ""}
        assert adm.freeze(build.reg(1, hotkey="hangs"), "r000001") is None and adm.freeze(build.reg(2, hotkey="also_good"), "r000001") is not None
        assert all(got[hotkey].reason == f"groups[2].rollouts[1]: {what} (check 1 of 3)" for hotkey, what in (("hangs", "the official grader did not finish this row inside its time limit"), ("dies", "the official grader fails on this row")))
        started = time.monotonic()
        assert all(adm.verdicts(list(files))[hotkey] is got[hotkey] for hotkey in files) and time.monotonic() - started < 1.0, "inside the hold no file is checked again"
        now[0] += FAULT_HOLD_S
        adm.verdicts(list(files))
        now[0] += FAULT_HOLD_S
        kept = adm.verdicts(list(files))  # the third failed check in a row, at the same row: the verdict stays
        assert all(kept[hotkey].reason.endswith(f"(checked {FAULT_TRIES} times: this verdict stays until a new file is uploaded)") for hotkey in ("hangs", "dies", "raises"))
        started = time.monotonic()
        assert all(adm.verdicts(list(files))[hotkey] is kept[hotkey] for hotkey in files) and time.monotonic() - started < 1.0, "no file is checked again: no wait at the hung row"
    finally:
        grade_pool.close()
    one_group = file(groups()[:1], "good")
    for job in (os._exit, hang):  # every job kills its worker, or never ends: the pool is dead, or too slow, for every row
        down = math_grader.GradePool(workers=4, timeout_s=0.5, job=job)
        adm = StoreAdmission(Store({layout.miner_object("good"): one_group}), build.params(), tokens=WordTokens(), grader=pool_grader(pool, down))
        try:
            for _ in range(2):  # loud each time, and nothing is held against the honest file
                with pytest.raises(InfraError, match="known-good"):
                    adm.verdicts(["good"])
            assert adm._tries == {} and adm._seen == {} and adm._held == {}
        finally:
            down.close()


def test_known_good_file_of_the_checker_passes_with_the_real_math_env(build, monkeypatch):
    pool, params = load_pool(), build.params(min_pool_prompt_frac=1.0)
    grade_pool = math_grader.GradePool(workers=2, timeout_s=20.0)
    try:
        for official in (env_grader(pool, math_grader.grade), pool_grader(pool, grade_pool)):
            files = list(admission._known_good(params, official))
            assert len(files) == admission.PROBE_TASKS
            for raw in files:  # every one passes: the right answer gets reward 1, and the try with no answer gets reward 0
                assert check_batch(raw, admission.PROBE_HOTKEY, params, tokens=WordTokens(), grader=official).checked == ("format", "tokens", "grader")
        real = admission._cap

        def buggy(groups, *limits):  # a bug of the checker that only a file with 3 groups hits
            return real(groups, *limits) if len(groups) != 3 else groups[3]

        monkeypatch.setattr(admission, "_cap", buggy)
        tasks = [pool.task(task_id) for task_id in pool.task_ids[:4]]
        content = [{"prompt": task.batch_prompt(), "rollouts": [row(task.answer, 1, "step " * 40), {"output": "step " * 40, "reward": 0}]} for task in tasks]
        adm = StoreAdmission(Store(), params, tokens=WordTokens(), grader=official)
        adm.store.put(layout.miner_object("good"), file(content, "good"))
        adm.store.put(layout.miner_object("hits_the_bug"), file(content[:3], "hits_the_bug"))
        got = adm.verdicts(["hits_the_bug", "good"])
        assert {hotkey: verdict.rule for hotkey, verdict in got.items()} == {"hits_the_bug": CHECK_FAULT, "good": ""}
        assert got["hits_the_bug"].reason.startswith("the checker fails on this file: IndexError in buggy, ")
    finally:
        grade_pool.close()
