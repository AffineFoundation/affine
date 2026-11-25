"""Parallel waiting-file checks preserve serial admission semantics."""

from __future__ import annotations

import dataclasses
import json
import os
import time

import pytest

from sn120_v2 import layout
from sn120_v2.admission import (
    CHECK_FAULT,
    FAULT_HOLD_S,
    MAX_WORKERS,
    WORKERS_ENV,
    GraderTimeLimit,
    StoreAdmission,
    check_workers,
    final_answer,
    pool_grader,
)
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import ConfigError

THINK = "step " * 40
HOTKEYS = tuple(f"5Miner{n}" for n in range(6))


class Tokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


class Task:
    def __init__(self, n: int) -> None:
        self.n, self.answer = n, str(2 * n)

    def batch_prompt(self) -> str:
        return f"double {self.n}"


class Pool:
    task_ids = tuple(f"t{n}" for n in range(10))

    def task(self, task_id: str) -> Task:
        return Task(int(task_id[1:]))

    def task_for_prompt(self, prompt: str) -> Task | None:
        return Task(int(prompt.split()[1])) if prompt.startswith("double ") else None


class Grades:
    """A top-level object that spawn can pickle; no parent state is shared."""

    def grade_many(self, jobs):
        grades = []
        for task, output, capped in jobs:
            if "slow" in output:
                time.sleep(0.02)
            grades.append(type("Grade", (), {})())
            grades[-1].machine_cause = "time-limit" if "slow" in output else None
            grades[-1].reward = float(not capped and final_answer(output) == task.answer)
        return grades


class CrashGrades(Grades):
    def grade_many(self, jobs):
        if any("crash" in output for _, output, _ in jobs):
            import sys
            print(f"CRASH pid={os.getpid()} ppid={os.getppid()} sid={os.getsid(0)} pgid={os.getpgrp()}", file=sys.stderr, flush=True)
            os.kill(os.getpid(), 9)
        return super().grade_many(jobs)


def file(hotkey: str, *, reward: bool = True, slow: bool = False, crash: bool = False) -> bytes:
    def rows(n: int):
        return [
            {"output": f"{THINK}\\boxed{{{2 * n}}}", "reward": 1},
            {"output": f"{THINK}{'slow ' if slow else ''}{'crash ' if crash else ''}\\boxed{{-1}}", "reward": 0 if reward else 1},
        ]
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey,
                       "groups": [{"prompt": f"double {n}", "rollouts": rows(n)} for n in range(10)]}).encode()


class Store(dict):
    put = dict.__setitem__

    def size(self, key):
        return len(self[key])

    def sha256(self, key):
        return sha256_hex(self[key])

    def delete(self, key):
        self.pop(key, None)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 1
        return self.now


def make(build, workers: int, crash: bool = False):
    raws = {
        HOTKEYS[0]: file(HOTKEYS[0]),
        HOTKEYS[1]: b"{",
        HOTKEYS[2]: file(HOTKEYS[2], reward=False),
        HOTKEYS[3]: b"x" * 100_000,
        HOTKEYS[4]: file(HOTKEYS[4], slow=True),
        HOTKEYS[5]: file(HOTKEYS[5], crash=True),
    }
    store = Store({layout.miner_object(key): raw for key, raw in raws.items()})
    grader = pool_grader(Pool(), CrashGrades() if crash else Grades(), chunk=8)
    return StoreAdmission(store, build.params(max_batch_bytes=50_000), tokens=Tokens(), grader=grader,
                          clock=Clock(), workers=workers)


def memory(adm):
    return dict(adm._seen), dict(adm._tries), dict(adm._held)


def test_mixed_verdicts_and_transient_holds_match_serial(build):
    serial, parallel = make(build, 1), make(build, 3)
    assert parallel.verdicts(HOTKEYS[:-1]) == serial.verdicts(HOTKEYS[:-1])
    assert memory(parallel) == memory(serial)
    assert serial.clock.now == parallel.clock.now
    for adm in (serial, parallel):
        adm.clock.now += FAULT_HOLD_S
    assert parallel.verdicts(HOTKEYS[:-1]) == serial.verdicts(HOTKEYS[:-1])
    assert memory(parallel) == memory(serial)
    for adm in (serial, parallel):
        adm.clock.now += FAULT_HOLD_S
    assert parallel.verdicts(HOTKEYS[:-1]) == serial.verdicts(HOTKEYS[:-1])
    assert memory(parallel) == memory(serial)
    assert "this verdict stays until a new file is uploaded" in parallel._seen[HOTKEYS[4]][1].reason


def test_worker_crash_is_unproven_and_other_files_continue(build, capsys):
    adm = make(build, 2, crash=True)
    found = adm.verdicts(HOTKEYS)
    assert found[HOTKEYS[5]].rule == CHECK_FAULT and "worker died" in found[HOTKEYS[5]].reason
    assert adm._tries[HOTKEYS[5]][2] == 0 and adm._held[HOTKEYS[5]][1].valid is False
    assert found[HOTKEYS[0]].valid and found[HOTKEYS[4]].rule == "grader_time_limit"
    assert "6 files checked, 2 workers" in capsys.readouterr().err
    before = memory(adm)
    assert adm.verdicts(HOTKEYS) == found and memory(adm) == before, "the ordinary hold prevents another crash/check"


def test_crash_does_not_increment_a_previous_proven_fault_count(build):
    adm = make(build, 2, crash=True)
    adm.verdicts(HOTKEYS)
    count = adm._tries[HOTKEYS[4]][2]
    assert count == 1
    raw = file(HOTKEYS[4], crash=True)
    adm.store[layout.miner_object(HOTKEYS[4])] = raw
    adm.clock.now += FAULT_HOLD_S
    adm.verdicts(HOTKEYS)
    assert adm._tries[HOTKEYS[4]][2] == 0, "different bytes start a new fault sequence"
    # Crash the same bytes after a proven fault; the following machine result
    # must leave its count exactly as it was, not increment or erase it.
    adm.store[layout.miner_object(HOTKEYS[4])] = file(HOTKEYS[4], slow=True)
    adm.clock.now += FAULT_HOLD_S
    adm.verdicts(HOTKEYS)
    assert adm._tries[HOTKEYS[4]][2] == 1
    adm.store[layout.miner_object(HOTKEYS[4])] = raw
    adm.clock.now += FAULT_HOLD_S
    adm.verdicts(HOTKEYS)
    assert adm._tries[HOTKEYS[4]][2] == 0


def test_worker_knob_is_strict_and_defaults_to_serial():
    assert check_workers({}) == check_workers({WORKERS_ENV: " "}) == 1
    assert check_workers({WORKERS_ENV: "64"}) == MAX_WORKERS == 64
    assert check_workers({WORKERS_ENV: " 8\n"}) == 8
    for bad in ("0", "65", "08", "+8", "8.0", "²", "８", "1" * 5000):
        with pytest.raises(ConfigError, match=WORKERS_ENV):
            check_workers({WORKERS_ENV: bad})
