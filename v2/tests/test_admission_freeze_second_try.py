"""The freeze after a valid verdict: each skip says why in the log, and a fault of the moment gets a second try.

The file check said ``valid``, and the freeze gives the seat no batch. Four cases:

  a  the file is gone at the freeze
  b  the file is now above the byte cap
  c  the miner put other bytes there after the check, and they break a rule
  d  the same bytes fail with a rule that depends on this machine at this moment (``TRANSIENT``)

Cases a to c are a skip at once. Case d gets one second try on the same bytes, inside ``FREEZE_RETRY_S`` for each
open. Each second try has its own line in the log, also when it passes.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import CHECK_FAULT, FAULT, FREEZE_RETRY_S, OTHER_BYTES, SAME_BYTES, SHA_SHOWN, TIME_LIMIT, TOKENIZER_FAULT, TRANSIENT, StoreAdmission, final_answer, pool_grader
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import InfraError
from sn120_v2.queue import select_round

HOTKEY = "5MinerA"
ROUND = "r000035"
THINK = "step " * 40
GROUPS = 10  # the groups of a miner file of this test; the known-good file of the checker has one
ROWS = 2 * GROUPS
DEAD = "dead"  # not a rule: the tokenizer fails on each text, also on its known-good case


class Moment:
    """This machine at this moment: the faults that the next checks of a whole miner file hit, one for each check, in order.

    A fault is a rule code in ``TRANSIENT``. The known-good cases of the tokenizer, the grader and the checker
    never hit one: the fault is of the moment, and the part that failed works again right after.
    """

    def __init__(self, monkeypatch, *faults: str) -> None:
        self.faults, self.checks = list(faults), 0
        real = admission._cap

        def cap(groups, *limits):
            if len(groups) == GROUPS:
                self.checks += 1
                if self.hit(CHECK_FAULT):
                    raise KeyError("secret text of the batch")
            return real(groups, *limits)

        monkeypatch.setattr(admission, "_cap", cap)

    def hit(self, code: str) -> bool:
        if self.faults[:1] != [code]:
            return False
        self.faults.pop(0)
        return True

    # the tokenizer
    def prompt_tokens(self, prompt: str) -> int:
        if self.faults[:1] == [DEAD]:
            raise ValueError("the tokenizer is gone")
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        if self.faults[:1] == [DEAD] or ("step" in text and self.hit(TOKENIZER_FAULT)):
            raise ValueError("secret text of the batch")
        return len(text.split())

    # the grade pool: the rows of a whole miner file come in one call
    def grade_many(self, jobs) -> list[SimpleNamespace]:
        grades = [SimpleNamespace(reward=float(not capped and final_answer(output) == task.answer), machine_cause=None) for task, output, capped in jobs]
        if len(jobs) == ROWS:
            grades[0].machine_cause = "time-limit" if self.hit(TIME_LIMIT) else "worker-died" if self.hit(FAULT) else None
        return grades


class Pool:
    """The task "double N" has the answer 2N."""

    task_ids = tuple(f"t{n}" for n in range(50))

    def task(self, task_id: str) -> SimpleNamespace:
        n = int(task_id[1:])
        return SimpleNamespace(n=n, answer=str(2 * n), batch_prompt=lambda: f"double {n}")

    def task_for_prompt(self, prompt: str) -> SimpleNamespace | None:
        return self.task(f"t{prompt.split()[1]}") if prompt.startswith("double ") else None


class Store(dict):
    """A stand-in store that counts its downloads."""

    reads = 0
    put = dict.__setitem__

    def get(self, key: str) -> bytes:
        self.reads += 1
        return self[key]

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


class Timer:
    """A timer that goes ``step`` seconds forward at each read. So each second try takes ``step`` seconds."""

    def __init__(self, step: float = 1.5) -> None:
        self.now, self.step = 100.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def file(hotkey: str = HOTKEY, wrong_reward: bool = False) -> bytes:
    """10 prompts "double N", each with a right and a wrong rollout. ``wrong_reward``: one reward is not the official grade (rule ``grader``)."""
    rows = lambda n: [{"output": f"{THINK}\\boxed{{{2 * n}}}", "reward": 1}, {"output": f"{THINK}\\boxed{{-1}}", "reward": int(wrong_reward and n == 3)}]
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": [{"prompt": f"double {n}", "rollouts": rows(n)} for n in range(GROUPS)]}).encode()


def start_of(raw: bytes) -> str:
    return sha256_hex(raw)[:SHA_SHOWN]


def checked(build, moment: Moment, files: dict[str, bytes], **inputs) -> StoreAdmission:
    """The store checker over a stand-in store, after a file check that said ``valid`` for each file."""
    store = Store({layout.miner_object(hotkey): raw for hotkey, raw in files.items()})
    inputs = {"params": build.params(), "timer": Timer()} | inputs
    adm = StoreAdmission(store, inputs.pop("params"), tokens=moment, grader=pool_grader(Pool(), moment), **inputs)
    found = adm.verdicts(list(files))
    assert all(found[hotkey].valid for hotkey in files) and moment.checks == len(files) == store.reads
    return adm


def freeze_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("freeze: ")]


def no_seat(uid: int, good: bytes) -> str:
    """The start of the line of a seat that gets no batch, after a check that said ``valid`` for ``good``."""
    return f"freeze: UID {uid} gets no seat in round {ROUND}. The check: {start_of(good)} valid, memory tag {start_of(good)}. The freeze: "


# --- cases a to c: a skip at once, with its reason in the log ---


def test_case_a_file_that_is_gone_at_the_freeze_is_a_skip_at_once(build, monkeypatch, capsys):
    moment, good = Moment(monkeypatch), file()
    adm = checked(build, moment, {HOTKEY: good})
    del adm.store[layout.miner_object(HOTKEY)]
    assert adm.freeze(build.reg(112, hotkey=HOTKEY), ROUND) is None
    assert freeze_lines(capsys) == [no_seat(112, good) + "no file"], "no bytes at the freeze: the line compares nothing"
    assert moment.checks == 1 and list(adm.store) == []


def test_case_b_file_above_the_byte_cap_is_a_skip_at_once_and_is_never_downloaded(build, monkeypatch, capsys):
    moment, good = Moment(monkeypatch), file()
    params = build.params(max_batch_bytes=len(good) + 8)
    adm = checked(build, moment, {HOTKEY: good}, params=params)
    adm.store[layout.miner_object(HOTKEY)] = good + b" " * 9
    assert adm.freeze(build.reg(151, hotkey=HOTKEY), ROUND) is None
    assert freeze_lines(capsys) == [no_seat(151, good) + f"not hashed not valid, rule size: file must be 1..{len(good) + 8} bytes"]
    assert moment.checks == 1 and adm.store.reads == 1, "an oversized object is not read, so the line compares no bytes"


def test_case_c_other_bytes_that_break_a_rule_of_the_miner_are_a_skip_at_once(build, monkeypatch, capsys):
    moment, good, other = Moment(monkeypatch), file(), file(wrong_reward=True)
    adm = checked(build, moment, {HOTKEY: good})
    adm.store[layout.miner_object(HOTKEY)] = other  # the miner wrote after the check
    assert adm.freeze(build.reg(12, hotkey=HOTKEY), ROUND) is None
    assert freeze_lines(capsys) == [
        no_seat(12, good) + f"{start_of(other)} not valid, rule grader: groups[3].rollouts[1].reward is not the official grade. The freeze read {OTHER_BYTES}"
    ]
    assert moment.checks == 2, "a rule break of the miner gets no second try"
    assert layout.frozen_batch(ROUND, HOTKEY) not in adm.store


# --- case d: the same bytes, a fault of the moment ---


@pytest.mark.parametrize("code", TRANSIENT)
def test_case_d_second_try_that_passes_gives_the_seat_and_leaves_one_line(code, build, monkeypatch, capsys):
    moment, good = Moment(monkeypatch), file()
    adm = checked(build, moment, {HOTKEY: good})
    moment.faults.append(code)  # the machine fails one time, at the freeze
    frozen = adm.freeze(build.reg(86, hotkey=HOTKEY), ROUND)
    assert frozen is not None and frozen.batch_sha256 == sha256_hex(good) and adm.store[frozen.uri] == good
    (line,) = freeze_lines(capsys)
    assert re.fullmatch(
        rf"freeze: UID 86, round {ROUND}: a second try on the same bytes\. The first try: {start_of(good)} not valid, rule {code}: .+\. "
        rf"The second try: {start_of(good)} valid, after 1\.5 s\. The second tries of this open have used 1\.5 of {FREEZE_RETRY_S} s",
        line,
    ), line
    assert "secret" not in line and "step" not in line, "the line never holds batch text"
    assert moment.checks == 3 and adm.store.reads == 2, "one check by the file check and two by the freeze, on one download of the freeze"


@pytest.mark.parametrize("code", TRANSIENT)
def test_case_d_second_try_that_fails_is_a_skip_with_two_lines(code, build, monkeypatch, capsys):
    moment, good = Moment(monkeypatch), file()
    adm = checked(build, moment, {HOTKEY: good})
    moment.faults += [code, code]  # the machine fails at both tries of the freeze
    assert adm.freeze(build.reg(86, hotkey=HOTKEY), ROUND) is None
    tried, skipped = freeze_lines(capsys)
    failed = rf"{start_of(good)} not valid, rule {code}: (.+)"
    first = re.fullmatch(
        rf"freeze: UID 86, round {ROUND}: a second try on the same bytes\. The first try: {failed}\. The second try: {failed}, after 1\.5 s\. "
        rf"The second tries of this open have used 1\.5 of {FREEZE_RETRY_S} s",
        tried,
    )
    assert first and first.group(1) == first.group(2), tried
    assert skipped == no_seat(86, good) + f"{start_of(good)} not valid, rule {code}: {first.group(2)}. The freeze read {SAME_BYTES}"
    assert "secret" not in tried + skipped and "step" not in tried + skipped, "the lines never hold batch text"
    assert moment.checks == 3 and moment.faults == [] and adm.store.reads == 2, "two tries and no third"
    assert layout.frozen_batch(ROUND, HOTKEY) not in adm.store


def test_result_of_the_second_try_counts_also_when_it_is_another_fault(build, monkeypatch, capsys):
    moment, good = Moment(monkeypatch), file()
    adm = checked(build, moment, {HOTKEY: good})
    moment.faults += [TIME_LIMIT, TOKENIZER_FAULT]
    assert adm.freeze(build.reg(86, hotkey=HOTKEY), ROUND) is None
    tried, skipped = freeze_lines(capsys)
    assert f"The first try: {start_of(good)} not valid, rule {TIME_LIMIT}: " in tried and f"The second try: {start_of(good)} not valid, rule {TOKENIZER_FAULT}: " in tried
    assert f"The freeze: {start_of(good)} not valid, rule {TOKENIZER_FAULT}: " in skipped and skipped.endswith(f"The freeze read {SAME_BYTES}")
    assert moment.checks == 3, "the second try gets no third try"


def test_failure_of_ours_at_the_second_try_stops_the_open(build, monkeypatch, capsys):
    moment, good = Moment(monkeypatch), file()
    adm = checked(build, moment, {HOTKEY: good})
    moment.faults += [TOKENIZER_FAULT, DEAD]  # one fault of the moment, then the tokenizer is gone
    with pytest.raises(InfraError, match="known-good"):
        adm.freeze(build.reg(86, hotkey=HOTKEY), ROUND)
    assert freeze_lines(capsys) == [] and layout.frozen_batch(ROUND, HOTKEY) not in adm.store


# --- the limit of the second tries of one open ---


def test_second_tries_of_one_open_stop_at_the_limit_and_a_started_try_is_not_cut(build, monkeypatch, capsys):
    moment = Moment(monkeypatch)
    files = {hotkey: file(hotkey) for hotkey in ("5A", "5B", "5C")}
    adm = checked(build, moment, files, timer=Timer(step=70.0))
    assert 70.0 < FREEZE_RETRY_S < 140.0
    moment.faults += [TIME_LIMIT] * 5  # 5A and 5B fail at both tries; 5C fails at its only try
    assert [adm.freeze(build.reg(uid, hotkey=hotkey), ROUND) for uid, hotkey in ((1, "5A"), (2, "5B"), (3, "5C"))] == [None, None, None]
    lines = freeze_lines(capsys)
    assert [line.split(": ")[1] for line in lines] == ["UID 1, round r000035", "UID 1 gets no seat in round r000035. The check", "UID 2, round r000035", "UID 2 gets no seat in round r000035. The check", "UID 3, round r000035", "UID 3 gets no seat in round r000035. The check"]
    assert lines[0].endswith(f"after 70.0 s. The second tries of this open have used 70.0 of {FREEZE_RETRY_S} s")
    assert lines[2].endswith(f"after 70.0 s. The second tries of this open have used 140.0 of {FREEZE_RETRY_S} s"), "it started below the limit and was not cut"
    assert lines[4] == (
        f"freeze: UID 3, round {ROUND}: no second try, the second tries of this open have used 140.0 of {FREEZE_RETRY_S} s. "
        f"The first try: {start_of(files['5C'])} not valid, rule {TIME_LIMIT}: groups[0].rollouts[0]: the official grader did not finish this row inside its time limit"
    )
    assert lines[5].endswith(f"The freeze read {SAME_BYTES}") and moment.faults == [] and moment.checks == 3 + 5

    assert all(verdict.valid for verdict in adm.verdicts(list(files)).values()) and moment.checks == 8, "the next try to open a round: the file check keeps its verdicts"
    moment.faults.append(TIME_LIMIT)
    assert adm.freeze(build.reg(3, hotkey="5C"), ROUND) is not None, "the limit starts again with the file check of each open"
    (line,) = freeze_lines(capsys)
    assert line.endswith(f"valid, after 70.0 s. The second tries of this open have used 70.0 of {FREEZE_RETRY_S} s")


# --- the walk of the queue ---


def test_walk_skips_the_seat_that_fails_two_times_and_takes_the_seat_that_the_second_try_mends(build, monkeypatch, capsys):
    """Four valid seats one after the other, and the machine fails at the first two."""
    moment = Moment(monkeypatch)
    hotkeys = {12: "5Twelve", 86: "5EightySix", 923: "5Nine", 161: "5One"}
    adm = checked(build, moment, {hotkey: file(hotkey) for hotkey in hotkeys.values()})
    queue = build.queue(*(build.seat(uid, block=block, hotkey=hotkey) for block, (uid, hotkey) in enumerate(hotkeys.items())))
    moment.faults += [TIME_LIMIT, TIME_LIMIT, FAULT]  # UID 12: a fault at both tries; UID 86: one fault
    picked = select_round(queue, adm.verdicts(list(hotkeys.values())), lambda reg: adm.freeze(reg, ROUND), cap=4)
    assert [entry.reg.uid for entry in picked.entries] == [86, 923, 161] and [key[0] for key in picked.skipped] == [12]
    assert moment.faults == [] and moment.checks == 4 + 6, "a seat with no fault gets one check"
    lines = freeze_lines(capsys)
    assert [line.split(": ")[1].split(",")[0].split(" gets")[0] for line in lines] == ["UID 12", "UID 12", "UID 86"]
    assert lines[1].endswith(f"The freeze read {SAME_BYTES}")
    assert f"The first try: {start_of(file('5EightySix'))} not valid, rule {FAULT}: " in lines[2] and f"The second try: {start_of(file('5EightySix'))} valid, after " in lines[2]
