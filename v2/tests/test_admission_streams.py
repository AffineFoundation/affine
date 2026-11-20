"""The live check of the waiting files with several read streams gives the verdicts of one stream.

``StoreAdmission.verdicts`` reads each waiting file from the store to hash it. ``streams`` is the count of files
that it reads at one time. These tests hold: the verdicts and the memory are the same for each count, a store call
that fails ends the call the same way, the files that wait in memory are bounded, and the count is a setting.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import threading
import time

import pytest

from sn120_v2 import cli, layout
from sn120_v2.admission import FAULT_HOLD_S, MAX_STREAMS, SERIAL_STREAMS, STREAMS_ENV, GraderTimeLimit, StoreAdmission, check_streams, final_answer
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.sim import SimWorld, sim_params

THINK = "step " * 40  # 40 reasoning tokens: above the least of 32
LIMIT = 50_000  # the byte cap of these tests
STREAMS = 8
KINDS = ("valid", "not json", "thief", "oversized", "slow", "absent")
HOTKEYS = [f"5Miner{n:02d}" for n in range(36)]


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


def grade(prompt: str, output: str, capped: bool) -> float | None:
    """The task "double N" has the answer 2N. An output with the word "slow" does not finish in the time limit."""
    if "slow" in output:
        raise GraderTimeLimit
    return float(not capped and final_answer(output) == str(2 * int(prompt.split()[1])))


def file(hotkey: str, word: str = "") -> bytes:
    """10 prompts with a right and a wrong rollout each. ``word`` goes into each wrong rollout."""
    rows = lambda n: [{"output": f"{THINK}\\boxed{{{2 * n}}}", "reward": 1}, {"output": f"{THINK}{word} \\boxed{{-1}}", "reward": 0}]
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": [{"prompt": f"double {n}", "rows": 0, "rollouts": rows(n)} for n in range(10)]}).replace('"rows": 0, ', "").encode()


def stored(kind: str, hotkey: str) -> bytes | None:
    return {
        "valid": file(hotkey), "not json": b"{", "thief": file("5Other"), "oversized": b" " * (LIMIT + 1), "slow": file(hotkey, "slow"), "absent": None,
    }[kind]


class Store(dict):
    """The store calls that admission makes, over a dict. It logs each call, counts the calls that run at one time and can fail a call."""

    def __init__(self, pause_s: float = 0.0) -> None:
        super().__init__()
        self.pause_s, self.log, self.fail = pause_s, [], {}
        self.threads: set[int] = set()
        self._lock, self._running, self.peak = threading.Lock(), 0, 0

    def _call(self, name: str, key: str):
        with self._lock:
            self.log.append((name, key))
            self.threads.add(threading.get_ident())
            self._running += 1
            self.peak = max(self.peak, self._running)
            count = sum(1 for entry in self.log if entry == (name, key))
        try:
            time.sleep(self.pause_s)
            error = self.fail.get((name, key, count)) or self.fail.get((name, key))
            if error is not None:
                raise error
            return self[key]
        finally:
            with self._lock:
                self._running -= 1

    put = dict.__setitem__

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def get(self, key: str) -> bytes:
        return self._call("get", key)

    def size(self, key: str) -> int:
        return len(self._call("size", key))

    def sha256(self, key: str) -> str:
        return sha256_hex(self._call("sha256", key))

    def calls(self) -> dict[str, list[str]]:
        """The calls of each key, in the order they were made."""
        found = collections.defaultdict(list)
        for name, key in self.log:
            found[key].append(name)
        return dict(found)


class Ticks:
    """A clock that goes on by one second at each read: a call that reads the clock more or less often shows in ``checked_at``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@dataclasses.dataclass
class Line:
    """One admission over its own store, with the same files as every other ``Line``."""

    streams: int
    build: type
    pause_s: float = 0.0

    def __post_init__(self) -> None:
        self.store, self.clock = Store(self.pause_s), Ticks()
        for n, hotkey in enumerate(HOTKEYS):
            raw = stored(KINDS[n % len(KINDS)], hotkey)
            if raw is not None:
                self.store.put(layout.miner_object(hotkey), raw)
        self.adm = StoreAdmission(self.store, self.build.params(max_batch_bytes=LIMIT), tokens=WordTokens(), grader=grade, clock=self.clock, streams=self.streams)

    def check(self, hotkeys=HOTKEYS) -> list[tuple[str, bytes]]:
        """The verdicts as the bytes of their records, in the order of the answer."""
        return [(hotkey, verdict.to_json()) for hotkey, verdict in self.adm.verdicts(hotkeys).items()]

    def memory(self) -> tuple:
        return dict(self.adm._seen), dict(self.adm._tries), dict(self.adm._held)

    def failed(self, hotkeys=HOTKEYS) -> str:
        with pytest.raises(InfraError) as caught:
            self.adm.verdicts(hotkeys)
        return str(caught.value)


def both(build, pause_s: float = 0.0) -> tuple[Line, Line]:
    return Line(SERIAL_STREAMS, build, pause_s), Line(STREAMS, build, pause_s)


def same(serial: Line, many: Line, got: tuple) -> None:
    assert got[0] == got[1], "the verdicts differ"
    assert serial.memory() == many.memory(), "the memory differs"
    assert serial.store.calls() == many.store.calls(), "the store calls of a file, or their order, differ"
    assert serial.clock.now == many.clock.now, "the clock was read more or less often"


def test_the_verdicts_with_many_streams_are_byte_for_byte_the_verdicts_of_one_stream(build):
    serial, many = lines = both(build)
    first = tuple(line.check() for line in lines)
    same(serial, many, first)
    by_hotkey = dict(first[1])
    assert list(by_hotkey) == [hotkey for n, hotkey in enumerate(HOTKEYS) if KINDS[n % len(KINDS)] != "absent"], "the order of the line; no file, no verdict"
    rules = {KINDS[n % len(KINDS)]: json.loads(by_hotkey[hotkey])["rule"] for n, hotkey in enumerate(HOTKEYS) if hotkey in by_hotkey}
    assert rules == {"valid": "", "not json": "json", "thief": "hotkey", "oversized": "size", "slow": "grader_time_limit"}
    assert many.store.calls()[layout.miner_object(HOTKEYS[0])] == ["size", "sha256", "size", "get"] and many.store.calls()[layout.miner_object(HOTKEYS[3])] == ["size", "size"]
    same(serial, many, tuple(line.check() for line in lines))  # nothing changed: the memory answers, and only the hash is read
    assert many.store.calls()[layout.miner_object(HOTKEYS[0])] == ["size", "sha256", "size", "get", "size", "sha256"]
    for line in lines:  # the files change: a new one, a replaced one, one that is gone; and the hold of the slow files ends
        line.store.put(layout.miner_object(HOTKEYS[5]), file(HOTKEYS[5]))
        line.store.put(layout.miner_object(HOTKEYS[0]), b"{}")
        line.store.put(layout.miner_object(HOTKEYS[1]), file(HOTKEYS[1], "slow"))
        line.store.delete(layout.miner_object(HOTKEYS[6]))
        line.clock.now += FAULT_HOLD_S
    third = tuple(line.check() for line in lines)
    same(serial, many, third)
    assert dict(third[1])[HOTKEYS[4]] != by_hotkey[HOTKEYS[4]], "the hold ended, so the slow file was checked again: a new time of the check"
    assert dict(third[1])[HOTKEYS[2]] == by_hotkey[HOTKEYS[2]] and HOTKEYS[6] not in dict(third[1]) and HOTKEYS[5] in dict(third[1])
    for line in lines:  # new parameters: every file is checked again
        line.adm.params = build.params(max_batch_bytes=LIMIT, max_rollouts=1)
    same(serial, many, tuple(line.check() for line in lines))
    repeated = [HOTKEYS[0], HOTKEYS[4], HOTKEYS[0], HOTKEYS[12], HOTKEYS[4], HOTKEYS[5]]  # a hotkey that is named two times
    for line in lines:
        line.store.put(layout.miner_object(HOTKEYS[0]), file(HOTKEYS[0]))
        line.store.put(layout.miner_object(HOTKEYS[4]), file(HOTKEYS[4], "slow") + b" ")
    got = tuple(line.check(repeated) for line in lines)
    assert got[0] == got[1] and serial.memory() == many.memory() and serial.clock.now == many.clock.now
    assert len(serial.store.threads) == 1 and len(many.store.threads) > 1, "one stream calls the store from the caller's thread only"


@pytest.mark.parametrize("call, at", [("size", 1), ("sha256", 1), ("size", 2), ("get", 1)])
def test_a_store_call_that_fails_ends_the_check_as_with_one_stream(build, call, at):
    """``at`` is the count of the call for that file: the second size call is the one before the download."""
    serial, many = lines = both(build)
    broken = layout.miner_object(HOTKEYS[18])  # a valid file in the middle of the line
    for line in lines:
        line.store.fail[(call, broken, at)] = InfraError(f"the store is down at {call}")
    texts = [line.failed() for line in lines]
    assert texts[0] == texts[1] == f"the store is down at {call}"
    assert serial.memory() == many.memory(), "the files before the failed one have their verdict in the memory, the others have none"
    assert set(serial.adm._seen) | set(serial.adm._held) == {hotkey for n, hotkey in enumerate(HOTKEYS[:18]) if KINDS[n % len(KINDS)] != "absent"}
    assert serial.clock.now == many.clock.now
    for line in lines:  # the store works again: the next call gives every verdict
        line.store.fail.clear()
    got = tuple(line.check() for line in lines)
    assert got[0] == got[1] and serial.memory() == many.memory() and HOTKEYS[18] in dict(got[1])


def test_a_file_that_goes_away_between_the_hash_and_the_download_has_no_verdict_as_with_one_stream(build):
    serial, many = lines = both(build)
    gone = layout.miner_object(HOTKEYS[18])
    for line in lines:
        line.store.fail[("get", gone)] = KeyError(gone)
    got = tuple(line.check() for line in lines)
    same(serial, many, got)
    assert HOTKEYS[18] not in dict(got[1]) and HOTKEYS[24] in dict(got[1]), "that file has no verdict; the files behind it have theirs"


def test_a_failed_download_is_no_error_when_the_check_does_not_need_it(build):
    """A hotkey that is named two times: the read ahead of its second place may download the file, and the check then has the verdict already."""
    many = Line(STREAMS, build)
    many.store.fail[("get", layout.miner_object(HOTKEYS[0]), 2)] = InfraError("the second download fails")
    got = many.check([HOTKEYS[0]] + HOTKEYS[6:12] + [HOTKEYS[0]])
    serial = Line(SERIAL_STREAMS, build)
    assert [hotkey for hotkey, _ in got] == [hotkey for hotkey, _ in serial.check([HOTKEYS[0]] + HOTKEYS[6:12] + [HOTKEYS[0]])]


def test_the_files_that_wait_in_memory_for_their_check_are_bounded_by_the_streams(build):
    many = Line(STREAMS, build, pause_s=0.002)
    lock, counts = threading.Lock(), {"waiting": 0, "peak": 0}
    raw, judged = many.adm._raw, many.adm._judged

    def downloaded(key):
        found = raw(key)
        with lock:
            counts["waiting"] += 1
            counts["peak"] = max(counts["peak"], counts["waiting"])
        return found

    def checked(hotkey, found):
        with lock:
            counts["waiting"] -= 1
        time.sleep(0.01)  # the check is slower than the store: the reads run ahead as far as they may
        return judged(hotkey, found)

    many.adm._raw, many.adm._judged = downloaded, checked
    assert len(many.check()) == 30
    assert 1 < counts["peak"] <= STREAMS, "at most one file for each stream waits for its check"
    assert 1 < many.store.peak <= STREAMS, "at most one store call for each stream runs at one time"
    serial = Line(SERIAL_STREAMS, build, pause_s=0.002)
    assert len(serial.check()) == 30 and serial.store.peak == 1


def test_many_streams_take_less_time_when_the_store_is_the_slow_part(build):
    serial, many = both(build, pause_s=0.01)
    spent = []
    for line in (serial, many):
        line.check()  # the first check downloads and judges; the second one only reads the hashes
        started = time.monotonic()
        line.check()
        spent.append(time.monotonic() - started)
    assert spent[1] < spent[0] / 2, f"one stream {spent[0]:.2f} s, {STREAMS} streams {spent[1]:.2f} s"


def test_the_count_of_streams_is_a_setting_and_one_stream_is_the_default():
    assert check_streams({}) == check_streams({STREAMS_ENV: " "}) == SERIAL_STREAMS == 1
    assert check_streams({STREAMS_ENV: "8"}) == 8 and check_streams({STREAMS_ENV: str(MAX_STREAMS)}) == MAX_STREAMS
    assert check_streams({STREAMS_ENV: " 16\n"}) == 16, "white space around the value is not part of it"
    for bad in ("0", str(MAX_STREAMS + 1), "-1", "many", "2.5", "8.0", "+8", "1_6", "8 8", "0x8"):
        with pytest.raises(ConfigError, match=STREAMS_ENV):
            check_streams({STREAMS_ENV: bad})


# ``str.isdigit()`` is true for each of these. ``int()`` cannot read the first two (a ValueError, which is a
# traceback and the exit code of a crash, so the run loop starts ``run`` again and again). It reads the next two
# as 8, and it refuses the last one for its length.
NOT_PLAIN_DIGITS = {
    "a superscript two": "\u00b2",
    "a circled one": "\u2460",
    "a full-width eight": "\uff18",
    "an Arabic-Indic eight": "\u0668",
    "a zero in front": "08",
    "more digits than int() reads": "1" * 5000,
}


@pytest.mark.parametrize("text", NOT_PLAIN_DIGITS.values(), ids=NOT_PLAIN_DIGITS.keys())
def test_a_count_of_streams_that_is_not_1_to_32_in_plain_digits_is_refused_as_a_setting_and_never_as_a_crash(text):
    assert text.isdigit(), "the test that let these values pass"
    with pytest.raises(ConfigError, match=f"^{STREAMS_ENV} must be a whole number from 1 to 32, written in the digits 0 to 9 with no zero in front") as refused:
        check_streams({STREAMS_ENV: text})
    assert text not in str(refused.value).replace("not 08", ""), "the refusal does not print the value"


def test_run_prints_each_setting_of_its_stack_at_the_start(tmp_path, monkeypatch, capsys):
    SimWorld(tmp_path, sim_params())  # a state with a genesis and no waiting seat
    load_stack = cli.load_stack
    lines = ("the check of the waiting files reads one file after the other (SN120_CHECK_STREAMS is NOT SET: this is the default)", "a second setting (X=1)")
    monkeypatch.setattr(cli, "load_stack", lambda spec, context: dataclasses.replace(load_stack(spec, context), settings=lines))
    assert cli.main(["run", "--state-dir", str(tmp_path), "--stack", "sn120_v2.sim:stack", "--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]) == 0
    shown = capsys.readouterr().err.splitlines()
    assert shown[:2] == [f"setting: {line}" for line in lines], "before the first pass of the loop"
    assert shown[2].startswith("RLIMIT_NOFILE: soft="), "the process limit is logged at start"
