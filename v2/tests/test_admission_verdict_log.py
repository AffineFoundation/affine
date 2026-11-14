"""The log of the file checks: one line for each seat whose file or verdict changed, and one summary line a check.

The verdict of a file, with its rule, its reason and the count of a fault, lives only in the memory of ``run``.
These lines put it in ``run.log``. They change no verdict.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from sn120_v2 import layout
from sn120_v2.admission import FAULT_HOLD_S, SHA_SHOWN, TOKENIZER_FAULT, StoreAdmission, verdict_log, verdict_note
from sn120_v2.canonical import sha256_hex
from sn120_v2.cli import main
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.sim import STORE_DIR, SimAdmission
from sn120_v2.state import load
from sn120_v2.store import LocalStore
from sn120_v2.types import Verdict

NOW = 1_794_626_162  # 2026-11-14 03:16:02 UTC
SEATS = [(809, "5A"), (877, "5B"), (948, "5C")]
THINK = "step " * 40


def valid(hotkey: str, label: str = "one") -> Verdict:
    return Verdict(hotkey, True, batch_sha256=sha256_hex(label.encode()))


def bad(hotkey: str, rule: str, reason: str, label: str = "one") -> Verdict:
    return Verdict(hotkey, False, reason, batch_sha256=sha256_hex(label.encode()), rule=rule)


def start(label: str = "one") -> str:
    return sha256_hex(label.encode())[:SHA_SHOWN]


def test_first_check_names_each_seat_that_has_a_file_and_counts_the_rest():
    verdicts = {"5A": valid("5A"), "5B": bad("5B", "train_cost", "the batch costs 9464226 token-steps, more than 8000000", "two")}
    lines, notes = verdict_log(SEATS, verdicts, {}, NOW)
    assert lines == [
        f"files: UID 809 {start()} valid; before: not checked by this process",
        f"files: UID 877 {start('two')} not valid, rule train_cost: the batch costs 9464226 token-steps, more than 8000000; before: not checked by this process",
        "files: check at 2026-11-14 03:16:02 UTC: 3 waiting seats, 1 valid, 1 not valid (train_cost 1), 1 with no file, 2 changed",
    ]
    assert notes == {"5A": f"{start()} valid", "5B": lines[1].split("UID 877 ")[1].split("; before")[0], "5C": "no file"}


def test_check_with_no_change_gives_the_summary_line_only():
    verdicts = {"5A": valid("5A"), "5B": bad("5B", "prompt", "groups[3].prompt: not in the task pool")}
    _, notes = verdict_log(SEATS, verdicts, {}, NOW)
    later = {hotkey: Verdict(**(verdict.to_dict() | {"checked_at": NOW + 30})) for hotkey, verdict in verdicts.items()}
    lines, same = verdict_log(SEATS, later, notes, NOW + 30)
    assert lines == ["files: check at 2026-11-14 03:16:32 UTC: 3 waiting seats, 1 valid, 1 not valid (prompt 1), 1 with no file, 0 changed"]
    assert same == notes


def test_each_kind_of_change_gives_one_line_with_what_the_log_said_before():
    _, notes = verdict_log(SEATS, {"5A": valid("5A"), "5B": valid("5B"), "5C": valid("5C")}, {}, NOW)
    after = {"5A": valid("5A", "new bytes"), "5B": bad("5B", "grader_fault", "groups[0].rollouts[0]: the official grader fails on this row (check 1 of 3)")}
    lines, notes = verdict_log(SEATS, after, notes, NOW)
    assert lines[:-1] == [
        f"files: UID 809 {start('new bytes')} valid; before: {start()} valid",
        f"files: UID 877 {start()} not valid, rule grader_fault: groups[0].rollouts[0]: the official grader fails on this row (check 1 of 3); before: {start()} valid",
        f"files: UID 948 no file; before: {start()} valid",
    ]
    assert lines[-1].endswith("3 waiting seats, 1 valid, 1 not valid (grader_fault 1), 1 with no file, 3 changed")
    counted = after | {"5B": bad("5B", "grader_fault", "groups[0].rollouts[0]: the official grader fails on this row (check 2 of 3)")}
    lines, _ = verdict_log(SEATS, counted, notes, NOW)
    assert len(lines) == 2 and "(check 2 of 3); before: " in lines[0] and "(check 1 of 3)" in lines[0], "a new count of the same fault is a change"


def test_seat_that_was_in_a_round_gets_no_line_when_it_comes_back_with_the_same_file():
    _, notes = verdict_log(SEATS, {"5A": valid("5A")}, {}, NOW)
    in_round, notes = verdict_log(SEATS[1:], {}, notes, NOW)
    back, _ = verdict_log(SEATS, {"5A": valid("5A")}, notes, NOW)
    assert len(in_round) == 1 and "2 waiting seats, 0 valid, 0 not valid, 2 with no file, 0 changed" in in_round[0]
    assert len(back) == 1 and back[0].endswith("0 changed")


def test_note_is_one_line_and_shows_a_file_that_has_no_hash():
    assert verdict_note(None) == "no file"
    assert verdict_note(Verdict("5A", False, "file must be\n1..64 bytes", rule="size")) == "not hashed not valid, rule size: file must be 1..64 bytes"
    lines, _ = verdict_log(SEATS[:1], {"5A": bad("5A", "json", "line 1\r\nline 2")}, {}, NOW)
    assert all("\n" not in line and "\r" not in line for line in lines)


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        if "poison" in text:
            raise ValueError("secret text of the batch")
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


def file(hotkey: str, word: str = "") -> bytes:
    rows = lambda n: [{"output": f"{THINK}\\boxed{{{2 * n}}}", "reward": 1}, {"output": f"{THINK}{word} \\boxed{{-1}}", "reward": 0}]
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": [{"prompt": f"double {n}", "rollouts": rows(n)} for n in range(10)]}).encode()


def test_real_checker_logs_the_count_of_a_fault_once_a_check_and_nothing_while_the_verdict_is_held(build):
    """The store checker with a file that breaks the tokenizer: the verdicts that ``run`` gets, as the log shows them."""
    clock = [7.0]
    store = Store({layout.miner_object("5A"): file("5A"), layout.miner_object("5B"): file("5B", "poison")})
    adm = StoreAdmission(store, build.params(), tokens=WordTokens(), grader=None, clock=lambda: clock[0])
    notes: dict[str, str] = {}
    seen = []
    for step in (0, 30, FAULT_HOLD_S, 30, FAULT_HOLD_S, FAULT_HOLD_S):
        clock[0] += step
        verdicts = adm.verdicts(["5A", "5B", "5C"])
        lines, notes = verdict_log(SEATS, verdicts, notes, int(clock[0]))
        seen.append(lines)
        assert verdicts["5A"].valid and verdicts["5B"].rule == TOKENIZER_FAULT, "the log changes no verdict"
    assert [len(lines) for lines in seen] == [3, 1, 2, 1, 2, 1], "a held verdict and a kept verdict give no line"
    assert [re.search(r"\((check.*?)\); before", lines[-2]).group(1) for lines in seen if len(lines) > 1] == [
        "check 1 of 3",
        "check 2 of 3",
        "checked 3 times: this verdict stays until a new file is uploaded",
    ]
    assert all("secret" not in line and "poison" not in line for lines in seen for line in lines), "a line never holds batch text"
    assert all(lines[-1].endswith(f"3 waiting seats, 1 valid, 1 not valid ({TOKENIZER_FAULT} 1), 1 with no file, {len(lines) - 1} changed") for lines in seen)


def test_run_writes_the_lines_to_its_log_and_takes_the_same_seats(tmp_path, capsys):
    """``run`` on a state with 57 waiting seats: two files that are not valid, then one valid file."""
    directory = tmp_path / "state"
    shutil.copytree(Path(__file__).parent / "data" / "state_515d7e9f" / "idle", directory)
    files = SimAdmission(LocalStore(directory / STORE_DIR))
    files.upload("hk4", valid=False)
    files.upload("hk9", valid=False)
    run = ["run", "--state-dir", str(directory), "--stack", "sn120_v2.sim:stack", "--rounds", "1", "--idle-sleep", "0"]
    rounds = len(load(directory / JOURNAL_FILE).rounds)
    assert main([*run, "--max-idle", "3"]) == 0
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("files: ")]
    plain = [re.sub(r" [0-9a-f]{12} ", " ", line) for line in lines]
    assert plain[:2] == [f"files: UID {uid} not valid, rule sim_invalid: ; before: not checked by this process" for uid in (4, 9)]
    summaries = [re.sub(r"check at .* UTC", "check", line) for line in lines[2:]]
    assert summaries == [f"files: check: 57 waiting seats, 0 valid, 2 not valid (sim_invalid 2), 55 with no file, {changed} changed" for changed in (2, 0, 0)]
    assert len(load(directory / JOURNAL_FILE).rounds) == rounds, "no valid file: no round"

    files.upload("hk9", 0.3)
    assert main([*run, "--max-idle", "3"]) == 0
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("files: ")]
    assert re.sub(r" [0-9a-f]{12} ", " ", lines[1]) == "files: UID 9 valid; before: not checked by this process", "a new process has no memory of the last check"
    assert lines[2].endswith("57 waiting seats, 1 valid, 1 not valid (sim_invalid 1), 55 with no file, 2 changed")
    state = load(directory / JOURNAL_FILE)
    assert len(state.rounds) == rounds + 1 and [entry.uid for entry in state.rounds[-1].decision.entries] == [9]
