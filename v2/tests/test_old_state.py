"""A state made by an older commit continues at this commit.

The smoke test keeps one bootstrap while fixes land: after a fix the operator
checks out the new commit and runs ``set-params`` and ``run`` on the old state.
``tests/data/state_515d7e9f`` holds two state directories that commit 515d7e9f
wrote through the command line with the CPU stack: ``idle`` (a bootstrap and
one closed round) and ``open_round`` (the same, then a round that was opened
and never run, as when a box dies after the lease).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from sn120_v2 import sim
from sn120_v2.cli import main
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.params import Params
from sn120_v2.sim import SIM_IMAGE, STORE_DIR, SimAdmission
from sn120_v2.state import load
from sn120_v2.store import LocalStore

STATES = Path(__file__).parent / "data" / "state_515d7e9f"
RUN = ["--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]


def old_state(tmp_path, name: str) -> tuple[Path, list[str], SimAdmission]:
    directory = tmp_path / name
    shutil.copytree(STATES / name, directory)
    return directory, ["--state-dir", str(directory), "--stack", "sn120_v2.sim:stack"], SimAdmission(LocalStore(directory / STORE_DIR))


def test_an_idle_state_of_an_older_commit_runs_a_round_takes_new_parameters_and_runs_again(tmp_path, capsys):
    directory, stack, admission = old_state(tmp_path, "idle")
    before = load(directory / JOURNAL_FILE)
    assert before.round is None and len(before.rounds) == 1 and [king.uid for king in before.kings] == [9, 1]
    assert before.params.batch_cap == 8, "the parameters that the old commit pinned; a field it did not have takes its default"
    assert main(["status", *stack]) == 0
    admission.upload("hk4", 0.3)
    admission.upload("hk5", 0.0)
    assert main(["run", *stack, *RUN]) == 0, "a round under the old pinned parameters"
    state = load(directory / JOURNAL_FILE)
    assert state.rounds[-1].outcome == "applied" and [king.uid for king in state.kings] == [9, 1, 4] and state.head.folded
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"batch_cap": 1, "n_tasks": 32, "min_scored_tasks": 8, "recipe": {"image_digests": [SIM_IMAGE]}}))
    assert main(["set-params", *stack, "--params", str(params)]) == 0
    admission.upload("hk6", 0.0)
    admission.upload("hk7", 0.0)
    assert main(["run", *stack, *RUN]) == 0
    record = load(directory / JOURNAL_FILE).rounds[-1]
    assert (record.round_id, record.n_tasks, len(record.decision.entries)) == ("r000003", 32, 1), "the new parameters are in force"


def test_a_round_that_an_older_commit_left_open_is_run_and_closed(tmp_path, monkeypatch):
    directory, stack, _ = old_state(tmp_path, "open_round")
    before = load(directory / JOURNAL_FILE)
    assert before.round is not None and before.round.decision is None, "the old commit issued the ticket and stopped"
    monkeypatch.setattr(time, "time", lambda: before.round.opened_at + 5.0)  # the lease of the stored ticket still runs
    assert main(["run", *stack, *RUN]) == 0
    state = load(directory / JOURNAL_FILE)
    record = state.rounds[-1]
    assert state.round is None and (record.round_id, record.outcome) == ("r000002", "applied")
    assert {entry.uid: entry.status for entry in record.decision.entries} == {4: "winner", 5: "lost"}
    assert state.head.folded and [king.uid for king in state.kings] == [9, 1, 4]


def test_an_open_round_whose_ticket_pins_another_image_is_closed_at_once_with_no_decision(tmp_path, monkeypatch, capsys):
    directory, stack, _ = old_state(tmp_path, "open_round")
    opened_at = load(directory / JOURNAL_FILE).round.opened_at
    monkeypatch.setattr(time, "time", lambda: opened_at + 5.0)
    monkeypatch.setattr(sim, "SIM_IMAGE", "the-image-after-a-fix")  # this box no longer has the image that the ticket pins
    assert main(["run", *stack, *RUN]) == 0
    state = load(directory / JOURNAL_FILE)
    assert state.round is None and state.rounds[-1].outcome == "no_decision" and len(state.heads) == 3
    text = capsys.readouterr().out
    assert "SN120 v2 round r000002: NO_DECISION" in text and text.count("no result") == 2, "run prints the round that got no result, and goes on"
    seats = [seat for seat in state.queue.values() if seat.reg.uid in (4, 5)]
    assert len(seats) == 2 and all(seat.front and not seat.leased and not seat.held for seat in seats), "nobody is charged; both are first again"


def test_the_operator_can_end_an_open_round_at_once(tmp_path, capsys, monkeypatch):
    directory, stack, _ = old_state(tmp_path, "open_round")
    assert main(["end-round", *stack]) == 0
    state = load(directory / JOURNAL_FILE)
    assert state.round is None and (state.rounds[-1].round_id, state.rounds[-1].outcome) == ("r000002", "no_decision")
    seats = [seat for seat in state.queue.values() if seat.reg.uid in (4, 5)]
    assert len(seats) == 2 and all(seat.front and not seat.leased and not seat.held and seat.infra_attempts == 0 for seat in seats)
    assert state.served_hotkeys == load(STATES / "open_round" / JOURNAL_FILE).served_hotkeys and len(state.heads) == 3
    assert main(["end-round", *stack]) == 2, "with no open round the command changes nothing"
    assert "no round is open" in capsys.readouterr().err
    assert main(["run", *stack, *RUN]) == 0 and load(directory / JOURNAL_FILE).rounds[-1].round_id == "r000002", "a new round waits apply_retry_s after a failure"
    ended_at = state.last_failure_at
    monkeypatch.setattr(time, "time", lambda: ended_at + Params().apply_retry_s + 1.0)
    assert main(["run", *stack, *RUN]) == 0
    state = load(directory / JOURNAL_FILE)
    assert (state.rounds[-1].round_id, state.rounds[-1].outcome) == ("r000003", "applied"), "the same seats run in the next round"
    assert {entry.uid: entry.status for entry in state.rounds[-1].decision.entries} == {4: "winner", 5: "lost"}
