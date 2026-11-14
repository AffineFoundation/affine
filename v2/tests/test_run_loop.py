"""The GPU box starts ``run`` again after a crash, and a restart costs no miner anything.

Three parts:

* The controller: a fault in its own judging, after the box result is in the
  store, stops ``run`` and keeps the round open with that result. The next
  ``run`` judges it. ``end-round`` does not throw such a result away unless told.
* ``systemd/sn120-v2-run-loop.sh`` with a fake ``run`` that crashes: what the
  loop does before a start, which exit codes stay stopped, when it gives up.
* The same loop with the real command line and the CPU stack, and the unit file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import sn120_v2
from sn120_v2.cli import RESULT_STORED, main
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.state import load
from sn120_v2.worker import LocalRunner

STATES = Path(__file__).parent / "data" / "state_515d7e9f"
SYSTEMD = Path(sn120_v2.__file__).resolve().parents[1] / "systemd"
LOOP = SYSTEMD / "sn120-v2-run-loop.sh"
RUN = ["--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]
# A child process must import this checkout, whatever the shell's PYTHONPATH says.
CHILD_ENV = {**os.environ, "PYTHONPATH": str(Path(sn120_v2.__file__).resolve().parents[1])}

# ``run`` with a controller whose judging raises: the clock is argv[1], the rest is the command line.
FAULTY_RUN = """
import sys, time
from sn120_v2 import controller
from sn120_v2.cli import main

def broken(pairs):
    raise RuntimeError("a fault in judging")

controller.decide = broken
time.time = lambda: float(sys.argv[1])
sys.exit(main(sys.argv[2:]))
"""


def open_round(tmp_path) -> tuple[Path, list[str]]:
    directory = tmp_path / "state"
    shutil.copytree(STATES / "open_round", directory)
    return directory, ["--state-dir", str(directory), "--stack", "sn120_v2.sim:stack"]


def result_then_judging_fault(tmp_path) -> tuple[Path, list[str], subprocess.CompletedProcess]:
    """An open round whose box result is in the store, left by a ``run`` that died in judging."""
    directory, stack = open_round(tmp_path)
    at = load(directory / JOURNAL_FILE).round.opened_at + 5.0  # the lease still runs
    done = subprocess.run([sys.executable, "-c", FAULTY_RUN, str(at), "run", *stack, *RUN], env=CHILD_ENV, capture_output=True, text=True)
    return directory, stack, done


def stored_results(directory: Path) -> list[str]:
    return sorted(path.parent.parent.name for path in directory.rglob("result.json"))


# ------------------------------------------------------------ the controller


@pytest.mark.parametrize("after_lease_end", [False, True])
def test_a_fault_in_judging_stops_run_and_keeps_the_result_and_the_next_run_judges_it(tmp_path, monkeypatch, after_lease_end):
    directory, stack, done = result_then_judging_fault(tmp_path)
    assert done.returncode == 1 and "RuntimeError: a fault in judging" in done.stderr, "run ends with a non-zero code"
    state = load(directory / JOURNAL_FILE)
    lease = state.round.leases[0]
    assert state.round.round_id == "r000002" and state.round.decision is None and lease.status == "open", "the round is open; the lease did not fail"
    assert state.failed_leases == 0 and stored_results(directory) == ["r000001", "r000002"], "the box result is in the store"
    # Then with a working ``decide``. One case starts ``run`` long after the lease end: a stored result is not timed out.
    monkeypatch.setattr(time, "time", lambda: lease.expires_at + 3600.0 if after_lease_end else state.round.opened_at + 60.0)
    assert main(["run", *stack, *RUN]) == 0
    state = load(directory / JOURNAL_FILE)
    record = state.rounds[-1]
    assert state.round is None and (record.round_id, record.outcome, record.leases[0].status) == ("r000002", "applied", "done")
    assert {entry.uid: entry.status for entry in record.decision.entries} == {4: "winner", 5: "lost"}, "the stored result was judged; no new round ran"
    assert state.failed_leases == 0 and len(state.rounds) == 2 and [king.uid for king in state.kings] == [9, 1, 4]


def test_a_failure_before_the_result_is_stored_fails_the_lease_and_run_goes_on(tmp_path, monkeypatch):
    directory, stack = open_round(tmp_path)
    opened_at = load(directory / JOURNAL_FILE).round.opened_at
    monkeypatch.setattr(time, "time", lambda: opened_at + 5.0)

    def dead_box(self, ticket):
        raise RuntimeError("the box died")

    monkeypatch.setattr(LocalRunner, "run", dead_box)
    assert main(["run", *stack, *RUN]) == 0, "as before: a box failure does not stop the controller"
    state = load(directory / JOURNAL_FILE)
    assert state.round is None and state.rounds[-1].outcome == "no_decision" and state.failed_leases == 1
    assert stored_results(directory) == ["r000001"]
    seats = [seat for seat in state.queue.values() if seat.reg.uid in (4, 5)]
    assert len(seats) == 2 and all(seat.front and not seat.held and seat.infra_attempts == 0 for seat in seats)


def test_end_round_keeps_a_stored_box_result_unless_the_operator_discards_it(tmp_path, capsys):
    directory, stack, _ = result_then_judging_fault(tmp_path)
    assert main(["end-round", *stack]) == RESULT_STORED == 3
    assert "kept: round r000002 has a box result in the store" in capsys.readouterr().err
    state = load(directory / JOURNAL_FILE)
    assert state.round is not None and state.round.decision is None and state.failed_leases == 0, "nothing was ended"
    assert main(["end-round", *stack, "--discard-result"]) == 0, "the operator ends a round whose stored result is itself bad"
    state = load(directory / JOURNAL_FILE)
    assert state.round is None and (state.rounds[-1].round_id, state.rounds[-1].outcome) == ("r000002", "no_decision")
    seats = [seat for seat in state.queue.values() if seat.reg.uid in (4, 5)]
    assert len(seats) == 2 and all(seat.front and not seat.leased and not seat.held and seat.infra_attempts == 0 for seat in seats), "nobody is charged"
    assert main(["end-round", *stack]) == 2 and "refused: V2Error: no round is open" in capsys.readouterr().err, "the text that the loop reads"


# ------------------------------------------------------------ the loop with a fake run

# The fake stands in for PYTHON. ``run`` and ``end-round`` each take the next line of their plan file.
FAKE_PYTHON = """#!{python}
import json, os, signal, subprocess, sys, time
from pathlib import Path

home = Path(os.environ["FAKE_DIR"])
command = sys.argv[3]

def take(default):
    path = home / (command + ".plan")
    lines = path.read_text().splitlines() if path.exists() else []
    path.write_text("".join(line + "\\n" for line in lines[1:]))
    return (lines[0] if lines else default).split()

def note(text):
    with (home / "calls").open("a") as handle:
        handle.write(text + "\\n")

def alive(pid):
    try:
        return Path(f"/proc/{{pid}}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False

def refuse(name):
    print(f"refused: {{name}}: the fake says no", file=sys.stderr)
    sys.exit(2)

rounds = home / "rounds"
if command == "status":
    print(json.dumps({{"phase": "IDLE", "rounds_closed": int(rounds.read_text()), "queue": 3}}, indent=2))
    sys.exit(0)
if command == "end-round":
    (answer,) = take("none")
    note(f"end-round {{answer}} mark={{'SN120_LOOP_MARK' in os.environ}}")
    if answer == "none":
        print("refused: V2Error: no round is open and undecided; a decided round is finished by run", file=sys.stderr)
        sys.exit(2)
    if answer.startswith("refuse:"):
        refuse(answer[7:])
    sys.exit({{"ended": 0, "kept": 3, "crash": 1}}[answer])
assert command == "run" and sys.argv[1:3] == ["-m", "sn120_v2"]
left = [pid for pid in (home / "children").read_text().split() if alive(int(pid))] if (home / "children").exists() else []
note(f"run {{' '.join(sys.argv[4:])}} mark={{os.environ.get('SN120_LOOP_MARK')}} left={{len(left)}}")
for step in take("exit:0"):
    name, _, value = step.partition(":")
    if name == "round":  # run closed one round
        rounds.write_text(str(int(rounds.read_text()) + 1))
        print("SN120 v2 round r000009: APPLIED", flush=True)
    elif name == "child":  # as a model server: its own session, it outlives run
        child = subprocess.Popen(["sleep", "300"], start_new_session=True)
        with (home / "children").open("a") as handle:
            handle.write(f"{{child.pid}}\\n")
    elif name == "exit":
        if value == "1":
            raise RuntimeError("the fake run crashed")
        sys.exit(int(value))
    elif name == "refuse":
        refuse(value)
    elif name == "kill":
        os.kill(os.getpid(), getattr(signal, "SIG" + value))
    elif name == "hang":
        (home / "hanging").write_text(str(os.getpid()))
        time.sleep(300)
"""

FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
[ "$*" = "--query-gpu=index,memory.used --format=csv,noheader,nounits" ] || exit 9
cat "$FAKE_DIR/gpus"
"""


class Box:
    """A work directory, a state directory, and a fake python and nvidia-smi for the loop."""

    def __init__(self, root: Path) -> None:
        self.home, self.work, self.state, self.bin = root / "fake", root / "work", root / "state", root / "bin"
        for directory in (self.home, self.state, self.bin):
            directory.mkdir(parents=True, exist_ok=True)
        self.python = self.bin / "python"
        self.python.write_text(FAKE_PYTHON.format(python=sys.executable))
        (self.bin / "nvidia-smi").write_text(FAKE_NVIDIA_SMI)
        for program in self.bin.iterdir():
            program.chmod(0o755)
        (self.home / "rounds").write_text("7")
        (self.home / "gpus").write_text("0, 3\n1, 12\n")
        settings = {"SN120_LOOP_PAUSE_S": "0", "SN120_LOOP_MAX_STARTS": "3", "SN120_LOOP_KILL_GRACE_S": "5", "SN120_LOOP_GPU_WAIT_S": "0", "SN120_LOOP_STOP_GRACE_S": "20"}
        self.env = {**CHILD_ENV, **settings, "FAKE_DIR": str(self.home), "SN120_WORK_DIR": str(self.work), "PATH": f"{self.bin}:{os.environ['PATH']}"}
        self.command = [str(LOOP), str(self.python), "--state-dir", str(self.state), "--stack", "fake:stack"]

    def plan(self, run: list[str] = (), end_round: list[str] = ()) -> None:
        (self.home / "run.plan").write_text("".join(line + "\n" for line in run))
        (self.home / "end-round.plan").write_text("".join(line + "\n" for line in end_round))

    def start(self, ignore_sigint: bool = False) -> subprocess.Popen:
        # A shell script that starts the loop with "&" gives it an ignored SIGINT, and bash cannot change that back.
        before = signal.signal(signal.SIGINT, signal.SIG_IGN) if ignore_sigint else None
        try:
            return subprocess.Popen(self.command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        finally:
            if ignore_sigint:
                signal.signal(signal.SIGINT, before)

    def loop(self, run: list[str] = (), end_round: list[str] = (), **env: str) -> int:
        self.plan(run, end_round)
        self.env.update(env)
        process = self.start()
        process.communicate(timeout=60)
        return process.returncode

    @property
    def calls(self) -> list[str]:
        return (self.home / "calls").read_text().splitlines() if (self.home / "calls").exists() else []

    @property
    def runs(self) -> int:
        return sum(call.startswith("run --") for call in self.calls)

    @property
    def log(self) -> str:
        return (self.work / "logs" / "run.log").read_text()

    def last_lines(self, count: int = 5) -> str:
        return "\n".join(self.log.splitlines()[-count:])

    @property
    def status(self) -> str:
        return (self.work / "logs" / "run-loop.status").read_text()


@pytest.fixture
def box(tmp_path) -> Box:
    return Box(tmp_path)


def test_the_loop_starts_run_again_after_a_crash_and_stops_when_run_finishes(box):
    assert box.loop(["exit:1", "kill:KILL", "exit:0"]) == 0
    mark = f"mark={box.state.resolve()}"
    assert box.calls == ["end-round none mark=False", f"run --state-dir {box.state} --stack fake:stack {mark} left=0"] * 3
    assert "RuntimeError: the fake run crashed" in box.log, "the output of run is in the log under $SN120_WORK_DIR/logs"
    assert "exit code 1 (a Python error" in box.log and "exit code 137 (SIGKILL" in box.log and box.log.count("next start in 0 s") == 2
    assert "STOPPED: run finished with exit code 0" in box.last_lines(2) and "stopped: run finished" in box.status
    assert not (box.work / "logs" / "run-loop.pid").exists()


@pytest.mark.parametrize("step, code, runs", [("refuse:FencedError", 2, 1), ("refuse:ConfigError", 2, 1), ("exit:130", 0, 1), ("exit:143", 0, 1), ("refuse:InfraError", 0, 2)])
def test_the_exit_codes_of_run_that_mean_do_not_start_again(box, step, code, runs):
    """0 is a finished run, 2 is refused, 130 and 143 are an operator's stop. A store that did not answer is tried again."""
    assert box.loop([step, "exit:0"]) == code and box.runs == runs
    if code == 2:
        assert f"STOPPED: run refused, and a new start cannot change that: refused: {step[7:]}: the fake says no" in box.last_lines(2)
        assert "start the loop again" in box.last_lines(1) and "refused" in box.status
    elif runs == 1:
        assert "STOPPED by the operator" in box.last_lines(2) and "start the loop again" in box.last_lines(1)
    else:
        assert "run ended with exit code 2, refused: InfraError: the fake says no." in box.log and "This try is not counted" in box.log


def test_the_loop_gives_up_after_the_set_starts_with_no_finished_round_and_a_finished_round_resets_the_count(box):
    crashes = ["exit:1", "exit:1", "round exit:1", "exit:1", "exit:1", "exit:1", "exit:0"]  # the third start closes a round, then crashes
    assert box.loop(crashes, SN120_LOOP_MAX_STARTS="3") == 3
    assert box.runs == 6, "two starts, a round, then three more starts: the limit counts the starts since the last finished round"
    assert "It closed 1 round(s) in this start: the count of starts goes back to zero." in box.log
    assert "start 3 of at most 3 with no finished round between them (rounds closed so far: 8)" in box.log
    why, where, then, discard = box.last_lines(4).splitlines()
    assert "STOPPED: 3 starts in a row and no round finished. The last one ended with: exit code 1" in why
    assert f"{box.python} -m sn120_v2 status --state-dir {box.state}" in where and "start the loop again" in then
    assert f"{box.python} -m sn120_v2 end-round --state-dir {box.state} --stack fake:stack --discard-result" in discard
    assert "THROWS THE RESULT AWAY" in discard and "gave up after 3 starts" in box.status


def test_before_a_new_start_the_loop_stops_what_the_dead_run_left(box):
    assert box.loop(["child child exit:1", "exit:0"]) == 0
    children = [int(pid) for pid in (box.home / "children").read_text().split()]
    assert len(children) == 2 and "stopping what the last run left: pids" in box.log
    assert [call.rsplit(" ", 1)[1] for call in box.calls if call.startswith("run --")] == ["left=0", "left=0"], "both were gone before the second start"
    for pid in children:
        with pytest.raises(OSError):
            os.kill(pid, 0)


def test_a_gpu_that_stays_in_use_blocks_the_start(box):
    (box.home / "gpus").write_text("0, 3\n1, 41230\n2, 1024\n3, 1025\n")
    assert box.loop(["exit:0"]) == 4 and box.calls == [], "neither end-round nor run was started"
    assert "STOPPED: run was not started, because after 0 s: GPUs 1 3 hold more than 1024 MiB" in box.last_lines(2)
    assert "nvidia-smi --query-compute-apps" in box.last_lines(1) and "a GPU stays in use" in box.status
    (box.home / "gpus").write_text("no devices were found\n")
    (box.bin / "nvidia-smi").write_text(FAKE_NVIDIA_SMI + "exit 6\n")
    assert box.loop(["exit:0"]) == 4 and box.calls == [], "a check that fails is not a free GPU"
    assert "nvidia-smi failed: no devices were found" in box.last_lines(2)


def test_the_loop_ends_an_open_round_before_a_start_but_never_a_round_with_a_stored_result(box):
    assert box.loop(["exit:1", "exit:1", "exit:0"], end_round=["ended", "kept", "none"]) == 0
    assert [call.split()[0] for call in box.calls] == ["end-round", "run"] * 3, "end-round comes before every start"
    assert "end-round: the open round is ended with no decision. Nobody is charged and no failure is counted." in box.log
    assert "end-round: a box result of the open round is in the store. The round stays open, and run judges that result." in box.log


def test_an_end_round_that_fails_counts_as_a_start_and_one_that_is_refused_stops_the_loop(box):
    assert box.loop(["exit:0"], end_round=["refuse:InfraError", "crash", "none"]) == 0
    assert [call.split()[0] for call in box.calls] == ["end-round", "end-round", "end-round", "run"]
    assert "end-round: refused: InfraError: the fake says no. run was not started." in box.log
    assert "end-round ended with exit code 1. run was not started. No round finished: this counts as start 1 of at most 3." in box.log
    assert box.loop(["exit:0"], end_round=["refuse:JournalCorrupt"]) == 2 and box.runs == 1, "run was not started again"
    assert "STOPPED: end-round refused, and a new start cannot change that: refused: JournalCorrupt" in box.last_lines(2)


def test_a_store_or_chain_that_does_not_answer_never_stops_the_loop_and_never_counts_as_a_start(box):
    """More such tries than the limit, from end-round and from run. Then a normal start."""
    refused = ["refuse:InfraError"]
    assert box.loop(refused * 3 + ["exit:0"], end_round=refused * 2, SN120_LOOP_MAX_STARTS="3") == 0
    assert [call.split()[0] for call in box.calls] == ["end-round"] * 3 + ["run", "end-round"] * 2 + ["run", "end-round", "run"]
    assert box.log.count("This try is not counted (the count of starts stays at 0 of at most 3)") == 5 and "try 5, no answer since" in box.log
    assert box.log.count("start 1 of at most 3 with no finished round") == 4 and "STOPPED: 3 starts" not in box.log
    assert "STOPPED: run finished with exit code 0" in box.last_lines(2) and "stopped: run finished" in box.status


def test_the_count_of_crashes_stays_through_the_tries_with_no_answer(box):
    """Two crashes, then tries with no answer, then the third crash: the loop gives up. Such tries do not set the count to zero."""
    refused = ["refuse:InfraError"]
    plan = ["exit:1", *refused * 2, "exit:1", *refused * 2, "exit:1", "exit:0"]
    assert box.loop(plan, end_round=["none", "none", "refuse:InfraError"], SN120_LOOP_MAX_STARTS="3") == 3
    assert box.runs == 7 and box.log.count("This try is not counted") == 5, "the last line of the plan never ran"
    assert "the count of starts stays at 1 of at most 3" in box.log and "the count of starts stays at 2 of at most 3" in box.log
    assert "No round finished: this counts as start 3 of at most 3." in box.log
    assert "STOPPED: 3 starts in a row and no round finished. The last one ended with: exit code 1" in box.last_lines(4)


def test_the_wait_for_the_store_or_the_chain_says_since_when_and_its_pause_grows_to_a_limit(box):
    box.plan(["refuse:InfraError"] * 3 + ["exit:0"])
    box.env.update(SN120_LOOP_PAUSE_S="1", SN120_LOOP_MAX_PAUSE_S="2")
    process = box.start()
    deadline = time.monotonic() + 20
    while "waiting for the store" not in (status := wait_for(box.work / "logs" / "run-loop.status")):
        assert time.monotonic() < deadline, status
        time.sleep(0.02)
    waiting = re.fullmatch(
        r"\S+ waiting for the store or the chain since (\S+Z) \(\d tries with no answer\); next try in \d s; "
        r"the last one ended with: exit code 2, refused: InfraError: the fake says no\n",
        status,
    )
    assert waiting, status
    process.communicate(timeout=60)
    assert process.returncode == 0
    lines = re.findall(r"try (\d), no answer since (\S+)\. .* Next try in (\d) s\.", box.log)
    assert [(line[0], line[2]) for line in lines] == [("1", "1"), ("2", "2"), ("3", "2")], "the pause doubles, up to the limit"
    assert {line[1] for line in lines} == {waiting.group(1)}, "every line names the time of the first try with no answer"
    assert "stop the loop and fix what the refused line names" in box.log


def wait_for(path: Path, seconds: float = 20.0) -> str:
    deadline = time.monotonic() + seconds
    while not path.exists() or not path.read_text():
        assert time.monotonic() < deadline, f"{path.name} did not appear"
        time.sleep(0.05)
    return path.read_text()


@pytest.mark.parametrize("ignore_sigint", [False, True])
def test_a_stop_of_the_loop_stops_run_and_what_it_started_and_nothing_starts_it_again(box, ignore_sigint):
    box.plan(["child hang", "exit:0"])
    process = box.start(ignore_sigint)
    run_pid = int(wait_for(box.home / "hanging"))
    assert int(wait_for(box.work / "logs" / "run-loop.pid")) == process.pid and f"run has pid {run_pid}" in box.status
    second = box.start()
    assert second.wait(timeout=20) == 64 and "another loop runs with this work directory" in second.stderr.read()
    process.send_signal(signal.SIGTERM)  # what "systemctl stop" and the operator's kill send
    assert process.wait(timeout=30) == 0 and box.runs == 1
    assert "STOPPED by the operator (SIGTERM to the loop; run ended with exit code 143)" in box.last_lines(2)
    for pid in (run_pid, int((box.home / "children").read_text())):
        with pytest.raises(OSError):
            os.kill(pid, 0)  # run is gone, and the loop stopped what run left behind


def test_a_wrong_call_changes_nothing(box, tmp_path):
    for command, env in (
        ([str(LOOP)], box.env),
        ([str(LOOP), str(box.python), "--stack", "fake:stack"], box.env),
        ([str(LOOP), str(box.python), "--state-dir", str(tmp_path / "none"), "--stack", "fake:stack"], box.env),
        (box.command, {name: value for name, value in box.env.items() if name != "SN120_WORK_DIR"}),
    ):
        done = subprocess.run(command, env=env, capture_output=True, text=True)
        assert done.returncode == 64 and "usage:" in done.stderr and box.calls == []


def test_on_a_new_box_the_loop_starts_from_an_empty_state_directory(box, tmp_path):
    """A new box has no journal yet: ``run`` pulls the copy from the store. The loop must not need a local journal."""
    box.state.rmdir()
    box.state.mkdir()
    (box.home / "rounds").write_text("no journal here")  # the fake ``status`` fails, as the real one does with no journal
    assert box.loop(["exit:0"]) == 0 and box.runs == 1
    assert "(rounds closed so far: unknown)" in box.log


# ------------------------------------------------------------ the loop with the real command line


def real_loop(box: Box, directory: Path) -> subprocess.CompletedProcess:
    command = [str(LOOP), sys.executable, "--state-dir", str(directory), "--stack", "sn120_v2.sim:stack", *RUN]
    return subprocess.run(command, env=box.env, capture_output=True, text=True, timeout=120)


def test_the_loop_with_the_real_commands_keeps_a_stored_result_and_run_judges_it(box, tmp_path):
    directory, _, _ = result_then_judging_fault(tmp_path / "real")
    done = real_loop(box, directory)  # the real clock: the lease of that round ended long ago
    assert done.returncode == 0, done.stderr
    assert "end-round: a box result of the open round is in the store." in box.log
    assert "rounds closed so far: 1)" in box.log, "the loop reads rounds_closed from the real status command"
    state = load(directory / JOURNAL_FILE)
    record = state.rounds[-1]
    assert state.round is None and (record.round_id, record.outcome) == ("r000002", "applied") and state.failed_leases == 0
    assert {entry.uid: entry.status for entry in record.decision.entries} == {4: "winner", 5: "lost"}


def test_the_loop_with_the_real_commands_ends_an_open_round_and_charges_nobody(box, tmp_path):
    directory, _ = open_round(tmp_path / "real")
    done = real_loop(box, directory)
    assert done.returncode == 0, done.stderr
    assert "end-round: the open round is ended with no decision." in box.log and "rounds closed so far: 2)" in box.log
    state = load(directory / JOURNAL_FILE)
    assert state.round is None and (state.rounds[-1].round_id, state.rounds[-1].outcome) == ("r000002", "no_decision")
    seats = [seat for seat in state.queue.values() if seat.reg.uid in (4, 5)]
    assert len(seats) == 2 and all(seat.front and not seat.leased and not seat.held and seat.infra_attempts == 0 for seat in seats)
    again = real_loop(box, directory)  # no round is open now: the real "refused" text of end-round means "go on"
    assert again.returncode == 0 and box.log.count("end-round:") == 1 and box.log.count("STOPPED: run finished") == 2


# ------------------------------------------------------------ the unit file


def test_the_unit_starts_the_same_loop_and_keeps_its_stops_stopped():
    unit = (SYSTEMD / "sn120-v2-run.service").read_text()
    body = unit.split("[Unit]")[1]
    assert set(re.findall(r"SN120_[A-Z0-9_]+", body)) == set(re.findall(r"^#   (SN120_[A-Z0-9_]+)", unit, re.M)), "every blank is listed"
    start = next(line for line in body.splitlines() if line.startswith("ExecStart="))
    assert start == "ExecStart=SN120_V2_DIR/systemd/sn120-v2-run-loop.sh SN120_PYTHON --state-dir SN120_STATE_DIR --stack sn120_v2.assembly:stack"
    assert os.access(LOOP, os.X_OK), "the unit starts the script directly"
    codes = re.search(r"^# Exit codes of the loop: (.*)$", LOOP.read_text(), re.M).group(1)
    stops = [code for code in re.findall(r"\d+", codes) if code != "0"]
    assert f"RestartPreventExitStatus={' '.join(stops)}\n" in body and "Restart=on-failure\n" in body, "systemd starts the loop again only when the loop itself died"
    assert "KillMode=mixed\n" in body and "WantedBy=multi-user.target" in body and "EnvironmentFile=SN120_ENV_FILE\n" in body
