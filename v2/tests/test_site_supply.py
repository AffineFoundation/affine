"""The miner supply record (``v2/site/supply.py``) on fixtures.

The validator's side is the real controller on CPU (``sim.SimWorld``): its
file check, its rounds, its frozen copies and its public snapshot. The store's
list is ``Bucket``: the sim's store as a list call of R2 shows it, with the
MD5 of one PUT as the ETag and the sim's clock as ``LastModified``. One test
runs the command itself against the S3 fake (moto).
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto.server import ThreadedMotoServer

from sn120_v2 import layout, sim
from sn120_v2.public import QUEUE_PREVIEW

SITE = Path(sim.__file__).resolve().parents[1] / "site"
supply = SimpleNamespace(**runpy.run_path(str(SITE / "supply.py"), run_name="supply"))  # a script of v2/site, not a module of the package

ROUND_FILES = layout.round_file("x").partition("x")[0]
ACCESS, SECRET, BUCKET = "supply-test-access-key", "supply-test-secret-key-0123456789", "sn120-supply-test"


class Bucket:
    """The list call on the sim's store. ``stamp`` gives each new or changed object the sim's time as its ``LastModified``."""

    def __init__(self, world: sim.SimWorld) -> None:
        self.world, self.seen, self.date = world, {}, None
        self.snapshot_extra: dict | None = None

    def stamp(self) -> None:
        keys = self.world.store.list("private/") + self.world.store.list(ROUND_FILES)
        for key in keys:
            etag = hashlib.md5(self.world.store.get(key), usedforsecurity=False).hexdigest()
            if self.seen.get(key, ("", 0, 0))[0] != etag:
                self.seen[key] = (etag, self.world.now, self.world.store.size(key))
        for key in set(self.seen) - set(keys):
            del self.seen[key]

    def touch(self, key: str) -> None:
        """The same bytes stored again: the ETag stays, ``LastModified`` is new."""
        self.seen[key] = (self.seen[key][0], self.world.now, self.seen[key][2])

    def list(self, prefix: str) -> list[supply.Stored]:
        self.stamp()
        return [supply.Stored(key, *self.seen[key]) for key in sorted(self.seen) if key.startswith(prefix)]

    def get(self, key: str) -> bytes:
        assert key == layout.MINING_JSON, "the recorder reads no other object: never a miner file"
        snap = json.loads(json.dumps(self.world.publisher.snapshots[-1]))
        if self.snapshot_extra:
            snap["queue"] |= self.snapshot_extra
        return json.dumps(snap).encode()


class Stage:
    """A validator, its store and one recorder with its record on disk."""

    def __init__(self, tmp_path: Path) -> None:
        self.world = sim.SimWorld(tmp_path / "validator", sim.sim_params())
        self.bucket = Bucket(self.world)
        self.record = tmp_path / "record" / supply.STATE
        self.record.parent.mkdir()
        self.doc: dict = {}

    def upload(self, uid: int, quality: float = 0.1, valid: bool = True) -> None:
        if uid in self.world.chain:
            self.world.admission.upload(f"hk{uid}", quality, valid=valid)
        else:
            self.world.register(uid, quality, valid=valid)
        self.bucket.stamp()

    def wait(self, seconds: int) -> None:
        self.world.now += seconds

    def check(self) -> None:
        """The validator checks every waiting file and publishes, as between two rounds."""
        self.world.controller.publish(self.world.clock(), check_files=True)

    def poll(self) -> dict:
        """One run of the recorder, as the command does it: the record is read from the disk and written back."""
        state, self.doc = supply.poll(self.bucket, supply.load_state(self.record), self.world.now)
        supply.write(self.record, json.dumps(state).encode(), 0o600)
        return self.doc

    def last(self, hours: int = 24) -> dict:
        return next(row for row in self.poll()["windows"] if row["hours"] == hours)

    def entry(self, uid: int, index: int = -1) -> dict:
        return json.loads(self.record.read_bytes())["files"][f"hk{uid}"][index]


def numbers(row: dict) -> tuple[int, int, int, int, int]:
    return tuple(row[name] for name in ("uploads", "validated", "no_verdict", "not_valid", "replaced"))


@pytest.fixture
def stage(tmp_path):
    return Stage(tmp_path)


def test_a_new_valid_file_is_an_upload_at_once_and_is_validated_only_after_the_validators_check(stage):
    stage.upload(11)
    stage.wait(60)
    assert numbers(stage.last()) == (1, 0, 1, 0, 0), "no check yet: the upload has no verdict"
    stage.wait(60)
    stage.check()
    assert numbers(stage.last()) == (1, 1, 0, 0, 0)
    entry = stage.entry(11)
    assert entry["proof"] == "file check" and entry["verdict"] == "valid"
    assert stage.doc["waiting"]["valid"] == 1 and stage.doc["waiting"]["source"] == "queue.next"


def test_a_changed_file_is_a_new_upload_and_waits_for_its_own_check(stage):
    stage.upload(11, quality=0.1)
    stage.wait(60)
    stage.check()
    assert numbers(stage.last()) == (1, 1, 0, 0, 0)
    stage.wait(600)
    stage.upload(11, quality=0.2)  # other bytes: another batch_sha256
    stage.wait(60)
    assert numbers(stage.last()) == (2, 1, 1, 0, 0), "the verdict of the check before the upload is not given to the new file"
    stage.check()
    assert numbers(stage.last()) == (2, 2, 0, 0, 0)
    assert stage.entry(11, 0)["ended_at"] is not None and stage.entry(11, 1)["ended_at"] is None


def test_an_invalid_file_is_an_upload_and_is_never_validated(stage):
    stage.upload(11, valid=False)
    stage.upload(12)
    stage.wait(60)
    stage.check()
    assert numbers(stage.last()) == (2, 1, 0, 1, 0)
    stage.wait(60)
    stage.check()
    assert numbers(stage.last()) == (2, 1, 0, 1, 0)
    assert stage.doc["waiting"]["valid"] == 1


def test_a_file_that_is_listed_again_counts_one_time_also_when_the_same_bytes_are_stored_again(stage):
    stage.upload(11)
    stage.wait(60)
    stage.check()
    for _ in range(5):  # the same list, five runs
        stage.wait(60)
        assert numbers(stage.last()) == (1, 1, 0, 0, 0)
    stage.wait(1800)
    first = stage.entry(11)["uploaded_at"]
    stage.bucket.touch(layout.miner_object("hk11"))  # the same ETag with a newer LastModified
    stage.upload(11)  # the same bytes through the miner's path: the ETag does not change
    assert numbers(stage.last()) == (1, 1, 0, 0, 0)
    assert stage.entry(11)["uploaded_at"] == first, "the file keeps its first upload time"
    assert sum(row["uploads"] for row in stage.doc["hours"]) == 1


def test_the_record_survives_a_restart_and_a_lost_record_starts_again_with_the_newest_files_only(stage):
    stage.upload(11)
    stage.upload(12, valid=False)
    stage.wait(60)
    stage.check()
    stage.poll()
    stage.wait(600)
    stage.upload(12, quality=0.3)  # 12 replaces its file: two files of 12 are in the record
    before = stage.last()
    saved = stage.record.read_bytes()
    assert numbers(before) == (3, 1, 1, 1, 0)
    stage.wait(60)
    again = stage.last()  # every run reads the record from the disk: this is a run after a restart
    assert numbers(again) == numbers(before) and stage.doc["counting_since"] == json.loads(saved)["since"]
    stage.record.unlink()  # the disk is lost
    stage.wait(60)
    fresh = stage.last()
    assert numbers(fresh) == (2, 1, 1, 0, 0), "the replaced file of 12 is gone; the store has only the newest file of each hotkey"
    assert stage.doc["counting_since"] == stage.doc["generated_at"] and not fresh["complete"]


def test_a_round_that_froze_the_file_is_a_proof_and_gives_the_batch_sha256(stage):
    stage.upload(11)
    stage.upload(12, valid=False)
    stage.poll()
    stage.wait(60)
    assert stage.world.round() is not None  # the round checks the files, freezes the valid one, and closes
    stage.wait(60)
    closed = stage.world.publisher.snapshots[-1]["history"][0]
    assert numbers(stage.last()) == (2, 1, 0, 1, 0)
    entry = stage.entry(11)
    assert entry["proof"] == f"round {closed['round_id']}" and entry["sha"] == closed["entries"][0]["batch_sha256"]
    assert entry["sha"] == hashlib.sha256(stage.world.store.get(layout.miner_object("hk11"))).hexdigest()
    lists = []
    stage.bucket.list = lambda prefix, real=stage.bucket.list: lists.append(prefix) or real(prefix)
    stage.poll()
    assert lists == [supply.MINERS], "the frozen copies of a closed round are listed one time"


def taken(stage, uid: int = 11) -> dict:
    """A round takes the file of ``uid``, then the miner stores another file. No run of the recorder saw the first file."""
    stage.upload(uid, quality=0.1)
    stage.wait(600)
    assert stage.world.round() is not None
    closed = stage.world.publisher.snapshots[-1]["history"][-1]
    stage.wait(7200)
    stage.upload(uid, quality=0.2)
    stage.wait(60)
    return closed


def test_a_file_that_a_round_took_counts_at_the_time_of_the_round_when_its_miner_replaced_it_before_the_record_saw_it(stage):
    lists = []
    stage.bucket.list = lambda prefix, real=stage.bucket.list: lists.append(prefix) or real(prefix)
    closed = taken(stage)
    row = stage.last()  # the first run: the store has only the new file of the miner
    assert numbers(row) == (2, 1, 1, 0, 0) and row["at_round_time"] == 1
    frozen = layout.frozen_batch(closed["round_id"], "x")[:-len("x.json")]
    assert lists == [supply.MINERS, frozen, layout.round_file(closed["round_id"])], "the list of the miners, the folder of the round, its round file"
    new, old = stage.entry(11, 0), stage.entry(11, 1)
    assert old["time_from"] == "round" and old["uploaded_at"] == closed["opened_at"] and old["ended_at"] is not None
    assert (old["verdict"], old["proof"], old["sha"]) == ("valid", f"round {closed['round_id']}", closed["entries"][0]["batch_sha256"])
    assert "time_from" not in new and new["verdict"] is None and new["ended_at"] is None
    hours = {row["t"]: row for row in stage.doc["hours"] if row["uploads"]}
    at_round = hours[closed["opened_at"] - closed["opened_at"] % 3600]
    assert (at_round["validated"], at_round["at_round_time"]) == (1, 1), "the count is in the hour in which the round opened"
    assert [row["complete"] for row in stage.doc["windows"]] == [False, False, False], "a file that no round took can still be missing"
    for _ in range(3):  # more runs: the file of the round stays one file
        stage.wait(60)
        row = stage.last()
        assert numbers(row) == (2, 1, 1, 0, 0) and row["at_round_time"] == 1
    assert lists.count(supply.MINERS) == 4 and len(lists) == 6, "the folder and the round file of the closed round are listed one time"
    stage.upload(11, quality=0.1)  # the miner stores the bytes of the round's file again: the same file, at the round's time
    row = stage.last()
    assert numbers(row) == (2, 1, 0, 0, 1) and row["at_round_time"] == 1
    assert stage.entry(11, 1)["ended_at"] is None and stage.entry(11, 1)["uploaded_at"] == closed["opened_at"]


def test_a_frozen_file_that_the_record_saw_as_the_stored_file_keeps_its_upload_time_and_counts_one_time(stage):
    stage.upload(11, quality=0.1)
    stage.wait(600)
    stage.poll()  # the record sees the file with its upload time
    uploaded = stage.entry(11)["uploaded_at"]
    assert stage.world.round() is not None
    stage.wait(600)
    row = stage.last()
    assert numbers(row) == (1, 1, 0, 0, 0) and row["at_round_time"] == 0, "still the stored file: one count, at the upload time"
    stage.upload(11, quality=0.2)  # the miner replaces it after the round
    stage.wait(60)
    row = stage.last()
    assert numbers(row) == (2, 1, 1, 0, 0) and row["at_round_time"] == 0
    assert stage.entry(11, 0)["uploaded_at"] == uploaded and len(json.loads(stage.record.read_bytes())["files"]["hk11"]) == 2


def test_a_file_that_came_into_two_rounds_counts_one_time_at_the_first_round(stage):
    stage.upload(11, quality=0.1)
    stage.wait(600)
    stage.world.trainer.fail_hotkeys = {"hk11"}  # no decision: nobody is charged, and the seat has its turn again
    assert stage.world.round().outcome == "no_decision"
    stage.wait(600)
    stage.world.trainer.fail_hotkeys = set()
    assert stage.world.round() is not None
    first, second = stage.world.publisher.snapshots[-1]["history"][-2:]
    assert first["opened_at"] < second["opened_at"]
    assert first["entries"][0]["batch_sha256"] == second["entries"][0]["batch_sha256"], "the same bytes of hk11 in two rounds"
    stage.wait(600)
    stage.upload(11, quality=0.3)
    stage.wait(60)
    for _ in range(2):
        stage.poll()
        entries = json.loads(stage.record.read_bytes())["files"]["hk11"]
        old = [entry for entry in entries if entry.get("time_from") == "round"]
        assert len(entries) == 2 and len(old) == 1 and old[0]["uploaded_at"] == first["opened_at"]
        day = stage.doc["windows"][-1]
        assert (day["uploads"], day["validated"], day["at_round_time"]) == (2, 1, 1), "two seats, one file: one count"
        stage.wait(60)


def test_a_record_of_the_first_form_gets_the_files_of_the_rounds_by_itself(stage):
    taken(stage)
    stage.poll()
    state = json.loads(stage.record.read_bytes())
    for hotkey, entries in state["files"].items():  # the record as the first form of the recorder wrote it
        state["files"][hotkey] = [entry for entry in entries if "time_from" not in entry]
    assert len(state["files"]["hk11"]) == 1 and "time_from" not in json.dumps(state)
    stage.record.write_text(json.dumps(state))
    stage.wait(60)
    row = stage.last()
    assert numbers(row) == (2, 1, 1, 0, 0) and row["at_round_time"] == 1
    after = json.loads(stage.record.read_bytes())
    assert (after["since"], after["gaps"], after["schema"]) == (state["since"], state["gaps"], state["schema"])
    assert stage.doc["counting_since"] == state["since"]


def test_a_stored_file_with_an_etag_that_is_not_an_md5_is_never_counted_a_second_time_from_the_round(stage):
    stage.upload(11, quality=0.1)
    stage.wait(600)
    assert stage.world.round() is not None
    closed = stage.world.publisher.snapshots[-1]["history"][-1]
    stage.wait(60)
    stage.bucket.stamp()
    key, real = layout.miner_object("hk11"), stage.bucket.stamp
    md5 = stage.bucket.seen[key][0]
    assert stage.bucket.seen[layout.frozen_batch(closed["round_id"], "hk11")][0] == md5

    def stamp() -> None:  # the miner's object as after a multipart upload: its ETag is not the MD5 of its bytes
        real()
        stage.bucket.seen[key] = (md5 + "-2", *stage.bucket.seen[key][1:])

    stage.bucket.stamp = stamp
    row = stage.last()
    assert numbers(row) == (1, 0, 1, 0, 0) and row["at_round_time"] == 0, "one file, as before: this ETag cannot prove the round"


def test_a_round_that_is_older_than_the_record_keeps_gives_no_file(stage):
    taken(stage)
    stage.wait(supply.KEEP_S)
    stage.poll()
    assert len(json.loads(stage.record.read_bytes())["files"]["hk11"]) == 1
    assert all(row["at_round_time"] == 0 for row in stage.doc["windows"])


def test_a_file_that_comes_after_the_check_has_no_verdict_until_the_next_check(stage):
    stage.upload(11)
    stage.wait(60)
    stage.check()
    stage.wait(3600)  # a round runs: no file check in this time
    stage.upload(12)
    stage.wait(1800)
    row = stage.last()
    assert numbers(row) == (2, 1, 1, 0, 0)
    assert stage.doc["waiting"]["checked_at"] < stage.entry(12)["uploaded_at"]
    stage.check()
    assert numbers(stage.last()) == (2, 2, 0, 0, 0), "the count of the past hour grew when the verdict came"
    hours = [row for row in stage.doc["hours"] if row["uploads"]]
    assert [row["validated"] for row in hours] == [1, 1] and len(stage.doc["hours"]) == 24


def test_a_file_that_was_replaced_before_any_verdict_is_counted_as_replaced(stage):
    stage.upload(11, quality=0.1)
    stage.wait(60)
    stage.poll()
    stage.upload(11, quality=0.2)
    stage.wait(60)
    stage.check()
    assert numbers(stage.last()) == (2, 1, 0, 0, 1)


def test_a_window_is_complete_only_when_the_record_covers_all_of_it(stage):
    stage.upload(11)
    stage.wait(7200)  # the file is two hours old at the first run
    first = stage.poll()
    assert [row["complete"] for row in first["windows"]] == [False, False, False]
    assert [row["uploads"] for row in first["windows"]] == [0, 1, 1], "the past is filled from LastModified"
    for _ in range(61):
        stage.wait(60)
        doc = stage.poll()
    assert [row["complete"] for row in doc["windows"]] == [True, False, False] and not doc["gaps"]
    stage.wait(1200)  # the recorder did not run for 20 minutes
    doc = stage.poll()
    assert len(doc["gaps"]) == 1 and [row["complete"] for row in doc["windows"]] == [False, False, False]
    for _ in range(61):
        stage.wait(60)
        doc = stage.poll()
    assert [row["complete"] for row in doc["windows"]] == [True, False, False], "the gap is behind the last hour now"


def test_only_the_first_seats_have_a_verdict_until_the_snapshot_has_the_table_of_the_whole_line(stage):
    seats = QUEUE_PREVIEW + 2
    for uid in range(100, 100 + seats):
        stage.upload(uid)
    stage.wait(60)
    stage.check()
    row = stage.last()
    assert numbers(row) == (seats, QUEUE_PREVIEW, 2, 0, 0)
    waiting = stage.doc["waiting"]
    assert (waiting["valid"], waiting["seats"], waiting["line"], waiting["whole_line"]) == (QUEUE_PREVIEW, QUEUE_PREVIEW, seats, False)
    checked = stage.world.publisher.snapshots[-1]["queue"]["files_checked_at"]
    rows = [{"uid": uid, "hotkey": f"hk{uid}", "file_valid": uid != 100, "file_verdict": "not_judged" if uid == 100 else "valid"}
            for uid in range(100, 100 + seats)]
    stage.bucket.snapshot_extra = {"files": {"checked_at": checked, "seats": seats, "hotkeys": seats, "valid": seats - 1, "not_valid": 0,
                                             "not_judged": 1, "verdicts": rows}}
    row = stage.last()
    assert numbers(row) == (seats, seats, 0, 0, 0), "a file that was called valid stays counted; the two last seats have a verdict now"
    waiting = stage.doc["waiting"]
    assert (waiting["valid"], waiting["source"], waiting["whole_line"]) == (seats - 1, "queue.files", True)
    stage.bucket.snapshot_extra = {"files": {"verdicts": [{"hotkey": "hk100", "file_verdict": "maybe"}]}}
    assert stage.poll()["waiting"]["source"] == "queue.next", "a table of another form is not used"


class RoundFiles:
    """A store with the public round files only: ``written`` is the write time of the file of each round."""

    def __init__(self, written: dict[str, int], error: Exception | None = None) -> None:
        self.written, self.error, self.calls, self.date = written, error, [], None

    def list(self, prefix: str) -> list[supply.Stored]:
        self.calls.append(prefix)
        if self.error:
            raise self.error
        return [supply.Stored(layout.round_file(name), "etag", at, 1) for name, at in self.written.items() if layout.round_file(name).startswith(prefix)]


OPEN = 1_000_000


def rounds(*closed: tuple[str, int], limit: object = 10800, open_gap: int = 4000) -> tuple[dict, dict[str, int]]:
    """A snapshot with one closed round for each (outcome, seconds from ``closed_at`` to the write of the round file), then an open round."""
    history, written, at = [], {}, OPEN
    for number, (outcome, extra) in enumerate(closed, 1):
        name = f"r{number:06d}"
        history.append({"round_id": name, "opened_at": at, "closed_at": at + 5400, "outcome": outcome})
        written[name] = at + 5400 + extra
        at += 5400 + open_gap
    running = {"round_id": f"r{len(closed) + 1:06d}", "opened_at": at, "closed_at": 0, "outcome": ""}
    return {"params": {"round_time_limit_s": limit}, "history": history, "round": running}, written


def test_the_time_of_a_round_comes_from_the_params_and_from_the_last_three_rounds_with_a_new_king():
    snap, written = rounds(("applied", 2615), ("applied", 2769), ("null", 38), ("applied", 2979), ("applied", 2818))
    store = RoundFiles(written)
    takes, kept = supply.round_time(store, {}, snap)
    assert takes["king_rounds"] == [{"round_id": "r000002", "extra_s": 2769}, {"round_id": "r000004", "extra_s": 2979}, {"round_id": "r000005", "extra_s": 2818}]
    assert (takes["limit_s"], takes["king_extra_s"], takes["king_read"]) == (10800, 2818, "ok"), "the median of the last three; a null round is no king"
    assert takes["words"] == "up to about 180 minutes, about 227 minutes with a king"
    assert store.calls == [layout.round_file(name) for name in ("r000002", "r000004", "r000005")] and kept == {name: written[name] for name in kept}
    again, kept_again = supply.round_time(store, kept, snap)
    assert again == takes and kept_again == kept and len(store.calls) == 3, "a round file with a write time in the record is not listed again"
    snap["params"]["round_time_limit_s"] = 77 * 60  # the operator changes the limit: the words follow at the next run
    assert supply.round_time(store, kept, snap)[0]["words"] == "up to about 77 minutes, about 124 minutes with a king"
    for none in (None, 0, -5, "10800", True, 10800.5):
        snap["params"]["round_time_limit_s"] = none
        takes = supply.round_time(store, kept, snap)[0]
        assert takes["limit_s"] is None and takes["words"] == "the public snapshot gives no time limit of a round now"
    assert supply.round_time(store, kept, {"history": [], "round": None})[0]["limit_s"] is None, "a snapshot with no params"


def test_fewer_than_three_rounds_with_a_king_give_the_median_of_those_and_a_new_round_moves_it():
    snap, written = rounds(("no_decision", 20), ("applied", 2600))
    assert supply.round_time(RoundFiles(written), {}, snap)[0]["king_extra_s"] == 2600
    snap, written = rounds(("applied", 2600), ("applied", 3000))
    takes = supply.round_time(RoundFiles(written), {}, snap)[0]
    assert takes["king_extra_s"] == 2800 and takes["words"] == "up to about 180 minutes, about 227 minutes with a king"
    snap, written = rounds(("applied", 2800), ("applied", 2800), ("applied", 2800), ("applied", 240), ("applied", 200))  # the fold got fast
    takes = supply.round_time(RoundFiles(written), {}, snap)[0]
    assert takes["king_extra_s"] == 240 and takes["words"] == "up to about 180 minutes, about 184 minutes with a king"


@pytest.mark.parametrize("closed", [(), (("null", 30), ("no_decision", 20), ("abandoned", 2000))])
def test_with_no_king_measured_the_words_give_the_limit_only_and_no_time_for_a_king(closed):
    snap, written = rounds(*closed)
    store = RoundFiles(written)
    takes, kept = supply.round_time(store, {}, snap)
    assert (takes["limit_s"], takes["king_extra_s"], takes["king_rounds"], takes["king_read"]) == (10800, None, [], "ok")
    assert takes["words"] == "up to about 180 minutes; a round that gives a new king takes longer, by a time that is not measured now"
    assert store.calls == [] and kept == {}, "no round with a king: no list call"


@pytest.mark.parametrize("error", [ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"), EndpointConnectionError(endpoint_url="https://store")])
def test_a_list_call_that_fails_does_not_stop_the_run_and_gives_no_time_for_a_king(error):
    snap, written = rounds(("applied", 2615), ("applied", 2769), ("applied", 2979))
    known = {"r000001": written["r000001"]}  # an older run measured the first round
    takes, kept = supply.round_time(RoundFiles(written, error), dict(known), snap)
    assert (takes["king_read"], takes["king_extra_s"]) == ("failed", None) and kept == known, "the record keeps what it has"
    assert takes["words"] == "up to about 180 minutes; a round that gives a new king takes longer, by a time that is not measured now"
    takes, kept = supply.round_time(RoundFiles(written), kept, snap)  # the next run: the store answers again
    assert (takes["king_read"], takes["king_extra_s"]) == ("ok", 2769) and len(kept) == 3


def test_a_round_file_that_was_written_again_is_no_measure_and_the_record_keeps_the_first_write_time():
    snap, written = rounds(("applied", 2615), ("applied", 2769), ("applied", 2979))
    late = written | {"r000002": snap["history"][2]["opened_at"] + supply.SKEW_S + 1}  # written again after the next round opened
    takes = supply.round_time(RoundFiles(late), {}, snap)[0]
    assert [row["round_id"] for row in takes["king_rounds"]] == ["r000001", "r000003"] and takes["king_extra_s"] == (2615 + 2979) // 2
    early = written | {"r000003": snap["history"][2]["closed_at"] - 1}  # a write time before the close is no measure
    assert [row["round_id"] for row in supply.round_time(RoundFiles(early), {}, snap)[0]["king_rounds"]] == ["r000001", "r000002"]
    takes, kept = supply.round_time(RoundFiles(written), {}, snap)
    later = {name: at + 86400 for name, at in written.items()}  # every file is written again a day later
    assert supply.round_time(RoundFiles(later), kept, snap)[0] == takes, "the first write time that a run saw stays"
    missing = {name: at for name, at in written.items() if name != "r000003"}
    takes, kept = supply.round_time(RoundFiles(missing), {}, snap)
    assert takes["king_extra_s"] == (2615 + 2769) // 2 and "r000003" not in kept, "a round file that is not in the list is asked for again at the next run"


def test_the_words_of_the_round_time_are_in_the_label_in_the_method_and_in_the_page(stage):
    closed = taken(stage)
    assert closed["outcome"] == "applied"
    key = layout.round_file(closed["round_id"])
    stage.bucket.stamp()
    stage.bucket.seen[key] = (stage.bucket.seen[key][0], closed["closed_at"] + 1500, stage.bucket.seen[key][2])  # the fold took 25 minutes
    doc, limit = stage.poll(), stage.world.params.round_time_limit_s
    takes = doc["round_time"]
    assert (takes["limit_s"], takes["king_extra_s"]) == (limit, 1500) and takes["king_rounds"] == [{"round_id": closed["round_id"], "extra_s": 1500}]
    words = f"up to about {round(limit / 60)} minutes, about {round((limit + 1500) / 60)} minutes with a king"
    assert takes["words"] == words and f"({words})" in doc["labels"]["no_verdict"] and f"({words}; see round_time)" in doc["method"]
    assert json.loads(stage.record.read_bytes())["round_files"] == {closed["round_id"]: closed["closed_at"] + 1500}
    assert any("a round takes " + words in line for line in supply.report(doc))
    first_form = json.loads(stage.record.read_bytes())
    del first_form["round_files"]  # a record of before this change
    stage.record.write_bytes(json.dumps(first_form).encode())
    assert stage.poll()["round_time"] == takes
    # No time of a round is a constant: not in the recorder, and not in the page. The page takes the words from the public file.
    source, page, script = ((SITE / name).read_text() for name in ("supply.py", "static/supply.html", "static/site.js"))
    for stale in ("90 minutes", "100 minutes", "137 minutes", "180 minutes", "227 minutes", "5400", "6000", "7200", "10800"):
        assert stale not in source and stale not in page
    assert 'data-f="round-time"' in page  # the page fills the words from the public file; no fixed minutes in the HTML
    assert "el.querySelectorAll('[data-f=round-time]').forEach(x => { x.textContent = S.round_time.words; });" in script


def test_the_public_file_has_counts_only(stage):
    stage.upload(11)
    stage.upload(12, valid=False)
    stage.world.round()
    stage.wait(60)
    text = json.dumps(stage.poll())
    record = stage.record.read_text()
    for private in ("hk11", "hk12", stage.entry(12)["etag"], stage.entry(11)["sha"]):
        assert private in record and private not in text
    assert set(stage.doc["labels"]) == {"uploads", "validated", "no_verdict", "not_valid", "replaced", "waiting"}
    heads = {"uploads": "Uploads per hour from R2 LastModified: ", "validated": "Validated so far: ", "no_verdict": "Uploads with no verdict yet: "}
    assert all(stage.doc["labels"][name].startswith(head) for name, head in heads.items()), "the operator's three names, word for word"
    assert "at the time the round took it" in stage.doc["labels"]["uploads"] and "at the time the round took it" in stage.doc["method"]
    for row in [*stage.doc["windows"], *stage.doc["hours"]]:
        assert row["uploads"] == row["validated"] + row["no_verdict"] + row["not_valid"] + row["replaced"] and row["at_round_time"] <= row["uploads"]
    assert stage.doc["clock"]["now"] == "host" and stage.doc["schema"] == supply.SCHEMA


@pytest.mark.parametrize("utc", ["2026-11-14 12:43:26", "2026-11-14 05:59:59", "2026-07-01 00:05:00", "2027-01-01 03:00:00"])
def test_a_time_in_words_is_in_utc(utc):
    t = datetime.strptime(utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    assert supply.utc_text(t) == utc + " UTC"


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


def test_the_command_reads_the_store_with_list_calls_and_writes_the_two_files(endpoint, tmp_path):
    s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=ACCESS, aws_secret_access_key=SECRET)
    s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "auto"})
    world = sim.SimWorld(tmp_path / "validator", sim.sim_params())
    world.register(11, 0.5)
    world.register(12, 0.5, valid=False)
    world.round()

    def mirror() -> None:  # the validator's store and its snapshot, in the bucket
        for key in world.store.list("private/"):
            s3.put_object(Bucket=BUCKET, Key=key, Body=world.store.get(key))
        snap = json.loads(json.dumps(world.publisher.snapshots[-1]))
        # The sim's clock is not the bucket's clock. Put the check after the uploads, as it was in the sim.
        snap["queue"]["files_checked_at"] = int(time.time()) + supply.SKEW_S + 1
        s3.put_object(Bucket=BUCKET, Key=layout.MINING_JSON, Body=json.dumps(snap).encode())
        s3.put_object(Bucket=BUCKET, Key=supply.MINERS + "hk11/other.json", Body=b"not a miner file")

    mirror()
    site, record = tmp_path / "site", tmp_path / "record"
    site.mkdir()
    env = {"PATH": os.environ["PATH"], "R2_ENDPOINT": endpoint, "R2_BUCKET": BUCKET, "R2_ACCESS_KEY_ID": ACCESS, "R2_SECRET_ACCESS_KEY": SECRET}

    def command(*args: str) -> subprocess.CompletedProcess:
        done = subprocess.run([sys.executable, str(SITE / "supply.py"), "--out", str(site / "miner-supply.json"), *args],
                              env=env, capture_output=True, text=True, timeout=120)
        assert ACCESS not in done.stdout + done.stderr and SECRET not in done.stdout + done.stderr
        return done

    first = command("--state-dir", str(record))
    assert first.returncode == 0, first.stderr
    doc = json.loads((site / "miner-supply.json").read_bytes())
    day = doc["windows"][-1]
    assert numbers(day) == (2, 1, 0, 1, 0) and doc["clock"]["now"] == "store" and abs(doc["clock"]["host_minus_store_s"]) <= 5
    assert stat.S_IMODE((site / "miner-supply.json").stat().st_mode) == 0o644
    assert stat.S_IMODE(record.stat().st_mode) == 0o700 and stat.S_IMODE((record / supply.STATE).stat().st_mode) == 0o600
    mirror()  # the same bytes stored again: a new LastModified, the same ETag
    second = command("--state-dir", str(record))
    assert second.returncode == 0 and numbers(json.loads((site / "miner-supply.json").read_bytes())["windows"][-1]) == (2, 1, 0, 1, 0)
    assert "2 uploads in 24 h, 1 validated so far" in second.stdout
    shown = command("--report")
    assert shown.returncode == 0 and " UTC; " in shown.stdout and "last 24 h: uploads 2 | validated so far 1" in shown.stdout
    assert "hk11" not in shown.stdout + first.stdout + second.stdout

    env["R2_SECRET_ACCESS_KEY"] = SECRET  # a store that does not answer: nothing is written, and no address is printed
    env["R2_ENDPOINT"] = "http://127.0.0.1:9"
    before = (site / "miner-supply.json").read_bytes(), (record / supply.STATE).read_bytes()
    down = command("--state-dir", str(record))
    assert down.returncode == 1 and "127.0.0.1" not in down.stderr and "nothing written" in down.stderr
    assert before == ((site / "miner-supply.json").read_bytes(), (record / supply.STATE).read_bytes())
