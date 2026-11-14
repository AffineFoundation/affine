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
from moto.server import ThreadedMotoServer

from sn120_v2 import layout, sim
from sn120_v2.public import QUEUE_PREVIEW

SITE = Path(sim.__file__).resolve().parents[1] / "site"
supply = SimpleNamespace(**runpy.run_path(str(SITE / "supply.py"), run_name="supply"))  # a script of v2/site, not a module of the package

ACCESS, SECRET, BUCKET = "supply-test-access-key", "supply-test-secret-key-0123456789", "sn120-supply-test"


class Bucket:
    """The list call on the sim's store. ``stamp`` gives each new or changed object the sim's time as its ``LastModified``."""

    def __init__(self, world: sim.SimWorld) -> None:
        self.world, self.seen, self.date = world, {}, None
        self.snapshot_extra: dict | None = None

    def stamp(self) -> None:
        keys = self.world.store.list("private/")
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
    assert stage.doc["clock"]["now"] == "host" and stage.doc["schema"] == supply.SCHEMA


@pytest.mark.parametrize("utc, words", [
    ("2026-11-14 12:43:26", "6:43 AM Costa Rica time, 14 Nov"),
    ("2026-11-14 18:00:00", "12:00 PM Costa Rica time, 14 Nov"),
    ("2026-11-14 06:00:00", "12:00 AM Costa Rica time, 14 Nov"),
    ("2026-11-14 05:59:59", "11:59 PM Costa Rica time, 13 Nov"),
    ("2026-07-01 00:05:00", "6:05 PM Costa Rica time, 30 Jun"),  # no daylight saving time in Costa Rica
    ("2027-01-01 03:00:00", "9:00 PM Costa Rica time, 31 Dec"),
])
def test_costa_rica_time_is_utc_minus_six_hours_all_year(utc, words):
    t = datetime.strptime(utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    assert supply.costa_rica(t) == words
    assert supply.utc_text(t) == utc + " UTC"
    assert supply.both(t) == f"{utc} UTC ({words})"


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
    assert shown.returncode == 0 and "Costa Rica time" in shown.stdout and "last 24 h: uploads 2 | validated so far 1" in shown.stdout
    assert "hk11" not in shown.stdout + first.stdout + second.stdout

    env["R2_SECRET_ACCESS_KEY"] = SECRET  # a store that does not answer: nothing is written, and no address is printed
    env["R2_ENDPOINT"] = "http://127.0.0.1:9"
    before = (site / "miner-supply.json").read_bytes(), (record / supply.STATE).read_bytes()
    down = command("--state-dir", str(record))
    assert down.returncode == 1 and "127.0.0.1" not in down.stderr and "nothing written" in down.stderr
    assert before == ((site / "miner-supply.json").read_bytes(), (record / supply.STATE).read_bytes())
