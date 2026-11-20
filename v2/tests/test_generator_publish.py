"""Publication, intake contract, atomic retries and once-ever payout."""
from __future__ import annotations

import json
import math

import pytest

from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import VerifyError
from sn120_v2.generator.common import PublicSink, atomic_json, json_bytes, jsonl_bytes, read_json, utc_text
from sn120_v2.generator.decisions import decision
from sn120_v2.generator.epoch import epoch_deadline, epoch_id, epoch_start
from sn120_v2.generator.payout import Claim, payout_weights, trailing_claims
from sn120_v2.generator.publish import publication, publish
from sn120_v2.store import LocalStore

EPOCH = "20261120T0000Z"
START = epoch_start(EPOCH).timestamp()
DEADLINE = epoch_deadline(EPOCH).timestamp()
SECRET = "test secret, never committed in an operator record"
COMMITMENT = sha256_hex(SECRET.encode())


def task_id(n):
    return f"{n:020x}"


class Pool:
    def row(self, task_id):
        return {"problem": f"question {task_id}", "answer": "42", "source_id": f"prior-{task_id}"}


def make_job(n=2):
    return {
        "epoch": EPOCH, "pool": "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115",
        "king": "frozen-head", "seed_commitment": COMMITMENT,
        "tasks": [
            {"task_id": task_id(i), "hotkey": "hk-a" if i == 1 else "hk-b", "uid": i,
             "uploaded_at": START + i, "file_sha256": sha256_hex(str(i).encode()), "head_version": "frozen-head"}
            for i in range(1, n + 1)
        ],
        "counts": {"hk-a": {"submitted": 2, "eligible": 1}, "hk-b": {"submitted": 1, "eligible": 1}},
    }


def make_output(n=2, passes=4, total=8):
    return {
        "epoch": EPOCH, "king": "frozen-head", "seed_commitment": COMMITMENT,
        "tasks": [
            {"task_id": task_id(i), "s": passes, "n": total, "q": decision(passes, total).q, "decision": decision(passes, total).action}
            for i in range(1, n + 1)
        ],
        "rollouts": {
            task_id(i): [{"text": f"full output {i}", "seed": i, "pass": True}]
            for i in range(1, n + 1) if decision(passes, total).accepted
        },
    }


def test_publication_accepts_grace_capture_and_rejects_late_king(tmp_path):
    publication(make_job(), make_output(), out_dir=tmp_path, secret=SECRET, history=[], registered={1: "hk-a", 2: "hk-b"}, now=DEADLINE + 1, pool_rows=Pool(), frozen_at=DEADLINE + 120)
    with pytest.raises(VerifyError, match="grace"):
        publication(make_job(), make_output(), out_dir=tmp_path, secret=SECRET, history=[], registered={1: "hk-a", 2: "hk-b"}, now=DEADLINE + 1, pool_rows=Pool(), frozen_at=DEADLINE + 121)


def test_publication_has_only_four_intake_fields_and_reveals_secret(tmp_path):
    plan = publication(
        make_job(), make_output(), out_dir=tmp_path, secret=SECRET, history=[], registered={1: "hk-a", 2: "hk-b"},
        now=DEADLINE + 1, pool_rows=Pool(), frozen_at=DEADLINE + 60,
    )
    assert len(plan["accepted"]) == 2
    assert all(set(row) == {"problem", "answer", "source_id", "prior_p"} for row in plan["intake"])
    assert all(row["prior_p"] == 0.5 for row in plan["intake"])
    assert plan["results"]["seed_secret"] == SECRET
    assert set(plan["rollouts"]) == {task_id(1), task_id(2)}
    assert all("id" not in row and "hotkey" not in row and "king" not in row for row in plan["intake"])
    assert all(row["task_id"] in plan["rollouts"] for row in plan["accepted"])


def test_rejected_and_unscored_public_rows_are_hashed_and_text_stays_private(tmp_path):
    job = make_job(3)
    output = make_output(3)
    output["tasks"][0] = {"task_id": task_id(1), "s": 4, "n": 8, "q": decision(4, 8).q, "decision": "accept"}
    output["tasks"][1] = {"task_id": task_id(2), "s": 0, "n": 8, "q": decision(0, 8).q, "decision": "reject"}
    output["tasks"][2] = {"task_id": task_id(3), "s": 2, "n": 4, "q": None, "decision": "unscored"}
    output["rollouts"] = {
        task_id(i): [{"text": f"private generated text {i}", "seed": i, "pass": i == 1}]
        for i in range(1, 4)
    }
    private_output = tmp_path / "private-out"
    (private_output / "rollouts").mkdir(parents=True)
    for i in range(1, 4):
        (private_output / "rollouts" / f"{task_id(i)}.jsonl").write_text(json.dumps(output["rollouts"][task_id(i)][0]) + "\n")
    plan = publication(job, output, out_dir=private_output, secret=SECRET, history=[], registered={1: "hk-a", 2: "hk-b", 3: "hk-c"}, now=DEADLINE + 60, pool_rows=Pool(), frozen_at=DEADLINE + 60)
    accepted_row, rejected_row, unscored_row = plan["results"]["tasks"]
    assert accepted_row["task_id"] == task_id(1)
    assert accepted_row["rollouts"].endswith(f"/{task_id(1)}.jsonl")
    assert accepted_row["decision"] == "accept" and accepted_row["s"] == 4 and accepted_row["n"] == 8
    assert rejected_row == {"task_sha256": sha256_hex(task_id(2).encode()), "decision": "reject", "n": 8, "s": 0}
    assert unscored_row == {"task_sha256": sha256_hex(task_id(3).encode()), "decision": "unscored", "n": 4, "s": 2, "unscored_reason": "time_budget"}
    assert set(plan["rollouts"]) == {task_id(1)}
    assert task_id(2).encode() not in json_bytes(plan["results"])
    assert task_id(3).encode() not in json_bytes(plan["results"])
    assert b"private generated text 2" not in b"".join(plan["rollouts"].values())
    assert (private_output / "rollouts" / f"{task_id(2)}.jsonl").is_file()


def test_publish_is_atomic_idempotent_and_writes_intake_last(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    mirror = LocalStore(tmp_path / "r2")
    sink = PublicSink(root, mirror, "https://site.test")
    events = []
    original = sink.put
    def tracked(relative, raw):
        events.append(relative)
        original(relative, raw)
    sink.put = tracked
    now = DEADLINE + 60
    job, output = make_job(), make_output()
    first = publish(sink=sink, job=job, output=output, out_dir=tmp_path, secret=SECRET, registered={1: "hk-a", 2: "hk-b"}, now=now, pool_rows=Pool(), frozen_at=DEADLINE)
    relative = f"epochs/{EPOCH}"
    assert events.index(f"{relative}/intake.jsonl") > events.index(f"{relative}/accepted.jsonl")
    assert events.index(f"{relative}/intake.jsonl") > events.index(f"{relative}/results.json")
    assert events[-1] == "state.json"
    expected = sha256_hex(sink.read(f"{relative}/intake.jsonl"))
    assert sink.read(f"{relative}/intake.sha256") == f"{expected}  intake.jsonl\n".encode()
    state = read_json(sink.path("state.json"))
    assert state["published_epochs"] == [{"epoch": EPOCH, "results": f"https://site.test/generator/{relative}/results.json", "intake": f"https://site.test/generator/{relative}/intake.jsonl", "intake_sha256": expected}]
    assert mirror.get(f"public/generator/{relative}/intake.jsonl") == sink.read(f"{relative}/intake.jsonl")
    before = list((root / "generator").rglob("*"))
    snapshot = {path: path.read_bytes() for path in before if path.is_file()}
    publish(sink=sink, job=job, output=output, out_dir=tmp_path, secret=SECRET, registered={1: "hk-a", 2: "hk-b"}, now=now + 1, pool_rows=Pool(), frozen_at=DEADLINE)
    assert {path: path.read_bytes() for path in snapshot} == snapshot
    assert len(sink.read("accepted-all.jsonl").splitlines()) == 2
    payout = read_json(sink.path("payout.json"))
    assert payout["weights"] == {"1": 0.5, "2": 0.5}


def test_empty_epoch_still_publishes_empty_intake_and_burns(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    sink = PublicSink(root)
    job = make_job(0)
    job["counts"] = {}
    plan = publish(
        sink=sink, job=job, output=make_output(0), out_dir=tmp_path, secret=SECRET,
        registered={1: "hk-a"}, now=DEADLINE, pool_rows=Pool(), frozen_at=DEADLINE,
    )
    relative = f"epochs/{EPOCH}"
    assert sink.read(f"{relative}/intake.jsonl") == b""
    assert sink.read(f"{relative}/accepted.jsonl") == b""
    assert read_json(sink.path("payout.json"))["weights"] == {"0": 1.0}
    assert read_json(sink.path("state.json"))["published_epochs"][0]["intake_sha256"] == sha256_hex(b"")


def test_scorer_must_cover_job_and_reproduce_decision(tmp_path):
    output = make_output()
    output["tasks"][0]["s"] = 7
    with pytest.raises(VerifyError):
        publication(make_job(), output, out_dir=tmp_path, secret=SECRET, history=[], registered={1: "hk-a", 2: "hk-b"}, now=DEADLINE, pool_rows=Pool(), frozen_at=DEADLINE)
    with pytest.raises(VerifyError):
        publication(make_job(), make_output(), out_dir=tmp_path, secret="wrong secret", history=[], registered={1: "hk-a", 2: "hk-b"}, now=DEADLINE, pool_rows=Pool(), frozen_at=DEADLINE)


def test_trailing_window_boundary_is_exclusive_and_once_ever_is_permanent():
    now = 1_000_000.0
    claim = Claim(task_id(1), 1, "hk-a", now - 86400, sha256_hex(b"one"))
    just_inside = Claim(task_id(2), 1, "hk-a", now - 86400 + 0.001, sha256_hex(b"two"))
    future = Claim(task_id(3), 1, "hk-a", now + 1, sha256_hex(b"three"))
    assert trailing_claims([claim, just_inside, future], now) == (just_inside,)
    assert payout_weights([claim, just_inside], {1: "hk-a"}, now=now, updated_at=now) == {1: 1.0}
    old = [claim, Claim(task_id(4), 2, "hk-b", now - 90000, sha256_hex(b"four"))]
    assert payout_weights(old, {1: "hk-a", 2: "hk-b"}, now=now, updated_at=now) == {0: 1.0}


def test_changed_hotkey_and_stale_payout_burn():
    claim = Claim(task_id(1), 1, "hk-a", 999_999, sha256_hex(b"one"))
    assert payout_weights([claim], {1: "hk-new"}, now=1_000_000, updated_at=1_000_000) == {0: 1.0}
    assert payout_weights([claim], {1: "hk-a"}, now=1_000_000, updated_at=1_000_000 - 21601) == {0: 1.0}
