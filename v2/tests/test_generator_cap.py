"""Per-hotkey scoring-capacity formula, immutable issuance and batch enforcement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nacl.signing import SigningKey

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.generator import cap
from sn120_v2.generator.config import Config
from sn120_v2.generator.control import issue_links, publish_epoch, snapshot_epoch, update_window_state, write_seed
from sn120_v2.generator.epoch import epoch_deadline, epoch_start, submission_object
from sn120_v2.generator.links import link_files
from sn120_v2.generator.protocol import sign_submission
from sn120_v2.generator.constants import POOL_ID
from sn120_v2.store import LocalStore
from test_generator_control import address, hotkey, tid
from test_generator_links import FakePresigner, open_sealed
from test_generator_publish import Pool

EPOCH = "20261120T2000Z"
NEXT = "20261121T0000Z"
DEADLINE = epoch_deadline(EPOCH).timestamp()
START = epoch_start(EPOCH).timestamp()
INPUTS = {"first_epoch": EPOCH, "rollouts_per_pod_s": 3.15, "pods": 1}


def inputs(**changes):
    return {**INPUTS, **changes}


def test_formula_defaults_clamps_and_helper_pods():
    parsed = cap.cap_inputs({key: value for key, value in INPUTS.items() if key != "pods"}, ssh={"helpers": [{}, {}]})
    assert parsed["pods"] == 3
    assert parsed["scoring_budget_s"] == 13500
    assert parsed["king_swap_s"] == 900
    assert parsed["safety_margin"] == 0.25
    assert parsed["samples_per_task"] == 8
    assert parsed["min_cap"] == 1 and parsed["max_cap"] == 32
    parsed = cap.cap_inputs(INPUTS, ssh={"helpers": [{}, {}]})
    assert cap.capacity_tasks(parsed) == 3720
    assert cap.cap_for(parsed, 1000) == (3, 3720)
    assert cap.cap_for(parsed, 100_000) == (1, 3720)
    assert cap.cap_for(parsed, 1) == (32, 3720)
    assert cap.cap_inputs(INPUTS, ssh=None)["pods"] == 1
    assert cap.cap_inputs({**INPUTS, "pods": 2}, ssh={"helpers": [{}]})["pods"] == 2
    assert cap.cap_inputs(None) is None


def test_pre_first_epoch_and_frozen_cap_never_change(tmp_path):
    parsed = cap.cap_inputs(inputs())
    assert cap.freeze(tmp_path, "20261120T1600Z", parsed, 2) is None
    assert cap.freeze(tmp_path, "20261120T1600Z", parsed, 1) is None
    first = cap.freeze(tmp_path, EPOCH, parsed, 1000, now=START)
    assert first["cap"] == 3 and first["active_hotkeys"] == 1000
    again = cap.freeze(tmp_path, EPOCH, parsed, 1, now=START + 3600)
    assert again == first
    assert cap.frozen(tmp_path, EPOCH) == first


def test_links_public_and_sealed_cap_without_presigned_material():
    miner, key = hotkey("cap-links")
    now = int(DEADLINE) - 300
    files = link_files([miner], FakePresigner(), now, caps={EPOCH: 13, NEXT: 7})
    document = json.loads(files[f"generator/links/{miner}.json"])
    assert document["max_tasks"] == 13 and document["next_max_tasks"] == 7
    assert open_sealed(type("Miner", (), {"key": key})(), document)["max_tasks"] == 13
    assert open_sealed(type("Miner", (), {"key": key})(), document, "next")["max_tasks"] == 7
    index = json.loads(files["generator/links.json"])
    assert index["max_tasks"] == 13 and index["next_max_tasks"] == 7
    assert "put_url" not in document and "X-Amz" not in json.dumps(document)


def make_config(tmp_path, submission_cap=INPUTS):
    site = tmp_path / "site"
    site.mkdir()
    return Config(site, tmp_path / "work", tmp_path / "pool", first_epoch="20261120T0000Z", submission_cap=submission_cap)


def issue(config, now, registered, activated, store, *, force=False):
    from sn120_v2.generator import control
    original = control.read_registrations
    control.read_registrations = lambda client, clock: (registered, activated)
    try:
        return issue_links(config, now=now, client=None, store=store, force=force)
    finally:
        control.read_registrations = original


def test_first_issuance_freezes_current_and_next_cap_and_state(tmp_path):
    config = make_config(tmp_path)
    a, ka = hotkey("cap-issue-a")
    b, kb = hotkey("cap-issue-b")
    store = LocalStore(tmp_path / "store")
    store.presign_get = lambda key, ttl: "https://fake.invalid/read"
    store.presign_put = lambda key, ttl: ("https://fake.invalid/write", {})
    now = DEADLINE - 300
    result = issue(config, now, {1: a, 2: b}, {a, b}, store, force=True)
    assert result["count"] == 2
    current = cap.frozen(config.work_dir, EPOCH)
    following = cap.frozen(config.work_dir, NEXT)
    assert current["active_hotkeys"] == 2 and following["active_hotkeys"] == 2
    assert current["cap"] == 32 and following["cap"] == 32
    # Later registration changes do not change either immutable record.
    c, kc = hotkey("cap-issue-c")
    issue(config, now + 600, {1: a, 2: b, 3: c}, {a, b, c}, store, force=True)
    assert cap.frozen(config.work_dir, EPOCH) == current
    assert cap.frozen(config.work_dir, NEXT) == following
    write_seed(config.work_dir, EPOCH)
    update_window_state(config, epoch=EPOCH, now=now, store=store)
    state = json.loads((config.site_dir / "generator/state.json").read_text())
    assert state["submission_cap"]["epoch"] == EPOCH
    assert state["next_submission_cap"]["epoch"] == NEXT
    index = json.loads((config.site_dir / "generator/links.json").read_text())
    assert index["max_tasks"] == 32
    assert "next_max_tasks" not in index


def upload(store, epoch, hotkey_value, key, count, when):
    ids = [f"{i:020x}" for i in range(1, count + 1)]
    submission = sign_submission(epoch, hotkey_value, ids, lambda message: key.sign(message).signature, require_activation=False)
    path = submission_object(epoch, hotkey_value)
    store.put(path, submission.canonical_json())
    import os
    os.utime(store._path(path), (when, when))
    return submission


def test_pre_first_epoch_1600_is_uncapped(tmp_path):
    config = make_config(tmp_path)
    a, ka = hotkey("cap-pre-first")
    store = LocalStore(tmp_path / "store")
    store.presign_get = lambda key, ttl: "https://fake.invalid/read"
    store.presign_put = lambda key, ttl: ("https://fake.invalid/write", {})
    epoch = "20261120T1600Z"
    now = epoch_deadline(epoch).timestamp() - 300
    result = issue(config, now, {1: a}, {a}, store, force=True)
    assert result["count"] == 1
    assert cap.frozen(config.work_dir, epoch) is None
    document = json.loads((config.site_dir / f"generator/links/{a}.json").read_text())
    assert "max_tasks" not in document and "max_tasks" not in json.loads((config.site_dir / "generator/links.json").read_text())


def test_snapshot_at_cap_accepted_and_cap_plus_one_rejected_whole(tmp_path):
    record = cap.freeze(tmp_path, EPOCH, cap.cap_inputs(inputs(min_cap=13)), 1000)
    assert record["cap"] == 13
    a, ka = hotkey("cap-snapshot-a")
    b, kb = hotkey("cap-snapshot-b")
    store = LocalStore(tmp_path / "store")
    upload(store, EPOCH, a, ka, 13, START + 1)
    upload(store, EPOCH, b, kb, 14, START + 1)
    from sn120_v2.generator.snapshot import build_snapshot
    manifest, job = build_snapshot(store, epoch=EPOCH, registered={1: a, 2: b}, activations={a, b}, known_task_ids=lambda task_id: int(task_id, 16) <= 32, previously_accepted=set(), king="head", seed_commitment="ab" * 32, submission_cap=record)
    assert len(job["tasks"]) == 13 and all(row["hotkey"] == a for row in job["tasks"])
    rejected = next(row for row in manifest["files"] if row["hotkey"] == b)
    assert rejected["valid"] is False and rejected["cap_rejected"] is True
    assert rejected["reason"] == cap.rejection_reason(14, EPOCH, 13)
    assert all(b not in winners for winners in job["tasks"])
    uncapped_manifest, uncapped_job = build_snapshot(store, epoch=EPOCH, registered={1: a, 2: b}, activations={a, b}, known_task_ids=lambda task_id: int(task_id, 16) <= 32, previously_accepted=set(), king="head", seed_commitment="ab" * 32)
    assert len(uncapped_job["tasks"]) == 14


def test_public_results_rejections_cap_payout_and_no_claims(tmp_path):
    from sn120_v2.generator.publish import publication
    secret = "private-cap-test"
    record = cap.freeze(tmp_path, EPOCH, cap.cap_inputs(inputs(min_cap=13)), 1000)
    job = {
        "epoch": EPOCH, "pool": POOL_ID,
        "king": "head", "seed_commitment": sha256_hex(secret.encode()),
        "submission_cap": cap.public(record),
        "tasks": [{"task_id": "01" * 10, "hotkey": "hk", "uid": 1, "uploaded_at": START + 1, "file_sha256": sha256_hex(b"one"), "head_version": "head"}],
    }
    output = {"epoch": EPOCH, "king": "head", "seed_commitment": job["seed_commitment"], "tasks": [{"task_id": "01" * 10, "s": 4, "n": 8, "q": 0.5, "decision": "accept"}], "rollouts": {"01" * 10: [{"text": "rollout", "seed": 1, "pass": True}]}}
    from sn120_v2.generator.decisions import decision
    output["tasks"][0]["q"] = decision(4, 8).q
    plan = publication(job, output, out_dir=tmp_path, secret=secret, history=[], registered={1: "hk", 2: "over"}, now=DEADLINE + 1, pool_rows=Pool(), frozen_at=DEADLINE, cap_rejections=[{"hotkey": "over", "reason": cap.rejection_reason(14, EPOCH, 13), "cap_rejected": True}])
    assert plan["results"]["submission_cap"]["cap"] == 13
    assert plan["results"]["submissions"] == [{"hotkey": "over", "reason": cap.rejection_reason(14, EPOCH, 13)}]
    assert "task_id" not in json.dumps(plan["results"]["submissions"])
    assert len(plan["claims"]) == 1 and plan["payout"]["weights"] == {"1": 1.0}


def test_control_plane_snapshot_and_publish_cap_rejection_and_payout(tmp_path):
    from sn120_v2.generator.decisions import decision
    parsed = cap.cap_inputs(inputs(min_cap=13))
    config = make_config(tmp_path, parsed)
    record = cap.freeze(config.work_dir, EPOCH, parsed, 1000, now=START)
    assert record["cap"] == 13
    seed = write_seed(config.work_dir, EPOCH)
    write_seed(config.work_dir, NEXT)
    king = config.work_dir / "king" / f"{EPOCH}.json"
    king.parent.mkdir(parents=True)
    king.write_text(json.dumps({"epoch": EPOCH, "version": "head", "captured_at_epoch": DEADLINE, "captured_at": "2026-11-20T20:00:00Z"}))
    a, ka = hotkey("cap-control-a")
    b, kb = hotkey("cap-control-b")
    store = LocalStore(tmp_path / "store")
    upload(store, EPOCH, a, ka, 13, START + 1)
    upload(store, EPOCH, b, kb, 14, START + 1)
    from sn120_v2.generator import control
    control.read_registrations = lambda client, clock: ({1: a, 2: b}, {a, b})
    assert snapshot_epoch(config, epoch=EPOCH, now=DEADLINE + 60, client=None, store=store, pool={f"{i:020x}" for i in range(1, 33)})["status"] == "snapshot"
    out = config.work_dir / "epochs" / EPOCH / "out"
    rollout = {"text": "accepted", "seed": 0, "pass": True}
    (out / "rollouts").mkdir(parents=True)
    for row in json.loads((config.work_dir / "jobs" / EPOCH / "evidence.json").read_text())["tasks"]:
        (out / "rollouts" / f"{row['task_id']}.jsonl").write_text(json.dumps(rollout) + "\n")
    (out / "results.json").write_text(json.dumps({
        "epoch": EPOCH, "king": "head", "seed_commitment": seed["seed_commitment"],
        "tasks": [{"task_id": row["task_id"], "s": 4, "n": 8, "q": decision(4, 8).q, "decision": "accept"} for row in json.loads((config.work_dir / "jobs" / EPOCH / "evidence.json").read_text())["tasks"]],
    }))
    result = publish_epoch(config, epoch=EPOCH, now=DEADLINE + 60, client={1: a, 2: b}, store=store, pool=Pool())
    assert result["accepted"] == 13
    public = json.loads((config.site_dir / "generator" / "epochs" / EPOCH / "results.json").read_text())
    assert public["submission_cap"]["cap"] == 13
    assert public["submissions"] == [{"hotkey": b, "reason": cap.rejection_reason(14, EPOCH, 13)}]
    assert len((config.site_dir / "generator" / "accepted-all.jsonl").read_text().splitlines()) == 13
    assert json.loads((config.site_dir / "generator" / "payout.json").read_text())["weights"] == {"1": 1.0}


def test_config_optional_cap_parsed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"site_dir": "site", "work_dir": "work", "pool_dir": "pool", "network": "finney", "public_url": "https://affine.io", "credentials_file": "", "head_url": "https://affine.io/public/head.json", "first_epoch": "20261120T0000Z", "submission_cap": INPUTS}))
    config = Config.load(path)
    assert config.submission_cap == INPUTS
    path.write_text(json.dumps({"site_dir": "site", "work_dir": "work", "pool_dir": "pool", "network": "finney", "public_url": "https://affine.io", "credentials_file": "", "head_url": "https://affine.io/public/head.json", "first_epoch": "20261120T0000Z"}))
    assert Config.load(path).submission_cap is None
