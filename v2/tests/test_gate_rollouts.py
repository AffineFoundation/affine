"""Capture parity and publication of every scored merged-model eval."""
import dataclasses
import gzip
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.errors import InfraError
from sn120_v2.gpu import plan
from sn120_v2.gpu.evaluator import ServeSettings, VllmEvaluator
from sn120_v2.rollouts import GateRolloutPublisher, RolloutCapture, capture_file, deterministic_gzip, published_rows


def test_gzip_hash_is_of_the_complete_closed_file(tmp_path):
    path = tmp_path / "one.gz"
    first = deterministic_gzip(path, [dict(text="x" * 10000)])
    assert first[1] == path.stat().st_size and first[2] == hashlib.sha256(path.read_bytes()).hexdigest()
    second = deterministic_gzip(tmp_path / "two.gz", [dict(text="x" * 10000)])
    assert first == second and path.read_bytes() == (tmp_path / "two.gz").read_bytes()

from sn120_v2.store import LocalStore
from sn120_v2.types import HeadVersion, NULL_MODEL, EvalRequest

POOL = SimpleNamespace(task=lambda task_id: SimpleNamespace(messages=lambda: [{"role": "user", "content": task_id}], answer="4"))
GRADES = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code="P" if text.startswith("P") else "F", reward=1.0 if text.startswith("P") else 0.0,
                                                                  answer="\\boxed{4}" if text.startswith("P") else "\\boxed{5}", machine_cause=None) for _, text, _ in jobs])


class Server:
    def spawn(self, argv, env, log):
        return SimpleNamespace(alive=lambda: True, stop=lambda: None)

    def http(self, url, body, timeout):
        if not url.endswith("chat/completions"):
            return {"data": [{"id": NULL_MODEL}, *({"id": lora.lora_hash} for lora in LORAS)]}
        return dict(choices=[dict(message={"content": "P \\boxed{4}" if body["model"] == NULL_MODEL else "F \\boxed{5}"},
                                  finish_reason="stop")], usage={"completion_tokens": 17})


LORAS = []


def request(build, head):
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_gpus=1, n_tasks=2))
    LORAS[:] = [build.artifact(head, entry, ticket.params) for entry in ticket.entries]
    return ticket, EvalRequest(ticket.round_id, head, tuple(LORAS), ticket.draw.task_ids, "seed-0", 0, ticket.params)


def test_outcomes_identical_with_and_without_capture_and_metadata_is_recorded(tmp_path, build):
    store = LocalStore(tmp_path / "store")
    base, master = manifest_digest({"model.bin": sha256_hex(b"bf16")}), manifest_digest({"master.bin": sha256_hex(b"fp32")})
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    store.put(f"{layout.public_base_prefix(base)}/model.bin", b"bf16")
    store.put(f"{layout.public_master_prefix(master)}/master.bin", b"fp32")
    config = plan.BoxConfig(tmp_path / "work", build.image)
    ticket, req = request(build, head)
    for lora in req.loras:
        for name, sha in lora.files.items():
            store.put(f"{lora.prefix}/{name}", sha.encode())
    server = Server()
    kwargs = dict(spawn=server.spawn, http=server.http, gpus_free=lambda *args: None, settings=ServeSettings(start_timeout_s=1), room=lambda sockets, spare: sockets)
    with_capture = VllmEvaluator(store, config, POOL, GRADES, **kwargs)
    # Cache verification is covered by the existing box tests; this fixture
    # isolates reply capture and scoring parity.
    local_lora = tmp_path / "lora"
    local_lora.mkdir()
    with_capture.resident.tree = lambda prefix, identity: local_lora
    outcome = with_capture.evaluate(req)
    with_capture.capture.close()
    rows = []
    for model in (NULL_MODEL, *(lora.lora_hash for lora in req.loras)):
        path = tmp_path / "work" / "rollouts" / req.round_id / "captures" / f"{model}.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    assert all(row["completion_text"] in ("P \\boxed{4}", "F \\boxed{5}") for row in rows)
    assert all(row["output_tokens"] == 17 and row["finish_reason"] == "stop" for row in rows)
    assert all(row["outcome"] in ("P", "F") for row in rows)
    assert len(rows) == 6
    with_capture.close()
    without_capture = VllmEvaluator(store, config, POOL, GRADES, **kwargs)
    without_capture.capture = None
    without_capture.resident.tree = lambda prefix, identity: local_lora
    assert without_capture.evaluate(req).to_json() == outcome.to_json()
    without_capture.close()
    broken = VllmEvaluator(store, config, POOL, GRADES, **kwargs)
    broken.resident.tree = lambda prefix, identity: local_lora
    broken.capture.graded = lambda *args: (_ for _ in ()).throw(OSError("full disk"))
    broken.capture.reply = lambda *args: (_ for _ in ()).throw(OSError("capture always broken"))
    assert broken.evaluate(req).to_json() == outcome.to_json()
    broken.close()


def make_round_captures(ticket, result, work, missing=()):
    root = work / "rollouts" / ticket.round_id / "captures"
    root.mkdir(parents=True, exist_ok=True)
    head = root / f"{NULL_MODEL}.jsonl"
    with head.open("wt") as handle:
        for sample, text in enumerate(result.null_outcomes):
            for pos, code in enumerate(text):
                handle.write(json.dumps(dict(round_id=ticket.round_id, sample_index=sample, task_pos=pos, outcome=code,
                                             prompt="head prompt", completion_text="head text", extracted_answer="4")) + "\n")
    for entry in result.entries:
        if not entry.outcomes or entry.queue_pos in missing:
            continue
        path = root / f"{entry.lora.lora_hash}.jsonl"
        with path.open("wt") as handle:
            for sample, text in enumerate(entry.outcomes):
                for pos, code in enumerate(text):
                    handle.write(json.dumps(dict(round_id=ticket.round_id, sample_index=sample, task_pos=pos, outcome=code,
                                                 prompt="entry prompt", completion_text=f"entry {entry.queue_pos} public text",
                                                 extracted_answer="4")) + "\n")
    return root


def test_settle_publishes_every_entry_and_deletes_captures_after_meta_last(tmp_path, build):
    ticket = build.ticket(n_entries=3, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF", 2: "PF"})
    winner_uid = 10
    store = LocalStore(tmp_path / "store")
    root = tmp_path / "work" / "rollouts"
    make_round_captures(ticket, result, tmp_path / "work", missing={2})
    (root / ticket.round_id / "captures" / "unscored.jsonl").write_text("private\n")

    class WatchStore(LocalStore):
        def __init__(self, root):
            super().__init__(root)
            self.meta_seen = False

        def put_file(self, key, path):
            if key.endswith("/meta.json"):
                raise AssertionError("meta must be a small put, after all data files")
            assert not self.exists(layout.public_rollout_gate_meta(ticket.round_id))
            assert hashlib.sha256(path.read_bytes()).hexdigest()
            return super().put_file(key, path)

        def put(self, key, data):
            if key == layout.public_rollout_gate_meta(ticket.round_id):
                self.meta_seen = True
                assert all(self.exists(layout.public_rollout_gate_entry(ticket.round_id, pos)) for pos in (0, 1))
                assert (root / ticket.round_id / "captures").exists()  # captures retire only after meta is durable
            return super().put(key, data)

    watched = WatchStore(tmp_path / "watched-store")
    publisher = GateRolloutPublisher(watched, tmp_path / "work")
    publisher.close_round([(ticket, result)], winner_uid, ticket.params.params_hash, ticket.draw.round_seed, 0)
    job = json.loads((root / "publication" / f"{ticket.round_id}.json").read_text())
    assert [row["queue_pos"] for row in job["entries"]] == [0, 1, 2]
    assert not (root / ticket.round_id / "captures" / "unscored.jsonl").exists()  # no outcomes: nothing to publish
    publisher.wait_idle()
    assert watched.meta_seen
    expected = {layout.public_rollout_gate(ticket.round_id), layout.public_rollout_gate(ticket.round_id, "winner"),
                layout.public_rollout_gate_meta(ticket.round_id),
                *(layout.public_rollout_gate_entry(ticket.round_id, pos) for pos in (0, 1))}
    assert set(watched.list("public/rollouts")) == expected
    meta = json.loads(watched.get(layout.public_rollout_gate_meta(ticket.round_id)))
    assert meta["schema"] == "sn120-v2-rollouts-gate-2" and meta["rows"] == 8
    assert [row["role"] for row in meta["files"]] == ["head", "winner", "entry", "entry", "entry"]
    combined = hashlib.sha256(json.dumps({row["key"]: row["sha256"] for row in meta["files"] if not row.get("missing")},
                                         sort_keys=True).encode()).hexdigest()
    assert meta["sha256"] == combined
    for row in meta["files"]:
        if row.get("missing"):
            assert row["queue_pos"] == 2 and row["rows"] == 0 and row["bytes"] == 0
            continue
        assert row["sha256"] == watched.sha256(row["key"])
        if row["role"] == "entry":
            seat = ticket.entries[row["queue_pos"]]
            assert (row["uid"], row["hotkey"]) == (seat.reg.uid, seat.reg.hotkey)
            text = gzip.decompress(watched.get(row["key"]))
            assert f'"queue_pos":{row["queue_pos"]}'.encode() in text
            assert f'"hotkey":"{seat.reg.hotkey}"'.encode() in text
            assert b'"role":"entry"' in text and b"entry prompt" in text and b"extracted_answer" in text
    assert b'"role":"winner"' in gzip.decompress(watched.get(layout.public_rollout_gate(ticket.round_id, "winner")))
    assert not (root / ticket.round_id).exists()


def test_entry_plan_uses_queue_positions_not_result_row_order(tmp_path, build):
    first = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    second = build.replace(first, lease_id="r000001-l1", entries=tuple(build.entry(pos) for pos in (2, 3, 4)))
    first_result = build.result(first, "PP", {0: "PP", 1: "FF"})
    second_result = build.result(second, "PP", {3: "PF", 2: "FP", 4: "skip"})
    for ticket, result in ((first, first_result), (second, second_result)):
        make_round_captures(ticket, result, tmp_path / "work")
    publisher = GateRolloutPublisher(LocalStore(tmp_path / "store"), tmp_path / "work")
    publisher.enqueue([(second, second_result), (first, first_result)], 13, first.params.params_hash, first.draw.round_seed, 0)
    job = json.loads((publisher.root / "publication" / "r000001.json").read_text())
    assert [(row["queue_pos"], row["uid"]) for row in job["entries"]] == [(0, 10), (1, 11), (2, 12), (3, 13)]
    publisher.wait_idle()
    winner_rows = gzip.decompress(publisher.store.get(layout.public_rollout_gate("r000001", "winner"))).splitlines()
    expected_model = next(entry.lora.lora_hash for entry in second_result.entries if entry.queue_pos == 3)
    assert [json.loads(row)["model_version_id"] for row in winner_rows] == [expected_model] * 2
    rows = [json.loads(line) for line in gzip.decompress(publisher.store.get(layout.public_rollout_gate_entry("r000001", 3))).splitlines()]
    assert [row["outcome"] for row in rows] == ["P", "F"] and all(row["uid"] == 13 for row in rows)


def test_publication_restart_with_and_without_entries_and_missing_capture(tmp_path, build):
    ticket = build.ticket(n_entries=3, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF", 2: "PF"})
    make_round_captures(ticket, result, tmp_path / "work", missing={1})
    root = tmp_path / "work" / "rollouts"
    store = LocalStore(tmp_path / "store")
    publisher = GateRolloutPublisher(store, tmp_path / "work")
    publisher.enqueue([(ticket, result)], 10, ticket.params.params_hash, ticket.draw.round_seed, 0)
    publisher.jobs.queue.clear()
    for _ in range(publisher.jobs.unfinished_tasks):
        publisher.jobs.task_done()
    restarted = GateRolloutPublisher(store, tmp_path / "work")
    restarted.wait_idle()
    meta = json.loads(store.get(layout.public_rollout_gate_meta(ticket.round_id)))
    missing = next(row for row in meta["files"] if row.get("queue_pos") == 1)
    assert missing["missing"] and missing["rows"] == 0
    assert store.exists(layout.public_rollout_gate_entry(ticket.round_id, 0))
    assert store.exists(layout.public_rollout_gate_entry(ticket.round_id, 2))

    # An in-flight gate-1 job never gains entries during a restart.
    old_ticket = build.replace(ticket, round_id="r000002", lease_id="r000002-l0")
    old_result = build.result(old_ticket, "PP", {0: "PP", 1: "FF", 2: "PF"})
    make_round_captures(old_ticket, old_result, tmp_path / "old-work")
    old_root = tmp_path / "old-work" / "rollouts"
    old = GateRolloutPublisher(store, tmp_path / "old-work")
    old.enqueue([(old_ticket, old_result)], 10, old_ticket.params.params_hash, old_ticket.draw.round_seed, 0)
    old.jobs.queue.clear()
    for _ in range(old.jobs.unfinished_tasks):
        old.jobs.task_done()
    job_path = old_root / "publication" / "r000002.json"
    job = json.loads(job_path.read_text())
    del job["entries"]
    job_path.write_text(json.dumps(job))
    GateRolloutPublisher(store, tmp_path / "old-work").wait_idle()
    old_meta = json.loads(store.get(layout.public_rollout_gate_meta("r000002")))
    assert old_meta["schema"] == "sn120-v2-rollouts-gate-1" and len(old_meta["files"]) == 2
    assert not store.list("public/rollouts/gate/r000002/entries")


def test_one_entry_upload_failure_does_not_block_the_other(tmp_path, build):
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF"})
    make_round_captures(ticket, result, tmp_path / "work")

    class OneEntryDown(LocalStore):
        def put_file(self, key, path):
            if key == layout.public_rollout_gate_entry(ticket.round_id, 0):
                raise InfraError("one entry upload failed")
            return super().put_file(key, path)

    publisher = GateRolloutPublisher(OneEntryDown(tmp_path / "store"), tmp_path / "work")
    publisher.enqueue([(ticket, result)], 10, ticket.params.params_hash, ticket.draw.round_seed, 0)
    publisher.wait_idle()
    meta = json.loads(publisher.store.get(layout.public_rollout_gate_meta(ticket.round_id)))
    failed = next(row for row in meta["files"] if row.get("queue_pos") == 0)
    assert failed["missing"] and failed["rows"] == 0 and failed["bytes"] == 0 and "sha256" not in failed
    assert publisher.store.exists(layout.public_rollout_gate_entry(ticket.round_id, 1))


def test_enqueue_never_gzips_or_uploads_in_the_settling_thread(tmp_path, build, monkeypatch):
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF"})
    make_round_captures(ticket, result, tmp_path / "work")
    store = LocalStore(tmp_path / "store")
    publisher = GateRolloutPublisher(store, tmp_path / "work")
    monkeypatch.setattr("sn120_v2.rollouts.deterministic_gzip", lambda *args: (_ for _ in ()).throw(AssertionError("gzip in enqueue")))
    monkeypatch.setattr(store, "put_file", lambda *args: (_ for _ in ()).throw(AssertionError("upload in enqueue")))
    publisher.enqueue([(ticket, result)], 10, ticket.params.params_hash, ticket.draw.round_seed, 0)
    assert publisher.jobs.unfinished_tasks == 1
    assert all(capture.exists() for capture in (publisher.root / ticket.round_id / "captures").glob("*.jsonl"))


def test_partial_entry_capture_publishes_existing_scored_rows(tmp_path, build):
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF"})
    root = make_round_captures(ticket, result, tmp_path / "work")
    partial = root / f"{result.entries[1].lora.lora_hash}.jsonl"
    lines = partial.read_text().splitlines()
    partial.write_text(lines[0] + "\n" + '{"sample_index":0,')  # torn final line is not fabricated
    publisher = GateRolloutPublisher(LocalStore(tmp_path / "store"), tmp_path / "work")
    publisher.enqueue([(ticket, result)], 10, ticket.params.params_hash, ticket.draw.round_seed, 0)
    publisher.wait_idle()
    rows = gzip.decompress(publisher.store.get(layout.public_rollout_gate_entry(ticket.round_id, 1))).splitlines()
    assert len(rows) == 1
    meta = json.loads(publisher.store.get(layout.public_rollout_gate_meta(ticket.round_id)))
    row = next(row for row in meta["files"] if row.get("queue_pos") == 1)
    assert row["rows"] == 1 and not row["missing"]


def test_settle_publishes_head_and_winner_and_deletes_losers_and_retries(tmp_path, build):
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF"})
    # The publisher consumes this decision; it does not re-score.
    decision = SimpleNamespace(queue_pos=0, uid=10)
    store = LocalStore(tmp_path / "store")
    root = tmp_path / "work" / "rollouts"
    for entry in result.entries:
        for sample, code in enumerate(entry.outcomes[0]):
            model = entry.lora.lora_hash
            path = root / ticket.round_id / "captures" / f"{model}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(round_id=ticket.round_id, sample_index=0, task_pos=sample, outcome=code,
                       completion_text="winner text" if entry.queue_pos == decision.queue_pos else "LOSER SECRET")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    head = root / ticket.round_id / "captures" / f"{NULL_MODEL}.jsonl"
    with open(head, "wt") as handle:
        for index in range(2):
            handle.write(json.dumps(dict(round_id=ticket.round_id, sample_index=0, task_pos=index, outcome="P", completion_text="head text")) + "\n")

    class DownStore(LocalStore):
        failures = 1

        def put_file(self, key, path):
            if key.endswith("head.jsonl.gz") and self.failures:
                self.failures -= 1
                raise InfraError("store down")
            return super().put_file(key, path)

    failing = DownStore(tmp_path / "failing-store")
    publisher = GateRolloutPublisher(failing, tmp_path / "work", retry_s=0.01)
    publisher.close_round([(ticket, result)], decision.uid, ticket.params.params_hash, ticket.draw.round_seed, 0)
    publisher.wait_idle()
    key = layout.public_rollout_gate(ticket.round_id)
    assert failing.exists(key)
    rows = [json.loads(line) for line in gzip.decompress(failing.get(key)).splitlines()]
    assert [row["role"] for row in rows] == ["head", "head"]
    assert all(row["winner_uid"] == decision.uid and row["model_version_id"] == ticket.head.version_id for row in rows)
    winner_model = next(entry.lora.lora_hash for entry in result.entries if entry.queue_pos == decision.queue_pos)
    assert failing.exists(layout.public_rollout_gate(ticket.round_id, "winner"))
    assert set(failing.list("public/rollouts")) == {
        layout.public_rollout_gate(ticket.round_id), layout.public_rollout_gate(ticket.round_id, "winner"),
        layout.public_rollout_gate_meta(ticket.round_id),
        *(layout.public_rollout_gate_entry(ticket.round_id, pos) for pos in (0, 1)),
    }  # every entry is public, not only head/winner
    assert not (root / ticket.round_id / "captures" / f"{winner_model}.jsonl").exists()  # whole local round retired after upload
    assert b"LOSER SECRET" in gzip.decompress(failing.get(layout.public_rollout_gate_entry(ticket.round_id, 1)))
    meta = json.loads(failing.get(layout.public_rollout_gate_meta(ticket.round_id)))
    assert meta["params_hash"] == ticket.params.params_hash and meta["closed_at"] == "1970-01-01T00:00:00Z"
    assert meta["rows"] == 8 and len(meta["files"]) == 4


def test_controller_settles_before_queueing_public_rollouts(build, tmp_path):
    from sn120_v2.journal import Journal
    from sn120_v2.controller import Controller

    world = __import__("sn120_v2.sim", fromlist=["SimWorld"]).SimWorld(tmp_path / "world", __import__("sn120_v2.sim", fromlist=["sim_params"]).sim_params())
    world.register(3, 0.3)
    ticket, result = world.controller.open_round(world.clock(), "entropy")[0], None
    events = []
    controller = Controller(world.controller.journal, world.store, world.controller.admission, world.controller.sampler, world.params,
                            folder=world.folder, rollout_publisher=SimpleNamespace(close_round=lambda *args: events.append(args)))
    # The journal is already the source of the replayed controller's state.
    sealed = world.runner.run(ticket)
    controller.submit_result(sealed, world.clock())
    assert controller.state.round is None and events and events[0][1] == 3
    assert events[0][2] == ticket.params.params_hash

    other = __import__("sn120_v2.sim", fromlist=["SimWorld"]).SimWorld(tmp_path / "other", __import__("sn120_v2.sim", fromlist=["sim_params"]).sim_params())
    other.register(3, 0.3)
    ticket = other.controller.open_round(other.clock(), "entropy")[0]
    controller = Controller(other.controller.journal, other.store, other.controller.admission, other.controller.sampler, other.params,
                            folder=other.folder, rollout_publisher=SimpleNamespace(close_round=lambda *args: (_ for _ in ()).throw(OSError("store down"))))
    controller.submit_result(other.runner.run(ticket), other.clock())
    assert controller.state.round is None and "public rollouts" in controller.last_error


def test_no_winner_discards_all_candidate_capture(tmp_path, build):
    ticket = build.ticket(n_entries=1, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {})
    root = tmp_path / "work" / "rollouts"
    loser = root / ticket.round_id / "captures" / (build.h("loser") + ".jsonl")
    loser.parent.mkdir(parents=True)
    loser.write_text("not public\n")
    head = root / ticket.round_id / "captures" / f"{NULL_MODEL}.jsonl"
    with open(head, "wt") as handle:
        for index in range(2):
            handle.write(json.dumps(dict(round_id=ticket.round_id, sample_index=0, task_pos=index, outcome="P", completion_text="head text")) + "\n")
    publisher = GateRolloutPublisher(LocalStore(tmp_path / "store"), tmp_path / "work")
    publisher.close_round([(ticket, result)], None, ticket.params.params_hash, ticket.draw.round_seed, 0)
    publisher.wait_idle()
    assert not loser.exists()
    assert not publisher.store.exists(layout.public_rollout_gate(ticket.round_id, "winner"))


def test_publication_is_one_pass_for_2000_tasks_two_samples_8kb(tmp_path, build):
    params = build.params(n_tasks=2000, samples_per_task=2)
    ticket = build.ticket(n_entries=1, n_tasks=2000, params=params)
    result = build.result(ticket, ("P" * 2000,) * 2, {})
    capture = tmp_path / "null.jsonl"
    with capture.open("w") as handle:
        for sample in range(2):
            for pos in range(2000):
                handle.write(json.dumps(dict(sample_index=sample, task_pos=pos, outcome="P", completion_text="x" * 8192)) + "\n")
    started = time.monotonic()
    output = tmp_path / "public.jsonl.gz"
    rows_n, _, _ = deterministic_gzip(output, published_rows(capture, ticket, result, "head", NULL_MODEL, None))
    elapsed = time.monotonic() - started
    assert rows_n == 4000
    assert elapsed < 5.0, f"publication took {elapsed:.3f}s instead of one bounded scan"


def test_live_plain_capture_is_readable_and_low_disk_skips_logging(tmp_path, build, capsys, monkeypatch):
    capture = RolloutCapture(tmp_path / "rollouts", POOL)
    ticket = build.ticket(n_entries=1, n_tasks=2, params=build.params(n_tasks=2))
    req = EvalRequest(ticket.round_id, ticket.head, (), ticket.draw.task_ids, "seed-0", 0, ticket.params)
    job = (NULL_MODEL, 0, 0)
    capture.graded(req, job, capture.reply(req, job, "P", False, 12, "stop"), "P", SimpleNamespace(answer="\\boxed{4}"))
    # No close(), no open gzip trailer: a publisher can read this immediately.
    rows = [json.loads(line) for line in capture_file(capture.root, ticket.round_id, NULL_MODEL).read_text().splitlines()]
    assert rows[0]["completion_text"] == "P" and rows[0]["output_tokens"] == 12
    capture.graded(req, (NULL_MODEL, 0, 1), capture.reply(req, job, "F", False, 9, "stop"), "F", SimpleNamespace(answer="\\boxed{5}"))
    assert len(capture_file(capture.root, ticket.round_id, NULL_MODEL).read_text().splitlines()) == 2
    monkeypatch.setattr("sn120_v2.rollouts.shutil.disk_usage", lambda path: SimpleNamespace(free=1))
    later = dataclasses.replace(req, round_id="r000002")
    capture.graded(later, job, ("P", False), "P")
    assert not capture_file(capture.root, "r000002", NULL_MODEL).exists()
    assert "public rollout capture disabled" in capsys.readouterr().out


def test_old_unpublished_rounds_are_cleaned_but_pending_and_previous_survive(tmp_path, build):
    root = tmp_path / "rollouts"
    for rid in ("r000001", "r000002", "r000003", "r000004"):
        (root / rid / "captures").mkdir(parents=True)
        (root / rid / "captures" / "loser.jsonl").write_text("never public\n")
    (root / "publication").mkdir()
    (root / "publication" / "r000001.json").write_text("{}")
    capture = RolloutCapture(root, POOL)
    ticket = build.ticket(n_entries=1, n_tasks=2, params=build.params(n_tasks=2))
    req = EvalRequest("r000005", ticket.head, (), ticket.draw.task_ids, "seed-0", 0, ticket.params)
    capture.start(req)
    assert (root / "r000001").exists()  # durable publication job
    assert not (root / "r000002").exists() and not (root / "r000003").exists()
    assert (root / "r000004").exists()  # previous round may still be awaiting settlement


def test_publisher_parks_after_twenty_failures_and_never_uploads_loser(tmp_path, build):
    ticket = build.ticket(n_entries=2, n_tasks=2, params=build.params(n_tasks=2))
    result = build.result(ticket, "PP", {0: "PP", 1: "FF"})
    root = tmp_path / "work" / "rollouts"
    for entry in result.entries:
        path = root / ticket.round_id / "captures" / f"{entry.lora.lora_hash}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(dict(sample_index=0, task_pos=pos, outcome=entry.outcomes[0][pos],
                                                 completion_text="winner" if entry.queue_pos == 0 else "LOSER")) + "\n" for pos in range(2)))
    head = root / ticket.round_id / "captures" / f"{NULL_MODEL}.jsonl"
    head.write_text("".join(json.dumps(dict(sample_index=0, task_pos=pos, outcome="P", completion_text="head")) + "\n" for pos in range(2)))

    class Down(LocalStore):
        def put_file(self, key, path):
            raise InfraError("down")

    store = Down(tmp_path / "store")
    publisher = GateRolloutPublisher(store, tmp_path / "work", retry_s=0)
    publisher.close_round([(ticket, result)], 10, ticket.params.params_hash, ticket.draw.round_seed, 0)
    publisher.wait_idle()
    assert publisher.attempts[ticket.round_id] == 20 and ticket.round_id in publisher.parked
    job = json.loads((root / "publication" / f"{ticket.round_id}.json").read_text())
    assert job["parked"] and job["attempts"] == 20 and job["error"]
    assert store.list("public/rollouts") == []
    assert (root / ticket.round_id / "captures" / f"{result.entries[1].lora.lora_hash}.jsonl").exists()  # retry evidence retained
    restarted = GateRolloutPublisher(store, tmp_path / "work", retry_s=0)
    assert ticket.round_id in restarted.parked and restarted.jobs.empty()
