"""Rollout publication is a bounded side effect of one row's score collection, never its gate."""
import gzip
import hashlib
import json
import types
from pathlib import Path

import pytest

from affine_bench import daemon as D
from sn120_v2 import layout
from test_daemon import world  # the deployed line's per-row world fixture


class Store:
    def __init__(self):
        self.data, self.read, self.fail = {}, [], 1

    def put_file(self, key, path):
        self.read.append(path)
        if self.fail:
            self.fail -= 1
            raise OSError("temporary")
        self.data[key] = Path(path).read_bytes()

    def put(self, key, data):
        self.data[key] = data

    def sha256(self, key):
        return hashlib.sha256(self.data[key]).hexdigest()


def row(tmp_path, monkeypatch, mid="row"):
    monkeypatch.setattr(D, "RES", str(tmp_path / "results"))
    monkeypatch.setattr(D, "PRIV", str(tmp_path / "private"))
    monkeypatch.setattr(D, "now", lambda: 1000.0)
    found = D.Row(mid)
    found.node = types.SimpleNamespace(ssh=str(tmp_path / "node-ssh"), out=lambda cmd, timeout=0: "")
    found.rec = {}
    return found


def test_stream_hash_retry_upload_and_score_not_blocked(tmp_path, monkeypatch):
    store = Store()
    monkeypatch.setattr(D.source, "store", lambda: store)
    worker = row(tmp_path, monkeypatch)
    data = gzip.compress(json.dumps(dict(model_id="row", benchmark_id="math", completion_text="public")).encode() + b"\n", mtime=0)
    path = tmp_path / "private" / "row" / "math.rollouts.jsonl.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    monkeypatch.setattr(D, "fetch_rollout", lambda node, rk, bid, local, tell: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    pending = dict(run_key="row", bid="math", attempts=0)
    worker.collect_rollout("math", dict(finished_at=0, bench_version="math/v1"), pending)
    assert store.fail == 0 and store.data[layout.public_rollout_bench("row", "math")] == data
    meta = json.loads(store.data[layout.public_rollout_bench_meta("row", "math")])
    assert meta["rows"] == 1 and meta["bytes"] == len(data) and meta["finished_at"] == "1970-01-01T00:00:00Z"
    assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert "row/math" not in worker.S["rollout_pending"] and not path.exists()

    class FailStore(Store):
        def put_file(self, key, path):
            raise OSError("down")

    monkeypatch.setattr(D.source, "store", lambda: FailStore())
    pending = dict(run_key="row", bid="math", attempts=0)
    worker.S["rollout_pending"]["row/math"] = pending
    worker.collect_rollout("math", dict(finished_at=0), pending)  # no exception reaches the already-collected score
    assert worker.S["rollout_pending"]["row/math"]["attempts"] == 1


def test_upload_helper_has_no_hidden_retry(tmp_path, monkeypatch):
    store = Store()
    monkeypatch.setattr(D.source, "store", lambda: store)
    store.fail = 0
    path = tmp_path / "r.gz"
    path.write_bytes(gzip.compress(b'{"model_id":"row","benchmark_id":"math"}\n', mtime=0))
    saved = []
    D.upload_rollout("row", "math", str(path), dict(finished_at=0, bench_version="v"), saved.append)
    assert store.fail == 0 and saved == [hashlib.sha256(path.read_bytes()).hexdigest()]


def test_transfer_streams_through_the_rows_ssh_and_checks_its_hash(tmp_path, monkeypatch):
    blob = b"large-public-data" * 1000
    wrapper = tmp_path / "row-ssh"
    output = tmp_path / "remote-bytes"
    output.write_bytes(blob)
    wrapper.write_text(f"#!/bin/sh\nprintf '%s' \"$1\" > {tmp_path / 'command'}\nexec /bin/cat {output}\n")
    wrapper.chmod(0o700)
    node = types.SimpleNamespace(ssh=str(wrapper), out=lambda cmd, timeout=0: f"{hashlib.sha256(blob).hexdigest()} {len(blob)}\n")
    found = D.fetch_rollout(node, "model", "math", str(tmp_path / "private" / "data.gz"))
    assert found["sha256"] == hashlib.sha256(blob).hexdigest()
    assert (tmp_path / "private" / "data.gz").read_bytes() == blob
    assert "cat " in (tmp_path / "command").read_text()
    monkeypatch.setattr(node, "out", lambda cmd, timeout=0: f"{'0' * 64} {len(blob)}\n")
    with pytest.raises(OSError):
        D.fetch_rollout(node, "model", "math", str(tmp_path / "bad.gz"))
    assert not (tmp_path / "bad.gz.part").exists()


def test_retry_only_pending_bounded_backoff_parks_and_never_scans_legacy(tmp_path, monkeypatch):
    worker = row(tmp_path, monkeypatch)
    calls, ssh = [], []
    for index in range(5):
        identity = f"row/bench{index}"
        worker.S["rollout_pending"][identity] = dict(model="row", run_key="row", bid=f"bench{index}", next_at=0, delay=300, attempts=0)
    results = tmp_path / "results" / "row"
    results.mkdir(parents=True)
    for bid in ("bench0", "bench1", "bench2", "legacy"):
        (results / f"{bid}.json").write_text(json.dumps(dict(model_id="row", benchmark=bid, finished_at=0)))
    monkeypatch.setattr(D, "fetch_rollout", lambda node, rk, bid, local, tell: calls.append(bid) or (_ for _ in ()).throw(OSError("still down")))
    monkeypatch.setattr(worker.node, "out", lambda cmd, timeout=0: ssh.append(cmd) or "")
    clock = [1000.0]
    monkeypatch.setattr(D, "now", lambda: clock[0])
    worker.retry_rollouts()
    assert calls == ["bench0", "bench1", "bench2"] and not ssh
    for at, attempts, next_at in ((1300.0, 2, 1900.0), (1900.0, 3, 1900.0)):
        clock[0] = at
        worker.retry_rollouts()
        current = [worker.S["rollout_pending"][f"row/bench{i}"] for i in range(3)]
        assert all(p["attempts"] == attempts and p["next_at"] == next_at for p in current), current
    pending = [worker.S["rollout_pending"][f"row/bench{i}"] for i in range(3)]
    assert all(p["attempts"] == D.ROLLOUT_MAX_ATTEMPTS and p["parked_at"] for p in pending)
    assert calls == ["bench0", "bench1", "bench2"] * 3
    assert [p["next_at"] for p in pending] == [1900.0, 1900.0, 1900.0]
    worker.retry_rollouts()
    assert calls == ["bench0", "bench1", "bench2"] * 3 and not ssh  # parked and legacy identities are not retried


def test_row_collect_uses_its_own_node_and_scores_survive_upload_failure(tmp_path, monkeypatch):
    store = Store()
    store.fail = 0
    monkeypatch.setattr(D.source, "store", lambda: store)
    worker = row(tmp_path, monkeypatch, "own-row")
    other = D.Row("other-row")
    other.node = types.SimpleNamespace(out=lambda cmd, timeout=0: pytest.fail("another row's node was used"))
    other.collect_rollout = lambda *args: pytest.fail("another row uploaded")
    model = dict(model_id="own-row", kind="genesis")
    summary = dict(benchmark="math", score=1.0, se=0.0, tasks=1, tasks_total=1, samples=1, rollouts=1,
                   cap_rate=0.0, mean_out_tokens=5, bench_version="math/v1/bench-v2", finished_at=0)
    data = gzip.compress(b'{"model_id":"own-row","benchmark_id":"math","completion_text":"public"}\n', mtime=0)
    assert D.R.benchmark("math")["version"] == 1
    seen = []

    def out(cmd, timeout=0):
        seen.append(cmd)
        if "test -f" in cmd:
            return "yes\n"
        if "summary.json" in cmd:
            return json.dumps(summary)
        if "per_task.jsonl" in cmd:
            return '{"idx":0,"correct":1.0}\n'
        return "{}"

    worker.node.out = out
    monkeypatch.setattr(D, "node_file", lambda node, remote, local: Path(local).write_bytes(data))
    # The remote hash comes from this row, not the shared legacy node or another worker.
    worker.node.out = lambda cmd, timeout=0: f"{hashlib.sha256(data).hexdigest()} {len(data)}\n" if "sha256sum" in cmd else out(cmd, timeout)
    got = worker.collect(dict(model=model, run_key="own-row", bids=["math"], started=0))
    assert got == {"math": 1.0} and Path(D.result_path("own-row", "math")).exists()
    assert store.data[layout.public_rollout_bench("own-row", "math")] == data
    assert any("runs/own-row/math/rollouts.jsonl.gz" in cmd for cmd in seen)
    assert not any("other-row" in cmd for cmd in seen)


def test_score_and_table_publication_succeed_when_store_rollout_upload_raises(world, monkeypatch):
    mid = next(m["model_id"] for m in D.model_rows() if m["kind"] == "genesis")
    worker = world.node(mid)
    original_out = worker.out

    def out(cmd, timeout=0):
        if "test -f" in cmd and "rollouts.jsonl.gz" in cmd:
            return "yes\n"
        if "sha256sum" in cmd:
            return f"{hashlib.sha256(b'public').hexdigest()} 6\n"
        return original_out(cmd, timeout)

    monkeypatch.setattr(worker, "out", out)
    monkeypatch.setattr(D, "node_file", lambda node, remote, local: Path(local).write_bytes(b"public"))
    monkeypatch.setattr(D, "upload_rollout", lambda *args: (_ for _ in ()).throw(OSError("store unavailable")))
    world.steps(3)  # launch, collect, aggregate-table confirmation and release
    assert json.loads(world.store.tables[-1])["models"]
    assert next(m for m in table_models(world.store.tables[-1]) if m["model_id"] == mid)["status"] == "scored"
    assert any(r["model_id"] == mid for r in json.loads(world.store.tables[-1])["results"])
    state = D.row_state(mid)
    assert state["rollout_pending"] and all(p["attempts"] == 1 for p in state["rollout_pending"].values())
    assert not state["fails"] and not world.lium.pods


def table_models(data):
    return json.loads(data)["models"]
