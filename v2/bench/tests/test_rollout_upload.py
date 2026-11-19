"""The rollout uploader is a side effect of score collection, never its gate."""
import gzip
import hashlib
import json
from pathlib import Path

from affine_bench import daemon as D
from sn120_v2 import layout


class Store:
    def __init__(self):
        self.data = {}
        self.fail = 1
        self.read = []

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


def test_stream_hash_retry_upload_and_score_not_blocked(tmp_path, monkeypatch):
    store = Store()
    monkeypatch.setattr(D, "PRIV", str(tmp_path))
    monkeypatch.setattr(D, "SF", str(tmp_path / "daemon.json"))
    monkeypatch.setattr(D.source, "store", lambda: store)
    D.S.clear(); D.S.update(fail={}, unavailable=[], history=[], rollout_uploads={}, rollout_pending={})
    data = gzip.compress(json.dumps(dict(model_id="genesis-a", benchmark_id="math", completion_text="public")).encode() + b"\n", mtime=0)
    path = tmp_path / "genesis-a" / "math.rollouts.jsonl.gz"
    path.parent.mkdir()
    path.write_bytes(data)
    monkeypatch.setattr(D, "fetch_rollout", lambda rk, bid, local: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    D.collect_rollout(dict(model=dict(model_id="genesis-a"), run_key="genesis-a"), "math", dict(finished_at=0, bench_version="math/v1"))
    assert store.fail == 0 and store.data[layout.public_rollout_bench("genesis-a", "math")] == data
    meta = json.loads(store.data[layout.public_rollout_bench_meta("genesis-a", "math")])
    assert meta["rows"] == 1 and meta["bytes"] == len(data) and meta["finished_at"] == "1970-01-01T00:00:00Z"
    assert meta["sha256"] == hashlib.sha256(data).hexdigest()

    class FailStore(Store):
        def put_file(self, key, path):
            raise OSError("down")

    failed = FailStore()
    monkeypatch.setattr(D.source, "store", lambda: failed)
    # collect() has already saved this aggregate before calling the uploader.
    D.collect_rollout(dict(model=dict(model_id="genesis-a"), run_key="genesis-a"), "math", dict(finished_at=0))
    assert failed.data == {}


def test_upload_helper_has_no_hidden_retry(tmp_path, monkeypatch):
    store = Store()
    monkeypatch.setattr(D.source, "store", lambda: store)
    D.S.clear(); D.S.update(fail={}, unavailable=[], history=[], rollout_uploads={}, rollout_pending={})
    store.fail = 0
    path = tmp_path / "r.gz"
    path.write_bytes(gzip.compress(b'{"model_id":"genesis-a","benchmark_id":"math"}\n', mtime=0))
    D.upload_rollout("genesis-a", "math", str(path), dict(finished_at=0, bench_version="v"))
    assert store.fail == 0


def test_transfer_uses_a_file_and_checks_the_node_hash(tmp_path, monkeypatch):
    blob = b"large-public-data" * 1000
    calls = []

    def fake_node_file(remote, local):
        calls.append((remote, local))
        Path(local).write_bytes(blob)

    monkeypatch.setattr(D, "node_file", fake_node_file)
    monkeypatch.setattr(D.NODE, "out", lambda cmd, timeout=0: f"{hashlib.sha256(blob).hexdigest()} {len(blob)}\n")
    found = D.fetch_rollout("model", "math", str(tmp_path / "data.gz"))
    assert found["sha256"] == hashlib.sha256(blob).hexdigest() and calls
    # A different byte count is rejected; the error is caught and tried again.
    monkeypatch.setattr(D.NODE, "out", lambda cmd, timeout=0: f"{'0' * 64} {len(blob)}\n")
    calls.clear()
    try:
        D.fetch_rollout("model", "math", str(tmp_path / "bad.gz"))
    except OSError:
        pass
    else:
        raise AssertionError("wrong node hash accepted")
    assert len(calls) == 3


def test_retry_only_pending_bounded_backoff_and_no_legacy_ssh(tmp_path, monkeypatch):
    D.S.clear(); D.S.update(fail={}, unavailable=[], history=[], rollout_uploads={}, rollout_pending={})
    monkeypatch.setattr(D, "RES", str(tmp_path / "results"))
    monkeypatch.setattr(D, "SF", str(tmp_path / "daemon.json"))
    monkeypatch.setattr(D, "PRIV", str(tmp_path / "private"))
    monkeypatch.setattr(D, "now", lambda: 1000.0)
    for index in range(5):
        identity = f"row{index}/math"
        D.S["rollout_pending"][identity] = dict(model=f"row{index}", run_key=f"row{index}", bid="math", next_at=0, delay=300, attempts=0)
        path = tmp_path / "results" / f"row{index}"
        path.mkdir(parents=True)
        (path / "math.json").write_text(json.dumps(dict(model_id=f"row{index}", benchmark="math", finished_at=0)))
    # A legacy result with no marker is never read, uploaded, or probed over ssh.
    legacy = tmp_path / "results" / "legacy"
    legacy.mkdir()
    (legacy / "math.json").write_text(json.dumps(dict(model_id="legacy", benchmark="math", finished_at=0)))
    calls, ssh = [], []
    monkeypatch.setattr(D, "collect_rollout", lambda run, bid, result: calls.append(run["run_key"]))
    monkeypatch.setattr(D.NODE, "out", lambda cmd, timeout=0: ssh.append((cmd, timeout)) or "")
    D.retry_rollouts()
    assert calls == ["row0", "row1", "row2"] and not ssh


def test_collect_marks_only_observed_rollout_and_defers_with_backoff(tmp_path, monkeypatch):
    D.S.clear(); D.S.update(fail={}, unavailable=[], history=[], rollout_uploads={}, rollout_pending={})
    monkeypatch.setattr(D, "RES", str(tmp_path / "results"))
    monkeypatch.setattr(D, "PRIV", str(tmp_path / "private"))
    monkeypatch.setattr(D, "SF", str(tmp_path / "daemon.json"))
    monkeypatch.setattr(D, "now", lambda: 1000.0)
    seen = []

    def node_out(cmd, timeout=0):
        seen.append((cmd, timeout))
        if "test -f" in cmd:
            return "yes\n" if "/math/" in cmd else ""
        if "summary.json" in cmd:
            return json.dumps(dict(benchmark="math", score=1.0, finished_at=0, bench_version="v"))
        if "per_task.jsonl" in cmd:
            return ""
        return "{}"

    monkeypatch.setattr(D.NODE, "out", node_out)
    monkeypatch.setattr(D, "fetch_rollout", lambda *args: (_ for _ in ()).throw(OSError("node is gone")))
    run = dict(model=dict(model_id="row", kind="genesis"), run_key="row", bids=["math", "aime22"], started=0)
    D.collect(run)
    assert set(D.S["rollout_pending"]) == {"row/math"}
    pending = D.S["rollout_pending"]["row/math"]
    assert pending["attempts"] == 1 and pending["next_at"] == 1300 and pending["delay"] == 600
    assert all("legacy" not in cmd for cmd, _ in seen)
    assert not any("sha256sum" in cmd for cmd, _ in seen if "/aime22/" in cmd)
