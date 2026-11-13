"""The per-green path on the host: a real v2 journal and store from the CPU simulation, a fake node and a fake Lium."""
import ast, importlib, json, pathlib, types
import pytest
from sn120_v2 import layout, sim
from sn120_v2.errors import ConfigError
from affine_bench import config as C, daemon as D, pod as P, registry as R, source

POD = dict(id="pod-exact-id", name=P.NAME, huid="fake-pod", gpu="8xB300", cost=1.0, bootstrapped=True)


class Store:
    """No copy and no delete, and a put to one key only: any other write by the service fails the test."""
    def __init__(self, inner): self.inner, self.signed, self.published, self.tables = inner, [], True, []
    def put(self, key, data): assert key == layout.BENCHMARKS; self.tables.append(data); self.inner.put(key, data)
    def presign_get(self, key, ttl_s): self.signed.append(key); return f"https://signed.example/{key}?ttl={ttl_s}"

    def get(self, key):
        if not self.published: raise KeyError(key)
        return self.inner.get(key)


def green(v2, uid):
    """One round that a miner wins. Returns the folded head that the controller leaves."""
    v2.register(uid, 0.3)
    assert v2.round().outcome == "applied"
    return v2.controller.state.head


class Node:
    """The rented box as the daemon sees it over ssh: a fetch, one runner, and its result files."""
    def __init__(self): self.cmds, self.stdin, self.fetched = [], [], False

    def run(self, cmd, inp=None, timeout=0):
        self.cmds.append(cmd); self.stdin.append(inp); self.fetched |= "gate_fetch.py" in cmd
        return types.SimpleNamespace(stdout=self.out(cmd), stderr="", returncode=0)

    def out(self, cmd, timeout=0):
        if ".verified" in cmd: return "yes" if self.fetched else ""
        if "echo ok=1" in cmd: return "done=1\nalive=\nlines=4\nok=1\n"
        if "summary.json" in cmd:
            bid = cmd.split(" 2>")[0].split("/")[-2]  # .../runs/<king>/<benchmark>/summary.json
            return json.dumps(dict(benchmark=bid, score=0.5, se=0.1, tasks=2, tasks_total=2, samples=2, rollouts=1, cap_rate=0.0,
                                   bench_version=R.bench_version(R.benchmark(bid)), finished_at=1, wall_seconds=1, mean_out_tokens=9))
        if "per_task.jsonl" in cmd: return '{"idx":0,"correct":1.0}\n{"idx":1,"correct":0.0}\n'
        return '{"wall_seconds": 60}' if cmd.endswith("/DONE") else "4321"  # 4321: the pid of the launched runner


@pytest.fixture
def world(monkeypatch, tmp_path):
    v2, node, removed = sim.SimWorld(tmp_path, sim.sim_params()), Node(), []
    store = Store(v2.store)
    monkeypatch.setattr(source, "store", lambda: store); monkeypatch.setattr(source, "_manifests", {})
    monkeypatch.setattr(C, "JOURNAL", str(tmp_path / sim.JOURNAL_FILE))
    monkeypatch.setattr(D, "NODE", node); monkeypatch.setattr(P, "IDLE_RELEASE_S", 0)
    monkeypatch.setattr(P, "rent", lambda S, log: (S.setdefault("rents", []).append(1), P.save(POD), dict(POD))[-1])
    monkeypatch.setattr(P, "listed", lambda: [dict(id=POD["id"], name=P.NAME), dict(id="someone-elses-pod", name="other")])
    monkeypatch.setattr(P, "lium", lambda *a, **k: (removed.append(a), types.SimpleNamespace(returncode=0, stdout="", stderr=""))[-1])
    return v2, store, node, removed


def test_a_king_is_listed_only_when_its_folded_head_is_published(world, monkeypatch, tmp_path):
    v2, store = world[:2]
    v2.register(1, 0.0)
    assert v2.round().outcome == "null" and source.kings() == []  # genesis is not a king, and a round with no winner makes none
    store.published = False
    folded = green(v2, 2)
    assert source.kings() == []  # the journal has the folded head, the store does not have its manifest yet
    store.published = True
    (row,) = source.kings()
    assert (row["king"], row["round_id"], row["head_version_id"]) == (folded.base_hash, folded.round_id, folded.version_id)
    blob = source.presigned(row["king"])
    assert blob["descriptor"] == dict(id=row["king"], files=dict(v2.controller.state.manifest.base_files))
    assert sorted(store.signed) == [layout.public_base_file(row["king"], name) for name in sorted(blob["urls"])]
    stacked = sim.SimWorld(tmp_path / "k2", sim.sim_params(fold_every_k=2))
    monkeypatch.setattr(C, "JOURNAL", str(stacked.root / sim.JOURNAL_FILE)); monkeypatch.setattr(store, "inner", stacked.store)
    assert green(stacked, 1).loras and source.kings() == []  # a king whose LoRA is still stacked on the base is not served


def test_one_green_is_benchmarked_once_on_its_own_pod_then_the_pod_is_released_by_exact_id(world, capsys):
    v2, store, node, removed = world
    D.step()
    assert not D.C_POD and not D.S.get("rents")  # no king yet: nothing is rented
    r1 = green(v2, 1)
    D.step(); D.step()  # rent, fetch and launch; then collect and release
    mid = "king-" + r1.base_hash[:16]
    assert D.missing(dict(model_id=mid), R.benchmarks(enabled_only=True, runnable_only=True)) == []
    assert removed == [("rm", POD["id"], "--name-only", "-y")] and not D.C_POD and len(D.S["rents"]) == 1
    table = json.load(open(C.SITE + "/benchmarks.json"))
    assert [(m["model_id"], m["status"], m["round_id"]) for m in table["models"]] == [(mid, "scored", r1.round_id)]
    for _ in range(3):  # the same king listed again (a retried apply, a restart): no second run, no second rent
        D.step()
    assert len(D.S["rents"]) == 1 and D.queue() == []
    # the expiring links reach the node on stdin only: never in a command line, never in a log line
    assert sum("signed.example" in (s or "") for s in node.stdin) == 1
    assert "signed.example" not in " ".join(node.cmds) + capsys.readouterr().out
    r2 = green(v2, 2)
    assert [m["king"] for m in D.queue()] == [r2.base_hash]  # only the new green is queued
    # the bucket holds the table: written once for the king, not at each poll; a host with no local copy starts from it
    assert len(store.tables) == 1 and [m["model_id"] for m in json.loads(store.tables[0])["models"]] == [mid]
    pathlib.Path(C.SITE, "benchmarks.json").unlink()
    source.restore_table(C.SITE + "/benchmarks.json")
    assert pathlib.Path(C.SITE, "benchmarks.json").read_bytes() == store.tables[0]


def test_only_one_daemon_per_state_directory():
    lock = D.only_instance()
    with pytest.raises(SystemExit):
        D.only_instance()
    lock.close()


def test_settings_come_from_the_environment_and_a_missing_one_stops_the_service(monkeypatch):
    assert source.store() == pathlib.Path.cwd()  # BENCH_STORE names a factory as "module:Class.method"; here pathlib:Path.cwd
    monkeypatch.delenv("BENCH_MAX_PRICE")
    with pytest.raises(ConfigError):
        importlib.reload(C)


def test_the_benchmark_is_informational_only():
    """The bench reads records, the key layout and the journal, nothing that decides. And nothing in v2's control plane reads the bench."""
    allowed = {"sn120_v2", "sn120_v2.layout", "sn120_v2.errors", "sn120_v2.types", "sn120_v2.params", "sn120_v2.state"}
    v2 = pathlib.Path(__file__).resolve().parents[2]
    for path in (v2 / "bench" / "affine_bench").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in getattr(node, "names", []) if isinstance(node, ast.Import)]
            assert not [n for n in names if n.startswith("sn120_v2") and n not in allowed], path
    for path in (v2 / "sn120_v2").rglob("*.py"):  # layout.py names the key of the table; no v2 module reads the table
        text = path.read_text()
        assert "affine_bench" not in text and ("BENCHMARKS" not in text and "benchmarks.json" not in text or path.name == "layout.py"), path
