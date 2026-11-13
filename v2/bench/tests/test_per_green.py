"""The per-green path on the host, against a fake store, a fake node and a fake Lium. No GPU and no network."""
import ast, importlib, json, pathlib, types
import pytest
from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.errors import ConfigError
from sn120_v2.types import HeadManifest, HeadVersion, LoRARef
from affine_bench import config as C, daemon as D, pod as P, registry as R, source

POD = dict(id="pod-exact-id", name=P.NAME, huid="fake-pod", gpu="8xB300", cost=1.0, bootstrapped=True)


def h(text): return sha256_hex(text.encode())


class Store:
    """Read-only on purpose: it has no put, copy or delete, so a write by the service would fail the test."""
    def __init__(self): self.objects, self.signed = {}, []
    def get(self, key): return self.objects[key]
    def presign_get(self, key, ttl_s): self.signed.append(key); return f"https://signed.example/{key}?ttl={ttl_s}"

    def head(self, parent, round_id, base, lora=None):
        """Publish one head version and move the current pointer to it. `lora` makes it a stacked (unfolded) head."""
        files, lf = {"config.json": h(base + "cfg"), "model.safetensors": h(base)}, {"adapter.safetensors": h(lora or "")}
        ref = LoRARef(manifest_digest(lf), round_id, "hotkey", 1, h("batch"), "recipe", 8)
        head = HeadVersion.make(parent_version_id=parent and parent.version_id, base_hash=manifest_digest(files), base_fp32_master_digest=h(base + "m"),
                                loras=(ref,) if lora else (), round_id=round_id, applied_at=len(self.objects))
        m = HeadManifest(head, files, {"master.safetensors": h(base + "m")}, {ref.lora_hash: lf} if lora else {})
        self.objects[layout.head_manifest(head.version_id)] = m.to_json()
        self.objects[layout.CURRENT] = json.dumps(dict(version_id=head.version_id)).encode()
        return head


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
def world(monkeypatch):
    store, node, removed = Store(), Node(), []
    monkeypatch.setattr(source, "store", lambda: store); monkeypatch.setattr(source, "_manifests", {})
    monkeypatch.setattr(D, "NODE", node); monkeypatch.setattr(P, "IDLE_RELEASE_S", 0)
    monkeypatch.setattr(P, "rent", lambda S, log: (S.setdefault("rents", []).append(1), P.save(POD), dict(POD))[-1])
    monkeypatch.setattr(P, "listed", lambda: [dict(id=POD["id"], name=P.NAME), dict(id="someone-elses-pod", name="other")])
    monkeypatch.setattr(P, "lium", lambda *a, **k: (removed.append(a), types.SimpleNamespace(returncode=0, stdout="", stderr=""))[-1])
    return store, node, removed


def test_a_king_is_listed_only_when_its_folded_head_is_published(world):
    store = world[0]
    assert source.kings() == []  # no head at all
    genesis = store.head(None, "genesis", "base0")
    stacked = store.head(genesis, "r1", "base0", lora="lora1")
    assert source.kings() == []  # genesis is not a king, and r1 still has a stacked LoRA
    folded = store.head(stacked, "r1", "base1")
    (row,) = source.kings()
    assert (row["king"], row["round_id"], row["head_version_id"]) == (folded.base_hash, "r1", folded.version_id)
    blob = source.presigned(row["king"])
    assert blob["descriptor"] == dict(id=row["king"], files={"config.json": h("base1cfg"), "model.safetensors": h("base1")})
    assert sorted(store.signed) == [f"public/bases/{row['king']}/{name}" for name in sorted(blob["urls"])]


def test_one_green_is_benchmarked_once_on_its_own_pod_then_the_pod_is_released_by_exact_id(world, capsys):
    store, node, removed = world
    stacked = store.head(store.head(None, "genesis", "base0"), "r1", "base0", lora="lora1")
    D.step()
    assert not D.C_POD and not D.S.get("rents")  # no folded king yet: nothing is rented
    r1 = store.head(stacked, "r1", "base1")
    D.step(); D.step()  # rent, fetch and launch; then collect and release
    mid = "king-" + r1.base_hash[:16]
    assert D.missing(dict(model_id=mid), R.benchmarks(enabled_only=True, runnable_only=True)) == []
    assert removed == [("rm", POD["id"], "--name-only", "-y")] and not D.C_POD and len(D.S["rents"]) == 1
    table = json.load(open(C.SITE + "/benchmarks.json"))
    assert [(m["model_id"], m["status"], m["round_id"]) for m in table["models"]] == [(mid, "scored", "r1")]
    for _ in range(3):  # the same king listed again (a retried apply, a restart): no second run, no second rent
        D.step()
    assert len(D.S["rents"]) == 1 and D.queue() == []
    # the expiring links reach the node on stdin only: never in a command line, never in a log line
    assert sum("signed.example" in (s or "") for s in node.stdin) == 1
    assert "signed.example" not in " ".join(node.cmds) + capsys.readouterr().out
    r2 = store.head(store.head(r1, "r2", "base1", lora="lora2"), "r2", "base2")
    assert [m["king"] for m in D.queue()] == [r2.base_hash]  # only the new green is queued


def test_only_one_daemon_per_state_directory():
    lock = D.only_instance()
    with pytest.raises(SystemExit):
        D.only_instance()
    lock.close()


def test_settings_come_from_the_environment_and_a_missing_one_stops_the_service(monkeypatch):
    assert type(source.store()) is types.SimpleNamespace  # BENCH_STORE names "module:function"; here types:SimpleNamespace
    monkeypatch.delenv("BENCH_MAX_PRICE")
    with pytest.raises(ConfigError):
        importlib.reload(C)


def test_the_benchmark_is_informational_only():
    """The bench reads records and the key layout, nothing that decides. And nothing in v2's control plane reads the bench."""
    allowed = {"sn120_v2", "sn120_v2.layout", "sn120_v2.errors", "sn120_v2.types", "sn120_v2.params", "sn120_v2.canonical"}
    v2 = pathlib.Path(__file__).resolve().parents[2]
    for path in (v2 / "bench" / "affine_bench").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in getattr(node, "names", []) if isinstance(node, ast.Import)]
            assert not [n for n in names if n.startswith("sn120_v2") and n not in allowed], path
    for path in (v2 / "sn120_v2").rglob("*.py"):
        assert "affine_bench" not in path.read_text() and "benchmarks.json" not in path.read_text(), path
