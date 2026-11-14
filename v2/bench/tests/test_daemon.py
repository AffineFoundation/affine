"""The host path: public files and a store from the CPU simulation of v2, a fake node and a fake Lium."""
import ast, dataclasses, importlib, json, os, pathlib, re, runpy, shutil, types
import pytest
from sn120_v2 import layout, sim
from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.params import GENESIS_BASE_HASH, Params
from affine_bench import config as C, daemon as D, pod as P, publish, registry as R, source

POD = dict(id="pod-exact-id", name=P.NAME, huid="fake-pod", gpu="8xB300", cost=1.0, bootstrapped=True)
TABLE = C.SITE + "/benchmarks.json"
RUNNABLE = R.benchmarks(enabled_only=True, runnable_only=True)
BENCH_VIEW = runpy.run_path(str(pathlib.Path(__file__).resolve().parents[2] / "site" / "build.py"))["bench_view"]


class Store:
    """No copy and no delete, and a put to one key only: any other write by the service fails the test."""
    def __init__(self, inner):
        self.inner, self.signed, self.hidden, self.down, self.tables, self.read = inner, [], set(), False, [], []
        self.gone, self.refused, self.asked = set(), False, []  # objects that are not uploaded yet; a key that may not ask; the questions

    def exists(self, key):
        assert key.startswith("public/bases/"); self.asked.append(key)
        if self.refused: raise InfraError("r2 head failed: 403")
        return key not in self.gone and self.inner.exists(key)

    def put(self, key, data): assert key == layout.BENCHMARKS; self.tables.append(data); self.inner.put(key, data)
    def presign_get(self, key, ttl_s): self.signed.append(key); return f"https://signed.example/{key}?ttl={ttl_s}"

    def get(self, key):
        self.read.append(key)
        if self.down: raise InfraError("the store does not answer")
        if key in self.hidden: raise KeyError(key)
        return self.inner.get(key)


def green(v2, uid):
    """One round that a miner wins. Returns the folded head that the controller leaves."""
    v2.register(uid, 0.3)
    assert v2.round().outcome == "applied"
    return v2.controller.state.head


class Node:
    """The rented box as the daemon sees it over ssh: a fetch, one runner, and its result files.

    A fetch leaves the mark `.verified` for a model only when every link answers and every sha256 is right, as
    deploy/gate_fetch.py does. `bad(links)` says that this fetch fails; `busy` is the answer of a run that is in work."""
    def __init__(self): self.cmds, self.stdin, self.have, self.dies, self.bad, self.busy = [], [], set(), False, lambda links: False, None

    def run(self, cmd, inp=None, timeout=0):
        self.cmds.append(cmd); self.stdin.append(inp)
        if "gate_fetch.py" in cmd and not self.bad(inp):
            self.have.add(json.loads(inp)["descriptor"]["id"])
        return types.SimpleNamespace(stdout=self.out(cmd), stderr="", returncode=0)

    def out(self, cmd, timeout=0):
        if ".verified" in cmd: return "yes" if cmd.split("/.verified")[0].rsplit("/", 1)[-1] in self.have else ""
        if "echo ok=1" in cmd and self.busy: return self.busy
        if "echo ok=1" in cmd: return "done=\nalive=\nlines=0\nok=1\n" if self.dies else "done=1\nalive=\nlines=4\nok=1\n"
        if "summary.json" in cmd:
            bid = cmd.split(" 2>")[0].split("/")[-2]  # .../runs/<model>/<benchmark>/summary.json
            return json.dumps(dict(benchmark=bid, score=0.5, se=0.1, tasks=2, tasks_total=2, samples=2, rollouts=1, cap_rate=0.0,
                                   bench_version=R.bench_version(R.benchmark(bid)), finished_at=1, wall_seconds=1, mean_out_tokens=9))
        if "per_task.jsonl" in cmd: return '{"idx":0,"correct":1.0}\n{"idx":1,"correct":0.0}\n'
        return '{"wall_seconds": 60}' if cmd.endswith("/DONE") else "4321"  # 4321: the pid of the launched runner


def empty_host(monkeypatch):
    """A host with nothing on its disk and a new daemon process. The store, and so the published table, is untouched."""
    for path in (D.RES, D.PRIV, D.SF, P.RECORD, TABLE):
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else pathlib.Path(path).unlink(missing_ok=True)
    os.makedirs(D.RES); os.makedirs(D.PRIV)
    D.S.clear(); D.S.update(fail={}, unavailable=[], history=[]); D.C_POD.clear(); D.SYNCED.clear()
    for cache in ("_manifests", "_rounds", "_served"):
        monkeypatch.setattr(source, cache, {})


@pytest.fixture
def world(monkeypatch, tmp_path):
    v2, node, removed = sim.SimWorld(tmp_path, sim.sim_params()), Node(), []
    v2.store.put(layout.public_base_file(v2.controller.state.head.base_hash, "model.bin"), b"sim base")  # as the genesis job leaves it
    store = Store(v2.store)
    empty_host(monkeypatch)
    monkeypatch.setattr(source, "store", lambda: store)
    node.bad = lambda links: any(key in links for key in store.gone)  # a link to an object that is not there gives 404
    monkeypatch.setattr(D, "NODE", node); monkeypatch.setattr(P, "IDLE_RELEASE_S", 0)
    monkeypatch.setattr(D, "WAIT_LATE_S", float("inf"))  # the clock of the simulation starts in 1970: a test sets the bound when it is about it
    monkeypatch.setattr(P, "rent", lambda S, log: (S.setdefault("rents", []).append(1), P.save(POD), dict(POD))[-1])
    monkeypatch.setattr(P, "listed", lambda: [dict(id=POD["id"], name=P.NAME), dict(id="someone-elses-pod", name="other")])
    monkeypatch.setattr(P, "lium", lambda *a, **k: (removed.append(a), types.SimpleNamespace(returncode=0, stdout="", stderr=""))[-1])
    return v2, store, node, removed


def new_process():
    """A restart of the unit: the new process has only the files of the state directory, loaded as the module loads them."""
    state = D.rj(D.SF, {}) or {}
    D.S.clear(); D.S.update(state); D.S.setdefault("fail", {}); D.S.setdefault("unavailable", []); D.S.setdefault("history", [])
    D.C_POD.clear(); D.C_POD.update(P.load()); D.SYNCED.clear()


def base_keys(v2, head): return {layout.public_base_file(head.base_hash, name) for name in v2.controller.state.manifest.base_files}
def events(capsys): return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
def table(): return json.load(open(TABLE))
def states(): return {m["model_id"]: m["status"] for m in table()["models"]}
def ids(head): return ("king-" if head.parent_version_id else "genesis-") + head.base_hash[:16]


def test_rows_are_the_launch_model_and_each_folded_king_read_from_public_files_only(world, monkeypatch, tmp_path):
    v2, store = world[:2]
    genesis = v2.controller.state.head
    v2.register(1, 0.0)
    assert v2.round().outcome == "null"
    (row,) = source.models()  # a round with no winner makes no row
    assert (row["model_id"], row["kind"], row["round_id"], row["uid"]) == (ids(genesis), "genesis", "genesis", None)
    assert (row["base_hash"], row["head_version_id"], row["crowned_at"]) == (genesis.base_hash, genesis.version_id, genesis.applied_at)
    first, second = green(v2, 2), green(v2, 3)
    rows = source.models()
    assert [(r["model_id"], r["kind"], r["uid"], r["hotkey"]) for r in rows] == [
        (ids(genesis), "genesis", None, None), (ids(first), "king", 2, "hk2"), (ids(second), "king", 3, "hk3")]
    assert [(r["round_id"], r["head_version_id"], r["crowned_at"]) for r in rows[1:]] == [(h.round_id, h.version_id, h.applied_at) for h in (first, second)]
    assert all(key.startswith("public/") for key in store.read) and not hasattr(C, "JOURNAL")  # no journal, no private object
    blob = source.presigned(second.base_hash)
    assert blob["descriptor"] == dict(id=second.base_hash, files=dict(v2.controller.state.manifest.base_files))
    assert sorted(store.signed) == [layout.public_base_file(second.base_hash, name) for name in sorted(blob["urls"])]
    assert list(source.presigned(genesis.base_hash)["descriptor"]["files"]) == ["model.bin"]  # the launch model is fetched the same way

    # the heads are cached: a second listing reads the pointer and nothing else
    store.read.clear()
    assert source.models() == rows and store.read == [layout.CURRENT]

    # a public file that is not there yet cuts the chain at that point; the newer rows are still listed
    empty_host(monkeypatch)
    store.hidden = {layout.round_file(second.round_id)}
    assert [(r["model_id"], r["uid"]) for r in source.models()] == [(ids(second), None)]
    store.hidden = {layout.head_manifest(genesis.version_id)}
    assert [r["model_id"] for r in source.models()] == [ids(first), ids(second)]
    store.hidden = {layout.CURRENT}
    assert source.models() == []  # no head is published: no row
    store.hidden, store.down = set(), True
    with pytest.raises(InfraError):  # a store that does not answer is an error, never an empty list
        source.models()

    # a king whose LoRA is still stacked on the base has no row: the node serves plain weights
    stacked = sim.SimWorld(tmp_path / "k2", sim.sim_params(fold_every_k=2))
    empty_host(monkeypatch); monkeypatch.setattr(store, "inner", stacked.store); store.down = False
    assert green(stacked, 1).loras and [r["kind"] for r in source.models()] == ["genesis"]


def test_genesis_is_benchmarked_once_and_each_new_king_once_each_on_a_pod_released_by_exact_id(world, capsys):
    v2, store, node, removed = world
    genesis = v2.controller.state.head
    D.step()  # rent, fetch and launch the launch model
    assert states() == {ids(genesis): "evaluating"} and len(D.S["rents"]) == 1
    D.step()  # collect and release
    assert states() == {ids(genesis): "scored"} and D.queue() == []
    assert removed == [("rm", POD["id"], "--name-only", "-y")] and not D.C_POD
    for _ in range(3):  # listed again at each poll: no second run, no second rent, no second write of the table
        D.step()
    assert len(D.S["rents"]) == 1
    assert [json.loads(t)["models"][0]["status"] for t in store.tables] == ["evaluating", "scored"]  # one write per change
    # the expiring links reach the node on stdin only: never in a command line, never in a log line
    assert sum("signed.example" in (s or "") for s in node.stdin) == 1
    assert "signed.example" not in " ".join(node.cmds) + capsys.readouterr().out

    king = green(v2, 1)
    assert [m["base_hash"] for m in D.queue()] == [king.base_hash]  # only the new king is queued
    D.step()
    assert states() == {ids(genesis): "scored", ids(king): "evaluating"}
    D.step()
    row = next(m for m in table()["models"] if m["model_id"] == ids(king))
    assert (row["status"], row["benchmarks_done"], row["benchmarks_total"], row["fails"]) == ("scored", len(RUNNABLE), len(RUNNABLE), 0)
    assert (row["round_id"], row["head_version_id"], row["uid"], row["hotkey"]) == (king.round_id, king.version_id, 1, "hk1")
    assert len(D.S["rents"]) == 2 and json.loads(store.tables[-1])["models"] == table()["models"]  # the store holds the table

    # the site's reader of today takes the file as it is: the king first, the launch model last
    view = BENCH_VIEW(table())
    assert [(r["kind"], r["status"], r["round_id"]) for r in view["rows"]] == [("king", "scored", king.round_id), ("genesis", "scored", "genesis")]
    assert all(set(r["cells"]) >= {b["id"] for b in RUNNABLE} and r["avg"] == 0.5 for r in view["rows"])


def test_a_host_with_an_empty_disk_reads_the_published_table_and_runs_no_row_again(world, monkeypatch):
    v2, store = world[:2]
    king = green(v2, 1)
    for _ in range(4):
        D.step()
    assert set(states().values()) == {"scored"} and len(states()) == 2 and len(D.S["rents"]) == 1  # one pod serves both waiting rows
    before = table()

    empty_host(monkeypatch)
    store.down = True
    with pytest.raises(InfraError):  # the published table cannot be read: the step stops before any rent
        D.step()
    assert not D.S.get("rents") and not D.C_POD and not os.path.exists(TABLE)
    store.down = False
    for _ in range(3):
        D.step()
    assert not D.S.get("rents") and D.queue() == [] and states() == {m["model_id"]: "scored" for m in before["models"]}
    assert table()["results"] == before["results"]  # every published result is still there

    # a new king on that host: only it is run, and the write to the store keeps the rows of the old host
    second = green(v2, 2)
    D.step(); D.step()
    assert len(D.S["rents"]) == 1 and states()[ids(second)] == "scored"
    published = json.loads(store.tables[-1])
    assert {r["model_id"] for r in published["results"]} == {m["model_id"] for m in before["models"]} | {ids(second)}

    # two writers by mistake: the table in the store is the base of each write, so a stale local copy drops nothing
    pathlib.Path(TABLE).write_text(json.dumps(dict(before, models=[], results=[])))
    shutil.rmtree(D.RES); os.makedirs(D.RES); D.S.pop("uploaded")
    D.publish_table(D.model_rows())
    assert {r["model_id"] for r in json.loads(store.tables[-1])["results"]} == {r["model_id"] for r in published["results"]}
    assert ids(king) in states()


@pytest.mark.parametrize("where", ["store", "disk"])
def test_reference_rows_that_an_operator_put_in_the_table_stay(world, where):
    """Rows of other models, scored or failed, in the table of the store or in the local file before the first start."""
    v2, store = world[:2]
    cell = dict(model_id="qwen3.5-9b", benchmark_id="math500", env_version="math500/v1/bench-v2", score=0.9, se=0.01,
                complete=True, samples=500, samples_expected=500, cap_rate=0.0, budget=65536)
    seed = json.dumps(dict(updated_at=1.0, benchmarks=[], results=[cell], models=[
        dict(model_id="qwen3.5-9b", label="Qwen3.5-9B", kind="reference", status="scored", crowned_at=None),
        dict(model_id="glm-z1-9b", label="GLM-Z1-9B-0414", kind="reference", status="failed", crowned_at=None)]))
    store.inner.put(layout.BENCHMARKS, seed.encode()) if where == "store" else pathlib.Path(TABLE).write_text(seed)
    D.step(); D.step()
    rows = BENCH_VIEW(json.loads(store.tables[-1]))["rows"]
    # the site builder puts the genesis head last, under the other models
    assert [(r["label"], r["kind"], r["status"]) for r in rows[:2]] == [("Qwen3.5-9B", "reference", "scored"), ("GLM-Z1-9B-0414", "reference", "failed")]
    assert (rows[2]["kind"], rows[2]["status"]) == ("genesis", "scored") and rows[0]["cells"]["math500"]["score"] == 0.9 and not rows[1]["cells"]
    assert len(D.S["rents"]) == 1 and D.queue() == []  # a reference row is never run by this service


def test_a_row_that_fails_three_times_is_failed_until_the_operator_queues_it_again(world):
    v2, store, node = world[:3]
    mid = ids(v2.controller.state.head)
    node.dies = True
    seen = []
    for _ in range(D.MAX_FAIL + 1):  # a runner that is found dead is started again in the same step, MAX_FAIL times in all
        D.step(); seen.append((states()[mid], table()["models"][0]["fails"]))
    assert seen == [("evaluating", 0), ("evaluating", 1), ("evaluating", 2), ("failed", 3)]
    D.step()
    assert states() == {mid: "failed"} and D.queue() == [] and not D.C_POD  # no further try, and the pod is given back
    assert json.loads(store.tables[-1])["models"][0]["status"] == "failed"  # the site sees the state
    node.dies = False
    D.cli("retry", [mid])
    D.step(); D.step()
    assert states() == {mid: "scored"}


def test_a_king_whose_weight_files_are_not_in_the_store_waits_and_starts_by_itself(world, capsys):
    """The head of a king is public before its weight files (the box sends them after the next round has opened)."""
    v2, store, node, removed = world
    genesis = v2.controller.state.head
    D.step(); D.step()
    assert states() == {ids(genesis): "scored"} and not D.C_POD
    king = green(v2, 1); mid = ids(king)
    store.gone = base_keys(v2, king)
    rents, cmds, signed, tables = len(D.S["rents"]), len(node.cmds), len(store.signed), len(store.tables)
    capsys.readouterr()
    for _ in range(D.MAX_FAIL + 2):  # more polls than the tries of a row
        D.step()
    row = next(m for m in table()["models"] if m["model_id"] == mid)
    assert (row["status"], row["fails"], row["waiting_for"]) == ("queued", 0, "weights")  # the table says why the row does not run
    assert D.S["fail"] == {} and D.S["unavailable"] == [] and D.queue() == []  # no failed try, and not marked unavailable
    assert len(D.S["rents"]) == rents and not D.C_POD and len(node.cmds) == cmds and len(store.signed) == signed  # no rent, no ssh, no link
    assert len(store.tables) == tables + 1 and table()["progress"]["queued"] == 1  # one write for the new row; it is still to do
    assert [(e["event"], e["model"]) for e in events(capsys) if e["event"] != "table-written"] == [("waiting", mid)]  # one line, not one per poll

    store.gone = set(sorted(store.gone)[-1:])  # every file but one is there: an object is at its name only when it is whole
    D.step()
    assert states()[mid] == "queued" and len(D.S["rents"]) == rents
    store.gone = set()
    D.step()  # the files are there: the row starts with no command of a person
    row = next(m for m in table()["models"] if m["model_id"] == mid)
    assert row["status"] == "evaluating" and "waiting_for" not in row and len(D.S["rents"]) == rents + 1 and "waiting" not in D.S.get("waiting", {})
    assert "waiting-over" in [e["event"] for e in events(capsys)]
    D.step()
    assert states() == {ids(genesis): "scored", mid: "scored"} and D.S["fail"] == {}


def test_a_king_that_waits_longer_than_the_bound_is_shown_as_waiting(world, monkeypatch, capsys):
    v2, store = world[:2]
    D.step(); D.step()
    king = green(v2, 1); mid = ids(king)
    store.gone = base_keys(v2, king)
    age = D.now() - king.applied_at
    monkeypatch.setattr(D, "WAIT_LATE_S", age + 600)
    D.step()
    assert states()[mid] == "queued"
    tables = len(store.tables); capsys.readouterr()
    monkeypatch.setattr(D, "WAIT_LATE_S", age - 600)  # the head was applied more than the bound ago
    D.step(); D.step()
    row = next(m for m in table()["models"] if m["model_id"] == mid)
    assert (row["status"], row["waiting_for"], row["fails"]) == ("waiting", "weights", 0) and len(store.tables) == tables + 1  # one write
    assert [e["event"] for e in events(capsys) if e["event"] != "table-written"] == ["waiting-late"]
    assert [(r["kind"], r["status"]) for r in BENCH_VIEW(table())["rows"]] == [("king", "waiting"), ("genesis", "scored")]  # the site passes the word on
    assert D.S["fail"] == {} and len(D.S["rents"]) == 1 and not D.C_POD
    store.gone = set()
    D.step(); D.step()
    assert states()[mid] == "scored"  # late is not failed: it still starts by itself


def test_a_waiting_king_gets_no_try_while_another_row_runs_and_holds_no_node(world, monkeypatch):
    v2, store, node, removed = world
    genesis = v2.controller.state.head
    D.step()
    assert states() == {ids(genesis): "evaluating"}
    king = green(v2, 1); mid = ids(king)
    store.gone = base_keys(v2, king)
    cmds = len(node.cmds)
    node.busy = "done=\nalive=1\nlines=9\nok=1\n"
    D.step()  # the launch model is in work; the new king has no files yet
    assert states() == {ids(genesis): "evaluating", mid: "queued"} and not any("gate_fetch.py" in c for c in node.cmds[cmds:])
    node.busy = None
    monkeypatch.setattr(P, "IDLE_RELEASE_S", 900)
    D.step()  # collected; nothing to start
    assert states() == {ids(genesis): "scored", mid: "queued"} and D.C_POD and D.S["idle_since"] and D.S["fail"] == {}
    monkeypatch.setattr(P, "IDLE_RELEASE_S", 0)
    D.step()  # the idle rule is the same as with an empty queue: a row that waits does not keep the node
    assert not D.C_POD and removed == [("rm", POD["id"], "--name-only", "-y")] and states()[mid] == "queued"
    store.gone = set()
    D.step(); D.step()
    assert states()[mid] == "scored" and len(D.S["rents"]) == 2


def test_a_real_fault_of_the_download_is_still_a_failed_try(world, capsys):
    """The files are in the store, and the download fails on the node: a wrong sha256, or a link that the store refuses."""
    v2, store, node = world[:3]
    D.step(); D.step()
    king = green(v2, 1); mid = ids(king)
    node.bad = lambda links: True
    seen = []
    for _ in range(D.MAX_FAIL):
        D.step(); seen.append((states()[mid], D.S["fail"].get(mid)))
    assert seen == [("queued", 1), ("queued", 2), ("failed", 3)] and D.queue() == [] and D.S.get("waiting", {}) == {}

    # a key that may not ask the store holds no row back: the download decides, and its failure counts
    D.cli("retry", [mid]); store.refused = True; capsys.readouterr()
    D.step()
    assert D.S["fail"] == {mid: 1} and "store-check-failed" in [e["event"] for e in events(capsys)]
    node.bad = lambda links: False
    D.step(); D.step()
    assert states()[mid] == "scored"  # and a download that works runs the row, whatever the question to the store gave


def test_the_table_says_each_five_minutes_that_the_service_looks_at_the_row_in_work(world, monkeypatch):
    v2, store, node = world[:3]
    D.step()
    assert table()["progress"]["checked_at"] is None  # launched in this step: no look at the node yet
    node.busy = "done=\nalive=1\nlines=7\nfinished=3\nok=1\n"
    before = D.now(); D.step()
    progress = table()["progress"]; tables = len(store.tables)
    assert progress["checked_at"] >= before and (progress["output_lines"], progress["benchmarks_finished"]) == (7, 3)
    assert before <= progress["output_grew_at"] <= progress["checked_at"] and json.loads(store.tables[-1])["progress"] == progress
    assert table()["models"][0]["benchmarks_done"] == 0  # the row's own count is of collected results
    node.busy = "done=\nalive=1\nlines=8\nfinished=4\nok=1\n"
    D.step(); D.step()
    assert len(store.tables) == tables and table()["progress"]["output_lines"] == 7  # not at each poll
    monkeypatch.setattr(D, "BEAT_S", 0)
    D.step()
    assert len(store.tables) == tables + 1 and table()["progress"]["output_lines"] == 8 and table()["progress"]["checked_at"] > progress["checked_at"]
    node.busy = None
    D.step()
    assert states() == {ids(v2.controller.state.head): "scored"} and table()["progress"]["checked_at"] is None


def test_a_restart_of_the_service_leaves_the_run_and_the_wait_as_they_are(world):
    v2, store, node, removed = world
    genesis = v2.controller.state.head
    D.step()
    king = green(v2, 1); mid = ids(king)
    store.gone = base_keys(v2, king)
    node.busy = "done=\nalive=1\nlines=9\nfinished=2\nok=1\n"
    D.step()
    started, launches = D.S["running"]["started"], sum("node-run" in c for c in node.cmds)
    new_process()
    assert D.S["running"]["started"] == started and D.C_POD["id"] == POD["id"] and list(D.S["waiting"]) == [mid]
    D.step()  # the new process finds the run in work
    assert states() == {ids(genesis): "evaluating", mid: "queued"} and sum("node-run" in c for c in node.cmds) == launches
    assert len(D.S["rents"]) == 1 and not removed and D.S["fail"] == {}
    node.busy = None; store.gone = set()
    D.step()  # it collects the run that the old process started, and starts the king whose files came
    assert states() == {ids(genesis): "scored", mid: "evaluating"} and sum("node-run" in c for c in node.cmds) == launches + 1 and len(D.S["rents"]) == 1


def test_release_gives_the_pod_back_and_the_next_start_runs_the_row_again(world):
    v2, store, node, removed = world
    D.step()
    assert D.C_POD and D.S["running"]
    D.cli("release", [])
    assert removed == [("rm", POD["id"], "--name-only", "-y")] and not D.C_POD and not D.S["running"]
    D.step(); D.step()
    assert states() == {ids(v2.controller.state.head): "scored"} and len(D.S["rents"]) == 2
    with pytest.raises(SystemExit):
        D.cli("no-such-command", [])


def test_only_one_daemon_per_state_directory():
    lock = D.only_instance()
    with pytest.raises(SystemExit):
        D.only_instance()
    with pytest.raises(SystemExit):  # a command that writes the daemon's state is refused while the daemon runs
        D.cli("retry", ["king-0"])
    lock.close()


def test_the_operator_document_and_the_unit_file_are_true_to_the_code():
    bench = pathlib.Path(__file__).resolve().parents[1]
    doc, unit = (bench / "OPERATIONS.md").read_text(), (bench / "deploy" / "sn120-v2-bench.service").read_text()
    lines = [line for line in unit.splitlines() if line and not line.startswith("#")]
    assert set(re.findall(r"SN120_[A-Z0-9_]+", "\n".join(lines))) == set(re.findall(r"^#   (SN120_[A-Z0-9_]+)", unit, re.M))
    assert "ExecStart=SN120_PYTHON -m affine_bench.cli daemon" in lines and not any(line.startswith("ExecStop") for line in lines)
    blanks = set(re.findall(r"s\|(SN120_[A-Z0-9_]+)\|", doc))  # the install line of the document fills every blank of the unit
    assert blanks == set(re.findall(r"^#   (SN120_[A-Z0-9_]+)", unit, re.M))
    required = set(re.findall(r'require_env\("(BENCH_[A-Z_]+)"\)', (bench / "affine_bench" / "config.py").read_text()))
    assert len(required) == 7 and all(f"\n   {name}=" in doc for name in required)  # the settings file has each required setting
    defaults = {f.name: f.default for f in dataclasses.fields(Params)}
    evaluator = (bench.parent / "sn120_v2" / "gpu" / "evaluator.py").read_text()
    for fact in (
        f"up to {P.IDLE_RELEASE_S:.0f} seconds of idle time", f"At most {P.MAX_RENTS_PER_DAY} rents in 24 hours", f"default `{P.GPU_PREF}`",
        f"the name `{P.NAME}`", f"asks again every {P.NO_OFFER_RETRY_S:.0f} seconds", f"no progress for {D.STALL_S} seconds",
        f"After {D.MAX_FAIL} fails", f"within one poll ({D.POLL_S} seconds)", f"more than {D.WAIT_LATE_S // 60} minutes after its head was applied",
        f"about each {D.BEAT_S // 60} minutes", f"After {D.POD_CHECK_AFTER_S // 60} minutes the service asks Lium",
        f"runs the {len(RUNNABLE)} benchmarks", f"`bids` names {len(RUNNABLE)} benchmarks", f"`n_gpus = {defaults['n_gpus']}`",
        f"at most {defaults['round_time_limit_s'] // 60} minutes (`round_time_limit_s = {defaults['round_time_limit_s']}`)",
        f"take {R.serving()['server']['gpu_memory_utilization']:.2f} of each GPU's memory", f"vLLM {R.serving()['server']['engine'].split('==')[1]} serves",
        "holds more than 1024 MiB. It waits 120 seconds", "genesis-" + GENESIS_BASE_HASH[:16],
    ):
        assert fact in doc, fact
    about = pathlib.Path(publish.__file__).read_text()  # the table's own text names the same two times
    assert (D.WAIT_LATE_S, D.BEAT_S) == (3600, 300) and "lasts more than an hour" in about and "about each 5 minutes" in about
    assert "free_limit_mib: int = 1024" in evaluator and "free_timeout_s: float = 120.0" in evaluator


def test_settings_come_from_the_environment_and_a_missing_one_stops_the_service(monkeypatch):
    assert source.store() == pathlib.Path.cwd()  # BENCH_STORE names a factory as "module:Class.method"; here pathlib:Path.cwd
    monkeypatch.delenv("BENCH_MAX_PRICE")
    with pytest.raises(ConfigError):
        importlib.reload(C)


def test_the_benchmark_is_informational_only():
    """The bench reads records and the key layout, nothing that decides and no journal. And nothing in v2's control plane reads the bench."""
    allowed = {"sn120_v2", "sn120_v2.layout", "sn120_v2.errors", "sn120_v2.types", "sn120_v2.params"}
    v2 = pathlib.Path(__file__).resolve().parents[2]
    for path in (v2 / "bench" / "affine_bench").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in getattr(node, "names", []) if isinstance(node, ast.Import)]
            assert not [n for n in names if n.startswith("sn120_v2") and n not in allowed], path
    for path in (v2 / "sn120_v2").rglob("*.py"):  # layout.py names the key of the table; no v2 module reads the table
        text = path.read_text()
        assert "affine_bench" not in text and ("BENCHMARKS" not in text and "benchmarks.json" not in text or path.name == "layout.py"), path
    assert publish.OUT == TABLE
