"""The host path: public files and a store from the CPU simulation of v2, a fake node for each row and a fake Lium.

The service starts one worker process for each row. Here a worker is one call in this process: each `D.step()` of
the service gives each row in work one poll, in the order in which the service starts the workers."""
import ast, dataclasses, hashlib, importlib, json, os, pathlib, re, runpy, shutil, subprocess, types
import pytest
from sn120_v2 import canonical, layout, sim
from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.params import GENESIS_BASE_HASH, Params
from affine_bench import config as C, daemon as D, derive, pod as P, publish, registry as R, source

TABLE = C.SITE + "/benchmarks.json"
RUNNABLE = R.benchmarks(enabled_only=True, runnable_only=True)
BENCH_VIEW = runpy.run_path(str(pathlib.Path(__file__).resolve().parents[2] / "site" / "build.py"))["bench_view"]
V1 = source.extra_rows()[0]["model_id"]
ARXIV = R.benchmark("arxivmath")


class Store:
    """No copy and no delete, and a put to one key only: any other write by the service fails the test."""
    def __init__(self, inner):
        self.inner, self.signed, self.hidden, self.down, self.tables, self.read = inner, [], set(), False, [], []
        self.gone, self.refused, self.asked = set(), False, []  # objects that are not uploaded yet; a key that may not ask; the questions
        self.read_only = False  # the table cannot be written

    def exists(self, key):
        assert key.startswith(("public/bases/", "mimo/")); self.asked.append(key)
        if self.refused: raise InfraError("r2 head failed: 403")
        return key not in self.gone and self.inner.exists(key)

    def put(self, key, data):
        assert key == layout.BENCHMARKS
        if self.read_only: raise InfraError("the store refuses the write")
        self.tables.append(data); self.inner.put(key, data)

    def presign_get(self, key, ttl_s): self.signed.append(key); return f"https://signed.example/{key}?ttl={ttl_s}"

    def get(self, key):
        self.read.append(key)
        if self.down: raise InfraError("the store does not answer")
        if key in self.hidden: raise KeyError(key)
        return self.inner.get(key)


class Lium:
    """The account at Lium as the service sees it through the CLI: offers, pods, and each command that was sent."""
    def __init__(self):
        self.offers = [self.offer(f"h100-{i}", "H100", 0.8) for i in range(8)]
        self.pods, self.sent, self.ups, self.n = [], [], [], 0
        self.up_fails = 0  # this many `lium up` fail and make no pod
        self.lost = False  # `lium up` makes the pod, and its answer does not come
        self.unlisted = set()  # ids of pods that are there, and that the listing leaves out (a listing that is empty for a moment)

    @staticmethod
    def offer(huid, kind, price, n=8, free=None, tier="secure"):
        return dict(huid=huid, gpu_type=kind, gpu_count=n, available_gpu_count=n if free is None else free, tier=tier, price_per_hour=price)

    def pod(self, name, id=None, huid=None, price=0.8, config="8xH100"):
        self.n += 1
        p = dict(id=id or f"lium-id-{self.n}", name=name, huid=huid or f"node-{self.n}", config=config, price_per_hour=price,
                 ip="203.0.113.7", ports={"22": 2200 + self.n}, status="RUNNING")
        self.pods.append(p); return p

    def cli(self, *a, **k):
        self.sent.append(a); out = ""
        there = {p["id"] for p in self.pods}; seen = there - self.unlisted
        if a[:1] == ("ps",): out = json.dumps([p for p in self.pods if p["id"] in seen])
        if a[:1] == ("ls",): out = json.dumps([o for o in self.offers if o["gpu_type"] == a[2]])
        if a[:1] == ("rm",):
            if a[1] not in seen: return types.SimpleNamespace(returncode=1, stdout="", stderr="No matching pods")
            self.pods = [p for p in self.pods if p["id"] != a[1]]
        if a[:1] == ("describe",):  # as the client 0.45.1: it looks in the listing, then at the events that Lium kept for the id
            out = ("listed" if a[1] in seen else f"Pod {a[1]} is no longer listed. Its last event: RUNNING · 2026-11-14T15:00:00" if a[1] in there
                   else f"Pod {a[1]} is no longer listed. Its last event: DELETED (user_request) · 2026-11-14T15:00:00")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    def up(self, offer, name, ssh):
        self.ups.append((offer["huid"], name))
        if self.up_fails:
            self.up_fails -= 1; return None, "lium up failed: the name was refused"
        p = self.pod(name, huid=offer["huid"], price=offer["price"], config=offer["kind"])
        self.offers = [o for o in self.offers if o["huid"] != offer["huid"]]
        pathlib.Path(ssh).write_text("#!/bin/bash\n")
        return (None, "the answer of lium up was lost") if self.lost else ({k: p[k] for k in ("huid", "id", "name", "config", "price_per_hour", "ip")}, "")

    @property
    def removed(self): return [a for a in self.sent if a[0] == "rm"]


def green(v2, uid):
    """One round that a miner wins. Returns the folded head that the controller leaves."""
    v2.register(uid, 0.3)
    assert v2.round().outcome == "applied"
    return v2.controller.state.head


class Node:
    """A rented box as a worker sees it over ssh: a fetch, one runner, and its result files.

    A fetch leaves the mark `.verified` for a model only when every link answers and every sha256 is right, as
    deploy/gate_fetch.py does. `bad(links)` says that this fetch fails; `busy` is the answer of a run that is in work.
    `results` gives a benchmark another summary and other per-problem lines than the plain ones."""
    def __init__(self):
        self.cmds, self.stdin, self.have, self.dies, self.bad, self.busy, self.results = [], [], set(), False, lambda links: False, None, {}
        self.saved = []  # the files that the host made from this node's output

    def save(self, cmd, path, timeout=0):
        self.cmds.append(cmd); pathlib.Path(path).write_bytes(b"the output of the node"); self.saved.append(path); return True

    def run(self, cmd, inp=None, timeout=0):
        self.cmds.append(cmd); self.stdin.append(inp)
        if "gate_fetch.py" in cmd and not self.bad(inp):
            self.have.add(json.loads(inp)["descriptor"]["id"])
        return types.SimpleNamespace(stdout=self.out(cmd), stderr="", returncode=0)

    def out(self, cmd, timeout=0):
        if ".verified" in cmd: return "yes" if cmd.split("/.verified")[0].rsplit("/", 1)[-1] in self.have else ""
        if "echo ok=1" in cmd and self.busy: return self.busy
        if "echo ok=1" in cmd: return "done=\nalive=\nlines=0\nok=1\n" if self.dies else "done=1\nalive=\nlines=4\nok=1\n"
        bid = cmd.split(" 2>")[0].split("/")[-2] if "summary.json" in cmd or "per_task.jsonl" in cmd else None  # .../runs/<model>/<benchmark>/<file>
        if bid in self.results: return self.results[bid]["summary.json" in cmd]
        if "summary.json" in cmd:
            return json.dumps(dict(benchmark=bid, score=0.5, se=0.1, tasks=2, tasks_total=2, samples=2, rollouts=1, cap_rate=0.0,
                                   bench_version=R.bench_version(R.benchmark(bid)), finished_at=1, wall_seconds=1, mean_out_tokens=9))
        if "per_task.jsonl" in cmd: return '{"idx":0,"correct":1.0}\n{"idx":1,"correct":0.0}\n'
        return '{"wall_seconds": 60}' if cmd.endswith("/DONE") else "4321"  # 4321: the pid of the launched runner

    @property
    def launches(self): return sum("node-run" in c for c in self.cmds)


def empty_host(monkeypatch):
    """A host with nothing on its disk and a new service process. The store, and so the published table, is untouched."""
    for path in (D.RES, D.PRIV, D.ROWS, D.SF, C.STATE + "/pod.json", TABLE):
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else pathlib.Path(path).unlink(missing_ok=True)
    for path in (D.RES, D.PRIV, D.ROWS): os.makedirs(path)
    D.S.clear(); D.S.update(fail={}, unavailable=[], history=[]); D.SYNCED.clear(); D.KIDS.clear()
    for cache in ("_manifests", "_rounds", "_served"):
        monkeypatch.setattr(source, cache, {})


class World:
    def __init__(self, monkeypatch, tmp_path):
        self.v2, self.lium, self.nodes, self.polled, self.setups = sim.SimWorld(tmp_path, sim.sim_params()), Lium(), {}, [], []
        v2 = self.v2
        v2.store.put(layout.public_base_file(v2.controller.state.head.base_hash, "model.bin"), b"sim base")  # as the genesis job leaves it
        self.store = store = Store(v2.store)
        self.genesis = v2.controller.state.head
        empty_host(monkeypatch)
        monkeypatch.delenv("BENCH_V1_KING_SOURCE", raising=False)
        monkeypatch.setattr(source, "store", lambda: store)
        monkeypatch.setattr(D, "WAIT_LATE_S", float("inf"))  # the clock of the simulation starts in 1970: a test sets the bound when it is about it
        monkeypatch.setattr(D, "node_of", lambda row: self.node(row.mid))
        monkeypatch.setattr(D, "start_worker", self.poll)
        monkeypatch.setattr(P, "lium", self.lium.cli); monkeypatch.setattr(P, "up", self.lium.up)
        monkeypatch.setattr(P, "setup", lambda ssh: (self.setups.append(ssh), (True, "BOOTSTRAP_OK"))[-1])
        monkeypatch.setattr(P, "MAX_PRICE", 1.0); monkeypatch.setattr(P, "RENT_WAIT_S", 0); monkeypatch.setattr(P, "NO_OFFER_RETRY_S", 0)
        monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)

    def node(self, mid):
        if mid not in self.nodes:
            self.nodes[mid] = Node(); self.nodes[mid].bad = lambda links: any(key in links for key in self.store.gone)  # a link to an object that is not there gives 404
        return self.nodes[mid]

    def poll(self, mid):
        """What the worker process of a row does in one poll."""
        row = D.Row(mid); self.polled.append(mid)
        row.S.setdefault("model", next(m for m in D.model_rows() if m["model_id"] == mid))
        row.step(row.S["model"], RUNNABLE); row.save()

    def steps(self, n=1):
        for _ in range(n): D.step()

    def pod_of(self, mid): return next((p for p in self.lium.pods if p["name"] == P.name_of(mid)), None)


@pytest.fixture
def world(monkeypatch, tmp_path): return World(monkeypatch, tmp_path)


def base_keys(v2, head): return {layout.public_base_file(head.base_hash, name) for name in v2.controller.state.manifest.base_files}
def events(capsys): return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
def table(): return json.load(open(TABLE))
def model(mid): return next(m for m in table()["models"] if m["model_id"] == mid)
def states(v1=False): return {m["model_id"]: m["status"] for m in table()["models"] if v1 or m["model_id"] != V1}
def ids(head): return ("king-" if head.parent_version_id else "genesis-") + head.base_hash[:16]
def rm(pod): return ("rm", pod["id"], "--name-only", "-y")


def test_rows_are_the_launch_model_and_each_folded_king_read_from_public_files_only(world, monkeypatch, tmp_path):
    v2, store, genesis = world.v2, world.store, world.genesis
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
    blob = source.presigned(rows[2])
    assert blob["descriptor"] == dict(id=second.base_hash, files=dict(v2.controller.state.manifest.base_files))
    assert sorted(store.signed) == [layout.public_base_file(second.base_hash, name) for name in sorted(blob["urls"])]
    assert list(source.presigned(rows[0])["descriptor"]["files"]) == ["model.bin"]  # the launch model is fetched the same way

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


# ---------------------------------------------------------------------------------------------- one pod for each row
def test_each_row_has_a_pod_of_its_own_named_by_the_row_and_the_rows_run_side_by_side(world, capsys):
    v2, store, lium = world.v2, world.store, world.lium
    first, second = green(v2, 1), green(v2, 2)
    rows = [ids(second), ids(first), ids(world.genesis)]  # the queue: newest first, the launch model last
    world.steps()  # one step of the service: each row rents, sets up, downloads and starts on a pod of its own
    assert world.polled == rows and states() == dict.fromkeys(rows, "evaluating")
    assert sorted(p["name"] for p in lium.pods) == sorted(P.name_of(mid) for mid in rows) and len(lium.ups) == 3
    # a name is the first 29 characters of affine-v2-bench-<model_id>: for a king, the 8 hex digits of the table's label
    assert all(P.name_of(mid) == f"affine-v2-bench-{mid}"[:29] and len(P.name_of(mid)) == P.NAME_MAX == 29 for mid in rows)
    assert P.name_of(ids(first)) == "affine-v2-bench-king-" + first.base_hash[:8] and first.base_hash[:8] in model(ids(first))["label"]
    assert len({huid for huid, name in lium.ups}) == 3  # no two rows on one offer
    assert [world.nodes[mid].launches for mid in rows] == [1, 1, 1] and len(world.setups) == 3
    assert len({D.pod_of(mid).record for mid in rows} | {D.pod_of(mid).ssh for mid in rows}) == 6  # a record and an ssh wrapper for each row
    pods = {mid: dict(world.pod_of(mid)) for mid in rows}
    for mid in rows:  # the table names the machine of each row, with no id and no address
        assert model(mid)["pod"] == dict(name=pods[mid]["huid"], gpu="8xH100", usd_h=0.8, state="running", since=model(mid)["pod"]["since"])
    assert table()["progress"]["evaluating"] == rows[0] and table()["progress"]["evaluating_all"] == rows
    assert "lium-id" not in open(TABLE).read() and "203.0.113.7" not in open(TABLE).read() + capsys.readouterr().out

    world.steps()  # each run has ended: the results are on the host, and the service writes them to the store
    assert states() == dict.fromkeys(rows, "scored") and len(lium.pods) == 3 and not lium.removed  # no pod goes before the store has the results
    world.steps()  # each worker reads the row's results in the table of the store, and gives its pod back by its exact id
    assert sorted(lium.removed) == sorted(rm(pods[mid]) for mid in rows) and not lium.pods
    assert all(model(mid)["pod"]["state"] == "released" and not D.pod_of(mid).load() for mid in rows)
    polled = len(world.polled)
    world.steps(3)  # listed again at each poll: no worker, no rent, no run, no write of the table
    assert len(world.polled) == polled and len(lium.ups) == 3 and [world.nodes[mid].launches for mid in rows] == [1, 1, 1]
    assert [sorted({m["status"] for m in json.loads(t)["models"] if m["model_id"] != V1}) for t in store.tables] == [["evaluating"], ["scored"], ["scored"]]
    # the expiring links reach a node on stdin only: never in a command line, never in a log line
    assert all(sum("signed.example" in (s or "") for s in world.nodes[mid].stdin) == 1 for mid in rows)
    assert "signed.example" not in " ".join(c for n in world.nodes.values() for c in n.cmds) + capsys.readouterr().out

    king = green(v2, 3)  # a later king: its own pod, by itself
    assert [m["base_hash"] for m in D.queue()] == [king.base_hash]
    world.steps(3)
    row = model(ids(king))
    assert (row["status"], row["benchmarks_done"], row["benchmarks_total"], row["fails"]) == ("scored", len(RUNNABLE), len(RUNNABLE), 0)
    assert (row["round_id"], row["head_version_id"], row["uid"], row["hotkey"]) == (king.round_id, king.version_id, 3, "hk3")
    assert len(lium.ups) == 4 and not lium.pods and json.loads(store.tables[-1])["models"] == table()["models"]  # the store holds the table

    # the site's reader of today takes the file as it is: the kings first, newest first, and the launch model last
    view = BENCH_VIEW(table())
    assert [(r["kind"], r["status"], r["round_id"]) for r in view["rows"] if r["kind"] != "v1-king"] == [
        ("king", "scored", h.round_id) for h in (king, second, first)] + [("genesis", "scored", "genesis")]
    assert all(set(r["cells"]) >= {b["id"] for b in RUNNABLE} and r["avg"] == 0.5 for r in view["rows"] if r["kind"] != "v1-king")


def test_a_row_that_cannot_rent_holds_up_no_other_row(world, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    lium.offers = lium.offers[:1]  # one machine for two rows
    world.steps()
    assert states() == {mid: "evaluating", gen: "queued"} and model(gen)["pod"]["state"] == "renting" and len(lium.pods) == 1
    assert "rent-no-offer" in [e["event"] for e in events(capsys) if e.get("row") == gen]
    world.steps(2)  # the king is scored and its pod is back while the other row still asks for an offer
    assert states() == {mid: "scored", gen: "queued"} and not lium.pods and D.row_state(gen)["fails"] == 0 and not D.stopped(gen)
    lium.offers = [Lium.offer("late-offer", "H100", 0.9)]
    world.steps(3)  # "no offer" is not a failed try: the row rents when an offer is there
    assert states() == {mid: "scored", gen: "scored"} and lium.ups[-1] == ("late-offer", P.name_of(gen))


def test_the_pod_is_given_back_only_when_the_table_in_the_store_has_the_rows_results(world, capsys):
    store, lium, gen = world.store, world.lium, ids(world.genesis)
    world.steps()
    store.read_only = True  # the results reach the host, and the table cannot be written to the store
    for _ in range(3):
        try: D.step()
        except InfraError: pass
    assert len(lium.pods) == 1 and not lium.removed and D.row_state(gen)["release_waits"]
    assert [e["benchmarks"] for e in events(capsys) if e["event"] == "release-waits"] == [[b["id"] for b in RUNNABLE]]  # one line, not one per poll
    store.read_only = False
    pod = dict(lium.pods[0])
    world.steps(2)
    assert lium.removed == [rm(pod)] and states() == {gen: "scored"}


def test_a_rent_takes_the_cheapest_whole_secure_node_of_an_allowed_kind_at_or_below_the_cap(world, monkeypatch, capsys):
    lium, gen = world.lium, ids(world.genesis)
    monkeypatch.setattr(P, "MAX_PRICE", 35.0)
    lium.offers = [
        Lium.offer("h200-secure", "H200", 33.6), Lium.offer("h100-dear", "H100", 23.2), Lium.offer("h100-cheap", "H100", 22.0),
        Lium.offer("h100-community", "H100", 9.0, tier="community"), Lium.offer("h100-part", "H100", 10.0, free=6), Lium.offer("h100-four", "H100", 8.0, n=4),
        Lium.offer("a100", "A100", 5.0), Lium.offer("b300-over", "B300", 35.01), Lium.offer("b200-same-price", "B200", 22.0)]
    offers, _ = P.offers()
    assert [(o["huid"], o["kind"], o["price"]) for o in offers] == [
        ("h100-cheap", "8xH100", 22.0), ("b200-same-price", "8xB200", 22.0), ("h100-dear", "8xH100", 23.2), ("h200-secure", "8xH200", 33.6)]
    D.cli("offers", [])  # the look of the operator: kind, price and count, no machine name and no address
    out = capsys.readouterr().out
    assert "8xH100  22.0 $/h  x1" in out and "8xH200  33.6 $/h  x1" in out and "4 offers fit" in out and "h100-cheap" not in out
    world.steps()
    assert lium.ups == [("h100-cheap", P.name_of(gen))] and D.pod_of(gen).load()["cost"] == 22.0
    assert all(a[0] in ("ps", "ls") for a in lium.sent)  # a rent sends no other command than its two readings

    # nothing fits: no pod, no failed try, and the row says that it still looks
    monkeypatch.setattr(P, "MAX_PRICE", 21.0); P.Pod(gen).save({}); lium.pods.clear(); D.wj(D.ROWS + f"/{gen}/daemon.json", {})
    world.steps(2)
    assert len(lium.ups) == 1 and not D.stopped(gen) and model(gen)["pod"]["state"] == "renting"
    # a pod that came dearer than the cap is given back at once, by its exact id
    monkeypatch.setattr(P, "offers", lambda: ([dict(huid="h200-secure", gpus=8, kind="8xH200", price=33.6, order=0)], []))
    world.steps()
    assert lium.ups[-1][0] == "h200-secure" and len(lium.removed) == 1 and not lium.pods and D.row_state(gen)["rent_tries"] == 1


def test_a_lost_answer_of_lium_never_makes_a_second_pod_for_a_row(world, capsys):
    lium, gen = world.lium, ids(world.genesis)
    lium.lost = True  # the pod is made, and the worker never hears of it
    world.steps()
    assert len(lium.pods) == 1 and len(lium.ups) == 1 and D.row_state(gen)["rent_tries"] == 1
    pod = dict(lium.pods[0])
    assert D.pod_of(gen).load()["id"] == pod["id"] and D.pod_of(gen).load()["bootstrapped"] is False  # found under the row's exact name
    lium.lost = False
    world.steps()  # the next try takes that pod: no second `lium up`
    assert len(lium.ups) == 1 and len(lium.pods) == 1 and states() == {gen: "evaluating"} and world.nodes[gen].launches == 1
    assert "203.0.113.7" in open(D.pod_of(gen).ssh).read() and "203.0.113.7" not in capsys.readouterr().out  # the wrapper is made from Lium's listing
    world.steps(2)
    assert lium.removed == [rm(pod)] and states() == {gen: "scored"}

    # a pod under the row's name that the row has no record of at all (the host lost its disk): taken, not doubled
    king = green(world.v2, 1); mid = ids(king)
    stray = dict(lium.pod(P.name_of(mid)))
    world.steps()
    assert len(lium.ups) == 1 and D.pod_of(mid).load()["id"] == stray["id"] and states()[mid] == "evaluating"
    world.steps(2)
    assert lium.removed[-1] == rm(stray)

    # two pods under one row's name: the service takes none, makes none and deletes none. The operator decides
    second = green(world.v2, 2); mid2 = ids(second)
    lium.pod(P.name_of(mid2)); lium.pod(P.name_of(mid2))
    removed, ups = len(lium.removed), len(lium.ups)
    world.steps(P.RENT_TRIES + 1)
    assert len(lium.ups) == ups and len(lium.removed) == removed and len(lium.pods) == 2 and D.stopped(mid2) == "rent_failed"
    D.cli("pods", [])
    assert capsys.readouterr().out.count("NO ROW CLAIMS THIS POD") == 2


def test_a_rent_that_fails_is_tried_again_with_a_growing_wait_and_then_the_row_says_so(world, monkeypatch, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    monkeypatch.setattr(P, "RENT_WAIT_S", 120)
    real_up = lium.up
    monkeypatch.setattr(P, "up", lambda offer, name, ssh: (lium.ups.append((offer["huid"], name)), (None, "Lium refused the name"))[-1] if name == P.name_of(mid) else real_up(offer, name, ssh))
    world.steps()
    s = D.row_state(mid)
    assert s["rent_tries"] == 1 and 100 < s["rent_wait_until"] - D.now() <= 120 and states() == {mid: "queued", gen: "evaluating"}
    world.steps()
    assert D.row_state(mid)["rent_tries"] == 1  # the wait runs: no try
    waits = []
    for _ in range(P.RENT_TRIES - 1):
        s = D.row_state(mid); s["rent_wait_until"] = 0; D.wj(D.ROWS + f"/{mid}/daemon.json", s)
        world.steps(); waits.append(D.row_state(mid).get("rent_wait_until", 0) - D.now())
    assert 230 < waits[0] <= 240 and 470 < waits[1] <= 480  # 2, 4 and 8 minutes
    row = model(mid)
    assert (row["status"], row["stopped"], row["pod"]["state"], row["fails"]) == ("failed", "rent_failed", "rent-failed", 0)
    assert D.row_state(mid)["rent_failed"]["tries"] == P.RENT_TRIES and "Lium refused the name" in D.row_state(mid)["rent_failed"]["why"]
    assert [e["tries"] for e in events(capsys) if e["event"] == "rent-failed"] == [1, 2, 3, 4]
    assert states()[gen] == "scored" and not lium.pods  # the other row ran, was scored and gave its pod back meanwhile
    polled, ups = len(world.polled), len(lium.ups)
    world.steps(3)
    assert len(world.polled) == polled and len(lium.ups) == ups  # no further try, no worker, no spend
    monkeypatch.setattr(P, "up", real_up)
    D.cli("retry", [mid])  # the operator's word; the service need not stop for it
    world.steps(3)
    assert states() == {mid: "scored", gen: "scored"}


def test_a_setup_that_fails_gives_the_pod_back_and_counts_as_a_failed_rent(world, monkeypatch):
    lium, gen = world.lium, ids(world.genesis)
    monkeypatch.setattr(P, "setup", lambda ssh: (False, "bootstrap.sh: line 9: error"))
    world.steps(P.RENT_TRIES)
    assert len(lium.ups) == P.RENT_TRIES and len(lium.removed) == P.RENT_TRIES and not lium.pods and D.stopped(gen) == "rent_failed"
    assert world.nodes.get(gen) is None or world.nodes[gen].launches == 0


# ---------------------------------------------------------------------------------------------- the one place that deletes
def test_a_delete_goes_to_lium_only_for_the_exact_id_of_the_rows_record_and_never_for_a_protected_pod(world, monkeypatch, capsys):
    lium = world.lium; row = "king-" + "a" * 16
    pod = P.Pod(row); other = lium.pod("someone-elses-pod"); twin = lium.pod(pod.name + "-old")
    mine = lium.pod(pod.name)
    pod.save(dict(id=mine["id"], name=pod.name, huid=mine["huid"], rented_at=D.now()))
    assert pod.release(D.log, why="test") and lium.removed == [rm(mine)] and lium.pods == [other, twin]  # one command, one id; a like name is not touched
    assert pod.release(D.log, why="again") and len(lium.removed) == 1  # no record: no command

    refused = lambda: [e["why"] for e in events(capsys) if e["event"] == "release-refused"]
    capsys.readouterr()
    # the prod GPU box and the operator's pod, by id: also when a record is wrong and names one of them, under any name
    for bad_id in ("6cb913e4-0000-4000-8000-000000000000", "c44cb565-0000-4000-8000-000000000000"):
        lium.pod(pod.name, id=bad_id); pod.save(dict(id=bad_id, name=pod.name, huid="x"))
        assert not pod.release(D.log, why="wrong record") and "is protected" in refused()[0]
        lium.pods.pop()
    # and by name: in the record, or as Lium lists the id
    for name in ("sn120-v2-prod-dublin", "casper-h100-ask-before-delete"):
        p = lium.pod(name); pod.save(dict(id=p["id"], name=pod.name, huid="x"))
        assert not pod.release(D.log, why="wrong record") and "is protected" in refused()[0]
        q = lium.pod(pod.name); pod.save(dict(id=q["id"], name=pod.name, huid=name))
        assert not pod.release(D.log, why="wrong record") and "is protected" in refused()[0]
        lium.pods = lium.pods[:-2]
    # a record that names the pod of another name: refused
    pod.save(dict(id=other["id"], name=pod.name, huid="x"))
    assert not pod.release(D.log, why="wrong record") and "is named 'someone-elses-pod'" in refused()[0]
    assert len(lium.removed) == 1 and other in lium.pods  # none of the refusals sent a command

    # the code has one place that sends a delete: one "rm" in the whole host code, with the id of the record
    src = {p.name: p.read_text() for p in pathlib.Path(P.__file__).parent.glob("*.py")}
    assert sum(text.count('"rm"') for text in src.values()) == 1 and 'lium("rm", pid, "--name-only", "-y"' in src["pod.py"]
    assert not any(word in text for text in src.values() for word in ("lium rm", "--all", '"down"'))


# ---------------------------------------------------------------------------------------------- money
def test_at_most_max_pods_rows_are_in_work_and_the_next_row_starts_when_one_is_done(world, monkeypatch):
    v2, lium = world.v2, world.lium
    first, second = green(v2, 1), green(v2, 2)
    monkeypatch.setattr(P, "MAX_PODS", 2)
    world.steps()
    assert states() == {ids(second): "evaluating", ids(first): "evaluating", ids(world.genesis): "queued"} and len(lium.pods) == 2
    assert table()["progress"]["queued"] == 3
    world.steps(2)
    assert len(lium.ups) == 2 and not lium.pods
    world.steps(3)
    assert states()[ids(world.genesis)] == "scored" and len(lium.ups) == 3 and not lium.pods
    # at most MAX_RENTS_PER_DAY pods are made in 24 hours, over all rows
    monkeypatch.setattr(P, "MAX_RENTS_PER_DAY", 3)
    king = green(v2, 3)
    world.steps(2)
    assert len(lium.ups) == 3 and states()[ids(king)] == "queued" and not D.stopped(ids(king))


def test_a_pod_lives_six_hours_at_most_and_a_row_that_fails_or_hangs_keeps_no_pod(world, monkeypatch, capsys):
    lium, gen = world.lium, ids(world.genesis)
    assert (P.POD_MAX_HOURS, P.MAX_PODS, P.MAX_RENTS_PER_DAY, P.RENT_TRIES, D.MAX_FAIL) == (6, 8, 16, 4, 3)
    world.steps()
    node = world.nodes[gen]; node.busy = "done=\nalive=1\nlines=9\nfinished=2\nok=1\n"  # a run that never ends: two benchmarks are finished
    for b in RUNNABLE[2:]: node.results[b["id"]] = {True: "", False: ""}
    pod = dict(lium.pods[0])
    rec = D.pod_of(gen).load(); rec["rented_at"] = D.now() - 6 * 3600 - 1; D.pod_of(gen).save(rec)
    world.steps()  # past its longest life: the run is stopped, what is finished is brought to the host
    assert any("kill -TERM 4321" in c for c in node.cmds) and D.stopped(gen) == "expired" and "pod-expired" in [e["event"] for e in events(capsys)]
    assert len(lium.pods) == 1 and model(gen)["benchmarks_done"] == 2  # the two results go to the store first
    world.steps()
    row = model(gen)
    assert lium.removed == [rm(pod)] and not lium.pods and (row["status"], row["stopped"], row["pod"]["state"], row["fails"]) == ("failed", "expired", "released", 0)
    world.steps(2)
    assert len(lium.ups) == 1 and node.launches == 1  # no new pod and no new run by itself
    node.busy = None; node.results.clear()
    D.cli("retry", [gen])  # the operator's word: a pod of its own again, and only the benchmarks that are missing
    world.steps()
    assert len(lium.ups) == 2 and "--benchmarks " + ",".join(b["id"] for b in RUNNABLE[2:]) + " " in world.nodes[gen].cmds[-1]


def test_a_row_that_fails_three_times_gives_its_pod_back_and_is_failed_until_the_operator_queues_it_again(world):
    lium, gen = world.lium, ids(world.genesis)
    world.steps()
    world.nodes[gen].dies = True; pod = dict(lium.pods[0])
    seen = []
    for _ in range(D.MAX_FAIL):  # a runner that is found dead is started again in the same poll, MAX_FAIL times in all
        world.steps(); seen.append((states()[gen], model(gen)["fails"]))
    assert seen == [("evaluating", 1), ("evaluating", 2), ("failed", 3)] and world.nodes[gen].launches == D.MAX_FAIL
    world.steps()
    assert lium.removed == [rm(pod)] and not lium.pods and model(gen)["stopped"] == "failed"  # no further try, and the pod is given back
    assert json.loads(world.store.tables[-1])["models"][0]["status"] == "failed"  # the site sees the state
    polled = len(world.polled); world.steps(2)
    assert len(world.polled) == polled and len(lium.ups) == 1
    world.nodes[gen].dies = False
    D.cli("retry", [gen])
    world.steps(3)
    assert states() == {gen: "scored"} and len(lium.ups) == 2 and not lium.pods


def test_a_run_that_ends_without_a_result_is_a_failed_try_and_a_result_without_its_lines_is_not_stored(world, capsys):
    gen = ids(world.genesis)
    world.steps()
    node = world.nodes[gen]
    node.results["gsm8k"] = {True: json.dumps(dict(benchmark="gsm8k", score=0.5, se=0.1, tasks=2, tasks_total=2, samples=2, rollouts=1, cap_rate=0.0,
                                                   bench_version=R.bench_version(R.benchmark("gsm8k")), finished_at=1)), False: ""}
    world.steps()
    assert not os.path.exists(D.result_path(gen, "gsm8k")) and D.row_state(gen)["fails"] == 1 and model(gen)["benchmarks_done"] == len(RUNNABLE) - 1
    assert [e["benchmarks"] for e in events(capsys) if e["event"] == "run-incomplete"] == [["gsm8k"]]
    assert node.launches == 2 and "--benchmarks gsm8k " in node.cmds[-1]  # only the missing benchmark is run again
    del node.results["gsm8k"]
    world.steps(2)
    assert states() == {gen: "scored"}


# ---------------------------------------------------------------------------------------------- the row of the single-node service
def single_node_state(world, mid, m):
    """The state directory as the single-node service leaves it while a row runs: its run in daemon.json, its pod in pod.json."""
    calm = world.lium.pod(P.NAME, id="296c5237-0000-4000-8000-000000000000", huid="calm-matrix-26", price=23.2)
    P.Pod(mid, first=True).save(dict(name=P.NAME, id=calm["id"], huid="calm-matrix-26", gpu="8xH100", cost=23.2, rented_at=D.now() - 12 * 3600, bootstrapped=True))
    D.S.update(running=dict(model=m, run_key=mid, bids=[b["id"] for b in RUNNABLE], started=D.now() - 12 * 3600, pid="4321", last_progress=D.now(),
                            progress=5, pod_id=calm["id"]), fail={}, history=[])
    D.save()
    return calm


def full_run(right):
    """An ArXivMath run of the version before: 16 samples for each of the 103 problems. right[i] of problem i's samples are right."""
    lines = [dict(idx=i, key=f"p{i}", n=16, correct=float(c), errors=0, capped=0, tokens=[1] * 16) for i, c in enumerate(right)]
    means = [c / 16 for c in right]; score = sum(means) / len(means)
    summary = dict(benchmark="arxivmath", score=score, se=0.03, tasks=len(lines), tasks_total=len(lines), samples=sum(p["n"] for p in lines), rollouts=16,
                   cap_rate=0.0, bench_version="arxivmath/v1/bench-v2", finished_at=1, wall_seconds=30000, mean_out_tokens=9, errors=0, complete=True)
    return {True: json.dumps(summary), False: "".join(json.dumps(p) + "\n" for p in lines)}


def scored_by_the_single_node_service(mid, right=None, lines=True, spoil=None):
    """The host as the single-node service leaves it when a king's run has ended: a result of each benchmark with the profile
    of then, ArXivMath at 16 samples a problem (the version of before), and its per-problem lines in the private directory."""
    for b in RUNNABLE:
        r = dict(benchmark=b["id"], score=0.5, se=0.03, tasks=10, tasks_total=10, samples=10 * b["rollouts"], rollouts=b["rollouts"], cap_rate=0.0,
                 bench_version=R.bench_version(b), finished_at=1, wall_seconds=600, mean_out_tokens=9, errors=0, complete=True)
        if b["id"] == "arxivmath":
            run = full_run(right or [(i * 7) % 17 for i in range(103)]); r = json.loads(run[True]); per = [json.loads(x) for x in run[False].splitlines()]
            if spoil: spoil(r, per)
            if lines:
                os.makedirs(f"{D.PRIV}/{mid}", exist_ok=True); pathlib.Path(f"{D.PRIV}/{mid}/arxivmath.per_task.jsonl").write_text("".join(json.dumps(x) + "\n" for x in per))
        r.update(model_id=mid, model_kind="king", profile=PP15, node=dict(pod=P.NAME, lium_id="296c5237-0000-4000-8000-000000000000", gpu="8xH100"), collected_at=1)
        os.makedirs(f"{D.RES}/{mid}", exist_ok=True); D.wj(D.result_path(mid, b["id"]), r)


def test_the_run_of_the_single_node_service_is_cut_at_the_first_start_and_its_pod_is_given_back(world, monkeypatch, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "MAX_PRICE", 35.0)
    calm = single_node_state(world, mid, m)
    # a command beside the stopped service: a run is recorded on that pod, so it stays, and that is a counted try
    assert not D.release_single(force=True) and not lium.removed and calm in lium.pods and D.S["single_release"]["tries"] == 1
    capsys.readouterr()
    world.steps()
    said = events(capsys)
    # the run is cut, not adopted: its record is kept under another name, and no row is a first row
    assert D.S["running"] is None and (D.S["cut_run"]["run_key"], D.S["cut_run"]["pid"]) == (mid, "4321")
    (cut,) = [e for e in said if e["event"] == "single-run-cut"]
    assert (cut["row"], cut["pod_id"]) == (mid, "296c5237")
    # the pod of the single-node service is given back by its exact id, with the one delete of the code
    assert lium.removed == [rm(calm)] and calm not in lium.pods and table()["node"]["up"] is False and not D.single_pod().load()
    # the row of the cut run is a row as any other: a pod of its own, every benchmark, the settings of now
    assert sorted(name for _, name in lium.ups) == sorted([P.name_of(mid), P.name_of(gen)]) and D.pod_of(mid).record != C.STATE + "/pod.json"
    assert world.node(mid).launches == 1 and all(f"{b['id']}" in world.node(mid).cmds[-1] for b in RUNNABLE) and model(mid)["profile"] == PP0
    world.steps(3)
    assert states() == {mid: "scored", gen: "scored"} and not lium.pods and world.node(mid).launches == 1
    cell = next(x for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath")
    assert (cell["env_version"], cell.get("derived_from")) == ("arxivmath/v2/bench-v2", None)
    # a later start with no run of that service cuts nothing and deletes nothing more
    capsys.readouterr(); world.steps()
    assert len(lium.removed) == 3 and not [e for e in events(capsys) if e["event"] == "single-run-cut"]


def test_a_run_of_the_single_node_service_is_cut_at_every_start_also_after_a_way_back_to_that_service(world, monkeypatch, capsys):
    """The way back: the single-node service keeps the key `rows_mode` that it does not know, rents its pod and starts a run.
    Then this code starts again: both keys are in the record."""
    lium = world.lium; king = green(world.v2, 1); mid, gen = ids(king), ids(world.genesis)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "MAX_PRICE", 35.0); monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    calm = single_node_state(world, mid, m); D.S["rows_mode"] = D.now() - 7200; D.save()
    capsys.readouterr(); world.steps()
    (cut,) = [e for e in events(capsys) if e["event"] == "single-run-cut"]
    assert cut["row"] == mid and D.S["running"] is None and D.S["cut_run"]["pid"] == "4321"
    # one pod for the row, and it is the row's own: the pod of that service is given back, the run on it is not adopted
    assert rm(calm) in lium.removed and calm not in lium.pods and not D.single_pod().load()
    assert sorted(name for _, name in lium.ups) == sorted([P.name_of(mid), P.name_of(gen)]) and world.node(mid).launches == 1
    assert not [a for a in table()["alarms"] if a["what"] == "single-pod-not-released"]


@pytest.mark.parametrize("marker", ["in the directory of a row that is not listed", "on a listed row with a run record of its own"])
def test_a_marker_file_of_a_first_row_has_no_meaning_the_pod_of_the_single_node_service_is_given_back(world, monkeypatch, capsys, marker):
    """Only the code of commit b1a89899 or a hand can have made the file `rows/<row>/first`."""
    lium = world.lium; king = green(world.v2, 1); mid, gen = ids(king), ids(world.genesis)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "MAX_PRICE", 35.0); monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    calm = single_node_state(world, mid, m); run = D.S.pop("running"); D.S["rows_mode"] = D.now() - 7200; D.save()
    row = mid if marker.startswith("on a listed row") else "king-0000000000000000"
    os.makedirs(f"{D.ROWS}/{row}"); pathlib.Path(f"{D.ROWS}/{row}/first").write_text("")
    if row == mid:  # as b1a89899 left it: the run on the pod of the single-node service is the row's, and its runner lives
        D.wj(f"{D.ROWS}/{mid}/daemon.json", dict(fails=0, history=[], running=run))
    capsys.readouterr(); world.steps()
    said = events(capsys)
    assert rm(calm) in lium.removed and calm not in lium.pods and not D.single_pod().load()
    # the row has one pod, its own, and one run, started by this code: the run on the other pod is not adopted
    assert sorted(name for _, name in lium.ups) == sorted([P.name_of(mid), P.name_of(gen)]) and D.pod_of(mid).record == f"{D.ROWS}/{mid}/pod.json"
    assert world.node(mid).launches == 1 and D.row_state(mid)["running"]["pod_id"] == world.pod_of(mid)["id"] != calm["id"]
    assert ([e["old_pod"] for e in said if e["event"] == "pod-changed"] == [calm["id"]]) == (row == mid)
    assert not table()["alarms"]


def test_the_single_node_record_that_names_the_pod_of_a_row_never_brings_a_delete_of_that_pod(world, monkeypatch, capsys):
    """BENCH_POD_NAME is the name of a row's pod, and pod.json of the single-node service names that pod."""
    lium = world.lium; king = green(world.v2, 1); mid = ids(king)
    monkeypatch.setattr(P, "MAX_PRICE", 35.0); monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    world.node(mid).busy = "done=\nalive=1\nlines=7\nfinished=1\nok=1\n"
    world.steps()
    pod = world.pod_of(mid); assert pod and D.row_state(mid)["running"]
    for name in (P.name_of(mid), "affine-v2-bench-1"):  # by the name, and by the id alone
        monkeypatch.setattr(P, "NAME", name)
        P.Pod("", first=True).save(dict(name=name, id=pod["id"], huid="x", gpu="8xH100", cost=23.2, rented_at=D.now() - 3600, bootstrapped=True))
        capsys.readouterr(); world.steps(D.ALARM_TRIES)
        assert not D.release_single(force=True)  # also not on the operator's word
        assert pod in lium.pods and rm(pod) not in lium.removed and D.row_state(mid)["running"] and D.pod_of(mid).load()["id"] == pod["id"]
        assert "single-pod-not-released" in [a["what"] for a in table()["alarms"]]
        assert all("that pod is the row's" in e["why"] for e in events(capsys) if e["event"] == "release-refused")
        D.S.pop("single_release", None); D.save()


def test_the_pod_of_the_single_node_service_that_cannot_be_given_back_is_an_alarm_and_no_pod_without_row(world, monkeypatch, capsys):
    lium = world.lium; king = green(world.v2, 1); mid = ids(king)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "MAX_PRICE", 35.0); monkeypatch.setattr(D, "LIUM_CHECK_S", 0); monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    calm = single_node_state(world, mid, m); D.S.pop("running"); D.save()  # the single-node service was between two rows
    calm["name"] = "another-name"  # Lium lists the id under a name that is not the service's: the delete guard refuses
    world.steps(D.ALARM_TRIES)
    assert rm(calm) not in lium.removed and calm in lium.pods and D.single_pod().load()["id"] == calm["id"]
    (alarm,) = [a for a in table()["alarms"] if a["what"] == "single-pod-not-released"]
    assert alarm["pod"] == P.NAME and alarm["tries"] >= D.ALARM_TRIES and not [a for a in table()["alarms"] if a["what"] == "pod-without-row"]
    capsys.readouterr(); D.cli("pods", [])
    assert "THE POD OF THE SINGLE-NODE SERVICE: no row runs on it, the service gives it back" in capsys.readouterr().out
    calm["name"] = P.NAME; world.steps()
    assert rm(calm) in lium.removed and not [a for a in table()["alarms"] if a["what"] == "single-pod-not-released"]


@pytest.mark.parametrize("record", ["6cb913e4", "c44cb565", "a pod of another name", "a name of the service that is not a bench name"])
def test_the_single_node_record_never_brings_a_delete_of_a_protected_pod_or_of_a_pod_that_is_not_a_bench_pod(world, monkeypatch, capsys, record):
    lium = world.lium; monkeypatch.setattr(P, "MAX_PRICE", 35.0); monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    if record == "a name of the service that is not a bench name": monkeypatch.setattr(P, "NAME", "someone-elses-pod")
    name = "sn120-v2-prod-dublin" if record == "6cb913e4" else "casper-h100-ask-before-delete" if record == "c44cb565" else "someone-elses-pod"
    other = lium.pod(name, id=(record if len(record) == 8 else "0a1b2c3d") + "-0000-4000-8000-000000000000")
    for said in (name, P.NAME):  # whatever name the record gives the pod
        P.Pod("", first=True).save(dict(name=said, id=other["id"], huid="x", gpu="8xH100", cost=23.2, rented_at=D.now() - 3600, bootstrapped=True))
        D.S.update(running=None, fail={}, history=[]); D.S.pop("rows_mode", None); D.save()
        capsys.readouterr(); world.steps(D.ALARM_TRIES)
        assert not D.release_single(force=True)  # also not on the operator's word
        assert other in lium.pods and not [c for c in lium.sent if c[0] == "rm" and c[1] == other["id"]]  # no delete call was made for it
        assert D.single_pod().load()["id"] == other["id"] and [a["what"] for a in table()["alarms"]].count("single-pod-not-released") == 1
        why = [e["why"] for e in events(capsys) if e["event"] == "release-refused"]
        assert why and all("is protected" in w or "is named" in w or "is not a bench name" in w for w in why)


def test_a_row_that_the_single_node_service_scored_is_not_queued_again_and_rents_nothing(world, monkeypatch, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    assert (ARXIV["rollouts"], ARXIV["problems"], R.bench_version(ARXIV)) == (2, 103, "arxivmath/v2/bench-v2")
    right = [(i * 7) % 17 for i in range(103)]
    scored_by_the_single_node_service(mid, right)
    assert D.missing(m, RUNNABLE) == ["arxivmath"]  # by the settings of now the row lacks ArXivMath: without the step below it would rent a pod
    D.sync_table(); D.publish_table(D.model_rows())  # the table in the store, as the single-node service wrote it: the full run at 16
    old = [x for x in json.loads(world.store.tables[-1])["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath"]
    assert [(x["env_version"], x["samples"]) for x in old] == [("arxivmath/v1/bench-v2", 1648)]
    calm = lium.pod(P.NAME, id="296c5237-0000-4000-8000-000000000000", huid="calm-matrix-26", price=23.2)  # the pod of that service, with no run on it
    P.Pod("", first=True).save(dict(name=P.NAME, id=calm["id"], huid="calm-matrix-26", gpu="8xH100", cost=23.2, rented_at=D.now() - 3600, bootstrapped=True))
    capsys.readouterr()
    world.steps(4)
    assert rm(calm) in lium.removed  # its lines are on the host: the pod of the single-node service goes back
    # no pod, no worker and no run for that row; the other row came and went
    assert [name for _, name in lium.ups] == [P.name_of(gen)] and mid not in world.nodes and not os.path.isdir(f"{D.ROWS}/{mid}")
    assert states() == {mid: "scored", gen: "scored"} and not D.missing(m, RUNNABLE) and mid not in [x["model_id"] for x in D.queue()]
    # its ArXivMath value is 2 of the 16, by the fixed rule, computed on the host; the full run's record is kept
    r = D.rj(D.result_path(mid, "arxivmath"))
    want = sum(sum(k < c for k in derive.picked("arxivmath", i, 16, 2)) / 2 for i, c in enumerate(right)) / 103
    assert (r["bench_version"], r["rollouts"], r["samples"], r["tasks"], r["complete"]) == ("arxivmath/v2/bench-v2", 2, 206, 103, True)
    assert abs(r["score"] - want) < 1e-12 and abs(r["score_full"] - sum(right) / 1648) < 1e-12
    assert (r["derived_from"], r["derived_by"], r["samples_full"]) == ("arxivmath/v1/bench-v2", derive.RULE, 1648)
    assert D.rj(f"{D.PRIV}/{mid}/arxivmath.full.json")["rollouts"] == 16
    cells = [x for x in table()["results"] if x["model_id"] == mid and not R.benchmark(x["benchmark_id"]).get("pooled_from")]
    assert {x["profile"] for x in cells} == {PP15}  # each result says the profile that it ran with
    # the published result of the full run stays in the table; the value at the settings' count comes after it, so the site shows that one
    arx = [x for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath"]
    assert [(x["env_version"], x["samples"], x.get("derived_from")) for x in arx] == [("arxivmath/v1/bench-v2", 1648, None), ("arxivmath/v2/bench-v2", 206, "arxivmath/v1/bench-v2")]
    assert [(x["env_version"], x["score"]) for x in table()["results_prior"] if x["model_id"] == mid] == [("arxivmath/v1/bench-v2", old[0]["score"])]
    assert len(cells) == len(RUNNABLE) + 1
    assert len([e for e in events(capsys) if e["event"] == "derived" and e["row"] == mid]) == 1  # once, not at each step


@pytest.mark.parametrize("fault", ["a sample ended with an error", "a problem is missing", "the per-problem lines are not on the host"])
def test_a_scored_row_whose_full_run_gives_no_value_is_held_and_rents_nothing_until_the_operators_word(world, monkeypatch, capsys, fault):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    def spoil(r, per):
        if fault == "a problem is missing": per.pop(); r.update(tasks=102, tasks_total=102, samples=1632)
        if fault == "a sample ended with an error": per[5].update(n=15, errors=1); r.update(samples=1647, errors=1, complete=False)
    scored_by_the_single_node_service(mid, lines=not fault.startswith("the per-problem lines"), spoil=spoil)
    capsys.readouterr()
    world.steps(4)
    (line,) = [e for e in events(capsys) if e["event"] == "no-value"]  # one line, not one at each step
    assert (line["row"], line["bench"]) == (mid, "arxivmath") and ("not whole" in line["why"] or "are not on this host" in line["why"])
    row = model(mid)
    assert (row["status"], row["stopped"], row["benchmarks_done"], row["fails"]) == ("failed", "held", len(RUNNABLE) - 1, 0)
    assert [name for _, name in lium.ups] == [P.name_of(gen)] and mid not in world.nodes and D.rj(D.result_path(mid, "arxivmath"))["rollouts"] == 16
    assert "pod" not in row  # the row never had a pod
    # the pod of the single-node service: it stays while the row's lines are not on this host (they may be on it), with the alarm; else it goes back
    monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    calm = lium.pod(P.NAME, id="296c5237-0000-4000-8000-000000000000", huid="calm-matrix-26", price=23.2)
    P.Pod("", first=True).save(dict(name=P.NAME, id=calm["id"], huid="calm-matrix-26", gpu="8xH100", cost=23.2, rented_at=D.now() - 3600, bootstrapped=True))
    world.steps(D.ALARM_TRIES)
    if fault.startswith("the per-problem lines"):
        assert calm in lium.pods and rm(calm) not in lium.removed and "single-pod-not-released" in [a["what"] for a in table()["alarms"]]
        assert D.release_single(force=True) and rm(calm) in lium.removed  # the operator's word
    else:
        assert rm(calm) in lium.removed and calm not in lium.pods
    D.cli("retry", [mid])  # the operator's word: a run of ArXivMath alone, at the settings' count, on a pod of the row's own
    world.steps()
    assert lium.ups[-1][1] == P.name_of(mid) and "--benchmarks arxivmath " in world.nodes[mid].cmds[-1]


@pytest.mark.parametrize("fault", ["the result file is cut", "the result file is not there and the lines are", "no result file of the row is on the host",
                                   "the result file has no count of samples"])
def test_a_row_with_a_published_full_run_whose_file_cannot_be_used_is_held_and_rents_nothing(world, monkeypatch, capsys, fault):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    scored_by_the_single_node_service(mid)
    D.sync_table(); D.publish_table(D.model_rows())  # the table in the store, as the single-node service wrote it: the full run at 16
    assert [x["env_version"] for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath"] == ["arxivmath/v1/bench-v2"]
    f = D.result_path(mid, "arxivmath"); lines = f"{D.PRIV}/{mid}/arxivmath.per_task.jsonl"
    if fault == "the result file is cut": pathlib.Path(f).write_text(pathlib.Path(f).read_text()[:40])
    if fault == "the result file is not there and the lines are": os.remove(f)
    if fault == "no result file of the row is on the host": shutil.rmtree(f"{D.RES}/{mid}"); shutil.rmtree(f"{D.PRIV}/{mid}")
    if fault == "the result file has no count of samples": D.wj(f, {k: v for k, v in D.rj(f).items() if k != "rollouts"})
    monkeypatch.setattr(D, "RELEASE_RETRY_S", 0)
    calm = lium.pod(P.NAME, id="296c5237-0000-4000-8000-000000000000", huid="calm-matrix-26", price=23.2)
    P.Pod("", first=True).save(dict(name=P.NAME, id=calm["id"], huid="calm-matrix-26", gpu="8xH100", cost=23.2, rented_at=D.now() - 3600, bootstrapped=True))
    capsys.readouterr()
    world.steps(4)
    (line,) = [e for e in events(capsys) if e["event"] == "no-value"]  # one line, not one at each step
    assert (line["row"], line["bench"]) == (mid, "arxivmath") and ("is not on this host" in line["why"] or "cannot be used" in line["why"])
    # nothing is rented for the row, and no run is started for it: it waits for the operator's word
    assert [name for _, name in lium.ups] == [P.name_of(gen)] and mid not in world.nodes and mid not in [x["model_id"] for x in D.queue()]
    assert (model(mid)["stopped"], D.stopped(mid)) == ("held", "held") and "pod" not in model(mid)
    # the published result of the full run stays in the table
    assert [(x["env_version"], x["samples"]) for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath"] == [("arxivmath/v1/bench-v2", 1648)]
    # the pod of the single-node service stays, with the alarm, while the row's lines are not on this host
    if os.path.exists(lines):
        assert rm(calm) in lium.removed and calm not in lium.pods
    else:
        assert calm in lium.pods and rm(calm) not in lium.removed and "single-pod-not-released" in [a["what"] for a in table()["alarms"]]
    D.cli("retry", [mid])  # the operator's word: a run of ArXivMath alone, at the settings' count, on a pod of the row's own
    world.steps()
    assert lium.ups[-1][1] == P.name_of(mid) and "--benchmarks arxivmath " in world.nodes[mid].cmds[-1]


def test_the_pick_rule_is_fixed_and_a_third_party_can_repeat_it():
    assert (derive.RULE, derive.CONSTANT) == ("pick-of-full/1", "sn120-v2-bench pick-of-full/1 2026-11-14")
    def by_hand(idx):  # the rule as the public text says it, with nothing of the code under test
        h = hashlib.sha256(f"sn120-v2-bench pick-of-full/1 2026-11-14|arxivmath|{idx}".encode()).digest()
        free = list(range(16)); a = free.pop(int.from_bytes(h[0:8], "big") % 16); b = free.pop(int.from_bytes(h[8:16], "big") % 15)
        return [a, b]
    picks = [derive.picked("arxivmath", i, 16, 2) for i in range(103)]
    assert picks == [by_hand(i) for i in range(103)] and all(a != b and 0 <= a < 16 and 0 <= b < 16 for a, b in picks)
    assert picks[:3] == [[1, 11], [3, 9], derive.picked("arxivmath", 2, 16, 2)] and len({tuple(p) for p in picks}) > 60
    # the pick looks at no result and at no order: the same lines in another order give the same value
    run = full_run([(i * 5) % 17 for i in range(103)]); summary = json.loads(run[True])
    a, _ = derive.from_full(summary, run[False], 2, "arxivmath/v2/bench-v2", 103)
    b, _ = derive.from_full(summary, "".join(reversed(run[False].splitlines(keepends=True))), 2, "arxivmath/v2/bench-v2", 103)
    assert a["score"] == b["score"] and a["samples"] == 206 and a["score_full"] == summary["score"]
    # all right stays all right, all wrong stays all wrong
    for c, want in ((16, 1.0), (0, 0.0)):
        run = full_run([c] * 103)
        assert derive.from_full(json.loads(run[True]), run[False], 2, "v", 103)[0]["score"] == want
    assert derive.from_full(summary, run[False], 2, "v", None)[0] is None and derive.from_full(dict(summary, rollouts=2), run[False], 2, "v", 103)[0] is None


# ---------------------------------------------------------------------------------------------- the settings profile
PP0, PP15 = "qwen3.5-family-pp0", "qwen3.5-family"


def test_a_king_and_the_v1_king_run_with_no_presence_penalty_and_the_launch_model_with_its_model_card(world):
    king = green(world.v2, 1); mid, gen = ids(king), ids(world.genesis)
    ms = R.model_settings(); profiles = ms["profiles"]
    assert (ms["king_profile"], ms["default_profile"]) == (PP0, PP15)
    differ = {k for k in profiles[PP15] if profiles[PP15][k] != profiles[PP0][k]}  # one difference, and the text that says so
    assert differ == {"presence_penalty", "source"} and set(profiles[PP15]) == set(profiles[PP0])
    for model_id, kind, name, penalty in ((mid, "king", PP0, 0.0), (V1, "v1-king", PP0, 0.0), (gen, "genesis", PP15, 1.5)):
        for st in (R.resolve_settings(model_id, kind), R.resolve_settings(model_id)):  # a node knows the id only
            assert (st["profile"], st["presence_penalty"], st["temperature"], st["top_p"], st["top_k"], st["budget"]) == (name, penalty, 1.0, 0.95, 20, 65536)
    world.steps(3)
    assert (model(mid)["profile"], model(gen)["profile"], model(V1)["profile"]) == (PP0, PP15, PP0)
    assert "qwen3.5-family-pp0 (presence_penalty 0.0" in table()["about"] and "qwen3.5-family (presence_penalty 1.5" in table()["about"]


SENT = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, max_tokens=65536)  # of both profiles


def test_the_table_gives_the_values_of_each_profile_that_a_row_names_and_no_sampling_that_no_run_uses(world):
    """A published row names its profile, so the values of a name stay: a change of a value takes a new name."""
    green(world.v2, 1); world.steps(3)
    t = table()
    assert t["profiles"] == {PP0: SENT | dict(presence_penalty=0.0, max_model_len=69632), PP15: SENT | dict(presence_penalty=1.5, max_model_len=69632)}
    assert {m["profile"] for m in t["models"]} == set(t["profiles"]) and all(set(v) == set(R.SENT) | {"max_model_len"} for v in t["profiles"].values())
    for name, values in t["profiles"].items():  # the same values as a node takes for a row of that profile
        st = next(st for st in (R.resolve_settings(m["model_id"], m["kind"]) for m in t["models"]) if st["profile"] == name)
        assert values == {k: st[k] for k in values}
    assert t["serving"] == R.serving() and set(t["serving"]) == {"version", "server", "client"}  # no block of sampling values that no code reads
    for text in ("serving has no sampling value", "profiles has each profile that a row or a result of this file names", ", ".join(R.SENT)):
        assert text in t["about"]
    assert BENCH_VIEW(t) == BENCH_VIEW({k: v for k, v in t.items() if k not in ("profiles", "serving")})  # the site's table does not read them


def test_a_run_sends_the_values_that_the_table_says_and_the_server_starts_as_the_table_says():
    bench = pathlib.Path(__file__).resolve().parents[1]
    flags = re.findall(r'"--sampling\.([a-z-]+)", str\(sp\["([a-z_]+)"\]\)', (bench / "affine_bench" / "node_run.py").read_text())
    assert sorted(v for _, v in flags) == sorted(R.SENT) and all(f.replace("-", "_") == v for f, v in flags)
    assert (bench / "affine_bench" / "node_run.py").read_text().count("--sampling.") == len(R.SENT)  # and no other sampling value
    sv, sh = R.serving()["server"], (bench / "deploy" / "serve.sh").read_text()
    for start in (f"--dtype {sv['dtype']} ", f"--max-num-seqs {sv['max_num_seqs']} ", f"--gpu-memory-utilization {sv['gpu_memory_utilization']:.2f} ",
                  f"--generation-config {sv['generation_config']} ", "NG=$(nvidia-smi -L | wc -l)", "--data-parallel-size $NG ", f"ML=${{3:-{sv['max_model_len']}}}"):
        assert start in sh, start
    assert sv["engine"] in (bench / "deploy" / "requirements-vllm.txt").read_text().split()


def test_two_rows_whose_short_names_are_equal_never_share_a_pod(world, monkeypatch, capsys):
    monkeypatch.setattr(P, "NAME_MAX", len(P.PREFIX) + 4)  # every king has the name affine-v2-bench-king
    first, second = green(world.v2, 1), green(world.v2, 2)
    assert P.name_of(ids(first)) == P.name_of(ids(second))
    world.node(ids(second)).busy = "done=\nalive=1\nlines=7\nfinished=1\nok=1\n"  # the newer row starts first, and its run goes on
    world.steps(P.RENT_TRIES)
    # the newer row has the pod; the other row takes no pod and makes none, and says so
    pods = [p for p in world.lium.pods if p["name"] == P.name_of(ids(first))]
    assert len(pods) == 1 and D.pod_of(ids(second)).load()["id"] == pods[0]["id"] and not D.pod_of(ids(first)).load()
    assert D.stopped(ids(first)) == "rent_failed" and "is the pod of another row's record" in D.row_state(ids(first))["rent_failed"]["why"]
    assert states()[ids(second)] == "evaluating" and model(ids(first))["stopped"] == "rent_failed" and world.node(ids(first)).cmds == []
    assert [name for _, name in world.lium.ups].count(P.name_of(ids(first))) == 1


# ---------------------------------------------------------------------------------------------- the store and the wait for weights
def test_a_host_with_an_empty_disk_reads_the_published_table_and_runs_no_row_again(world, monkeypatch):
    v2, store, lium = world.v2, world.store, world.lium
    king = green(v2, 1)
    world.steps(3)
    assert set(states().values()) == {"scored"} and len(states()) == 2 and len(lium.ups) == 2
    before = table()

    empty_host(monkeypatch)
    store.down = True
    with pytest.raises(InfraError):  # the published table cannot be read: the step stops before any worker
        D.step()
    assert len(lium.ups) == 2 and not os.path.exists(TABLE) and len(world.polled) == 6
    store.down = False
    world.steps(3)
    assert len(lium.ups) == 2 and D.queue() == [] and states() == {m["model_id"]: "scored" for m in before["models"] if m["model_id"] != V1}
    assert table()["results"] == before["results"]  # every published result is still there

    # a new king on that host: only it is run, and the write to the store keeps the rows of the old host
    second = green(v2, 2)
    world.steps(3)
    assert len(lium.ups) == 3 and states()[ids(second)] == "scored"
    published = json.loads(store.tables[-1])
    assert {r["model_id"] for r in published["results"]} == {m["model_id"] for m in before["models"] if m["model_id"] != V1} | {ids(second)}

    # two writers by mistake: the table in the store is the base of each write, so a stale local copy drops nothing
    pathlib.Path(TABLE).write_text(json.dumps(dict(before, models=[], results=[])))
    shutil.rmtree(D.RES); os.makedirs(D.RES); D.S.pop("uploaded")
    D.publish_table(D.model_rows())
    assert {r["model_id"] for r in json.loads(store.tables[-1])["results"]} == {r["model_id"] for r in published["results"]}
    assert ids(king) in states()


@pytest.mark.parametrize("where", ["store", "disk"])
def test_reference_rows_that_an_operator_put_in_the_table_stay(world, where):
    """Rows of other models, scored or failed, in the table of the store or in the local file before the first start."""
    store = world.store
    cell = dict(model_id="qwen3.5-9b", benchmark_id="math500", env_version="math500/v1/bench-v2", score=0.9, se=0.01,
                complete=True, samples=500, samples_expected=500, cap_rate=0.0, budget=65536)
    seed = json.dumps(dict(updated_at=1.0, benchmarks=[], results=[cell], models=[
        dict(model_id="qwen3.5-9b", label="Qwen3.5-9B", kind="reference", status="scored", crowned_at=None),
        dict(model_id="glm-z1-9b", label="GLM-Z1-9B-0414", kind="reference", status="failed", crowned_at=None)]))
    store.inner.put(layout.BENCHMARKS, seed.encode()) if where == "store" else pathlib.Path(TABLE).write_text(seed)
    world.steps(3)
    rows = [r for r in BENCH_VIEW(json.loads(store.tables[-1]))["rows"] if r["kind"] != "v1-king"]
    # the site builder puts the genesis head last, under the other models
    assert [(r["label"], r["kind"], r["status"]) for r in rows[:2]] == [("Qwen3.5-9B", "reference", "scored"), ("GLM-Z1-9B-0414", "reference", "failed")]
    assert (rows[2]["kind"], rows[2]["status"]) == ("genesis", "scored") and rows[0]["cells"]["math500"]["score"] == 0.9 and not rows[1]["cells"]
    assert len(world.lium.ups) == 1 and D.queue() == []  # a reference row is never run by this service


def test_a_king_whose_weight_files_are_not_in_the_store_waits_and_starts_by_itself(world, capsys):
    """The head of a king is public before its weight files (the box sends them after the next round has opened)."""
    v2, store, lium, genesis = world.v2, world.store, world.lium, world.genesis
    world.steps(3)
    assert states() == {ids(genesis): "scored"} and not lium.pods
    king = green(v2, 1); mid = ids(king)
    store.gone = base_keys(v2, king)
    ups, polled, signed, tables = len(lium.ups), len(world.polled), len(store.signed), len(store.tables)
    capsys.readouterr()
    world.steps(D.MAX_FAIL + 2)  # more polls than the tries of a row
    row = model(mid)
    assert (row["status"], row["fails"], row["waiting_for"]) == ("queued", 0, "weights") and "pod" not in row  # the table says why the row does not run
    assert not os.path.isdir(f"{D.ROWS}/{mid}") and D.queue() == []  # no failed try, no state of a worker
    assert len(lium.ups) == ups and not lium.pods and len(world.polled) == polled and len(store.signed) == signed  # no rent, no worker, no link
    assert len(store.tables) == tables + 1 and table()["progress"]["queued"] == 1  # one write for the new row; it is still to do
    assert [(e["event"], e["model"]) for e in events(capsys) if e["event"] != "table-written"] == [("waiting", mid)]  # one line, not one per poll

    store.gone = set(sorted(store.gone)[-1:])  # every file but one is there: an object is at its name only when it is whole
    world.steps()
    assert states()[mid] == "queued" and len(lium.ups) == ups
    store.gone = set()
    world.steps()  # the files are there: the row starts with no command of a person
    row = model(mid)
    assert row["status"] == "evaluating" and "waiting_for" not in row and len(lium.ups) == ups + 1 and mid not in D.S.get("waiting", {})
    assert "waiting-over" in [e["event"] for e in events(capsys)]
    world.steps()
    assert states() == {ids(genesis): "scored", mid: "scored"} and D.row_state(mid)["fails"] == 0


def test_a_king_that_waits_longer_than_the_bound_is_shown_as_waiting(world, monkeypatch, capsys):
    v2, store, lium = world.v2, world.store, world.lium
    world.steps(3)
    king = green(v2, 1); mid = ids(king)
    store.gone = base_keys(v2, king)
    age = D.now() - king.applied_at
    monkeypatch.setattr(D, "WAIT_LATE_S", age + 600)
    world.steps()
    assert states()[mid] == "queued"
    tables = len(store.tables); capsys.readouterr()
    monkeypatch.setattr(D, "WAIT_LATE_S", age - 600)  # the head was applied more than the bound ago
    world.steps(2)
    row = model(mid)
    assert (row["status"], row["waiting_for"], row["fails"]) == ("waiting", "weights", 0) and len(store.tables) == tables + 1  # one write
    assert [e["event"] for e in events(capsys) if e["event"] != "table-written"] == ["waiting-late"]
    assert [(r["kind"], r["status"]) for r in BENCH_VIEW(table())["rows"] if r["kind"] != "v1-king"] == [("king", "waiting"), ("genesis", "scored")]  # the site passes the word on
    assert len(lium.ups) == 1 and not lium.pods
    store.gone = set()
    world.steps(2)
    assert states()[mid] == "scored"  # late is not failed: it still starts by itself


def test_a_real_fault_of_the_download_is_a_failed_try_and_after_three_the_pod_goes_back_by_its_exact_id(world, capsys):
    """The files are in the store, and the download fails on the node: a wrong sha256, or a link that the store refuses."""
    v2, store, lium = world.v2, world.store, world.lium
    world.steps(3)
    king = green(v2, 1); mid = ids(king)
    world.node(mid).bad = lambda links: True
    seen = []
    for _ in range(D.MAX_FAIL):
        world.steps(); seen.append((states()[mid], D.row_state(mid)["fails"]))
    assert seen == [("queued", 1), ("queued", 2), ("failed", 3)] and D.queue() == [] and D.S.get("waiting", {}).get(mid) is None
    pod = dict(world.pod_of(mid)); assert len(lium.ups) == 2  # one pod for the three tries
    world.steps()
    assert lium.removed[-1] == rm(pod) and not lium.pods and world.nodes[mid].launches == 0

    # a key that may not ask the store holds no king back: the download decides, and its failure counts
    D.cli("retry", [mid]); store.refused = True; capsys.readouterr()
    world.steps()
    assert D.row_state(mid)["fails"] == 1 and "store-check-failed" in [e["event"] for e in events(capsys)]
    world.node(mid).bad = lambda links: False
    world.steps(2)
    assert states()[mid] == "scored"  # and a download that works runs the row, whatever the question to the store gave


def test_the_table_says_each_five_minutes_that_a_worker_looks_at_its_row_in_work(world, monkeypatch):
    store, gen = world.store, ids(world.genesis)
    world.steps()
    assert table()["progress"]["checked_at"] is None  # launched in this step: no look at the node yet
    node = world.nodes[gen]; node.busy = "done=\nalive=1\nlines=7\nfinished=3\nok=1\n"
    before = D.now(); world.steps()
    progress = table()["progress"]; tables = len(store.tables)
    assert progress["checked_at"] >= before and (progress["output_lines"], progress["benchmarks_finished"]) == (7, 3)
    assert before <= progress["output_grew_at"] <= progress["checked_at"] and json.loads(store.tables[-1])["progress"] == progress
    assert model(gen)["benchmarks_done"] == 0 and model(gen)["progress"]["output_lines"] == 7  # the row's own count is of collected results
    node.busy = "done=\nalive=1\nlines=8\nfinished=4\nok=1\n"
    world.steps(2)
    assert len(store.tables) == tables and table()["progress"]["output_lines"] == 7  # not at each poll
    monkeypatch.setattr(D, "BEAT_S", 0)
    world.steps()
    assert len(store.tables) == tables + 1 and table()["progress"]["output_lines"] == 8 and table()["progress"]["checked_at"] > progress["checked_at"]
    node.busy = None
    world.steps()
    assert states() == {gen: "scored"} and table()["progress"]["checked_at"] is None and "progress" not in model(gen)


def test_a_restart_of_the_service_leaves_each_run_and_each_wait_as_it_is(world):
    v2, store, lium, genesis = world.v2, world.store, world.lium, world.genesis
    world.steps()
    king = green(v2, 1); mid, gen = ids(king), ids(genesis)
    store.gone = base_keys(v2, king)
    world.nodes[gen].busy = "done=\nalive=1\nlines=9\nfinished=2\nok=1\n"
    world.steps()
    started = D.row_state(gen)["running"]["started"]
    state = D.rj(D.SF); D.S.clear(); D.S.update(state); D.SYNCED.clear(); D.KIDS.clear()  # a new process has only the files of the state directory
    assert sorted(D.S["waiting"]) == [mid, V1]
    world.steps()  # the new process finds the run in work
    assert states() == {gen: "evaluating", mid: "queued"} and world.nodes[gen].launches == 1 and D.row_state(gen)["running"]["started"] == started
    assert len(lium.ups) == 1 and not lium.removed and D.row_state(gen)["fails"] == 0
    world.nodes[gen].busy = None; store.gone = set()
    world.steps()  # it collects the run that the old process started, and the king whose files came starts on its own pod
    assert states() == {gen: "scored", mid: "evaluating"} and world.nodes[gen].launches == 1 and len(lium.ups) == 2


# ---------------------------------------------------------------------------------------------- the last king of v1
V1_SOURCE = "mimo/v1-run/checkpoints/" + source.V1_KING["checkpoint"] + "/model"  # the form of v1's own place in the bucket


def v1_source(world, monkeypatch):
    """The operator sets the source: the prefix in the store where the 16 files of the checkpoint are."""
    for n in source.V1_KING_FILES: world.store.inner.put(f"{V1_SOURCE}/{n}", b"v1 " + n.encode())
    monkeypatch.setenv("BENCH_V1_KING_SOURCE", V1_SOURCE + "/")
    return dict(id=source.V1_KING["checkpoint"], files=dict(source.V1_KING_FILES))


def test_the_v1_king_has_a_row_of_its_own_kind_and_changes_no_king_row_and_not_the_kings_order(world, monkeypatch):
    v2, lium = world.v2, world.lium
    first, second = green(v2, 1), green(v2, 2)
    row = source.extra_rows()[0]
    assert (row["model_id"], row["kind"], row["uid"], row["v1_epoch"]) == ("v1king-b6d592c3bb1b4393", "v1-king", 920, 1211)
    assert row["hotkey"] == "5FaTFWogz3AiQbEFRnL9CZcqR1keeypEYkd7f1L5YLbWH1cP" and row["base_hash"] == "b6d592c3bb1b439365e82147a47e65d93c083a13444aceae078f8a2053cd5ce3"
    assert P.name_of(V1) == "affine-v2-bench-v1king-b6d592" and "v1-king" in publish.OWN_KINDS
    # the file map in the code is the checkpoint's: its id by the rule of v1 and v2 is the checkpoint
    assert len(source.V1_KING_FILES) == 16 and canonical.manifest_digest(source.V1_KING_FILES) == row["base_hash"]
    # the same settings and budget as a king, and the same ten benchmarks
    assert R.resolve_settings(V1, "v1-king") == R.resolve_settings(ids(first), "king") == R.resolve_settings(V1) and R.resolve_settings(V1, "v1-king")["budget"] == 65536
    world.steps(3)
    without = table()
    assert [m["model_id"] for m in without["models"]] == [ids(world.genesis), ids(first), ids(second), V1]
    row = model(V1)
    assert (row["kind"], row["label"], row["uid"], row["v1_epoch"], row["status"], row["waiting_for"]) == ("v1-king", "v1 king E1211 b6d592c3", 920, 1211, "waiting", "source")
    assert row["benchmarks_total"] == len(RUNNABLE) and "pod" not in row and without["progress"]["queued"] == 0

    v1_source(world, monkeypatch)
    world.steps(4)
    assert states(v1=True)[V1] == "scored" and len(lium.ups) == 4
    kings = lambda t: ([m for m in t["models"] if m["model_id"] != V1], [r for r in t["results"] if r["model_id"] != V1])
    assert kings(table()) == kings(without)  # no king row and no result of a king changed
    view = lambda t: [(r["kind"], r["round_id"], r["status"], r["avg"]) for r in BENCH_VIEW(t)["rows"] if r["kind"] != "v1-king"]
    assert view(table()) == view(without) and [k for k, *_ in view(table())] == ["king", "king", "genesis"]  # and not their order
    assert D.candidates(D.model_rows()) == []


def test_the_v1_king_is_queued_once_and_only_once_and_rents_nothing_without_a_readable_source(world, monkeypatch, capsys):
    store, lium = world.store, world.lium
    world.steps(3)
    ups, polled = len(lium.ups), len(world.polled)
    capsys.readouterr()
    world.steps(3)  # the source is not set
    assert (model(V1)["status"], model(V1)["waiting_for"]) == ("waiting", "source") and D.S["waiting"][V1]["file"] == "the source is not set (BENCH_V1_KING_SOURCE)"
    assert [e["event"] for e in events(capsys) if e.get("model") == V1] == []  # it waited before: no line at each poll
    desc = v1_source(world, monkeypatch)
    for fault in ("a small file is not there", "a weight file is not there", "the store does not answer"):
        store.gone, store.refused = set(), False
        if fault == "a small file is not there": store.gone = {f"{V1_SOURCE}/config.json"}
        if fault == "a weight file is not there": store.gone = {f"{V1_SOURCE}/model-00010-of-00010.safetensors"}
        if fault == "the store does not answer": store.refused = True
        world.steps(2)
        assert (model(V1)["status"], model(V1)["waiting_for"]) == ("waiting", "source"), fault
        assert len(lium.ups) == ups and not lium.pods and len(world.polled) == polled and not os.path.isdir(f"{D.ROWS}/{V1}"), fault  # no rent, no worker, no pod that waits
    assert not any(V1_SOURCE in key for key in store.signed)
    store.gone, store.refused = set(), False
    world.steps()  # readable: queued by this step, with no word of a person
    assert states(v1=True)[V1] == "evaluating" and lium.ups[-1][1] == P.name_of(V1) and len(lium.ups) == ups + 1
    node = world.nodes[V1]
    blob = json.loads(next(s for s in node.stdin if s))
    assert blob["descriptor"] == desc and sorted(blob["urls"]) == sorted(desc["files"]) and all(V1_SOURCE + "/" in u for u in blob["urls"].values())
    assert f"--model-dir {C.NODE_KINGS}/{desc['id']} " in node.cmds[-1] and "--reasoning-parser qwen3 " in node.cmds[-1]
    world.steps(2)
    assert states(v1=True)[V1] == "scored" and not lium.pods
    world.steps(5)  # listed at every poll for ever: one run, one rent
    assert len(lium.ups) == ups + 1 and node.launches == 1 and D.queue() == []
    D.SYNCED.clear(); D.KIDS.clear(); shutil.rmtree(D.RES); os.makedirs(D.RES); shutil.rmtree(D.ROWS); os.makedirs(D.ROWS)  # also after the host lost its disk
    world.steps(3)
    assert len(lium.ups) == ups + 1 and states(v1=True)[V1] == "scored"


def test_a_failed_fetch_of_the_v1_king_gives_its_pod_back_by_its_exact_id(world, monkeypatch):
    lium = world.lium
    world.steps(3)
    v1_source(world, monkeypatch)
    world.node(V1).bad = lambda links: True  # the source could be read on the host, and the download on the node fails (a wrong sha256)
    other = lium.pod(P.name_of("v1king-0000000000000000"))  # a pod with a like name is not this row's pod
    world.steps(D.MAX_FAIL)
    pod = dict(world.pod_of(V1))
    assert D.stopped(V1) == "failed" and world.nodes[V1].launches == 0
    world.steps()
    assert lium.removed[-1] == rm(pod) and lium.pods == [other] and model(V1)["stopped"] == "failed" and model(V1)["pod"]["state"] == "released"
    ups = len(lium.ups); world.steps(3)
    assert len(lium.ups) == ups and states() == {ids(world.genesis): "scored"}


# ---------------------------------------------------------------------------------------------- commands and documents
def test_the_pods_check_lists_each_bench_pod_with_its_row_and_reads_only(world, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    world.steps()
    lium.pod("sn120-v2-prod-dublin", id="6cb913e4-0000-4000-8000-000000000000"); lium.pod(P.name_of("king-" + "f" * 16))  # not a bench pod; a bench pod of no row
    lium.pods = [p for p in lium.pods if p["name"] != P.name_of(gen)]  # a row whose pod is not there
    sent = len(lium.sent); capsys.readouterr()
    D.cli("pods", [])
    out = capsys.readouterr().out
    lines = {line.split()[1]: line for line in out.splitlines()[1:-1]}
    assert world.pod_of(mid)["id"][:8] in lines[P.name_of(mid)] and f" {mid} " in lines[P.name_of(mid)] and " running " in lines[P.name_of(mid)] and " 0.8" in lines[P.name_of(mid)]
    assert "NO ROW CLAIMS THIS POD" in lines[P.name_of("king-" + "f" * 16)] and "NOT LISTED BY LIUM" in lines[P.name_of(gen)]
    assert "sn120-v2-prod-dublin" not in out and "203.0.113.7" not in out and "2 bench pods at Lium" in out
    assert [a[0] for a in lium.sent[sent:]] == ["ps"]  # one reading; nothing is deleted
    D.cli("status", [])
    assert json.loads(capsys.readouterr().out)["rows"][mid]["state"] == "running"


def test_release_gives_the_pod_of_a_row_back_and_the_next_start_runs_the_row_again(world):
    lium, gen = world.lium, ids(world.genesis)
    world.steps()
    pod = dict(lium.pods[0])
    D.cli("release", [gen])
    assert lium.removed == [rm(pod)] and not D.pod_of(gen).load() and not D.row_state(gen)["running"]
    world.steps(3)
    assert states() == {gen: "scored"} and len(lium.ups) == 2
    with pytest.raises(SystemExit):
        D.cli("no-such-command", [])


def test_only_one_service_per_state_directory_and_one_worker_per_row():
    lock = D.only_instance()
    with pytest.raises(SystemExit):
        D.only_instance()
    with pytest.raises(SystemExit):  # a command that does the service's work is refused while the service runs
        D.cli("release", ["king-0"])
    lock.close()
    worker = D.row_lock("king-0")
    assert worker is not None and D.row_lock("king-0") is None
    with pytest.raises(SystemExit):
        D.row_main("king-0")
    with pytest.raises(SystemExit):  # a row whose worker is alive is not changed under it
        D.cli("retry", ["king-0"])
    worker.close()
    assert D.row_lock("king-0") is not None


def test_the_table_says_how_arxivmath_is_scored_and_what_a_derived_row_is(world):
    world.steps()
    about = table()["about"]
    assert f"ArXivMath is scored with {ARXIV['rollouts']} samples for each of its {ARXIV['problems']} problems" in about
    for words in ("derived_from", f"`{derive.RULE}`", "v2/bench/affine_bench/derive.py", "score_full, se_full and samples_full", "waiting_for=source",
                  "kind v1-king", "A row of any other kind (a reference model) is not run by this service", "Merge-only publish", "models[].pod"):
        assert words in about, words
    assert "A row of another kind" not in about


def test_the_operator_document_and_the_unit_file_are_true_to_the_code():
    bench = pathlib.Path(__file__).resolve().parents[1]
    doc, unit = (bench / "OPERATIONS.md").read_text(), (bench / "deploy" / "sn120-v2-bench.service").read_text()
    lines = [line for line in unit.splitlines() if line and not line.startswith("#")]
    assert set(re.findall(r"SN120_[A-Z0-9_]+", "\n".join(lines))) == set(re.findall(r"^#   (SN120_[A-Z0-9_]+)", unit, re.M))
    assert "ExecStart=SN120_PYTHON -m affine_bench.cli daemon" in lines and not any(line.startswith("ExecStop") for line in lines)
    assert "KillMode=control-group" in lines  # a stop of the unit ends the workers too; the runs on the nodes go on
    blanks = set(re.findall(r"s\|(SN120_[A-Z0-9_]+)\|", doc))  # the install line of the document fills every blank of the unit
    assert blanks == set(re.findall(r"^#   (SN120_[A-Z0-9_]+)", unit, re.M))
    required = set(re.findall(r'require_env\("(BENCH_[A-Z_]+)"\)', (bench / "affine_bench" / "config.py").read_text()))
    assert len(required) == 7 and all(f"\n   {name}=" in doc for name in required)  # the settings file has each required setting
    defaults = {f.name: f.default for f in dataclasses.fields(Params)}
    evaluator = (bench.parent / "sn120_v2" / "gpu" / "evaluator.py").read_text()
    for fact in (
        f"At most {P.MAX_PODS} rows are in work", f"At most {P.MAX_RENTS_PER_DAY} pods are made in 24 hours", f"default `{P.GPU_PREF}`",
        f"A pod lives {P.POD_MAX_HOURS:.0f} hours at most", f"{P.RENT_TRIES} tries", f"the name `{P.NAME}`", f"the first {P.NAME_MAX} characters of `affine-v2-bench-<model_id>`",
        f"asks again every {P.NO_OFFER_RETRY_S:.0f} seconds", f"no progress for {D.STALL_S} seconds",
        f"After {D.MAX_FAIL} fails", f"within one poll ({D.POLL_S} seconds)", f"more than {D.WAIT_LATE_S // 60} minutes after its head was applied",
        f"about each {D.BEAT_S // 60} minutes", f"After {D.POD_CHECK_AFTER_S // 60} minutes the worker asks Lium",
        f"waits {D.CONFIRM_WAIT_S // 60} minutes at most", f"`{derive.RULE}`", "BENCH_V1_KING_SOURCE",
        *(f"`{x}`" for x in P.PROTECTED_IDS + P.PROTECTED_NAMES),
        f"runs the {len(RUNNABLE)} benchmarks", f"`bids` names {len(RUNNABLE)} benchmarks", f"`n_gpus = {defaults['n_gpus']}`",
        f"at most {defaults['round_time_limit_s'] // 60} minutes (`round_time_limit_s = {defaults['round_time_limit_s']}`)",
        f"take {R.serving()['server']['gpu_memory_utilization']:.2f} of each GPU's memory", f"vLLM {R.serving()['server']['engine'].split('==')[1]} serves",
        "holds more than 1024 MiB. It waits 120 seconds", "genesis-" + GENESIS_BASE_HASH[:16],
        f"again every {D.RELEASE_RETRY_S // 60} minutes if Lium or a guard refuses", f"After {D.ALARM_TRIES} failed tries the table has the alarm `single-pod-not-released`", f"{D.ALARM_TRIES} or more tries to give a node back have failed",
        f"checks this each {D.LIUM_CHECK_S // 60} minutes", *(f"`{w}`" for w in P.GONE_WORDS), "Do not stop or restart the service while a row says `renting`",
    ):
        assert fact in doc, fact
    about = pathlib.Path(publish.__file__).read_text()  # the table's own text names the same two times
    assert (D.WAIT_LATE_S, D.BEAT_S) == (3600, 300) and "lasts more than an hour" in about and "about each 5 minutes" in about
    assert "free_limit_mib: int = 1024" in evaluator and "free_timeout_s: float = 120.0" in evaluator
    rent = (bench / "deploy" / "rent.sh").read_text()  # the rent script picks nothing and deletes nothing
    assert "lium ls" not in rent and " ls " not in rent and " rm " not in rent and "${1:?" in rent and "NAME=${NAME:?}" in rent


def test_settings_come_from_the_environment_and_a_missing_one_stops_the_service(monkeypatch):
    assert source.store() == pathlib.Path.cwd()  # BENCH_STORE names a factory as "module:Class.method"; here pathlib:Path.cwd
    monkeypatch.delenv("BENCH_MAX_PRICE")  # no price cap: the service does not start, so nothing is rented
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


# ---------------------------------------------------------------------------------------------- faults: no pod costs money with no end and no word
def one_row_in_work(world, busy="done=\nalive=1\nlines=9\nfinished=2\nok=1\n"):
    """The launch model's row with a pod and a run that goes on. Returns (row id, its node, a copy of its pod)."""
    gen = ids(world.genesis); world.steps()
    node = world.nodes[gen]; node.busy = busy
    return gen, node, dict(world.lium.pods[0])


def test_the_six_hours_hold_when_ssh_hangs_and_when_a_result_file_is_cut(world, capsys):
    gen, node, pod = one_row_in_work(world)
    plain = node.run
    def run(cmd, inp=None, timeout=0):
        if "kill -TERM" in cmd: raise subprocess.TimeoutExpired(cmd, timeout)  # the node does not answer the stop of the run
        return plain(cmd, inp, timeout)
    node.run = run; node.results["math500"] = {True: '{"benchmark": "math500", "sco', False: '{"idx":0,"correct":1.0}\n'}  # a summary that is cut
    rec = D.pod_of(gen).load(); rec["rented_at"] = D.now() - 6 * 3600 - 1; D.pod_of(gen).save(rec)
    world.steps(3)
    ev = [e["event"] for e in events(capsys)]
    assert D.stopped(gen) == "expired" and ev.count("pod-expired") == 1 and "collect-incomplete" in ev
    assert world.lium.removed == [rm(pod)] and not world.lium.pods and model(gen)["pod"]["state"] == "released"
    assert not os.path.exists(D.result_path(gen, "math500")) and model(gen)["benchmarks_done"] == len(RUNNABLE) - 1  # the other results came


@pytest.mark.parametrize("listing", ["lacks the pod", "is empty", "is not a list of pods"])
def test_a_listing_without_the_pod_is_no_proof_that_the_pod_is_gone(world, monkeypatch, capsys, listing):
    lium = world.lium; gen, node, pod = one_row_in_work(world, busy=None)
    world.steps()  # the run is done and its results are on the host
    plain = lium.cli
    if listing == "lacks the pod": lium.unlisted.add(pod["id"])
    if listing == "is empty": lium.unlisted.update(p["id"] for p in lium.pods)
    if listing == "is not a list of pods": monkeypatch.setattr(P, "lium", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout='[null, "x"]', stderr="") if a[0] == "ps" else plain(*a, **k))
    world.steps(4)
    rec = D.pod_of(gen).load(); row = model(gen)
    assert rec["id"] == pod["id"] and not lium.removed and len(lium.pods) == 1  # the record stays; nothing says `released`
    assert (row["status"], row["pod"]["state"]) == ("scored", "release-refused") and "release-deferred" in [e["event"] for e in events(capsys)]
    refused = D.row_state(gen)["release_refused"]
    assert table()["alarms"] == [dict(what="release-refused", row=gen, since=refused["since"], tries=refused["tries"])] and abs(refused["since"] - row["pod"]["since"]) < 1
    assert D.row_state(gen)["release_refused"]["tries"] >= D.ALARM_TRIES and gen in D.plan(D.model_rows(), {})  # it keeps its place of BENCH_MAX_PODS
    lium.unlisted.clear(); monkeypatch.setattr(P, "lium", plain)
    world.steps()
    assert lium.removed == [rm(pod)] and not lium.pods and model(gen)["pod"]["state"] == "released" and table()["alarms"] == []
    assert "release_refused" not in D.row_state(gen)


def test_a_pod_that_lium_says_it_deleted_is_released_with_no_delete_and_only_then(world):
    lium = world.lium; gen, node, pod = one_row_in_work(world, busy=None)
    world.steps()
    lium.pods.clear()  # Lium took the pod: `describe` of its id says DELETED
    world.steps()
    assert not lium.removed and model(gen)["pod"]["state"] == "released" and not D.pod_of(gen).load()
    lium.pod("p", id="there"); lium.unlisted.add("there")
    assert P.state("there") is None and P.state("gone-id") == "gone"  # not listed and no DELETED event: unknown, never gone
    lium.unlisted.clear(); assert P.state("there") == "listed"


def test_an_empty_listing_in_a_rent_never_makes_a_second_pod_beside_the_rows_record(world, monkeypatch, capsys):
    lium = world.lium; gen = ids(world.genesis)
    monkeypatch.setattr(P, "setup", lambda ssh: (False, "root@203.0.113.7: Permission denied"))
    monkeypatch.setattr(P, "lium", lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="no") if a[0] == "rm" else lium.cli(*a, **k))
    world.steps()  # the setup fails, and so does the delete: the row has a record of a pod that is not set up
    (pod,) = lium.pods; assert D.pod_of(gen).load()["id"] == pod["id"] and len(lium.ups) == 1
    lium.unlisted.add(pod["id"])  # the next listing is empty
    world.steps(6)
    assert len(lium.ups) == 1 and len(lium.pods) == 1 and D.pod_of(gen).load()["id"] == pod["id"]  # no second `lium up`; the record stays
    assert D.stopped(gen) == "rent_failed" and "did not say that it was deleted" in D.row_state(gen)["rent_failed"]["why"]
    out = capsys.readouterr().out
    assert "pod-record-dropped" not in out and "203.0.113.7" not in out and "root@ADDRESS: Permission denied" in out  # and no address in a journal line


def test_only_the_last_event_of_a_pod_says_that_lium_deleted_it(monkeypatch):
    pid = "0d1f3c5a-0000-4000-8000-000000000001"
    said = lambda text: monkeypatch.setattr(P, "lium", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=text, stderr=""))
    old = "When                 Event            What\n2026-11-13T09:00:00  pod-lifecycle    REMOVED (migration) — the node was REMOVED and TERMINATED\n2026-11-14T15:00:00  pod-lifecycle    RUNNING\n"
    said(f"Pod {pid} is no longer listed. Its last event: RUNNING · 2026-11-14T15:00:00\n" + old)
    assert P.state(pid) is None  # a living pod that one listing lacks, with an old event that has the words
    said(f"Pod {pid} is no longer listed. Its last event: FAILED (reboot) — the container was REMOVED · 2026-11-14T15:00:00\n" + old)
    assert P.state(pid) is None  # the words in the text of the last event are not its status
    said(f"Pod {pid} is no longer listed. Its last event:\nDELETED (user_request) · 2026-11-14T15:00:00\n" + old)
    assert P.state(pid) == "gone"  # the client folds the line at 80 columns
    said(f"Pod {pid} is no longer listed. Its last event: pod-delete.success · 2026-11-14T15:00:00\n" + old)
    assert P.state(pid) is None  # a last event with no status: not proven, so not gone
    said("Pod        calm-matrix-26\nStatus     RUNNING\nLast event DELETED (old)\n"); assert P.state(pid) == "listed"


def test_a_pod_past_its_hours_goes_back_whatever_the_table_in_the_store_gives(world, monkeypatch, capsys):
    gen, node, pod = one_row_in_work(world)
    def not_json(self): raise ValueError("the table object of the store is not JSON")
    monkeypatch.setattr(D.Row, "results_left", not_json); monkeypatch.setattr(D, "CONFIRM_WAIT_S", 0)
    rec = D.pod_of(gen).load(); rec["rented_at"] = D.now() - 6 * 3600 - 1; D.pod_of(gen).save(rec)
    world.steps(3)
    ev = [e["event"] for e in events(capsys)]
    assert "store-check-failed" in ev and D.stopped(gen) == "expired" and world.lium.removed == [rm(pod)] and not world.lium.pods


def test_a_lium_up_with_no_answer_is_never_tried_again_by_the_row_and_is_an_alarm_at_once(world, monkeypatch, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    real_up = lium.up
    monkeypatch.setattr(P, "up", lambda offer, name, ssh: (lium.ups.append((offer["huid"], name)), (None, P.NO_ANSWER))[-1] if name == P.name_of(mid) else real_up(offer, name, ssh))
    world.steps()
    row = model(mid); s = D.row_state(mid)["rent_failed"]
    assert (row["stopped"], row["pod"]["state"]) == ("rent_failed", "rent-failed") and s["no_answer"] and s["tries"] == 1 and "no answer" in s["why"]
    assert [a["row"] for a in table()["alarms"] if a["what"] == "rent-no-answer"] == [mid]  # at once, and it says where to look
    assert "bench pods" in next(e["do"] for e in events(capsys) if e.get("what") == "rent-no-answer")
    count = len(P.rents()); world.steps(4)
    assert [n for _, n in lium.ups].count(P.name_of(mid)) == 1 and len(P.rents()) == count == 2  # no second `lium up`, and the count of the day stays
    late = lium.pod(P.name_of(mid), price=0.8, config="8xH100")  # Lium made the pod after the cut
    monkeypatch.setattr(P, "up", real_up); D.cli("retry", [mid]); world.steps(3)
    assert [n for _, n in lium.ups].count(P.name_of(mid)) == 1 and not [p for p in lium.pods if p["name"] == P.name_of(mid) and p is not late]  # the retry took that pod: no second one
    assert states()[mid] == "scored" and not table()["alarms"]


def test_a_cut_of_lium_up_kills_it_and_leaves_no_file_with_the_address(tmp_path, monkeypatch):
    fake = tmp_path / "lium"; ssh = tmp_path / "row" / "node-ssh"; ssh.parent.mkdir()
    fake.write_text(f"#!/bin/bash\necho $$ > {tmp_path}/pid\necho '{{\"pod\": {{\"ip\": \"203.0.113.7\"}}}}'\nsleep 60\n"); fake.chmod(0o755)
    monkeypatch.setattr(P, "LIUM", str(fake)); monkeypatch.setattr(P, "UP_TIMEOUT_S", 2.0); monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert P.up(dict(huid="h", gpus=8, kind="8xH100", price=1), "affine-v2-bench-x", str(ssh)) == (None, P.NO_ANSWER)
    pid = int((tmp_path / "pid").read_text())
    assert not os.path.exists(f"/proc/{pid}")  # `lium up` itself is dead, not only the script around it
    assert sorted(x.name for x in tmp_path.iterdir()) == ["lium", "pid", "row"] and not list(ssh.parent.iterdir())  # and Lium's answer is on no disk


def test_a_bench_pod_that_no_row_claims_is_an_alarm_in_the_table_and_a_pod_in_its_rent_is_not(world, monkeypatch):
    lium = world.lium; gen = ids(world.genesis); monkeypatch.setattr(D, "LIUM_CHECK_S", 0)
    lium.pod(P.name_of("king-" + "f" * 16)); lium.pod("sn120-v2-prod-dublin", id="6cb913e4-0000-4000-8000-000000000000")
    lium.pod(P.name_of(gen))  # a pod under the name of a row that is in its rent: Lium has it, the record is not written yet
    os.makedirs(f"{D.ROWS}/{gen}"); D.wj(f"{D.ROWS}/{gen}/daemon.json", dict(fails=0, history=[], phase=dict(state="renting", since=1)))
    sent = len(lium.sent); D.lium_check()
    assert D.S["alarms"] == [dict(what="pod-without-row", pod=P.name_of("king-" + "f" * 16))] and [a[0] for a in lium.sent[sent:]] == ["ps"]
    world.steps()
    assert table()["alarms"] == D.S["alarms"] and not lium.removed  # a person decides; nothing is deleted


def test_one_result_file_that_lacks_a_field_stops_neither_the_table_nor_the_other_rows(world, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    os.makedirs(f"{D.RES}/{gen}")  # a file of before on the host, without `se`
    D.wj(D.result_path(gen, "gsm8k"), dict(model_id=gen, benchmark="gsm8k", score=0.5, tasks=2, samples=2, rollouts=1, cap_rate=0.0, bench_version="gsm8k/v0"))
    world.steps(3)
    assert states()[mid] == "scored" and not world.pod_of(mid)  # the other row is scored and its pod is given back
    told = [e for e in events(capsys) if e["event"] == "result-unreadable"]
    assert len(told) == 1 and told[0]["file"].endswith(f"{gen}/gsm8k.json") and "se" in told[0]["error"]
    # and a result of a node that lacks a field is not stored: a failed try, as a result that is missing
    king2 = green(v2, 2); mid2 = ids(king2)
    world.node(mid2).results["gsm8k"] = {True: json.dumps(dict(benchmark="gsm8k", score=0.5, tasks=2, samples=2, rollouts=1, cap_rate=0.0,
                                                               bench_version=R.bench_version(R.benchmark("gsm8k")))), False: '{"idx":0,"correct":1.0}\n'}
    world.steps(2)
    assert not os.path.exists(D.result_path(mid2, "gsm8k")) and D.row_state(mid2)["fails"] >= 1 and model(mid2)["benchmarks_done"] == len(RUNNABLE) - 1


def test_a_result_of_another_benchmark_version_is_a_failed_try_and_not_a_run_with_no_end(world, capsys):
    gen = ids(world.genesis); world.steps()
    node = world.nodes[gen]
    node.results["gsm8k"] = {True: json.dumps(dict(benchmark="gsm8k", score=0.5, se=0.1, tasks=2, tasks_total=2, samples=2, rollouts=1, cap_rate=0.0,
                                                   bench_version="gsm8k/v0/of-another-commit", finished_at=1)), False: '{"idx":0,"correct":1.0}\n'}
    pod = dict(world.lium.pods[0])
    world.steps(D.MAX_FAIL + 1)
    assert node.launches == D.MAX_FAIL and D.stopped(gen) == "failed" and world.lium.removed == [rm(pod)]
    assert "version-differs" in [e["event"] for e in events(capsys)]


def test_a_fetch_of_weights_that_passes_its_hour_is_a_failed_try_and_after_three_the_pod_goes_back(world, monkeypatch):
    lium = world.lium; gen = ids(world.genesis)
    def fetch(m, node): raise subprocess.TimeoutExpired("gate_fetch.py", 3600)
    monkeypatch.setattr(D, "fetch_weights", fetch)
    seen = []
    for _ in range(D.MAX_FAIL):
        world.steps(); seen.append((D.row_state(gen)["fails"], len(lium.pods)))
    assert seen == [(1, 1), (2, 1), (3, 1)] and len(lium.ups) == 1
    pod = dict(lium.pods[0]); world.steps()
    assert lium.removed == [rm(pod)] and not lium.pods and model(gen)["stopped"] == "failed" and world.nodes[gen].launches == 0


def test_the_ceiling_of_a_day_is_one_step_with_the_rent_and_a_rent_that_makes_no_pod_gives_its_count_back(world, monkeypatch):
    lium = world.lium; gen = ids(world.genesis)
    assert P.rents(add=True, cap=2) and P.rents(add=True, cap=2) and P.rents(add=True, cap=2) is None and len(P.rents()) == 2  # the third finds the day full
    P.rents(drop=P.rents()[0]); assert len(P.rents()) == 1
    monkeypatch.setattr(P, "MAX_RENTS_PER_DAY", 2); lium.up_fails = 1; monkeypatch.setattr(P, "RENT_TRIES", 9)
    world.steps()
    assert len(lium.ups) == 1 and not lium.pods and len(P.rents()) == 1  # a `lium up` that made no pod does not count
    world.steps()
    assert len(lium.pods) == 1 and len(P.rents()) == 2  # the pod that was made counts, once


def test_the_rent_script_leaves_no_file_with_the_address_and_the_ssh_scripts_are_private(tmp_path):
    fake = tmp_path / "lium"; ssh = tmp_path / "ssh" / "node-ssh"
    fake.write_text("#!/bin/bash\necho '{\"pod\": {\"id\": \"i1\", \"huid\": \"h\", \"name\": \"affine-v2-bench-x\", \"config\": \"8xH100\", \"price_per_hour\": 1, "
                    "\"ip\": \"203.0.113.7\", \"ports\": {\"22\": 2201}}}'\n"); fake.chmod(0o755)
    before = set(os.listdir("/tmp"))
    p = subprocess.run(["bash", P.RENT, "h 8 8xH100 1"], capture_output=True, text=True, env=dict(os.environ, LIUM=str(fake), NAME="affine-v2-bench-x", MAX_PRICE="2",
                                                                                                    NODE_SSH=str(ssh), TMPDIR=str(tmp_path)))
    assert p.returncode == 0 and json.loads(p.stdout.strip().splitlines()[-1])["id"] == "i1" and "203.0.113.7" not in p.stdout + p.stderr
    assert oct(ssh.stat().st_mode & 0o777) == "0o700" and "203.0.113.7" in ssh.read_text()
    assert set(os.listdir("/tmp")) == before and sorted(x.name for x in tmp_path.iterdir()) == ["lium", "ssh"]  # no file with Lium's answer is left
    row = P.Pod("king-" + "a" * 16); os.makedirs(row.dir, exist_ok=True)
    assert row._write_ssh(dict(ip="203.0.113.7", ports={"22": 2201}, id="i1")) and oct(os.stat(row.ssh).st_mode & 0o777) == "0o700"
    bad = tmp_path / "bad-ssh"; bad.write_text("#!/bin/bash\necho 'ssh: connect to host 203.0.113.7 port 2201: Connection refused' >&2; exit 255\n"); bad.chmod(0o700)
    assert P.setup(str(bad)) == (False, "ssh: connect to host ADDRESS port 2201: Connection refused\n")  # the text for the journal has no address
