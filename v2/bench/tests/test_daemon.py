"""The host path: public files and a store from the CPU simulation of v2, a fake node for each row and a fake Lium.

The service starts one worker process for each row. Here a worker is one call in this process: each `D.step()` of
the service gives each row in work one poll, in the order in which the service starts the workers."""
import ast, dataclasses, hashlib, importlib, json, os, pathlib, re, runpy, shutil, types
import pytest
from sn120_v2 import layout, sim
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
        assert key.startswith(("public/bases/", "v1-king/")); self.asked.append(key)
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
        if a[:1] == ("ps",): out = json.dumps(self.pods)
        if a[:1] == ("ls",): out = json.dumps([o for o in self.offers if o["gpu_type"] == a[2]])
        if a[:1] == ("rm",): self.pods = [p for p in self.pods if p["id"] != a[1]]
        if a[:1] == ("describe",): out = "listed" if any(p["id"] == a[1] for p in self.pods) else "the pod is no longer listed"
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

    def pod_of(self, mid): return next((p for p in self.lium.pods if p["name"] == P.PREFIX + mid), None)


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
    assert sorted(p["name"] for p in lium.pods) == sorted(P.PREFIX + mid for mid in rows) and len(lium.ups) == 3
    assert all(P.PREFIX + mid == f"affine-v2-bench-{mid}" and 37 <= len(P.PREFIX + mid) <= 40 for mid in rows)
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
    assert states() == {mid: "scored", gen: "scored"} and lium.ups[-1] == ("late-offer", P.PREFIX + gen)


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
    assert lium.ups == [("h100-cheap", P.PREFIX + gen)] and D.pod_of(gen).load()["cost"] == 22.0
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
    stray = dict(lium.pod(P.PREFIX + mid))
    world.steps()
    assert len(lium.ups) == 1 and D.pod_of(mid).load()["id"] == stray["id"] and states()[mid] == "evaluating"
    world.steps(2)
    assert lium.removed[-1] == rm(stray)

    # two pods under one row's name: the service takes none, makes none and deletes none. The operator decides
    second = green(world.v2, 2); mid2 = ids(second)
    lium.pod(P.PREFIX + mid2); lium.pod(P.PREFIX + mid2)
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
    monkeypatch.setattr(P, "up", lambda offer, name, ssh: (lium.ups.append((offer["huid"], name)), (None, "Lium refused the name"))[-1] if name.endswith(mid) else real_up(offer, name, ssh))
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
    pod = P.Pod(row); other = lium.pod("someone-elses-pod"); twin = lium.pod(P.PREFIX + row + "-old")
    mine = lium.pod(pod.name)
    pod.save(dict(id=mine["id"], name=pod.name, huid=mine["huid"], rented_at=D.now()))
    assert pod.release(D.log, why="test") and lium.removed == [rm(mine)] and lium.pods == [other, twin]  # one command, one id; a like name is not touched
    assert pod.release(D.log, why="again") and len(lium.removed) == 1  # no record: no command

    refused = lambda: [e["why"] for e in events(capsys) if e["event"] == "release-refused"]
    capsys.readouterr()
    # the prod GPU box and the operator's pod, by id: also when a record is wrong and names one of them, under any name
    for bad_id in ("6cb913e4-0000-4000-8000-000000000000", "c44cb565-0000-4000-8000-000000000000"):
        lium.pod(pod.name, id=bad_id); pod.save(dict(id=bad_id, name=pod.name, huid="x"))
        assert not pod.release(D.log, why="wrong record", row_done=True) and "is protected" in refused()[0]
        lium.pods.pop()
    # and by name: in the record, or as Lium lists the id
    for name in ("sn120-v2-prod-dublin", "casper-h100-ask-before-delete"):
        p = lium.pod(name); pod.save(dict(id=p["id"], name=pod.name, huid="x"))
        assert not pod.release(D.log, why="wrong record", row_done=True) and "is protected" in refused()[0]
        q = lium.pod(pod.name); pod.save(dict(id=q["id"], name=pod.name, huid=name))
        assert not pod.release(D.log, why="wrong record", row_done=True) and "is protected" in refused()[0]
        lium.pods = lium.pods[:-2]
    # a record that names the pod of another name: refused
    pod.save(dict(id=other["id"], name=pod.name, huid="x"))
    assert not pod.release(D.log, why="wrong record") and "is named 'someone-elses-pod'" in refused()[0]
    # the pod of the single-node service: only the worker of its row, and only when that row's results are published
    calm = lium.pod(P.NAME, id="296c5237-0000-4000-8000-000000000000", huid="calm-matrix-26")
    held = P.HELD_FOR_ROW["296c5237"]; assert held == "king-10757e879efe5b91"
    for who, done in ((P.Pod(row, first=True), True), (P.Pod(held, first=True), False)):
        who.save(dict(id=calm["id"], name=P.NAME, huid="calm-matrix-26"))
        assert not who.release(D.log, why="test", row_done=done) and "is kept until row king-10757e879efe5b91 is done" in refused()[0]
    assert len(lium.removed) == 1 and calm in lium.pods and other in lium.pods  # none of the refusals sent a command
    assert P.Pod(held, first=True).release(D.log, why="row done", row_done=True) and lium.removed[-1] == rm(calm)

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


def test_the_run_of_the_single_node_service_is_the_first_row_it_keeps_its_pod_and_is_not_started_again(world, monkeypatch, capsys):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid, gen = ids(king), ids(world.genesis)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "HELD_FOR_ROW", {"296c5237": mid}); monkeypatch.setattr(P, "MAX_PRICE", 35.0)
    calm = single_node_state(world, mid, m)
    assert (ARXIV["rollouts"], ARXIV["problems"], R.bench_version(ARXIV)) == (2, 103, "arxivmath/v2/bench-v2")
    node = world.node(mid); node.busy = "done=\nalive=1\nlines=1400\nfinished=9\nok=1\n"
    world.steps()
    # the present record is the pod of that row: no rent for it, no new commit on it, no start of a runner on it
    assert D.is_first(mid) and D.pod_of(mid).record == C.STATE + "/pod.json" and D.pod_of(mid).name == P.NAME and D.pod_of(mid).ssh == C.NODE_SSH
    assert states() == {mid: "evaluating", gen: "evaluating"} and lium.ups == [(lium.ups[0][0], P.PREFIX + gen)]
    assert world.setups == [D.pod_of(gen).ssh] and node.launches == 0 and not any("git" in c or "gate_fetch" in c for c in node.cmds)
    assert model(mid)["pod"] | dict(since=0) == dict(name="calm-matrix-26", gpu="8xH100", usd_h=23.2, state="running", since=0)
    assert table()["node"]["pod"] == P.NAME and table()["node"]["up"] is True  # the top-level node object is still that pod
    assert D.S["running"]["run_key"] == mid  # the single-node state is left as it was, for the way back
    world.steps(2)
    assert states()[gen] == "scored" and node.launches == 0 and calm in lium.pods  # the other row came and went; this one runs on

    # its full run ends: 16 samples a problem. The ArXivMath value of the row is 2 of the 16, by the fixed rule
    right = [(i * 7) % 17 for i in range(103)]
    node.busy = None; node.results["arxivmath"] = full_run(right)
    world.steps()
    r = D.rj(D.result_path(mid, "arxivmath"))
    want = sum(sum(k < c for k in derive.picked("arxivmath", i, 16, 2)) / 2 for i, c in enumerate(right)) / 103
    assert (r["bench_version"], r["rollouts"], r["samples"], r["tasks"], r["complete"]) == ("arxivmath/v2/bench-v2", 2, 206, 103, True)
    assert abs(r["score"] - want) < 1e-12 and r["score"] != r["score_full"] and abs(r["score_full"] - sum(right) / 1648) < 1e-12
    assert (r["derived_from"], r["derived_by"], r["samples_full"]) == ("arxivmath/v1/bench-v2", derive.RULE, 1648)
    assert D.rj(f"{D.PRIV}/{mid}/arxivmath.full.json")["rollouts"] == 16  # the full run's record is kept on the host
    cell = next(x for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath")
    assert (cell["env_version"], cell["samples"], cell["samples_expected"], cell["n"], cell["complete"]) == ("arxivmath/v2/bench-v2", 206, 206, 103, True)
    assert (cell["derived_from"], cell["derived_by"], cell["samples_full"], cell["score_full"]) == ("arxivmath/v1/bench-v2", derive.RULE, 1648, r["score_full"])
    assert len([x for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath"]) == 1  # one ArXivMath row, like each other row
    assert states()[mid] == "scored" and node.launches == 0 and calm in lium.pods and not D.missing(m, RUNNABLE)  # no second ArXivMath run
    world.steps()  # the table of the store has the row: its pod is given back by its exact id
    assert lium.removed[-1] == rm(calm) and calm not in lium.pods and not D.is_first(mid) and table()["node"]["up"] is False
    ups = len(lium.ups); world.steps(3)
    assert len(lium.ups) == ups and node.launches == 0 and model(mid)["pod"]["state"] == "released"


@pytest.mark.parametrize("fault", ["a sample is missing", "a sample ended with an error", "a problem is missing"])
def test_a_full_run_that_is_not_whole_gives_no_arxivmath_value_and_a_plain_line(world, monkeypatch, capsys, fault):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid = ids(king)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "HELD_FOR_ROW", {"296c5237": mid}); monkeypatch.setattr(P, "MAX_PRICE", 35.0)
    calm = single_node_state(world, mid, m)
    run = full_run([8] * 103); summary = json.loads(run[True]); lines = [json.loads(x) for x in run[False].splitlines()]
    if fault == "a problem is missing": lines.pop(); summary.update(tasks=102, tasks_total=102, samples=1632)
    else: lines[5].update(n=15, errors=1); summary.update(samples=1647, errors=1, complete=False)
    node = world.node(mid); node.results["arxivmath"] = {True: json.dumps(summary), False: "".join(json.dumps(p) + "\n" for p in lines)}
    capsys.readouterr()
    world.steps()
    (line,) = [e for e in events(capsys) if e["event"] == "no-value"]
    assert line["bench"] == "arxivmath" and "the full run is not whole" in line["why"] and ("1647" in line["why"] or "1632" in line["why"])
    assert not os.path.exists(D.result_path(mid, "arxivmath")) and D.rj(f"{D.PRIV}/{mid}/arxivmath.full.json")["samples"] == summary["samples"]
    row = model(mid)
    assert (row["status"], row["stopped"], row["benchmarks_done"], row["fails"]) == ("failed", "held", len(RUNNABLE) - 1, 0)
    assert not [x for x in table()["results"] if x["model_id"] == mid and x["benchmark_id"] == "arxivmath"]  # no value at all
    world.steps(2)  # the run is over: the pod is given back when the nine results are in the store, and no run is started by itself
    assert rm(calm) in lium.removed and not lium.pods and node.launches == 0 and [name for _, name in lium.ups] == [P.PREFIX + ids(world.genesis)]
    D.cli("retry", [mid])  # the operator's word: a new run of ArXivMath alone, at the settings' count, on a pod of the row's own
    world.steps()
    assert lium.ups[-1][1] == P.PREFIX + mid and "--benchmarks arxivmath " in world.nodes[mid].cmds[-1]


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


def test_a_first_pod_that_lium_took_back_makes_the_row_a_row_as_any_other(world, monkeypatch):
    v2, lium = world.v2, world.lium
    king = green(v2, 1); mid = ids(king)
    m = next(r for r in D.model_rows() if r["model_id"] == mid)
    monkeypatch.setattr(P, "HELD_FOR_ROW", {"296c5237": mid}); monkeypatch.setattr(P, "MAX_PRICE", 35.0)
    monkeypatch.setattr(D, "POD_CHECK_AFTER_S", 0); monkeypatch.setattr(D, "POD_CHECK_EVERY_S", 0)
    calm = single_node_state(world, mid, m)
    lium.pods.remove(calm); world.node(mid).busy = "no answer"
    world.steps(2)
    assert not D.is_first(mid) and not lium.removed and D.row_state(mid)["fails"] == 0
    world.steps()
    assert lium.ups[-1][1] == P.PREFIX + mid and states()[mid] == "evaluating"  # its own pod, the run from the start, ArXivMath at the settings' count


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
def v1_source(world, monkeypatch, files=None):
    """The operator sets the source: a prefix in the store with the descriptor and each file."""
    row = source.extra_rows()[0]; files = {"model.safetensors": b"v1 weights", "config.json": b"{}"} if files is None else files
    desc = dict(id=row["base_hash"], files={n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    world.store.inner.put("v1-king/b6d592c3/descriptor.json", json.dumps(desc).encode())
    for n, d in files.items(): world.store.inner.put(f"v1-king/b6d592c3/{n}", d)
    monkeypatch.setenv("BENCH_V1_KING_SOURCE", "v1-king/b6d592c3/")
    return desc


def test_the_v1_king_has_a_row_of_its_own_kind_and_changes_no_king_row_and_not_the_kings_order(world, monkeypatch):
    v2, lium = world.v2, world.lium
    first, second = green(v2, 1), green(v2, 2)
    row = source.extra_rows()[0]
    assert (row["model_id"], row["kind"], row["uid"], row["v1_epoch"]) == ("v1king-b6d592c3bb1b4393", "v1-king", 920, 1211)
    assert row["hotkey"] == "5FaTFWogz3AiQbEFRnL9CZcqR1keeypEYkd7f1L5YLbWH1cP" and row["base_hash"] == "b6d592c3bb1b439365e82147a47e65d93c083a13444aceae078f8a2053cd5ce3"
    assert 37 <= len(P.PREFIX + V1) <= 40 and "v1-king" in publish.OWN_KINDS
    # the same settings and budget as a king, and the same ten benchmarks
    assert R.resolve_settings(V1, "v1-king") == R.resolve_settings(ids(first), "king") and R.resolve_settings(V1, "v1-king")["budget"] == 65536
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
    for fault in ("no descriptor", "another checkpoint", "a file is not there", "the store does not answer"):
        store.hidden, store.gone, store.refused = set(), set(), False
        if fault == "no descriptor": store.hidden = {"v1-king/b6d592c3/descriptor.json"}
        if fault == "another checkpoint": store.inner.put("v1-king/b6d592c3/descriptor.json", json.dumps(dict(desc, id="0" * 64)).encode())
        if fault == "a file is not there": store.inner.put("v1-king/b6d592c3/descriptor.json", json.dumps(desc).encode()); store.gone = {"v1-king/b6d592c3/config.json"}
        if fault == "the store does not answer": store.refused = True
        world.steps(2)
        assert (model(V1)["status"], model(V1)["waiting_for"]) == ("waiting", "source"), fault
        assert len(lium.ups) == ups and not lium.pods and len(world.polled) == polled and not os.path.isdir(f"{D.ROWS}/{V1}"), fault  # no rent, no worker, no pod that waits
    assert not any("v1-king/" in key for key in store.signed)
    store.refused = False
    world.steps()  # readable: queued by this step, with no word of a person
    assert states(v1=True)[V1] == "evaluating" and lium.ups[-1][1] == P.PREFIX + V1 and len(lium.ups) == ups + 1
    node = world.nodes[V1]
    blob = json.loads(next(s for s in node.stdin if s))
    assert blob["descriptor"] == desc and sorted(blob["urls"]) == sorted(desc["files"]) and all("v1-king/b6d592c3/" in u for u in blob["urls"].values())
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
    other = lium.pod(P.PREFIX + "v1king-0000000000000000")  # a pod with a like name is not this row's pod
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
    lium.pod("sn120-v2-prod-dublin", id="6cb913e4-0000-4000-8000-000000000000"); lium.pod(P.PREFIX + "king-" + "f" * 16)  # not a bench pod; a bench pod of no row
    lium.pods = [p for p in lium.pods if p["name"] != P.PREFIX + gen]  # a row whose pod is not there
    sent = len(lium.sent); capsys.readouterr()
    D.cli("pods", [])
    out = capsys.readouterr().out
    lines = {line.split()[1]: line for line in out.splitlines()[1:-1]}
    assert world.pod_of(mid)["id"][:8] in lines[P.PREFIX + mid] and f" {mid} " in lines[P.PREFIX + mid] and " running " in lines[P.PREFIX + mid] and " 0.8" in lines[P.PREFIX + mid]
    assert "NO ROW CLAIMS THIS POD" in lines[P.PREFIX + "king-" + "f" * 16] and "NOT LISTED BY LIUM" in lines[P.PREFIX + gen]
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
        f"A pod lives {P.POD_MAX_HOURS:.0f} hours at most", f"{P.RENT_TRIES} tries", f"the name `{P.NAME}`", "`affine-v2-bench-<model_id>`",
        f"asks again every {P.NO_OFFER_RETRY_S:.0f} seconds", f"no progress for {D.STALL_S} seconds",
        f"After {D.MAX_FAIL} fails", f"within one poll ({D.POLL_S} seconds)", f"more than {D.WAIT_LATE_S // 60} minutes after its head was applied",
        f"about each {D.BEAT_S // 60} minutes", f"After {D.POD_CHECK_AFTER_S // 60} minutes the worker asks Lium",
        f"waits {D.CONFIRM_WAIT_S // 60} minutes at most", f"`{derive.RULE}`", "BENCH_V1_KING_SOURCE",
        *(f"`{x}`" for x in P.PROTECTED_IDS + P.PROTECTED_NAMES + tuple(P.HELD_FOR_ROW)),
        f"runs the {len(RUNNABLE)} benchmarks", f"`bids` names {len(RUNNABLE)} benchmarks", f"`n_gpus = {defaults['n_gpus']}`",
        f"at most {defaults['round_time_limit_s'] // 60} minutes (`round_time_limit_s = {defaults['round_time_limit_s']}`)",
        f"take {R.serving()['server']['gpu_memory_utilization']:.2f} of each GPU's memory", f"vLLM {R.serving()['server']['engine'].split('==')[1]} serves",
        "holds more than 1024 MiB. It waits 120 seconds", "genesis-" + GENESIS_BASE_HASH[:16],
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
