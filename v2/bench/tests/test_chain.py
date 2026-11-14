"""A king whose full files are not in the store: its row's node rebuilds it from the newest checkpoint and the LoRAs of
the steps after it (affine_bench/chain.py, deploy/chain_rebuild.py).

The host path uses the store fake, the fake Lium and the fake node of test_daemon.py. The box in `chain` mode is the
CPU simulation of v2 with the weight files of a king taken out of the store, and a chain record written from its state.
The node's script runs here for real, on files of this machine, with a stand-in for the rebuild tool."""
import hashlib, json, os, pathlib, re, subprocess, sys, types
import pytest
from sn120_v2 import layout
from sn120_v2.errors import InfraError
from affine_bench import chain, daemon as D, fetch
import test_daemon as TD
from test_daemon import world  # noqa: F401 - the fixture

REF = "c" * 40
SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "chain_rebuild.py"


class ChainStore(TD.Store):
    """The store fake, and the size of an object. `exists` is still asked for a base file only."""
    def size(self, key):
        assert key.startswith(("public/bases/", "public/masters/", "public/loras/")); self.asked.append(key)
        if self.refused: raise InfraError("r2 head failed: 403")
        if key in self.gone: raise KeyError(key)
        return self.inner.size(key)


class ChainNode(TD.Node):
    """A node that also has deploy/chain_rebuild.py: `tool(blob)` says whether the rebuild ends with every sha256 right.
    Only then the mark .verified is there, as the script leaves it."""
    def __init__(self):
        super().__init__(); self.rebuilds, self.tool = [], lambda blob: True

    def run(self, cmd, inp=None, timeout=0):
        if "chain_rebuild.py" not in cmd:
            return super().run(cmd, inp, timeout)
        self.cmds.append(cmd); self.stdin.append(inp); blob = json.loads(inp); self.rebuilds.append(dict(blob=blob, cmd=cmd, timeout=timeout))
        ok = self.tool(blob)
        if ok: self.have.add(blob["descriptor"]["id"])
        return types.SimpleNamespace(stdout=json.dumps(dict(status="rebuilt" if ok else "refused")), stderr="", returncode=0 if ok else 2)

    @property
    def fetches(self): return sum("gate_fetch.py" in c for c in self.cmds)


@pytest.fixture
def cw(world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(TD, "Node", ChainNode)
    world.store = store = ChainStore(world.v2.store)
    monkeypatch.setattr(chain.source, "store", lambda: store)
    for name, value in dict(REF=REF, _sizes={}, WEIGHT_SUFFIX=".bin", GRACE_S=300, UPLOAD_WAIT_S=3600).items():  # the simulation's weight file is model.bin
        monkeypatch.setattr(chain, name, value)
    world.v2.store.put(layout.public_master_file(world.genesis.base_fp32_master_digest, "master.bin"), b"sim master")  # as the genesis job leaves it
    return world


def weight_keys(head):
    return {layout.public_base_file(head.base_hash, "model.bin"), layout.public_master_file(head.base_fp32_master_digest, "master.bin")}


def box(world, kings, checkpoints=(), kernel="fixed-order-v1", flags=None):
    """What the box in `chain` mode leaves in the store: the chain record, and the weight files only of a checkpoint."""
    v2, heads = world.v2, world.v2.controller.state.heads
    by_id = {h.version_id: h for h in heads}; marked = {h.version_id for h in checkpoints}; flags = flags or {}
    ids = lambda h: dict(version_id=h.version_id, base_hash=h.base_hash, base_fp32_master_digest=h.base_fp32_master_digest)
    steps, last = [], heads[0].version_id
    for head in [h for h in heads[1:] if h.folded]:
        loras = [dict(lora_hash=ref.lora_hash, round_id=ref.round_id, miner_uid=ref.miner_uid, rank=ref.rank,
                      files={key.rsplit("/", 1)[-1]: v2.store.sha256(key) for key in v2.store.list(layout.public_lora_prefix(ref.lora_hash) + "/")})
                 for ref in by_id[head.parent_version_id].loras]
        steps.append(dict(from_version_id=last, stacked_version_id=head.parent_version_id, round_id=head.round_id, loras=loras, kernel=kernel,
                          checkpoint=head.version_id in marked) | ids(head) | ({chain.PLANNED: flags[head.version_id]} if head.version_id in flags else {}))
        last = head.version_id
    doc = dict(schema=chain.SCHEMA, head_version_id=heads[-1].version_id, recipe={}, genesis=ids(heads[0]) | dict(checkpoint=True), steps=steps, stacked=None)
    v2.store.put(chain.HEAD_CHAIN, json.dumps(doc).encode())
    world.store.gone = set().union(*[weight_keys(k) for k in kings if k.version_id not in marked])
    return doc


def kings_of(world, n): return [TD.green(world.v2, uid) for uid in range(1, n + 1)]
def lora_keys(world, king):
    ref = next(h for h in world.v2.controller.state.heads if h.version_id == king.parent_version_id).loras[0]
    return world.v2.store.list(layout.public_lora_prefix(ref.lora_hash) + "/")


@pytest.mark.parametrize("n", [1, 3, 12])
def test_a_king_without_its_full_files_is_rebuilt_on_its_node_from_the_checkpoint_and_each_lora_after_it(cw, n):
    world, store, genesis = cw, cw.store, cw.genesis
    kings = kings_of(world, n); king = kings[-1]; mid = TD.ids(king)
    box(world, kings)
    world.steps(3)
    node = world.nodes[mid]; (call,) = node.rebuilds; blob = call["blob"]
    assert TD.states()[mid] == "scored" and D.row_state(mid)["fails"] == 0 and node.launches == 1 and node.fetches == 0
    manifest = world.v2.controller.state.manifest
    assert blob["descriptor"] == dict(id=king.base_hash, files=dict(manifest.base_files)) and blob["head"] == king.version_id  # the hashes of the king's manifest
    tail = blob["chain"]
    assert tail["start"]["version_id"] == genesis.version_id and tail["start"]["checkpoint"] is True and tail["stacked"] is None
    assert [s["version_id"] for s in tail["steps"]] == [k.version_id for k in kings] and tail["head_version_id"] == king.version_id
    links = blob["links"]; (point,) = links["checkpoints"]
    assert point["version_id"] == genesis.version_id and point["base"]["id"] == genesis.base_hash and point["master"]["id"] == genesis.base_fp32_master_digest
    assert [f["name"] for f in point["master"]["files"]] == ["master.bin"] and point["master"]["bytes"] == len(b"sim master")
    assert [s["id"] for s in links["chain_loras"]] == [l["lora_hash"] for s in tail["steps"] for l in s["loras"]] and len(links["chain_loras"]) == n
    assert all(f["url"].startswith("https://signed.example/public/") and f["size"] > 0 and len(f["sha256"]) == 64 for s in links["chain_loras"] for f in s["files"])
    limit = chain.BASE_S + n * chain.STEP_S
    assert call["timeout"] == limit and links["expires_at"] - D.now() > limit  # a link lives longer than the rebuild may take
    assert not [key for key in store.signed if king.base_hash in key]  # no link to a file that is not in the store
    want = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()  # the node runs the script only when it is the host's, and with the pinned commit
    assert f"echo '{want}  /data/bench/repo/v2/bench/deploy/chain_rebuild.py' | sha256sum -c --status" in call["cmd"]
    assert f"--ref {REF} " in call["cmd"] and call["cmd"].endswith("--device cpu") and "signed.example" not in call["cmd"]
    assert "alarms" not in TD.table() or not TD.table()["alarms"]


def test_a_king_that_is_a_checkpoint_is_fetched_as_before_and_a_later_king_starts_from_it(cw):
    world, store, genesis = cw, cw.store, cw.genesis
    kings = kings_of(world, 3); first, second, third = kings
    box(world, kings, checkpoints=[first])
    signed = len(store.signed)
    world.steps(3)
    assert set(TD.states().values()) == {"scored"}
    for head in (genesis, first):  # a chain of 0 steps: the node downloads the full base, and no rebuild runs
        node = world.nodes[TD.ids(head)]
        assert node.fetches == 1 and not node.rebuilds and json.loads(node.stdin[node.cmds.index(next(c for c in node.cmds if "gate_fetch.py" in c))])["descriptor"]["id"] == head.base_hash
    for king, steps in ((second, [second]), (third, [second, third])):
        (call,) = world.nodes[TD.ids(king)].rebuilds; tail = call["blob"]["chain"]
        assert tail["start"]["version_id"] == first.version_id and [s["version_id"] for s in tail["steps"]] == [k.version_id for k in steps]
        assert call["blob"]["links"]["checkpoints"][0]["master"]["id"] == first.base_fp32_master_digest and world.nodes[TD.ids(king)].fetches == 0
    assert len(store.signed) > signed


def test_a_missing_lora_makes_the_row_wait_with_no_failed_try_and_no_rent(cw, capsys):
    world, store, lium = cw, cw.store, cw.lium
    world.steps(3)
    kings = kings_of(world, 3); box(world, kings)
    lost = set(lora_keys(world, kings[1])[:1]); store.gone |= lost
    ups = len(lium.ups); capsys.readouterr()
    world.steps(D.MAX_FAIL + 2)
    states = TD.states()
    assert states[TD.ids(kings[0])] == "scored" and len(lium.ups) == ups + 1  # the way to the first king does not need that LoRA
    for king in kings[1:]:
        row = TD.model(TD.ids(king))
        assert (row["status"], row["fails"], row["waiting_for"]) == ("queued", 0, "weights") and "pod" not in row
        assert not os.path.isdir(f"{D.ROWS}/{TD.ids(king)}") and TD.ids(king) not in world.nodes
    waits = [e for e in TD.events(capsys) if e["event"] == "waiting"]
    assert len(waits) == 2 and all(sorted(lost)[0] + " is not in the store" in e["file"] for e in waits)  # the journal says which file
    store.gone -= lost
    world.steps(2)
    assert set(TD.states().values()) == {"scored"} and all(D.row_state(TD.ids(k))["fails"] == 0 for k in kings)


def test_a_rebuild_with_a_wrong_file_hash_is_refused_and_the_row_never_runs_on_those_weights(cw, capsys):
    world, lium = cw, cw.lium
    world.steps(3)
    kings = kings_of(world, 2); king = kings[-1]; mid = TD.ids(king); box(world, kings)
    world.node(mid).tool = lambda blob: False  # the script found another sha256: it leaves no mark
    seen = []
    for _ in range(D.MAX_FAIL):
        world.steps(); seen.append((TD.states()[mid], D.row_state(mid)["fails"]))
    assert seen == [("queued", 1), ("queued", 2), ("failed", 3)] and world.nodes[mid].launches == 0 and len(world.nodes[mid].rebuilds) == D.MAX_FAIL
    pod = dict(world.pod_of(mid)); world.steps()
    assert lium.removed[-1] == TD.rm(pod) and not world.pod_of(mid) and world.nodes[mid].launches == 0  # the pod goes back by its exact id, as after a failed fetch
    assert [e["status"] for e in TD.events(capsys) if e["event"] == "rebuild" and e["row"] == mid] == ["rc2"] * D.MAX_FAIL


def test_a_rebuild_that_is_cut_is_a_failed_try_and_the_next_try_goes_on_in_the_same_work_directory_with_new_links(cw, monkeypatch):
    world, store = cw, cw.store
    world.steps(3)
    kings = kings_of(world, 3); king = kings[-1]; mid = TD.ids(king); box(world, kings)
    node = world.node(mid); calls = []
    def cut(blob):
        calls.append(blob)
        if len(calls) == 1: raise subprocess.TimeoutExpired("chain_rebuild.py", 1)
        return True
    node.tool = cut
    world.steps()
    assert D.row_state(mid)["fails"] == 1 and node.launches == 0 and TD.states()[mid] == "queued" and len(world.lium.ups) >= 2
    pods = len(world.lium.ups); signed = len(store.signed)
    monkeypatch.setattr(D, "now", lambda t=D.now(): t + 5000)
    world.steps(2)
    first, second = node.rebuilds
    assert first["cmd"] == second["cmd"] and " /data/bench/models/kings/.chain " in second["cmd"]  # the work directory of the tool stays
    assert len(store.signed) > signed and second["blob"]["chain"] == first["blob"]["chain"]  # new links for the same way
    assert TD.states()[mid] == "scored" and node.launches == 1 and len(world.lium.ups) == pods  # on the same pod: no second rent


def test_a_king_whose_checkpoint_is_in_upload_waits_and_is_fetched_when_the_upload_is_whole(cw, monkeypatch, capsys):
    world, store, lium = cw, cw.store, cw.lium
    world.steps(3)
    (king,) = kings = kings_of(world, 1); mid = TD.ids(king); box(world, kings)
    monkeypatch.setattr(chain, "UPLOAD_WAIT_S", float("inf"))  # the clock of the simulation starts in 1970
    store.gone = {layout.public_base_file(king.base_hash, "model.bin")}  # one weight file of the checkpoint is at its name, the base is not whole
    ups = len(lium.ups); capsys.readouterr()
    world.steps(D.MAX_FAIL + 2)
    row = TD.model(mid)
    assert (row["status"], row["fails"], row["waiting_for"]) == ("queued", 0, "weights") and len(lium.ups) == ups and mid not in world.nodes
    assert ["in upload" in e["file"] for e in TD.events(capsys) if e["event"] == "waiting"] == [True]
    store.gone = set()
    world.steps(2)
    node = world.nodes[mid]
    assert TD.states()[mid] == "scored" and node.fetches == 1 and not node.rebuilds and D.row_state(mid)["fails"] == 0


def test_a_checkpoint_upload_that_stalls_is_rebuilt_after_the_bound_and_the_table_has_an_alarm(cw, monkeypatch, capsys):
    world, store = cw, cw.store
    world.steps(3)
    (king,) = kings = kings_of(world, 1); mid = TD.ids(king); box(world, kings)
    store.gone = {layout.public_base_file(king.base_hash, "model.bin")}
    age = D.now() - king.applied_at
    monkeypatch.setattr(chain, "UPLOAD_WAIT_S", age + 600)
    world.steps(2)
    assert TD.states()[mid] == "queued" and mid not in world.nodes
    monkeypatch.setattr(chain, "UPLOAD_WAIT_S", age - 600); capsys.readouterr()
    world.steps()
    (alarm,) = TD.table()["alarms"]
    assert (alarm["what"], alarm["row"]) == ("checkpoint-upload-stalled", mid) and len(world.nodes[mid].rebuilds) == 1
    assert [e["what"] for e in TD.events(capsys) if e["event"] == "alarm"] == ["checkpoint-upload-stalled"]
    world.steps(2)
    assert TD.states()[mid] == "scored" and not TD.table()["alarms"]  # the alarm is there while the row has its pod


def test_the_flag_of_the_box_and_the_grace_time_decide_when_a_new_head_is_rebuilt(cw, monkeypatch):
    world = cw
    world.steps(3)
    (king,) = kings = kings_of(world, 1); mid = TD.ids(king); m = next(r for r in D.model_rows() if r["model_id"] == mid)
    age = D.now() - king.applied_at
    monkeypatch.setattr(chain, "UPLOAD_WAIT_S", float("inf"))
    box(world, kings)
    monkeypatch.setattr(chain, "GRACE_S", age + 600)  # no flag in the record: a new head waits for the grace time
    assert "a checkpoint upload may still start" in chain.blocked(m) and chain.order(m) is None
    box(world, kings, flags={king.version_id: False})  # the box says that this head gets no checkpoint: no grace time
    assert chain.blocked(m) is None and chain.order(m)["steps"] == 1
    box(world, kings, flags={king.version_id: True})  # the box says that it gets one: the row waits for it, with no file in the store yet
    monkeypatch.setattr(chain, "GRACE_S", 0)
    assert "in upload" in chain.blocked(m)
    monkeypatch.setattr(chain, "UPLOAD_WAIT_S", age - 600)
    assert chain.blocked(m) is None and chain.order(m)["stalled"] is True


def test_no_rebuild_without_a_full_commit_id_or_through_a_step_that_only_the_box_repeats(cw, monkeypatch, capsys):
    world, store, lium = cw, cw.store, cw.lium
    world.steps(3)
    kings = kings_of(world, 2); king = kings[-1]; mid = TD.ids(king); m = next(r for r in D.model_rows() if r["model_id"] == mid)
    box(world, kings, kernel="torch-matmul-v1")
    assert "has the kernel torch-matmul-v1: the node does not replay it" in chain.blocked(m)
    box(world, kings)
    for ref in ("c" * 39, "main", "C" * 40, "c" * 40 + " x"):
        monkeypatch.setattr(chain, "REF", ref)
        assert "BENCH_CHAIN_REF is not a full commit id" in chain.blocked(m) and chain.order(m) is None
    monkeypatch.setattr(chain, "REF", REF); monkeypatch.setattr(chain, "DEVICE", "cuda:0; rm")
    assert "no rebuild" in chain.blocked(m) and chain.order(m) is None
    monkeypatch.setattr(chain, "DEVICE", "cuda:0")
    assert chain.blocked(m) is None and chain.order(m)["device"] == "cuda:0"
    monkeypatch.setattr(chain, "REF", "")  # not set: as before this code. The chain record is not read
    read = len(store.read); ups = len(lium.ups)
    world.steps(D.MAX_FAIL + 2)
    assert chain.blocked(m) == "model.bin" and len(store.read) >= read and layout.public_base_file(king.base_hash, "model.bin") in store.gone
    assert not [k for k in store.read[read:] if k == chain.HEAD_CHAIN] and len(lium.ups) == ups and TD.model(mid)["waiting_for"] == "weights"
    for change in (dict(schema="other"), dict(steps=[{"version_id": king.version_id}])):  # a record that is wrong holds the row, and stops nothing else
        monkeypatch.setattr(chain, "REF", REF)
        store.inner.put(chain.HEAD_CHAIN, json.dumps(json.loads(store.inner.get(chain.HEAD_CHAIN)) | change).encode())
        assert "no rebuild: the head chain" in chain.blocked(m) and chain.order(m) is None
    world.steps()
    assert TD.states()[mid] == "queued" and D.row_state(mid) == {}


# ------------------------------------------------------------------------------------------------ the node's script, for real
TOOL = '''
import argparse, hashlib, json, os, pathlib, shutil, sys, time, urllib.request
p = argparse.ArgumentParser(); b = p.add_subparsers(dest="c").add_parser("rebuild")
for n in ("--chain", "--work", "--links", "--head", "--device"): b.add_argument(n)
a = p.parse_args(); doc, links, work = json.load(open(a.chain)), json.load(open(a.links)), pathlib.Path(a.work); cache = work / "cache"
(work / "seen.json").write_text(json.dumps(dict(mode=oct(os.stat(a.links).st_mode & 0o777), device=a.device, boto3="boto3" in sys.modules or bool(__import__("boto3")))))
point = links["checkpoints"][0]
url = {(s["id"], f["name"]): f["url"] for s in [point["base"], point["master"], *links["chain_loras"]] for f in s["files"]}
size = {(s["id"], f["name"]): f["size"] for s in [point["base"], point["master"], *links["chain_loras"]] for f in s["files"]}
say = lambda text: print(text, file=sys.stderr, flush=True)  # the lines of the tool, in its words
def get(i, n): say(f"fetch {n} ({size[(i, n)]} bytes)"); return urllib.request.urlopen(url[(i, n)]).read()
heads = [doc["start"], *doc["steps"]]
at = max((i for i, h in enumerate(heads) if (cache / h["base_hash"] / "model.bin").exists()), default=None)
if at is None:
    say(f"start: checkpoint {heads[0]['version_id'][:12]}, then {len(heads) - 1} step(s)")
    at = 0; d = cache / heads[0]["base_hash"]; d.mkdir(parents=True); get(point["master"]["id"], "master.bin"); (d / "model.bin").write_bytes(get(heads[0]["base_hash"], "model.bin"))
started = heads[at]["version_id"]
for i in range(at + 1, len(heads)):
    if os.environ.get("TOOL_DIES_AT") == str(i): sys.exit(1)
    if os.environ.get("TOOL_REFUSES_AT") == str(i): print("refused: VerifyError: the step gave files with other hashes than the chain record", file=sys.stderr); sys.exit(2)
    before = (cache / heads[i - 1]["base_hash"] / "model.bin").read_bytes()
    lora = b"".join(get(l["lora_hash"], n) for l in heads[i]["loras"] for n in sorted(l["files"]))
    t = time.monotonic(); time.sleep(0.15)
    for n in (1, 2) if os.environ.get("TOOL_FOLDS_TWICE") else (1,):
        say(f"timing {time.strftime('%H:%M:%S', time.gmtime())} UTC: step to head {heads[i]['version_id'][:12]}: fold {n} of 2, fixed-order-v1 on {a.device}: {time.monotonic() - t:.1f} s")
    d = cache / heads[i]["base_hash"]; d.mkdir(parents=True); (d / "model.bin").write_bytes(hashlib.sha256(before + lora).hexdigest().encode())
    shutil.rmtree(cache / heads[i - 1]["base_hash"])
print("a line that is not the result")
print(json.dumps(dict(version_id=heads[-1]["version_id"], base_hash=heads[-1]["base_hash"], base_dir=str(cache / heads[-1]["base_hash"]), started_from=started, steps=len(heads) - 1 - at)))
'''


def sh(*a, cwd): return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
def hx(text): return hashlib.sha256(text.encode()).hexdigest()


class Pod:
    """The disk of a node: a repository with the chain code (the stand-in tool), the main checkout, and a store of files."""
    def __init__(self, tmp, steps=3):
        self.tmp = tmp; origin = tmp / "origin"; (origin / "v2" / "sn120_v2").mkdir(parents=True)
        (origin / "v2" / "sn120_v2" / "__init__.py").write_text(""); (origin / "v2" / "sn120_v2" / "headfold.py").write_text(TOOL)
        (origin / "v2" / "bench" / "deploy").mkdir(parents=True); (origin / "v2" / "bench" / "deploy" / "requirements-harness.txt").write_text("boto3==1.43.123\n")
        git = ("git", "-c", "user.name=t", "-c", "user.email=t@example.org")
        sh("git", "init", "-q", cwd=origin); sh("git", "add", ".", cwd=origin); sh(*git, "commit", "-q", "-m", "chain code", cwd=origin)
        self.ref = sh("git", "rev-parse", "HEAD", cwd=origin)
        (origin / "v2" / "note.txt").write_text("a later commit"); sh("git", "add", ".", cwd=origin); sh(*git, "commit", "-q", "-m", "later", cwd=origin)
        self.later, self.origin = sh("git", "rev-parse", "HEAD", cwd=origin), origin
        sh("git", "clone", "-q", str(origin), str(tmp / "main"), cwd=tmp)  # the main checkout is at the later commit
        store = tmp / "store"; store.mkdir()
        def put(name, data): (store / name).write_bytes(data); return dict(name=name.split("-", 1)[1], sha256=hashlib.sha256(data).hexdigest(), size=len(data), url=(store / name).as_uri())
        heads = [dict(version_id=hx("v0"), base_hash=hx("b0"), base_fp32_master_digest=hx("m0"), checkpoint=True)]
        loras, content = [], b"the base of the checkpoint"
        base = dict(id=hx("b0"), files=[put("b0-model.bin", content)]); master = dict(id=hx("m0"), files=[put("m0-master.bin", b"the master")])
        for i in range(1, steps + 1):
            row = put(f"l{i}-adapter_model.bin", b"lora %d" % i); content = hashlib.sha256(content + b"lora %d" % i).hexdigest().encode()
            loras.append(dict(id=hx(f"l{i}"), files=[row]))
            heads.append(dict(version_id=hx(f"v{i}"), base_hash=hx(f"b{i}"), base_fp32_master_digest=hx(f"m{i}"), checkpoint=False, kernel="fixed-order-v1",
                              from_version_id=heads[-1]["version_id"], loras=[dict(lora_hash=hx(f"l{i}"), rank=16, files={"adapter_model.bin": row["sha256"]})]))
        self.king = heads[-1]
        self.blob = dict(descriptor=dict(id=self.king["base_hash"], files={"model.bin": hashlib.sha256(content).hexdigest()}), head=self.king["version_id"],
                         chain=dict(schema=chain.SCHEMA, head_version_id=self.king["version_id"], stacked=None, start=heads[0], steps=heads[1:]),
                         links=dict(expires_at=2 ** 40, checkpoints=[dict(version_id=heads[0]["version_id"], base=base, master=master)], chain_loras=loras))
        self.dest, self.work = tmp / "kings", tmp / "kings" / ".chain"; self.dest.mkdir()

    def run(self, ref=None, blob=None, env=None, device="cpu"):
        p = subprocess.run([sys.executable, str(SCRIPT), str(self.dest), str(self.work), "--python", sys.executable, "--repo-url", str(self.origin), "--ref", ref or self.ref,
                            "--clone", str(self.tmp / "chain-repo"), "--main-clone", str(self.tmp / "main"), "--deps", str(self.tmp / "deps"), "--device", device],
                           input=json.dumps(blob or self.blob), capture_output=True, text=True, env=dict(os.environ, **(env or {})))
        return p.returncode, json.loads(p.stdout.strip().splitlines()[-1])

    @property
    def model(self): return self.dest / self.king["base_hash"]
    def left(self): return sorted(p.name for p in self.work.iterdir()) if self.work.is_dir() else []


def test_the_script_of_the_node_marks_the_base_only_when_the_checkout_is_the_pinned_commit_and_each_sha256_is_right(tmp_path):
    pod = Pod(tmp_path)
    rc, out = pod.run()
    assert (rc, out["status"], out["steps"], out["started_from"], out["chain_ref"]) == (0, "rebuilt", 3, pod.blob["chain"]["start"]["version_id"], pod.ref)
    mark = json.loads((pod.model / ".verified").read_text())
    assert mark["files"] == pod.blob["descriptor"]["files"] and (mark["by"], mark["chain_ref"], mark["steps"]) == ("chain-rebuild", pod.ref, 3)
    assert hashlib.sha256((pod.model / "model.bin").read_bytes()).hexdigest() == mark["files"]["model.bin"] and sorted(p.name for p in pod.model.iterdir()) == [".verified", "model.bin"]
    seen = json.loads((pod.work / "seen.json").read_text())
    assert seen == dict(mode="0o600", device="cpu", boto3=True)  # the links file was private while the tool ran ...
    assert not (pod.work / "links.private.json").exists() and not [n for n in pod.left() if n in ("cache", "fetch", "fold")]  # ... and it is gone, with the tool's files
    assert sh("git", "rev-parse", "HEAD", cwd=tmp_path / "chain-repo") == pod.ref != sh("git", "rev-parse", "HEAD", cwd=tmp_path / "main")  # a second checkout, at the pin
    assert pod.run()[1] == dict(ckpt=pod.king["base_hash"], status="present")  # a second call does nothing


@pytest.mark.parametrize("twice", [False, True])
def test_the_script_says_the_seconds_of_each_part_and_a_tool_with_other_lines_changes_no_result(tmp_path, twice):
    pod = Pod(tmp_path)
    rc, out = pod.run(env=dict(TOOL_FOLDS_TWICE="1") if twice else {})
    t = out["times"]
    assert rc == 0 and list(out)[0] == "times" and out["status"] == "rebuilt"  # the end of the line, which the row's journal keeps, has the status
    assert t["master_bytes"] == len(b"the master") and t["cores"] == os.cpu_count() and t["disk_free_bytes"] > 0 and t["disk_used_bytes"] >= 0
    assert len(t["steps"]) == 3 and all(sorted(s) == ["after_s", "fold_s", "lora_s"] and 0.1 <= s["fold_s"] < 5 for s in t["steps"])  # one fold time for a step, also with two folds side by side
    parts = t["master_s"] + t["cast_s"] + sum(sum(s.values()) for s in t["steps"])
    assert all(t[k] >= 0 for k in ("checkout_s", "install_s", "tool_s", "check_s", "master_s", "cast_s")) and parts <= t["tool_s"] + 0.5
    assert "timing " in (pod.work / "rebuild.log").read_text() and "fetch master.bin" in (pod.work / "rebuild.log").read_text()  # the tool's lines are still in the log
    quiet = Pod(tmp_path / "quiet"); tool = quiet.origin / "v2" / "sn120_v2" / "headfold.py"
    tool.write_text(TOOL.replace("say = lambda text: print(text, file=sys.stderr, flush=True)", "say = lambda text: print('another form: ' + text, file=sys.stderr, flush=True)"))
    sh("git", "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-am", "other lines", cwd=quiet.origin)
    rc, out = quiet.run(ref=sh("git", "rev-parse", "HEAD", cwd=quiet.origin))
    assert (rc, out["status"], out["times"]["steps"], out["times"]["master_bytes"]) == (0, "rebuilt", [], 0) and (quiet.model / ".verified").exists()
    rc, out = Pod(tmp_path / "cut").run(env=dict(TOOL_REFUSES_AT="2"))  # a rebuild that is refused says its times too
    assert (rc, out["status"]) == (2, "refused") and len(out["times"]["steps"]) == 1 and out["times"]["tool_s"] >= 0.1


@pytest.mark.parametrize("fault", ["no such commit", "changed file", "other commit"])
def test_the_tool_never_runs_from_a_checkout_that_is_not_exactly_the_pinned_commit(tmp_path, fault):
    pod = Pod(tmp_path); ref = pod.ref; second = tmp_path / "chain-repo"
    if fault == "no such commit":
        ref = "0" * 40
    else:
        assert pod.run(env=dict(TOOL_DIES_AT="1"))[0] == 1 and sh("git", "rev-parse", "HEAD", cwd=second) == ref  # the second checkout is there now, at the pin
        if fault == "changed file":  # a tool that would say yes to everything
            (second / "v2" / "sn120_v2" / "headfold.py").write_text("import sys; sys.exit(0)")
        else:
            sh("git", "checkout", "-q", "--detach", pod.later, cwd=second)
    rc, out = pod.run(ref=ref)
    if fault == "no such commit":
        assert (rc, out["status"]) == (2, "refused") and "is not at commit " + "0" * 40 in out["why"] and "the tool was not run" in out["why"]
        assert not pod.model.exists() and not (pod.work / "seen.json").exists()
    else:  # the checkout is brought to the pin, and only then the tool runs: the result names the commit that ran
        assert (rc, out["status"], out["chain_ref"]) == (0, "rebuilt", ref) and sh("git", "rev-parse", "HEAD", cwd=second) == ref
        assert sh("git", "status", "--porcelain", "--untracked-files=no", cwd=second) == "" and (second / "v2" / "sn120_v2" / "headfold.py").read_text() == TOOL


@pytest.mark.parametrize("fault", ["the tool refuses a step", "the result has another sha256", "the result has one file more", "the tool made another head"])
def test_a_wrong_file_hash_leaves_no_base_and_no_mark_on_the_node(tmp_path, fault):
    pod = Pod(tmp_path); env = {}; blob = json.loads(json.dumps(pod.blob))
    if fault == "the tool refuses a step": env = dict(TOOL_REFUSES_AT="2")
    if fault == "the result has another sha256": blob["descriptor"]["files"]["model.bin"] = hx("the base of the head manifest")
    if fault == "the result has one file more": blob["descriptor"]["files"] = dict(blob["descriptor"]["files"], **{"config.json": hx("config")})
    if fault == "the tool made another head": blob["head"] = hx("another head")
    rc, out = pod.run(blob=blob, env=env)
    assert (rc, out["status"]) == (2, "refused") and not pod.model.exists() and not list(pod.dest.glob("*/.verified"))
    assert not (pod.work / "links.private.json").exists()
    if fault == "the tool refuses a step": assert out["why"].startswith("refused: VerifyError: the step gave files with other hashes")
    if "sha256" in fault or "one file more" in fault:
        assert "sha256 mismatch" in out["why"] and not (pod.work / "cache" / pod.king["base_hash"]).exists()  # the wrong files were deleted
    assert pod.run()[0] == 0 and (pod.model / ".verified").exists()  # the right input after it gives the base


def test_a_cut_in_the_middle_of_a_rebuild_keeps_the_work_and_the_next_call_goes_on_from_the_last_whole_step(tmp_path):
    pod = Pod(tmp_path, steps=4)
    rc, out = pod.run(env=dict(TOOL_DIES_AT="3"))  # the tool dies before the third step
    assert (rc, out["status"]) == (1, "failed") and "ended with code 1" in out["why"] and not pod.model.exists()
    assert not (pod.work / "links.private.json").exists()  # the links do not stay on the disk after a cut
    held = pod.blob["chain"]["steps"][1]
    assert sorted(p.name for p in (pod.work / "cache").iterdir()) == [held["base_hash"]]  # the head after the second step is kept
    rc, out = pod.run()
    assert (rc, out["status"], out["started_from"], out["steps"]) == (0, "rebuilt", held["version_id"], 2) and (pod.model / ".verified").exists()


def test_the_host_sends_the_rebuild_only_to_a_script_that_is_its_own(tmp_path):
    """The command of the host, run by a shell on files of this machine: a script with another content is not run."""
    node = types.SimpleNamespace(cmds=[], run=lambda cmd, inp=None, timeout=0: (node.cmds.append(cmd), types.SimpleNamespace(stdout="", stderr="", returncode=0))[1])
    fetch.rebuild(dict(base_hash="b" * 64), node, dict(blob={}, limit_s=1, ref=REF, device="cpu"))
    check = re.search(r"echo '[0-9a-f]{64}  .*?exit 3; \}", node.cmds[0]).group(0) + "; echo ran"
    for content, ran in ((SCRIPT.read_bytes(), True), (SCRIPT.read_bytes() + b"\n# changed", False)):
        copy = tmp_path / "chain_rebuild.py"; copy.write_bytes(content)
        p = subprocess.run(["bash", "-c", check.replace("/data/bench/repo/v2/bench/deploy/chain_rebuild.py", str(copy))], capture_output=True, text=True)
        assert (p.returncode == 0 and p.stdout.strip() == "ran") is ran
        if not ran: assert p.returncode == 3 and json.loads(p.stdout)["status"] == "refused"
