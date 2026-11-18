"""The measured dry run of a rebuild (affine_bench/dryrun.py): the fake Lium and the store fake of test_daemon.py, the
box in `chain` mode of test_chain.py, and a node whose rebuild script says its times.

Each king is a checkpoint here, as on mainnet while the box makes a checkpoint for every king. The clock of the test
moves when the setup of a pod and a rebuild run, so that the report has minutes and dollars to say."""
import gc, glob, json, os, pathlib, signal, stat, subprocess, sys, time, types
import pytest
from affine_bench import chain, cli, config as C, daemon as D, dryrun, pod as P
import test_chain as TC
import test_daemon as TD
from test_chain import cw  # noqa: F401 - the fixture
from test_daemon import world  # noqa: F401 - the fixture

ROW, NAME = dryrun.ROW, "affine-v2-bench-dry-rebuild"
SETUP_S, REBUILD_S = 900, 600


class Killed(BaseException):
    """The process of the dry run is killed: no code of it runs after this."""


class DryNode(TC.ChainNode):
    """The node of the dry run. `fault[device]`: `refused` (a sha256 is wrong) or `no answer`. `during` runs while the
    script runs on the node."""
    def __init__(self, clock):
        super().__init__(); self.clock, self.fault, self.during = clock, {}, lambda: None

    def run(self, cmd, inp=None, timeout=0):
        if "chain_rebuild.py" not in cmd:
            return super().run(cmd, inp, timeout)
        blob = json.loads(inp); device = cmd.rsplit("--device ", 1)[1]
        self.cmds.append(cmd); self.rebuilds.append(dict(blob=blob, cmd=cmd, timeout=timeout, device=device))
        self.during(); self.clock.skip(REBUILD_S)
        if self.fault.get(device) == "no answer":
            raise subprocess.TimeoutExpired("ssh to 203.0.113.7", timeout)
        ok = self.fault.get(device) != "refused"; steps = len(blob["chain"]["steps"])
        times = dict(cores=64, disk_free_bytes=900e9, disk_used_bytes=75e9, checkout_s=6.0, install_s=24.0, tool_s=540.0, master_s=180.0, master_bytes=37_640_000_000,
                     cast_s=60.0, steps=[dict(lora_s=3.0, fold_s=240.0, after_s=30.0)] * steps, check_s=27.0)
        line = dict(times=times, ckpt=blob["descriptor"]["id"], status="rebuilt" if ok else "refused") | \
            (dict(steps=steps, started_from=blob["chain"]["start"]["version_id"]) if ok else dict(why="sha256 mismatch for model.bin, read from 203.0.113.7"))
        return types.SimpleNamespace(stdout="a line of the shell\n" + json.dumps(line), stderr="", returncode=0 if ok else 2)


class Clock:
    def __init__(self, monkeypatch):
        self.real, self.skew = time.time, 0.0; monkeypatch.setattr(time, "time", lambda: self.real() + self.skew)
    def skip(self, seconds): self.skew += seconds


@pytest.fixture
def dw(cw, monkeypatch):
    world = cw; world.clock = Clock(monkeypatch); world.setup_ok = (True, "BOOTSTRAP_OK")
    monkeypatch.setattr(TD, "Node", lambda: DryNode(world.clock))
    monkeypatch.setattr(P, "setup", lambda ssh: (world.setups.append(ssh), world.clock.skip(SETUP_S), world.setup_ok)[-1])
    monkeypatch.setattr(dryrun, "RELEASE_WAIT_S", 0)
    pathlib.Path(D.ST + "/heartbeat").unlink(missing_ok=True)  # no service runs beside the command, when a test does not say so

    def worker(mid):
        """As daemon.start_worker and one poll of daemon.row_main: no worker while another process holds the row's lock."""
        lock = D.row_lock(mid)
        if lock is None:
            return
        try:
            row = D.Row(mid); world.polled.append(mid)
            row.S["model"] = row.S.get("model") or next(m for m in D.model_rows() if m["model_id"] == mid)
            row.step(row.S["model"], TD.RUNNABLE); row.save()
        finally:
            lock.close()
    monkeypatch.setattr(D, "start_worker", worker)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    yield world
    for s, h in handlers.items(): signal.signal(s, h)


def kings(world, n=3, scored=True):
    """n kings, each one a checkpoint. `scored`: each row has its results and its pod is given back, so no row waits for a pod."""
    ks = TC.kings_of(world, n); TC.box(world, ks, checkpoints=ks)
    if scored:
        world.steps(4)
        assert set(TD.states().values()) == {"scored"} and not world.lium.pods
    return ks


def dry(capsys, *argv):
    """Run the command. Returns (exit code, the report, each JSON line of the output, the output)."""
    capsys.readouterr()
    try:
        code = dryrun.main(list(argv))
    except SystemExit as e:
        code = e
    out = capsys.readouterr().out; docs = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    return code, next((d["dry_run"] for d in docs if "dry_run" in d), None), docs, out


def record(): return json.load(open(f"{D.ROWS}/{ROW}/pod.json"))
def public(world): return json.dumps(TD.table()) + "".join(t.decode() if isinstance(t, bytes) else str(t) for t in world.store.tables)
def counts(): return {k: TD.table()["progress"][k] for k in ("models_total", "models_scored", "queued")}


def test_without_the_word_rent_the_command_says_its_plan_and_rents_nothing(dw, capsys):
    ks = kings(dw); lium = dw.lium; sent = len(lium.sent)
    code, rep, docs, out = dry(capsys)
    plan = docs[0]["dry_run_plan"]
    assert code == 0 and rep is None and "nothing was rented" in out
    assert (plan["king"], plan["start"], plan["steps"], plan["devices"], plan["rent"]) == (ks[2].version_id, ks[1].version_id, 1, ["cpu", "cuda:0"], False)
    assert (plan["pod_name"], plan["max_usd_h"], plan["chain_ref"]) == (NAME, 1.0, TC.REF)
    limit_s = P.UP_TIMEOUT_S + 2400 + 2 * (chain.BASE_S + chain.STEP_S) + dryrun.RELEASE_TRIES * dryrun.RELEASE_WAIT_S  # the longest that the pod can be there
    assert plan["bound_minutes"] == round(limit_s / 60) and plan["bound_usd"] == round(limit_s / 3600 * 1.0, 2) and limit_s < P.POD_MAX_HOURS * 3600
    assert len(lium.sent) == sent and not os.path.isdir(f"{D.ROWS}/{ROW}") and ROW not in dw.nodes  # no question to Lium, no state, no node


@pytest.mark.parametrize("argv, king, start, steps", [(["--steps", "2"], 2, 0, 2), (["--from", "genesis"], 2, None, 3), (["--head", "second"], 1, 0, 1)])
def test_the_king_and_the_older_checkpoint_can_be_named(dw, capsys, argv, king, start, steps):
    ks = kings(dw)
    names = dict(genesis=dw.genesis.version_id[:12], second=TD.ids(ks[1]))
    code, _, docs, _ = dry(capsys, *[names.get(a, a) for a in argv])
    plan = docs[0]["dry_run_plan"]
    assert code == 0 and (plan["king"], plan["start"], plan["steps"]) == (ks[king].version_id, (dw.genesis if start is None else ks[start]).version_id, steps)


def test_the_dry_run_makes_a_checkpoint_king_from_an_older_checkpoint_on_a_pod_of_its_own_and_gives_the_pod_back(dw, capsys):
    ks = kings(dw); lium, store = dw.lium, dw.store
    ups, removed, tables, before, rents = len(lium.ups), len(lium.removed), len(store.tables), counts(), len(P.rents())
    code, rep, docs, out = dry(capsys, "--rent")
    node = dw.nodes[ROW]; cpu, gpu = node.rebuilds; king, older = ks[2], ks[1]
    assert code == 0 and rep["result"] == "done" and rep["released"] is True and (rep["king"], rep["start"], rep["steps"]) == (king.version_id, older.version_id, 1)
    # one pod, rented as the pod of a row is, under the name of the dry run, and given back by its exact id
    rec = record()
    assert [name for _, name in lium.ups[ups:]] == [NAME] and len(P.rents()) == rents + 1 and len(dw.setups) == 5
    assert lium.removed[removed:] == [("rm", rec["last"]["id"], "--name-only", "-y")] and not lium.pods and (rec["why"], rec["last"]["name"]) == ("dry run", NAME)
    # each rebuild is an order from the older checkpoint, and the result is compared with the hashes of the king's own head manifest
    manifest = dw.v2.controller.state.manifest
    for call, device, tag in ((cpu, "cpu", "cpu"), (gpu, "cuda:0", "cuda0")):
        blob = call["blob"]; tail = blob["chain"]; (point,) = blob["links"]["checkpoints"]
        assert blob["descriptor"] == dict(id=king.base_hash, files=dict(manifest.base_files)) and blob["head"] == king.version_id
        assert tail["start"]["version_id"] == older.version_id == point["version_id"] and [s["version_id"] for s in tail["steps"]] == [king.version_id]
        assert point["master"]["id"] == older.base_fp32_master_digest and call["timeout"] == chain.BASE_S + chain.STEP_S
        root = f"{C.NODE_ROOT}/dry/{tag}"  # a directory of its own: never the weights of a row
        assert f" {root} {root}/.chain " in call["cmd"] and call["cmd"].endswith("--device " + device) and f"--ref {TC.REF} " in call["cmd"] and C.NODE_KINGS not in call["cmd"]
    scripts = [i for i, c in enumerate(node.cmds) if "chain_rebuild.py" in c]
    assert all(node.cmds[i - 1] == f"rm -rf {C.NODE_ROOT}/dry" for i in scripts) and node.launches == 0 and node.fetches == 0  # no rebuild finds the result of another; no benchmark
    # the minutes of each part, and the dollars at the pod's price
    runs = {r["device"]: r for r in rep["runs"]}
    assert rep["pod"]["usd_h"] == 0.8 and rep["minutes"] == dict(rent=0.0, setup=15.0, give_back=0.0, total=35.0) and rep["usd"] == round(35.0 * 0.8 / 60, 2)
    for r in runs.values():
        t = r["times"]
        assert (r["status"], r["same_files"], r["minutes"], r["steps"], r["started_from"]) == ("rebuilt", True, 10.0, 1, older.version_id)
        assert (t["master_s"], t["cast_s"], t["steps"], t["check_s"], t["cores"]) == (180.0, 60.0, [dict(lora_s=3.0, fold_s=240.0, after_s=30.0)], 27.0, 64)
    for text in ("device cpu: rebuilt, 64 CPU cores", "device cuda:0: rebuilt", "setup of the pod                         15.0 min     0.20 $\n", "37.64 GB, 209 MB/s",
                 "fold 1 (with hash check 1)                4.0 min     0.05 $  LoRA download 0.1 min, after the fold 0.5 min\n", "hash check 2 (the king's manifest)        0.5 min     0.01 $\n", "GIVEN BACK",
                 "total, from the rent to the give-back    35.0 min     0.47 $\n", "hash check 1: the tool makes the sha256 of each folded file while it writes it"):
        assert text in out, text
    # nothing public and nothing in the store; the report is private
    assert len(store.tables) == tables and "203.0.113" not in out and "signed.example" not in out
    (path,) = glob.glob(f"{D.PRIV}/{ROW}/report-*.json")
    assert json.load(open(path)) == rep and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    dw.steps(2)
    assert "dry" not in public(dw) and counts() == before and ROW not in dw.polled  # the service has nothing to do for it, and the table does not know it
    state = D.row_state(ROW)
    assert "held" not in state and state["phase"]["state"] == "dry-run-ended" and "base_hash" not in state["model"]


def test_while_the_dry_run_has_its_pod_the_service_counts_it_as_a_row_in_work_and_the_table_does_not_name_it(dw, capsys, monkeypatch):
    ks = kings(dw); lium = dw.lium; seen = {}
    monkeypatch.setattr(P, "MAX_PODS", 1)

    def during():  # a king is crowned while the dry run has the one pod that may be there
        if not seen:
            new = TD.green(dw.v2, 4); TC.box(dw, [*ks, new], checkpoints=[*ks, new]); dw.steps(2)
            seen.update(new=TD.ids(new), pods=[p["name"] for p in lium.pods], state=TD.states()[TD.ids(new)], table=public(dw), polled=ROW in dw.polled,
                        phase=D.row_state(ROW)["phase"]["state"], held=bool(D.row_state(ROW).get("held")), queued=counts()["queued"])
    dw.node(ROW).during = during
    code, rep, _, _ = dry(capsys, "--rent")
    assert code == 0 and rep["released"] is True
    assert seen["pods"] == [NAME] and seen["state"] == "queued" and seen["queued"] == 1  # the row waits: the pod of the dry run counts
    assert not seen["polled"] and "dry" not in seen["table"]  # the service starts no worker for it while the command lives, and the table says nothing of it
    assert (seen["phase"], seen["held"]) == ("dry-run", True)  # the state that makes the service give the pod back if the command dies
    dw.steps(4)
    assert TD.states()[seen["new"]] == "scored" and not lium.pods and "dry" not in public(dw)


def test_a_row_that_comes_while_lium_makes_the_pod_goes_first(dw, capsys, monkeypatch):
    ks = kings(dw); lium = dw.lium; up = P.up
    monkeypatch.setattr(P, "MAX_PODS", 1)

    def slow_up(offer, name, ssh):
        new = TD.green(dw.v2, 4); TC.box(dw, [*ks, new], checkpoints=[*ks, new]); return up(offer, name, ssh)
    monkeypatch.setattr(P, "up", slow_up)
    code, rep, _, _ = dry(capsys, "--rent")
    assert code == 2 and rep["result"] == "no-place" and rep["released"] is True and not lium.pods and not dw.nodes[ROW].rebuilds
    assert lium.removed[-1][1] == record()["last"]["id"] and rep["minutes"]["total"] == 15.0  # it had the setup only


@pytest.mark.parametrize("case", ["no pinned chain code", "the king is not a checkpoint", "the start is not a checkpoint", "a step with a kernel that no node replays",
                                  "each pod is taken by a row", "another dry run is alive", "an earlier dry run still has a pod", "an earlier rent got no answer"])
def test_a_dry_run_that_may_not_start_rents_nothing(dw, capsys, monkeypatch, case):
    ks = kings(dw, scored=case != "each pod is taken by a row"); lium = dw.lium; argv = ["--rent"]; word = case
    if case == "no pinned chain code":
        monkeypatch.setattr(chain, "REF", "v2"); word = "BENCH_CHAIN_REF"
    if case == "the king is not a checkpoint":
        TC.box(dw, ks, checkpoints=ks[:2]); argv += ["--head", ks[2].version_id[:12]]; word = "is not a checkpoint with each base file in the store"
    if case == "the start is not a checkpoint":
        TC.box(dw, ks, checkpoints=[ks[0], ks[2]]); argv += ["--head", ks[2].version_id]; word = f"head {ks[1].version_id[:12]} is not a checkpoint"
    if case == "a step with a kernel that no node replays":
        TC.box(dw, ks, checkpoints=ks, kernel="torch-matmul-v1"); word = "the node does not replay it"
    if case == "each pod is taken by a row":
        monkeypatch.setattr(P, "MAX_PODS", 4); word = "4 rows have or wait for a pod"  # the launch model and three kings wait for their pods
    if case == "another dry run is alive":
        lock = D.row_lock(ROW); assert lock; word = "another dry run is alive"
    if case == "an earlier dry run still has a pod":
        P.Pod(ROW).save(dict(id="lium-id-of-before", name=NAME)); word = "the record still names the pod of an earlier dry run"
    if case == "an earlier rent got no answer":
        os.makedirs(f"{D.ROWS}/{ROW}", exist_ok=True); D.wj(f"{D.ROWS}/{ROW}/daemon.json", dict(rent_failed=dict(at=1, why="x", tries=1, no_answer=True)))
        word = "`lium up` gave no answer in an earlier dry run"
    ups, removed = len(lium.ups), len(lium.removed)
    code, rep, _, _ = dry(capsys, *argv)
    assert code == 2 and rep["result"] == "refused" and word in rep["why"], rep
    assert len(lium.ups) == ups and len(lium.removed) == removed and not (dw.nodes.get(ROW) and dw.nodes[ROW].rebuilds)
    if case == "an earlier rent got no answer":  # the operator's word, as for a row; then the dry run starts
        D.cli("retry", [ROW])
        assert dry(capsys, "--rent")[1]["result"] == "done" and not lium.pods


@pytest.mark.parametrize("case", ["the day's count of pods is full", "no offer at or below the price cap", "lium refuses the rent", "the setup of the pod fails",
                                  "the answer of lium up is lost", "lium up gives no answer"])
def test_a_rent_under_the_guards_of_a_row_that_gives_no_pod_is_tried_once_and_leaves_no_pod(dw, capsys, monkeypatch, case):
    kings(dw); lium = dw.lium; ups, made = len(lium.ups), 0
    if case == "the day's count of pods is full": monkeypatch.setattr(P, "MAX_RENTS_PER_DAY", len(P.rents()))
    if case == "no offer at or below the price cap": lium.offers = [lium.offer("h100-dear", "H100", 1.5), lium.offer("a100", "A100", 0.5), lium.offer("h100-open", "H100", 0.5, tier="community")]
    if case == "lium refuses the rent": lium.up_fails = 1
    if case == "the setup of the pod fails": dw.setup_ok = (False, "the bootstrap failed"); made = 1
    if case == "the answer of lium up is lost": lium.lost = True; made = 1
    if case == "lium up gives no answer": monkeypatch.setattr(P, "up", lambda offer, name, ssh: (lium.ups.append((offer["huid"], name)), (None, P.NO_ANSWER))[1])
    removed = len(lium.removed)
    code, rep, docs, _ = dry(capsys, "--rent")
    assert code == 2 and rep["result"] == "no-pod" and rep["released"] is True and not lium.pods and not dw.nodes[ROW].rebuilds
    assert len(lium.ups) - ups == (0 if "day" in case or "offer" in case else 1) and len(lium.removed) - removed == made  # one try, and a pod that was made went back
    assert "held" not in D.row_state(ROW)
    if case == "lium up gives no answer":  # Lium may still make that pod: the alarm of a row, and no second `lium up` without the operator's word
        assert any(d.get("what") == "rent-no-answer" for d in docs) and dry(capsys, "--rent")[1]["why"].startswith("`lium up` gave no answer") and len(lium.ups) - ups == 1


@pytest.mark.parametrize("case", ["a sha256 of the result is wrong", "the node gives no answer"])
def test_a_rebuild_that_fails_gives_the_pod_back_and_the_report_says_which_device(dw, capsys, case):
    kings(dw); lium = dw.lium; node = dw.node(ROW)
    node.fault["cpu"] = "refused" if "sha256" in case else "no answer"
    code, rep, _, out = dry(capsys, "--rent")
    cpu, gpu = rep["runs"]
    assert code == 1 and rep["result"] == "failed" and rep["released"] is True and not lium.pods and lium.removed[-1][1] == record()["last"]["id"]
    assert "203.0.113" not in out and "ADDRESS" in out  # an error text of the node brings no address into the output
    if "sha256" in case:  # the script ended: the next device is measured
        assert (cpu["status"], cpu["same_files"], gpu["status"], gpu["same_files"]) == ("refused", False, "rebuilt", True) and "sha256 mismatch" in cpu["why"] and len(node.rebuilds) == 2
        assert cpu["times"]["steps"][0]["fold_s"] == 240.0  # the times of a rebuild that was refused are in the report too
    else:  # the script may still run on the node: no rebuild beside it
        assert (cpu["status"], gpu["status"]) == ("cut", "not-run") and "TimeoutExpired" in cpu["why"] and len(node.rebuilds) == 1


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_a_signal_gives_the_pod_back_before_the_command_ends(dw, capsys, sig):
    kings(dw); lium = dw.lium
    dw.node(ROW).during = lambda: signal.raise_signal(sig)
    code, rep, _, _ = dry(capsys, "--rent")
    assert isinstance(code, SystemExit) and rep["result"] == "cut" and rep["released"] is True and not lium.pods and lium.removed[-1][1] == record()["last"]["id"]
    assert "held" not in D.row_state(ROW) and signal.getsignal(sig) is not signal.SIG_IGN


@pytest.mark.parametrize("pod_made", [True, False])
def test_a_signal_while_lium_makes_the_pod_leaves_no_pod_without_a_word(dw, capsys, monkeypatch, pod_made):
    kings(dw); lium = dw.lium; up = P.up

    def cut_up(offer, name, ssh):
        if pod_made: up(offer, name, ssh)
        signal.raise_signal(signal.SIGHUP)
    monkeypatch.setattr(P, "up", cut_up)
    code, rep, docs, _ = dry(capsys, "--rent")
    assert isinstance(code, SystemExit) and rep["result"] == "cut" and not lium.pods and not dw.setups[4:]
    if pod_made:  # Lium lists the pod under the name of the dry run: it gets the record, and goes back by its id
        assert rep["released"] is True and lium.removed[-1][1] == record()["last"]["id"] and record()["last"]["taken_by_name"] is True
    else:  # nothing is listed now, and Lium may still make it: the output says what to look at
        assert [d["do"] for d in docs if d.get("what") == "dry-run-rent-cut"] and NAME in next(d["do"] for d in docs if d.get("what") == "dry-run-rent-cut")


def killed_dry_run(dw, monkeypatch):
    """A dry run that is killed while a rebuild runs: its pod is at Lium, and none of its code gives it back."""
    def kill(*a): raise Killed
    dw.node(ROW).during = kill
    with monkeypatch.context() as mp:
        mp.setattr(dryrun, "give_back", kill)
        try:
            dryrun.main(["--rent"])
        except Killed:
            pass
    gc.collect()  # the lock of the row dies with the process
    dw.node(ROW).during = lambda: None
    assert [p["name"] for p in dw.lium.pods] == [NAME] and D.row_state(ROW)["held"] and record()["id"] == dw.lium.pods[0]["id"]


@pytest.mark.parametrize("by", ["a step of the service", "the worker process of the row"])
def test_the_service_gives_the_pod_of_a_dry_run_that_was_killed_back_and_runs_nothing_on_it(dw, capsys, monkeypatch, by):
    kings(dw); lium = dw.lium; before = counts()
    killed_dry_run(dw, monkeypatch); pod = lium.pods[0]; node = dw.nodes[ROW]; cmds = len(node.cmds)
    if by == "a step of the service":
        dw.steps(1)
        assert ROW in dw.polled
    else:
        monkeypatch.setattr(time, "sleep", lambda s: pytest.fail("the worker waits: the pod was not given back at its first poll"))
        D.row_main(ROW)
    assert not lium.pods and lium.removed[-1] == TD.rm(pod) and record()["why"] == "held"
    assert len(node.cmds) == cmds and node.launches == 0  # nothing was sent to the node
    polled = len(dw.polled); dw.steps(2)
    assert ROW not in dw.polled[polled:] and "dry" not in public(dw) and counts() == before  # the row has no pod: no worker, and the table never named it
    assert dry(capsys, "--rent")[1]["result"] == "done" and not lium.pods  # the next dry run starts


def test_a_retry_by_mistake_starts_no_benchmark_on_the_pod_of_a_dry_run(dw, capsys, monkeypatch):
    kings(dw); lium = dw.lium
    killed_dry_run(dw, monkeypatch); node = dw.nodes[ROW]; cmds = len(node.cmds)
    D.cli("retry", [ROW])  # `held` is gone: the worker would start work on the row
    dw.steps(D.MAX_FAIL + 1)
    assert not lium.pods and D.row_state(ROW)["fails"] == D.MAX_FAIL and node.launches == 0 and node.fetches == 0 and len(node.rebuilds) == 1
    assert not [c for c in node.cmds[cmds:] if "node-run" in c or "gate_fetch" in c or "chain_rebuild" in c] and "dry" not in public(dw)


@pytest.mark.parametrize("case", ["lium refuses the delete", "the lium client breaks"])
def test_a_pod_that_cannot_be_given_back_is_said_loudly_and_the_service_goes_on_with_the_tries(dw, capsys, monkeypatch, case):
    kings(dw); lium = dw.lium; real = P.lium; broken = [True]

    def cli_(*a, **k):
        if a[:1] == ("rm",) and broken[0]:
            lium.sent.append(a)
            if "breaks" in case: raise subprocess.TimeoutExpired("lium rm", 180)
            return types.SimpleNamespace(returncode=1, stdout="", stderr="the API of Lium does not answer")
        return real(*a, **k)
    monkeypatch.setattr(P, "lium", cli_)
    removed = len(lium.removed)
    code, rep, docs, out = dry(capsys, "--rent")
    assert code == 3 and rep["result"] == "done" and rep["released"] is False and "NOT GIVEN BACK" in out
    assert [d for d in docs if d.get("event") == "alarm" and d.get("what") == "dry-run-pod-not-given-back"]
    assert len(lium.removed) - removed == dryrun.RELEASE_TRIES and [p["name"] for p in lium.pods] == [NAME] and D.row_state(ROW)["held"] and record()["id"]
    assert dry(capsys, "--rent")[1]["why"].startswith("the record still names the pod")  # no second pod beside it
    broken[0] = False; dw.steps(1)  # the service's worker of the row: the same delete, by the id of the record
    assert not lium.pods and record()["why"] == "held" and "dry" not in public(dw)


def test_the_rent_starts_right_after_a_look_of_the_service_at_liums_list(dw):
    beat, slept = pathlib.Path(D.ST + "/heartbeat"), []
    try:
        dryrun.settle(slept.append)
        assert not slept  # no service runs: nothing looks at Lium's list, and there is nothing to wait for
        beat.write_text("now"); D.wj(D.SF, dict(lium_checked=D.now() - 500))

        def looks(seconds):
            slept.append(seconds)
            if len(slept) == 3: D.wj(D.SF, dict(lium_checked=D.now()))
        dryrun.settle(looks)
        assert slept == [5, 5, 5]  # until the service has looked
        dryrun.settle(looks)
        assert slept == [5, 5, 5]  # it looked a moment ago: no wait
        D.wj(D.SF, dict(lium_checked=D.now() - 500)); slept.clear()
        dryrun.settle(lambda seconds: (slept.append(seconds), dw.clock.skip(60)))
        assert len(slept) == (D.LIUM_CHECK_S + dryrun.QUIET_S) // 60  # a service that never looks holds the dry run for one period of its look, no longer
        old = time.time() - 10 * D.POLL_S; os.utime(beat, (old, old)); slept.clear()
        dryrun.settle(slept.append)
        assert not slept  # a heartbeat that is old: the service does not run
    finally:
        beat.unlink(missing_ok=True); pathlib.Path(D.SF).unlink(missing_ok=True)


def test_the_command_is_a_command_of_the_cli_and_it_is_no_part_of_the_service(dw, capsys, monkeypatch):
    kings(dw)
    monkeypatch.setattr(sys, "argv", ["affine-bench", "rebuild-dry-run", "--devices", "cpu"])
    with pytest.raises(SystemExit) as end:
        cli.main()
    plan = next(json.loads(line)["dry_run_plan"] for line in capsys.readouterr().out.splitlines() if line.startswith('{"dry_run_plan"'))
    assert end.value.code == 0 and plan["devices"] == ["cpu"] and plan["rent"] is False and "rebuild-dry-run" in cli.__doc__
    bench = pathlib.Path(__file__).resolve().parents[1]
    for name in ("daemon.py", "pod.py", "publish.py", "fetch.py", "chain.py", "source.py"):  # nothing that the service runs knows the dry run
        assert "dryrun" not in (bench / "affine_bench" / name).read_text()
    assert "rebuild-dry-run" not in (bench / "deploy" / "sn120-v2-bench.service").read_text()
    for argv in (["--devices", "gpu"], ["--steps", "0"]):
        code, rep, _, _ = dry(capsys, *argv)
        assert code == 2 and rep["result"] == "refused" and rep["rented"] is False
