"""Host-side controller of the prior-table sampler (runs on the validator host; holds the only credentials, the pod never does).
Every POLL_S: (1) follow the gate's head (promotion-gate state/status.json 'head'): on a change, presign R2 GET URLs on the host
(gate presign.py), materialize the checkpoint on the prior pod (gate_fetch.py: sha256-checked, URL file deleted), re-serve all
replicas (pod serve.sh), and tell the table about the new head (old-head counts discounted); (2) pull new lines of the pod's
grades.jsonl and ingest them into the PRIVATE prior table; (3) write a private status file with throughput + table stats.
Read-only towards the gate/lifecycle: it never writes anything outside its own state dir and the prior pod.
usage: controller.py CONFIG.json [--once]"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from priors.table import PriorTable

CFG = json.load(open(sys.argv[1])); ONCE = "--once" in sys.argv
ST = Path(CFG["state_dir"]); ST.mkdir(parents=True, exist_ok=True); os.chmod(ST, 0o700)
POD = CFG["pod_ssh"]                        # ssh wrapper script
GATE_STATUS = Path(CFG.get("gate_status", "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/status.json"))
LIFE = Path(CFG.get("lifecycle", "/home/const/subnet120-rewrite/state/mimo-mainnet-v1"))
PRESIGN = CFG.get("presign", "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code/presign.py")
HOST_PY = CFG.get("host_python", "/home/const/subnet120-rewrite/.venv/bin/python")
POLL_S = int(CFG.get("poll_s", 60))


def cr(ts=None):
    return time.strftime("%Y-%m-%d %H:%M:%S CR (UTC-6)", time.gmtime((ts or time.time()) - 6 * 3600))


def log(**kw):
    kw["at"] = time.time(); kw["at_cr"] = cr(kw["at"]); print(json.dumps(kw, default=str), flush=True)


def pod(cmd, inp=None, timeout=300):
    p = subprocess.run([POD, cmd], input=inp, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0: raise RuntimeError("pod rc=%d %s" % (p.returncode, (p.stderr or p.stdout)[-300:]))
    return p.stdout


def wj(path, obj, mode=0o600):
    t = Path(str(path) + ".tmp"); t.write_text(json.dumps(obj, indent=1, default=str)); os.chmod(t, mode); os.replace(t, path)


def descriptor(ck):
    try:
        d = json.loads((LIFE / "CURRENT.json").read_text())
        if d["checkpoint"]["id"] == ck: return d["checkpoint"]["files"]
    except Exception: pass
    eps = sorted((p for p in (LIFE / "epochs").glob("*") if p.name.isdigit()), key=lambda p: -int(p.name))
    for p in eps[:200]:
        try: r = json.loads((p / "TRAINING-RESULT.json").read_text())
        except Exception: continue
        desc = (((r.get("payload") or r).get("report") or {}).get("publication_receipt") or {}).get("payload", {}).get("descriptor") or {}
        if desc.get("id") == ck: return desc["files"]
    return None


def gate_head():
    try:
        h = json.loads(GATE_STATUS.read_text()).get("head")
        return h if isinstance(h, str) and len(h) == 64 else None
    except Exception: return None


def serving():
    try: return json.loads(pod("cat /opt/prior/serve/SERVING.json 2>/dev/null || echo null", timeout=60))
    except Exception: return None


def switch_head(ck):
    files = descriptor(ck)
    if not files: log(event="no_descriptor", head=ck[:12]); return False
    t0 = time.time()
    blob = subprocess.run([HOST_PY, "-B", PRESIGN, ck], input=json.dumps(files), capture_output=True, text=True, timeout=120)
    if blob.returncode != 0: log(event="presign_failed", err=blob.stderr[-200:]); return False
    K = "/opt/prior/ckpt"
    out = pod(("set -e; umask 077; mkdir -p {K}/.work; cd {K}/.work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
               "json.dump(d['descriptor'],open('{K}/{c}.descriptor.json','w'));json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
               "(python3 /opt/prior/code/gate_fetch.py {K}/{c}.descriptor.json {K} {c}.urls.private.json --threads 32 >> {K}/{c}.fetch.log 2>&1; "
               "rm -f {c}.urls.private.json; tail -1 {K}/{c}.fetch.log)").format(K=K, c=ck), inp=blob.stdout, timeout=3600)
    log(event="fetched", head=ck[:12], out=out.strip()[-200:], s=round(time.time() - t0))
    out = pod("/opt/prior/code/serve.sh %s %s > /opt/prior/logs/serve-ctl.log 2>&1; tail -1 /opt/prior/logs/serve-ctl.log" % (ck, CFG.get("gmu", "0.90")),
              timeout=2400)
    log(event="served", head=ck[:12], out=out.strip()[-200:], s=round(time.time() - t0))
    s = serving()
    return bool(s and s.get("ckpt") == ck and s.get("replicas"))


def pull(table):
    off = int((json.loads((ST / "pull.json").read_text()) if (ST / "pull.json").exists() else {}).get("offset", 0))
    raw = pod("tail -c +%d /opt/prior/runs/sampler/grades.jsonl 2>/dev/null | head -c 50000000" % (off + 1), timeout=300)
    data = raw.encode(); cut = data.rfind(b"\n") + 1
    recs = [json.loads(l) for l in data[:cut].decode().splitlines() if l.strip()]
    if recs:
        with open(ST / "grades.private.jsonl", "a") as f: f.write(data[:cut].decode())
        os.chmod(ST / "grades.private.jsonl", 0o600)
    k = table.ingest(recs)
    wj(ST / "pull.json", dict(offset=off + cut, at=time.time()))
    return k


def stats(table):
    rows = table.rows(); out = {}
    for env in sorted({r["env"] for r in rows}):
        rs = [r for r in rows if r["env"] == env]; seen = [r for r in rs if r["n_head"] > 0]
        ps = [r["s_head"] / r["n_head"] for r in seen]
        bins = [0] * 5
        for p in ps: bins[min(4, int(p * 5))] += 1
        nb = sum(r["b_head"] for r in seen); nn = sum(r["n_head"] for r in seen)
        out[env] = dict(tasks=len(rs), seen_head=len(seen), rollouts_head=nn, mean_p=(sum(ps) / len(ps) if ps else None),
                        cap_rate=(nb / nn if nn else None), frac_p0=(sum(p == 0 for p in ps) / len(ps) if ps else None),
                        frac_p1=(sum(p == 1 for p in ps) / len(ps) if ps else None), p_hist_5bins=bins)
    return out


def main():
    table = PriorTable(str(ST / "prior_table.private.sqlite"), discount=float(CFG.get("discount", 0.5)))
    os.chmod(ST / "prior_table.private.sqlite", 0o600)
    if CFG.get("envs"):     # register the whole active set so unseen tasks are in the table (prior weight only, n = 0)
        from priors.tasks import load_active
        table.ensure_tasks([(t["task_id"], t["env"]) for t in load_active(CFG["envs"])])
    while True:
        try:
            s = serving(); g = gate_head()
            if s and s.get("ckpt") and table.head() is None:
                table.set_head(s["ckpt"], note="initial"); log(event="table_head", head=s["ckpt"][:12])
            if g and s is not None and g != (s or {}).get("ckpt"):
                log(event="head_change", gate_head=g[:12], served=((s or {}).get("ckpt") or "")[:12])
                k = pull(table)    # ingest everything the old head produced first
                if switch_head(g):
                    table.set_head(g, note="gate head change"); log(event="table_head", head=g[:12], discounted=True)
            k = pull(table)
            try: prog = json.loads(pod("cat /opt/prior/runs/sampler/progress.json", timeout=60))
            except Exception: prog = None
            wj(ST / "STATUS.private.json", dict(at=time.time(), at_cr=cr(), gate_head=g, table_head=table.head(), ingested_now=k,
                                               progress=prog, table=stats(table)))
        except Exception as x:
            log(event="error", error="%s: %s" % (type(x).__name__, x), tb=traceback.format_exc()[-600:])
        if ONCE: break
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
