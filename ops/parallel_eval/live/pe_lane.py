#!/usr/bin/env python3
"""One PE lane: boot -> bundle -> fetch -> serve -> live-identical probe (fixed indices, k, plan_seed).
Sample-count based (NOT time-based). Jacob 2026-11-06 13:58 CR correction.
"""
import gzip, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
import pe_rent
from pe_probe_plan import build_probe_plan  # shared with detached eval (no drift)

BUNDLE = PRIV / "probe-pe.tar.zst"
HARNESS = rj(PRIV / "public_harness.json") or {}
PRESIGN = GATE_CODE / "presign.py"
PST_CODE = RA / "20261102-per-sub-pod-test/code"

def note(jd, msg):
    jn = jd.name if hasattr(jd, "name") else str(jd)
    print("[%s] %s %s" % (cr(), jn, msg), flush=True)
    append(LOGS / "lane.jsonl", dict(job=jn, msg=msg, at=time.time(), at_cr=cr()))

def boot(P):
    links = {
        "/root/affine-env-probe": "/opt/plain/probe/affine-env-probe",
        "/root/affine-mimo-20261013": "/opt/plain/probe/affine-mimo-20261013",
        "/root/affine-mimo-public-evaluator-20261013": "/opt/plain/probe/affine-mimo-public-evaluator-20261013",
        "/workspace": "/opt/plain/workspace",
    }
    cmds = []
    for l, t in links.items():
        cmds.append(
            "mkdir -p {T}; if [ -d {L} ] && [ ! -L {L} ]; then mkdir -p {T}; cp -a {L}/. {T}/ 2>/dev/null || true; rm -rf {L}; fi; ln -sfn {T} {L}".format(L=l, T=t)
        )
    P.run("; ".join(cmds) + "; mkdir -p %s %s/ckpt %s/runs; echo boot-ok" % (PODD, PODD, PODD), timeout=300)
    P.run("command -v zstd >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd >/dev/null); zstd -V | head -1", timeout=600)

def push_bundle(P):
    bun = rj(ST / "BUNDLE.json") or {}
    part = bun.get("probe") or {}
    PST = "/root/pe-tools"
    # push bundle_fetch + gate_fetch helpers
    for fname in ("bundle_fetch.py", "gate_fetch.py"):
        src = PST_CODE / fname
        if src.exists():
            P.run("mkdir -p %s && cat > %s/%s" % (PST, PST, fname), inp=src.read_bytes(), timeout=60)
    if part.get("key") and part.get("sha256"):
        # R2 presign via presign_keys {name: key}
        from pe_common import scrub as _scrub
        helper = PST_CODE / "presign_keys.py"
        pr = subprocess.run([HOST_PY, "-B", str(helper)], input=json.dumps({"probe": part["key"]}),
                            capture_output=True, text=True, timeout=120)
        if pr.returncode != 0:
            raise RuntimeError("presign bundle failed: " + scrub(pr.stderr or pr.stdout)[-300:])
        url = json.loads(pr.stdout)["probe"]
        marker = P.run("test -f /root/.pe-bundle-probe && cat /root/.pe-bundle-probe || echo NO").strip()
        if marker != part["sha256"]:
            out = P.run("python3 %s/bundle_fetch.py probe %d %s /" % (PST, int(part["bytes"]), part["sha256"]),
                        inp=json.dumps(dict(url=url)).encode(), timeout=3600)
            res = json.loads(out.strip().splitlines()[-1])
            if not res.get("sha_ok") or res.get("tar_rc"):
                raise RuntimeError("bundle fetch: %s" % res)
            P.run("echo %s > /root/.pe-bundle-probe" % part["sha256"])
        else:
            print("bundle already present", flush=True)
    elif BUNDLE.exists():
        # fallback: stream local zstd (slow)
        data = BUNDLE.read_bytes()
        P.run("umask 022; zstd -dc | tar --no-same-owner -C / -xf - && echo bundle-ok", inp=data, timeout=1800)
    else:
        raise RuntimeError("no bundle")
    # harden + 64K
    P.run(
        "set -e; A=%s; rm -f $A/live_math_manifest_subset.json; "
        "sed -i -E 's/--max-model-len [0-9]+/--max-model-len 69632/' $A/serve.sh; "
        "grep max-model-len $A/serve.sh | head -3; test ! -f $A/live_math_manifest_subset.json; echo SCAN_OK" % PODD,
        timeout=120,
    )
    # Ensure mimo venv (EVPY) + source_root + probe_worker exist (FileNotFound caused e815/e816 fail)
    P.run(
        "set -e; "
        "test -x /root/affine-mimo-20261013/venv/bin/python || "
        "  (test -x /opt/plain/probe/affine-mimo-20261013/venv/bin/python && "
        "   ln -sfn /opt/plain/probe/affine-mimo-20261013 /root/affine-mimo-20261013); "
        "test -d /root/affine-mimo-public-evaluator-20261013/source-final-v822b || "
        "  (test -d /opt/plain/probe/affine-mimo-public-evaluator-20261013/source-final-v822b && "
        "   ln -sfn /opt/plain/probe/affine-mimo-public-evaluator-20261013 /root/affine-mimo-public-evaluator-20261013); "
        "test -f /root/affine-env-probe/probe_worker.py; "
        "test -x /root/affine-mimo-20261013/venv/bin/python; "
        "test -d /root/affine-mimo-public-evaluator-20261013/source-final-v822b; "
        "echo EVPY_OK",
        timeout=60,
    )
    ensure_harness64(P)
    ensure_live_envs(P)

# Live probe env registry + gate-pool-v2 task snapshot, byte-identical to noble-shark-d6 (read 2026-11-06 16:12 CR).
# The slim PE bundle shipped envs.json with affine_gatepool pool 19309 (v1 snapshot) -> 151/493 drawn tasks (index >= 19309)
# silently dropped (e817/e819b partial 342 tasks). Live: pool 21578, snap-affine_gatepool-v2.json, source_hash 72443a03...
LIVE_ENVS = [
    ("envs.json", PRIV / "livesync/envs.live.json", "4af963e88e62ee57ccd956863f41113879938859a77f0b2826ac29cac8f7c74d"),
    ("envs/snap-affine_gatepool-v2.json", PRIV / "livesync/snap-affine_gatepool-v2.json", None),
]

def ensure_live_envs(P):
    for rel, src, pin in LIVE_ENVS:
        data = src.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if pin and h != pin:
            raise RuntimeError("local %s sha %s != live pin %s" % (rel, h[:16], pin[:16]))
        dst = "%s/%s" % (PODD, rel)
        # PE_GZ_UPLOAD (2026-11-07 CR): 26 MB snapshot timed out (300 s) on a slow-uplink executor; ship gzip (~3 MB), verify sha after.
        P.run("set -e; mkdir -p $(dirname %s); python3 -c 'import gzip,sys;sys.stdout.buffer.write(gzip.decompress(sys.stdin.buffer.read()))' > %s.pe-tmp && mv %s.pe-tmp %s && chmod 644 %s" % (dst, dst, dst, dst, dst), inp=gzip.compress(data, 6), timeout=900)
        got = P.run("sha256sum %s | cut -d' ' -f1" % dst, timeout=60).strip()
        if got != h:
            raise RuntimeError("pod %s sha %s != %s" % (rel, got[:16], h[:16]))
    pool = P.run("python3 -c \"import json;d=json.load(open('%s/envs.json'))['affine_gatepool'];print(d['pool'], d['spec']['config']['task_snapshot'])\"" % PODD, timeout=60).strip()
    if not pool.startswith("21578 "):
        raise RuntimeError("gatepool pool after live envs: %s" % pool)
    print("live envs ok:", pool, flush=True)

def ensure_harness64(P):
    """Same normalize_mimo bound raise as live gate-64k raise_probe_harness.sh (harness ONLY).
    Do NOT patch environments.py — that breaks trusted env source_hash (live also reverted it).
    Miner-verifiable: identical regex to published raise_probe_harness switch mode on complete_group_harness.py.
    """
    script = r"""
import pathlib, re, stat
H = pathlib.Path("/opt/plain/probe/affine-mimo-public-evaluator-20261013/source-final-v822b/subnet/complete_group_harness.py")
ht = H.read_text()
ht2, n1 = re.subn(r"(1 <= value\['max_output_tokens'\] <= )\d+", r"\g<1>65536", ht, count=1)
ht2, n2 = re.subn(r"(value\['max_output_tokens'\] < value\['max_context_tokens'\] <= )\d+", r"\g<1>69632", ht2, count=1)
assert n1 == 1 and n2 == 1, (n1, n2)
mode = H.stat().st_mode
if not (mode & stat.S_IWUSR):
    H.chmod(mode | stat.S_IWUSR)
tmp = H.with_name("." + H.name + ".g64tmp")
tmp.write_text(ht2); tmp.chmod(H.stat().st_mode); tmp.replace(H)
v = H.read_text()
assert "65536" in v and "69632" in v, "post-write verify failed"
print("harness64", n1, n2)
"""
    out = P.run("python3 -", inp=script.encode(), timeout=60)
    if "harness64" not in out:
        raise RuntimeError("harness64 failed: " + out[-400:])

def ckpt_files(cid):
    for e_dir in sorted((LIFE / "epochs").glob("*"), reverse=True):
        if not e_dir.is_dir():
            continue
        for name in ("TRAINING-RESULT.json", "CANDIDATE.json", "RESULT.json"):
            p = e_dir / name
            if not p.exists():
                continue
            d = rj(p) or {}
            for path in (
                d.get("checkpoint"),
                (d.get("payload") or {}).get("checkpoint"),
                (d.get("result") or {}).get("checkpoint") if isinstance(d.get("result"), dict) else None,
            ):
                if isinstance(path, dict) and path.get("id") == cid:
                    return path.get("files") or {}
    cur = rj(LIFE / "CURRENT.json") or {}
    ck = cur.get("checkpoint") or {}
    if ck.get("id") == cid:
        return ck.get("files") or {}
    for ed in EP.glob("*"):
        tl = rj(ed / "TIMELINE.json") or {}
        if tl.get("candidate") == cid and tl.get("files"):
            return tl["files"]
    raise RuntimeError("ckpt files not found for %s" % cid[:16])

def fetch_ckpt(P, cid, files=None):
    files = files or ckpt_files(cid)
    t0 = time.time()
    b = subprocess.run([HOST_PY, "-B", str(PRESIGN), cid], input=json.dumps(files), capture_output=True, text=True, timeout=120)
    if b.returncode != 0:
        raise RuntimeError("presign failed: " + scrub(b.stderr or b.stdout)[-300:])
    payload = json.loads(b.stdout)
    blob = json.dumps(payload).encode()
    fetch_py = GATE_CODE / "pod" / "gate_fetch.py"
    if not fetch_py.exists():
        fetch_py = PST_CODE / "gate_fetch.py"
    P.run("mkdir -p %s/ckpt %s/gate" % (PODD, PODD))
    P.run("cat > %s/gate/gate_fetch.py" % PODD, inp=fetch_py.read_bytes(), timeout=30)
    K = PODD + "/ckpt"
    c = (
        "set -e; umask 077; mkdir -p {K}/.work; cd {K}/.work && "
        "python3 -c \"import json,sys;d=json.load(sys.stdin);"
        "json.dump(d['descriptor'],open('{K}/{c}.descriptor.json','w'));"
        "json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
        "(python3 {PODD}/gate/gate_fetch.py {K}/{c}.descriptor.json {K} {c}.urls.private.json --threads 32 >> {K}/{c}.fetch.log 2>&1; "
        "rc=$?; rm -f {c}.urls.private.json; tail -3 {K}/{c}.fetch.log; test -f {K}/{c}/.verified; du -sb {K}/{c} | cut -f1; exit $rc)"
    ).format(K=K, c=cid, PODD=PODD)
    out = P.run(c, inp=blob, timeout=5400).strip().splitlines()
    return dict(seconds=round(time.time() - t0, 1), bytes=int(out[-1]) if out and out[-1].isdigit() else None, ckpt=cid)

def serve(P, cid, gmu="0.9", max_num_seqs=512):
    """Exact live max_num_seqs=512; no low-memory/5090 variant."""
    if int(max_num_seqs) != 512:
        raise ValueError("PE live replica requires MAX_SEQS=512")
    out = P.run("MAX_SEQS=%d python3 %s/podctl.py serve %s %s" % (int(max_num_seqs), PODD, cid, gmu), timeout=900)
    print("serve out:", out.strip()[-300:], flush=True)
    return serve_wait(P, cid)


def serve_wait(P, cid):
    """Readiness gate shared by cold serve and warm (pre-served) nodes: all replicas up serving cid + OpenAI endpoint, stable 2 polls."""
    t0 = time.time()
    ready = 0
    while time.time() - t0 < 900:
        st_raw = P.run("python3 %s/podctl.py status" % PODD, retries=2).strip().splitlines()[-1]
        st = json.loads(st_raw)
        up = int(st.get("replicas_up") or st.get("serve_procs") or 0)
        need = int(st.get("replicas") or 8)
        # confirm OpenAI models endpoint on at least one replica port
        models = P.run(
            "curl -s -m 3 http://127.0.0.1:18555/v1/models 2>/dev/null | head -c 200 || true",
            check=False, timeout=15,
        )
        ok = ("data" in (models or "")) or ("model" in (models or "").lower())
        print("serve wait up=%d/%d models=%s" % (up, need, (models or "")[:60]), flush=True)
        if (st.get("served") and st.get("served") != cid) or (st.get("serve_ckpt") and st.get("serve_ckpt") != cid):
            raise RuntimeError("pod serves %s, expected %s" % (str(st.get("served"))[:12], cid[:12]))
        if up >= need and ok:
            ready += 1
            if ready >= 2:  # stable for 2 polls
                flags = verify_serve_flags(P, cid, need)
                return dict(serve_s=round(time.time() - t0, 1), status=st, flags=flags)
        else:
            ready = 0
        time.sleep(5)
    raise RuntimeError("serve timeout after %.0fs" % (time.time() - t0))

def verify_serve_flags(P, cid, replicas=8):
    """throughput-v4 parity guard (cold AND warm nodes): every `vllm serve` process serves ckpt/<cid> with the public live flags
    bf16, --max-model-len 69632, --max-num-seqs 512, no fp8; one per replica."""
    out = P.run("ps -eo args | grep -F 'vllm serve' | grep -v grep || true", check=False, timeout=60)
    procs = [l for l in (out or "").splitlines() if " serve " in l and "/vllm" in l]
    bad = [l[:200] for l in procs if ("/ckpt/%s" % cid) not in l or "--dtype bfloat16" not in l or "--max-model-len 69632" not in l
           or "--max-num-seqs 512" not in l or "fp8" in l]
    if len(procs) < int(replicas) or bad:
        raise RuntimeError("serve parity failed: procs=%d need=%d bad=%s" % (len(procs), int(replicas), bad[:1]))
    return dict(procs=len(procs), bf16=True, max_model_len=69632, max_num_seqs=512, fp8=False)


def run_probe(P, cid, indices, plan_seed, k=8, max_tokens=65536, env_id="affine_gatepool", run_name=None, max_num_seqs=512, beat=None):
    plan = build_probe_plan(
        cid, indices, plan_seed, k=k, max_tokens=max_tokens, env_id=env_id,
        run_name=run_name, max_num_seqs=max_num_seqs, harness=HARNESS,
    )
    name = plan["run_name"]
    P.run("test ! -f %s/live_math_manifest_subset.json" % PODD)
    P.run("python3 %s/podctl.py start-run %s" % (PODD, cid), inp=json.dumps(plan).encode(), timeout=120)
    t0 = time.time()
    last_p, last_t = None, time.time()
    while True:
        st = json.loads(P.run("python3 %s/podctl.py status" % PODD, retries=3).strip().splitlines()[-1])
        ru = (st.get("runs") or {}).get(name, {})
        if ru.get("result"):
            break
        pg = ru.get("progress") or {}
        ph = (pg.get("generated"), pg.get("samples"), pg.get("done"))
        if ph != last_p:
            last_p, last_t = ph, time.time()
            print("  progress", name, ph, flush=True)
        if beat:
            beat(progress=dict(generated=pg.get("generated"), samples=pg.get("samples"), done=pg.get("done")), progress_at=last_t)
        if time.time() - last_t > 1200:
            P.run("python3 %s/podctl.py finalize-run %s 60" % (PODD, name), check=False, timeout=180)
            time.sleep(20)
            break
        if time.time() - t0 > 5400:
            P.run("python3 %s/podctl.py finalize-run %s 60" % (PODD, name), check=False, timeout=180)
            break
        time.sleep(10)
    res = json.loads(P.run("cat %s/runs/%s/result.json" % (PODD, name), timeout=600))
    return res, dict(run_name=name, wall_s=round(time.time() - t0, 1), plan_seed=plan_seed, k=k, n=len(indices))

def _sigterm(signum, frame):
    raise SystemExit("lane got signal %d" % signum)


# ------------------------------------------------------------------ v5: second node for a 2-node shard
SHARD_WAIT_S = 480          # after the primary is serving, wait at most this long for the shard node, else run single-node


def other_lanes_waiting(job):
    """Any other live job whose lane is still trying to get a node: sharding must never take a node another candidate is waiting for."""
    me = job.get("id"); n = 0
    for sp in JOBS.glob("live-e*/STATUS.json"):
        if sp.parent.name == me:
            continue
        s = rj(sp) or {}
        if s.get("status") in ("starting", "renting", "warm-wait") and time.time() - float(s.get("beat") or 0) < 120:
            n += 1
    return n


def shard_node(job, att, C, files, box):
    """Thread: get + prepare a SECOND eval node for this attempt. Only free capacity: a generic ready warm node, else a cold rent with
    one slot of headroom (cap MAX_EVAL_NODES-1) and only while no other lane is waiting. Never raises; fills box."""
    name = PREFIX + (job["id"].replace("/", "-")[:30] + "-a%d-s1" % att)
    box.update(name=name, status="starting")
    try:
        if other_lanes_waiting(job):
            box.update(status="skipped", why="another lane is waiting for a node"); return
        d = None
        try:
            import pe_warm
            w = pe_warm.acquire(dict(job, id=job["id"] + "-s1"), generic_only=True)
            if w:
                d = w["pod"]; box.update(warm=w["name"])
        except Exception as ex:
            box.update(warm_err=str(ex)[:120])
        if not d:
            if box.get("abandon"):
                box.update(status="abandoned", released=release_shard(box, "shard-abandoned")); return
            try:
                d = pe_rent.rent(name, ttl="6h", gpu=job.get("gpu"), max_eval_nodes=MAX_EVAL_NODES - 1)
            except pe_rent.CapReached as ex:
                box.update(status="skipped", why="no free capacity: %s" % str(ex)[:80]); return
        box["d"] = d
        if not d:
            box.update(status="skipped", why="rent failed"); return
        if "5090" in str(d.get("gpu") or "").upper():
            raise ValueError("5090 eval nodes are forbidden")
        P2 = pod_of(d)
        for _ in range(30):
            try:
                P2.run("echo ssh-ok", timeout=30); break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("shard ssh never ready")
        if box.get("abandon"):
            box.update(status="abandoned", released=release_shard(box, "shard-abandoned")); return
        boot(P2); push_bundle(P2)
        if box.get("abandon"):
            box.update(status="abandoned", released=release_shard(box, "shard-abandoned")); return
        fi = fetch_ckpt(P2, C, files=files)
        if box.get("abandon"):
            box.update(status="abandoned", released=release_shard(box, "shard-abandoned")); return
        si = serve(P2, C, max_num_seqs=512)
        box.update(P=P2, status="ready", fetch_s=fi.get("seconds"), serve_s=si.get("serve_s"), ready_at=time.time())
    except BaseException as ex:
        box.update(status="failed", why=("%s: %s" % (type(ex).__name__, ex))[:200])
        box.update(released=release_shard(box, "lane-failed-shard"))


def release_shard(box, reason):
    if box.get("released") not in (None, False) and not str(box.get("released")).startswith("release err"):
        return box["released"]
    d = box.get("d")
    try:
        if d and d.get("id"):
            return pe_rent.release(d["id"], reason=reason)
        if box.get("name") and box.get("status") not in ("skipped",) and not box.get("warm"):
            return pe_rent.release(box["name"], reason=reason + "-noid")   # rent may have completed server-side without returning
    except Exception as ex:
        return "release err %s" % str(ex)[:80]


def run_lane(job):
    """One attempt. Spec in JOB.json (never rewritten here); attempt state in STATUS.json; pod ALWAYS released in finally."""
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGHUP, _sigterm)
    jd = Path(job["dir"]) if "dir" in job else (JOBS / job["id"])
    jd.mkdir(parents=True, exist_ok=True, mode=0o700)
    att = int(job.get("attempt") or 0)
    st = dict(status="starting", started=time.time(), started_cr=cr(), attempt=att, lane_pid=os.getpid(), beat=time.time())
    wj(jd / "STATUS.json", st)
    name = PREFIX + (job["id"].replace("/", "-")[:34] + "-a%d" % att)
    run_name = job.get("run_name") or ("pe-%s-a%d-%s" % (job["id"], att, job["candidate"][:12]))

    def beat(**kw):
        st.update(kw); st["beat"] = time.time(); wj(jd / "STATUS.json", st)
    d = None
    warm = None
    shard_box, shard_thread = {}, None
    try:
        # throughput-v4: claim a pre-warmed node (same public bundle/envs/harness; maybe candidate already fetched+served);
        # else rent cold (B300-first default order) under the five-eval-node cap, waiting for headroom instead of exceeding it.
        t_wait = time.time()
        while True:
            st.update(status="renting"); beat()
            try:
                import pe_warm
                warm = pe_warm.acquire(job, beat=beat)
            except Exception as ex:
                note(jd, "warm acquire error %s" % str(ex)[:200]); warm = None
            if warm:
                d = warm["pod"]
                st.update(warm=dict({k: v for k, v in warm.items() if k != "pod"})); beat()
                note(jd, "claimed warm %s fetched=%s served=%s" % (warm["name"], warm["fetched"], warm["served"]))
                break
            note(jd, "rent %s" % name)
            try:
                d = pe_rent.rent(name, ttl="6h", gpu=job.get("gpu"), max_eval_nodes=MAX_EVAL_NODES)
                break
            except pe_rent.CapReached as ex:
                if time.time() - t_wait > 3 * 3600:
                    raise RuntimeError("no eval-node headroom for 3h")
                st.update(status="warm-wait", cap=str(ex)); beat(); time.sleep(20)
        if not d:
            st.update(status="rent-failed"); beat(); return "rent-failed"
        cfg5 = pe_cfg()
        if cfg5.get("shard2") is True and mode() == "live":
            import threading
            shard_thread = threading.Thread(target=shard_node, args=(job, att, job["candidate"], job.get("files"), shard_box), daemon=True)
            shard_thread.start()
        st.update(status="booting", pod={k: v for k, v in d.items() if k != "ip"}); beat()
        P = pod_of(d)
        for i in range(30):
            try:
                P.run("echo ssh-ok", timeout=30); break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("ssh never ready")
        note(jd, "boot+bundle")
        boot(P); push_bundle(P)
        st.update(status="fetching"); beat()
        t_f = time.time()
        C = job["candidate"]
        if warm and warm.get("fetched") and P.run("test -f %s/ckpt/%s/.verified && echo V || echo N" % (PODD, C), timeout=60).strip().endswith("V"):
            finfo = dict(seconds=0.0, prefetched=True, ckpt=C)
        else:
            finfo = fetch_ckpt(P, C, files=job.get("files"))
        st["fetch_s"] = finfo.get("seconds")
        st.update(status="serving"); beat()
        gpu = (d.get("gpu") or job.get("gpu") or "").upper()
        if "5090" in gpu:
            raise ValueError("5090 eval nodes are forbidden")
        mns = int(job.get("max_num_seqs") or 512)
        if mns != 512:
            raise ValueError("PE live replica requires MAX_SEQS=512")
        if warm and warm.get("served") and finfo.get("prefetched"):
            sinfo = serve_wait(P, C); sinfo["preserved"] = True
        else:
            sinfo = serve(P, C, max_num_seqs=mns)
        st["serve_s"] = sinfo.get("serve_s"); st["max_num_seqs"] = mns
        # v5: longest-first dispatch (same indices/k/seed; only the send order) and optional 2-node shard (flags lpt_dispatch / shard2)
        import pe_shard
        K = int(job.get("k") or 8); ENV = job.get("env_id") or "affine_gatepool"; MT = int(job.get("max_tokens") or 65536)
        exp = pe_shard.expected_tokens(job) if (cfg5.get("lpt_dispatch") is True or shard_thread) else {}
        order, oinfo = pe_shard.lpt_order(job, exp) if cfg5.get("lpt_dispatch") is True else ([int(i) for i in job["indices"]], dict(lpt=False))
        st["dispatch"] = oinfo
        if shard_thread:
            t_sw = time.time()
            while shard_thread.is_alive() and shard_box.get("status") != "ready" and time.time() - t_sw < SHARD_WAIT_S:
                st.update(status="shard-wait", shard={k: v for k, v in shard_box.items() if k not in ("P", "d")}); beat(); time.sleep(5)
            if shard_box.get("status") != "ready":
                shard_box["abandon"] = True
                note(jd, "shard node not ready (%s %s): single-node" % (shard_box.get("status"), shard_box.get("why")))
            st["shard_wait_s"] = round(time.time() - t_sw, 1)
        if shard_box.get("status") == "ready" and not shard_box.get("abandon"):
            shards, sinfo2 = pe_shard.split2(order, exp)
            st.update(status="probing", shard=dict({k: v for k, v in shard_box.items() if k not in ("P", "d")}, split=sinfo2,
                                                   tasks=[len(x) for x in shards])); beat()
            note(jd, "2-node shard %s tasks est_tokens=%s" % ([len(x) for x in shards], sinfo2.get("est_tokens")))
            import threading
            outs, errs, prog = [None, None], [None, None], [{}, {}]
            def _run(i, PP):
                def b(**kw):
                    prog[i] = kw.get("progress") or {}
                    tot = {f: sum(int((x or {}).get(f) or 0) for x in prog) for f in ("generated", "samples")}
                    beat(progress=dict(tot, shards=list(prog)), progress_at=kw.get("progress_at"))
                try:
                    outs[i] = run_probe(PP, job["candidate"], shards[i], job["plan_seed"], k=K, max_tokens=MT, env_id=ENV,
                                        run_name="%s-s%d" % (run_name, i), max_num_seqs=mns, beat=b)
                except BaseException as ex:
                    errs[i] = "%s: %s" % (type(ex).__name__, str(ex)[:200])
            ths = [threading.Thread(target=_run, args=(i, PP), daemon=True) for i, PP in enumerate((P, shard_box["P"]))]
            for t in ths: t.start()
            for t in ths: t.join()
            if any(errs) or any(o is None for o in outs):
                raise RuntimeError("shard failed: %s" % errs)
            for i, o in enumerate(outs):
                wj(jd / ("RESULT-shard%d-a%d.json" % (i, att)), o[0], 0o600)
            res = pe_shard.merge([o[0] for o in outs], shards, ENV, K)
            meta = dict(run_name=run_name, wall_s=max(o[1]["wall_s"] for o in outs), plan_seed=job["plan_seed"], k=K, n=len(order),
                        shards=[o[1] for o in outs], dispatch=oinfo)
        else:
            st.update(status="probing"); beat()
            res, meta = run_probe(
                P, job["candidate"], order, job["plan_seed"],
                k=K, max_tokens=MT, env_id=ENV,
                run_name=run_name, max_num_seqs=mns, beat=beat,
            )
            meta["dispatch"] = oinfo
        planned = len(job["indices"]) * int(job.get("k") or 8)
        if res.get("partial") or int(res.get("samples") or 0) < planned:
            wj(jd / ("RESULT.incomplete-a%d.json" % att), dict(partial=res.get("partial"), samples=res.get("samples"), planned=planned), 0o600)
            raise RuntimeError("incomplete result: partial=%s samples=%s planned=%d" % (res.get("partial"), res.get("samples"), planned))
        wj(jd / "RESULT.json", res, 0o600)
        wj(jd / "META.json", meta)
        st.update(status="done", done_at=time.time(), done_at_cr=cr(), meta=meta)
        wj(jd / "STATUS.json", st)
        note(jd, "done wall=%s" % meta.get("wall_s"))
        return "done"
    except BaseException as ex:
        st.update(status="failed", error=("%s: %s" % (type(ex).__name__, ex))[:500], failed_at=time.time(), failed_at_cr=cr())
        wj(jd / "STATUS.json", st)
        note(jd, "FAILED %s" % str(ex)[:200])
        ev(event="lane_failed", job=job.get("id"), err=str(ex)[:300])
        return "failed"
    finally:
        if shard_thread is not None:
            shard_box["abandon"] = True
            shard_thread.join(timeout=300)
            st.update(shard_released=release_shard(shard_box, "lane-shard-%s" % st.get("status")))
        rel = None
        for _ in range(3):
            try:
                if d and d.get("id"):
                    rel = pe_rent.release(d["id"], reason="lane-%s" % st.get("status"))
                else:   # rent may have succeeded server-side without returning: release by exact name
                    rel = pe_rent.release(name, reason="lane-%s-noid" % st.get("status"))
                break
            except Exception as ex:
                note(jd, "release err %s" % ex); time.sleep(10)
        st.update(released=rel, released_at_cr=cr()); wj(jd / "STATUS.json", st)

if __name__ == "__main__":
    job = rj(sys.argv[1])
    job["dir"] = str(Path(sys.argv[1]).parent)
    print(run_lane(job))
