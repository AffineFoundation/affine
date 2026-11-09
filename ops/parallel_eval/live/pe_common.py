#!/usr/bin/env python3
"""SN120 parallel-eval (PE) shared helpers. Host-only. Pods = affine-pe-* only.
Probe is live-identical (sample-count); no EVAL_SECONDS. Jacob 2026-11-06 13:55+13:58 CR.
"""
import fcntl, json, os, re, subprocess, sys, threading, time
from pathlib import Path
sys.dont_write_bytecode = True

HOME = Path.home()
REPO = HOME / "subnet120-rewrite"
RA = REPO / "state/root-audits"
HERE = Path(os.environ.get("PE_HERE") or (RA / "20261106-parallel-cutover"))
CODE = Path(__file__).resolve().parent
ST = Path(os.environ.get("PE_ST") or (HERE / "state"))
PRIV = RA / "20261106-parallel-cutover" / "private"   # real private bundle/harness (never sandboxed copies)
JOBS = Path(os.environ.get("PE_JOBS") or (HERE / "jobs"))
LOGS = HERE / "logs"
GATE = RA / "20261026-promotion-gate"
GATE_CODE = GATE / "code"
GATE_ST = Path(os.environ.get("HG_GATE_STATE") or (GATE / "state"))
EP = GATE_ST / "epochs"
HELDOUT = GATE_ST / "heldout"
LIFE = Path(os.environ.get("PE_LIFE") or (REPO / "state/mimo-mainnet-v1"))
FLAG = Path(os.environ.get("PE_FLAG") or (RA / "20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json"))
HOST_PY = str(REPO / ".venv/bin/python")
LIUM = str(HOME / ".lium/bin/lium")
os.environ["LIUM"] = LIUM
PREFIX = "affine-pe-"
NEVER_NAMES = (
    "noble-shark-d6", "swift-wolf-bd", "calm-eagle-fe", "golden-raven-7a", "eager-wolf-98",
    "calm-orbit-ef", "gentle-lion-27", "zesty-orbit-0a", "gentle-shark-06", "gentle-wolf-f6",
    "lunar-shark-c5", "affine-bench-4", "affine-prior-1", "affine-trainer-1", "affine-probe-8xb200-1",
    "affine-pks-5090-a", "affine-pks-5090-t64", "casper-h100-ask-before-delete", "affineminer-sim-1",
)
# GPU classes PE may rent (live-replica MAX_SEQS 512 at 64K; never 5090s).
ALLOWED_GPUS = ("B300", "B200", "H200", "H100")
# throughput-v4 (2026-11-07 12:10 CR, t1365u): fastest first. B300 measured 27-29 min full-plan sampling vs 82 min on H100
# (state/THROUGHPUT-SUMMARY-1200.json); B200/H200 fallback; H100 only when nothing faster has an 8-GPU node free.
# Probe semantics are hardware-independent (same public serve/probe code, bf16, MAX_SEQS 512, n/k/head/64K/z>1/floor 0).
GPU_ORDER = ("B300", "B200", "H200", "H100")
PODD = "/root/affine-env-probe"
STALENESS_MAX = 2  # GREENs applied since training head (merge window)
APPLY_STALENESS_MAX = 0  # fresh-only phase 1 (Jacob 2026-11-07 07:27 CR Option B)
WARM_POOL_TARGET = 2     # idle pre-warmed eval nodes kept ready (flag parallel_eval.warm_pool overrides)
WARM_TTL = "12h"         # provider TTL for warm nodes
WARM_MAX_AGE_S = 9 * 3600  # recycle an unclaimed warm node before its TTL can bite a lane
MAX_INFLIGHT = 32
MAX_EVAL_NODES = 5  # Jacob 2026-11-07 07:28 CR t1365u: parallel evals up to five at a time

sys.path.insert(0, str(REPO / "ops/pods"))
os.environ.setdefault("PODS_ROLES", str(REPO / "ops/pods/roles.json"))
import replace_pod as R

_lk = threading.Lock()

def cr(t=None):
    return time.strftime("%Y-%m-%d %H:%M:%S CR", time.gmtime((t or time.time()) - 6 * 3600))

def rj(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return d

def wj(p, obj, mode=0o600):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(".%s.%d.%d.tmp" % (p.name, os.getpid(), threading.get_ident()))
    t.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.chmod(t, mode)
    os.replace(t, p)

def append(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lk, open(p, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")

def ev(**kw):
    kw.setdefault("at", time.time())
    kw["at_cr"] = cr(kw["at"])
    append(ST / "events.jsonl", kw)
    return kw

def scrub(s):
    return re.sub(r"sk_[A-Za-z0-9_\-\.]{4,}", "sk_<redacted>", s or "")

def lium(*a, timeout=180):
    try:
        p = subprocess.run([LIUM, *a], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return p.returncode, p.stdout, scrub(p.stderr)

def lium_ps():
    rc, o, e = lium("ps", "--format", "json", timeout=90)
    if rc != 0:
        raise RuntimeError("lium ps failed: " + e[-300:])
    return json.loads(o)

def our_pods(ps=None):
    return [p for p in (ps if ps is not None else lium_ps()) if (p.get("name") or "").startswith(PREFIX)]

def lium_ls(gpu):
    rc, o, e = lium("ls", "--gpu", gpu, "--format", "json", timeout=120)
    if rc != 0:
        raise RuntimeError("lium ls %s failed: %s" % (gpu, (e or o)[-300:]))
    return json.loads(o)

class Pod(R.Pod):
    def run(self, cmd, inp=None, timeout=600, check=True, retries=0):
        for a in range(retries + 1):
            try:
                return R.Pod.run(self, cmd, inp=inp, timeout=timeout, check=check)
            except Exception:
                if a == retries:
                    raise
                time.sleep(10)

def pod_of(d):
    return Pod(d["id"], d["ip"], d["port"], d.get("huid"), d.get("name"))

def pe_cfg():
    f = rj(FLAG, {}) or {}
    return dict(f.get("parallel_eval") or {})

def mode():
    """shadow | live | off"""
    c = pe_cfg()
    if c.get("enabled") is True:
        return "live"
    if c.get("shadow") is True or not c:
        return "shadow"  # default until cutover flag
    return "off"

def live_head():
    cur = rj(LIFE / "CURRENT.json") or {}
    ck = cur.get("checkpoint") or (cur.get("payload") or {}).get("checkpoint") or {}
    return ck.get("id") if isinstance(ck, dict) else None

def current_round():
    cur = rj(HELDOUT / "CURRENT.private.json") or {}
    rid = cur.get("id")
    if not rid:
        return None, None
    return rid, rj(HELDOUT / rid / "ROUND.private.json")


def gpu_order():
    """Effective rental order: flag parallel_eval.gpu_order if it is a valid subset of ALLOWED_GPUS, else GPU_ORDER.
    Any 5090 / unknown class in the flag -> ignore the flag entirely (fail safe to the code default)."""
    o = pe_cfg().get("gpu_order")
    if isinstance(o, list) and o and all(isinstance(g, str) and g in ALLOWED_GPUS for g in o) and len(set(o)) == len(o):
        return tuple(o)
    return GPU_ORDER


def warm_target():
    try:
        n = int(pe_cfg().get("warm_pool", WARM_POOL_TARGET))
    except Exception:
        n = WARM_POOL_TARGET
    return max(0, min(n, MAX_EVAL_NODES))
