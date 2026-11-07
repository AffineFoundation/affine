#!/usr/bin/env python3
"""PE stale-GREEN apply (Jacob 2026-11-06 13:55 CR): new_head = current_head + (candidate - candidate_training_head), fp32.
- staleness = number of GREENs applied since candidate_training_head became head; apply only if <= STALENESS_MAX (2).
- GREENs are applied strictly in green-time order (decision finish), one at a time, under a host lock.
- fresh (staleness 0): new head == candidate exactly (bitwise; same as today's promotion).
- merge is written to <out>.tmp and renamed atomically; journal updates are tmp+rename under flock.
Host-side math. No chain.
"""
import fcntl, hashlib, json, os, shutil, sys, time
from contextlib import contextmanager
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
STALENESS_MAX = 2        # merge window Jacob ruled (stale <= 2 would be applied as an fp32 delta once the signed merge exists)
APPLY_STALENESS_MAX = 0  # FRESH-ONLY phase 1 (Jacob 2026-11-07 07:27 CR, Option B): only staleness 0 is applied; staleness 1-2
                         # -> merge_blocked (GREEN credited + paid via K1, head unchanged) until the trainer-signed merge is built
_ST_DEFAULT = Path("/home/const/subnet120-rewrite/state/root-audits/20261106-parallel-cutover/state")


def _st():
    return Path(os.environ.get("PE_STATE") or _ST_DEFAULT)


def cr(t=None):
    return time.strftime("%Y-%m-%d %H:%M:%S CR", time.gmtime((t or time.time()) - 21600))


def wj(p, o, mode=0o600):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(".%s.%d.tmp" % (p.name, os.getpid()))
    t.write_text(json.dumps(o, indent=1, sort_keys=True, default=str) + "\n"); os.chmod(t, mode); os.replace(t, p)


def rj(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return d


@contextmanager
def locked():
    _st().mkdir(parents=True, exist_ok=True)
    with open(_st() / "apply.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        yield


# ---------------- head chain (journal of applied GREENs) ----------------
def chain():
    return rj(_st() / "HEAD-CHAIN.json", None) or {"genesis": None, "applied": []}


def set_genesis(head):
    with locked():
        c = chain()
        if c.get("genesis") is None:
            c["genesis"] = head; c["genesis_at_cr"] = cr(); wj(_st() / "HEAD-CHAIN.json", c)
        return c


def reset_genesis(head, why):
    """Only while PE has applied nothing: the head moved through the serial gate (pre-from_epoch epoch)."""
    with locked():
        c = chain()
        if c["applied"]:
            raise RuntimeError("reset_genesis refused: PE applies exist")
        c.setdefault("genesis_history", []).append(dict(genesis=c.get("genesis"), replaced_at_cr=cr(), why=why))
        c["genesis"] = head; c["genesis_at_cr"] = cr(); wj(_st() / "HEAD-CHAIN.json", c)
        return c


def current_head_of(c):
    return c["applied"][-1]["new_head"] if c["applied"] else c.get("genesis")


def staleness(training_head, c=None):
    """GREENs applied since training_head was head. None = not in chain (unknown / older than PE) -> overflow."""
    c = c or chain()
    heads = [c.get("genesis")] + [a["new_head"] for a in c["applied"]]
    idx = None
    for i, h in enumerate(heads):
        if h == training_head:
            idx = i
    return None if idx is None else len(heads) - 1 - idx


# ---------------- pending GREEN queue (green-time order) ----------------
def add_green(epoch, candidate, training_head, green_at, files=None, extra=None):
    with locked():
        q = rj(_st() / "GREENS.json", []) or []
        if any(x["candidate"] == candidate for x in q):
            return False
        q.append(dict(epoch=epoch, candidate=candidate, training_head=training_head, green_at=float(green_at),
                      green_at_cr=cr(green_at), files=files, state="pending", **(extra or {})))
        q.sort(key=lambda x: (x["green_at"], x["epoch"]))
        wj(_st() / "GREENS.json", q)
        return True


def greens():
    return rj(_st() / "GREENS.json", []) or []


def next_green():
    for x in sorted(greens(), key=lambda x: (x["green_at"], x["epoch"])):
        if x["state"] == "pending":
            return x
    return None


def _mark(candidate, **kw):
    q = greens()
    for x in q:
        if x["candidate"] == candidate:
            x.update(kw)
    wj(_st() / "GREENS.json", q)


def mark(candidate, **kw):
    """Locked state change for a queued GREEN (wiring-v1: merge_blocked / unrestorable)."""
    with locked():
        _mark(candidate, resolved_at_cr=cr(), **kw)


def decide_apply(item, c=None):
    """-> ("fresh"|"merge"|"stale", staleness)"""
    c = c or chain()
    s = staleness(item["training_head"], c)
    if s is None or s > STALENESS_MAX:
        return "stale", s
    if s <= APPLY_STALENESS_MAX:
        return "fresh", s
    return "merge", s   # fresh-only: the caller marks it merge_blocked (never applied)


def record_applied(item, kind, s, new_head, base_head, merge_info=None):
    """Atomic head-chain append + green state; idempotent per candidate; refuses if head moved under us."""
    with locked():
        c = chain()
        if any(a["candidate"] == item["candidate"] for a in c["applied"]):
            return c
        if current_head_of(c) != base_head:
            raise RuntimeError("base head moved under apply (%s != %s)" % (str(current_head_of(c))[:12], base_head[:12]))
        older = [x for x in greens() if x["state"] == "pending" and (x["green_at"], x["epoch"]) < (item["green_at"], item["epoch"])]
        if older:
            raise RuntimeError("green-time order: %d older pending GREEN(s) first" % len(older))
        c["applied"].append(dict(candidate=item["candidate"], epoch=item["epoch"], training_head=item["training_head"],
                                 base_head=base_head, new_head=new_head, kind=kind, staleness=s,
                                 green_at=item["green_at"], applied_at=time.time(), applied_at_cr=cr(), merge=merge_info))
        wj(_st() / "HEAD-CHAIN.json", c)
        _mark(item["candidate"], state="applied", kind=kind, staleness=s, new_head=new_head, applied_at_cr=cr())
        return c


def record_stale(item, s):
    with locked():
        _mark(item["candidate"], state="stale_overflow", staleness=s, resolved_at_cr=cr())


# ---------------- fp32 delta merge on safetensors dirs ----------------
WEIGHT_SUFFIX = ".safetensors"


def _sha(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def merge_dirs(head_dir, cand_dir, train_dir, out_dir, fp32_out_dir=None):
    """out = head + (cand - train) per tensor, computed in fp32; stored in head's dtype (bf16 publish) and optionally fp32.
    Shard layout + metadata from head_dir; non-weight files from cand_dir. Written to *.tmp then renamed (atomic)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    head_dir, cand_dir, train_dir, out_dir = map(Path, (head_dir, cand_dir, train_dir, out_dir))
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ftmp = None
    if fp32_out_dir:
        ftmp = Path(fp32_out_dir).with_name(Path(fp32_out_dir).name + ".tmp")
        if ftmp.exists():
            shutil.rmtree(ftmp)
        ftmp.mkdir(parents=True)

    def tmap(d):
        m = {}
        for f in sorted(d.glob("*" + WEIGHT_SUFFIX)):
            with safe_open(str(f), "pt") as s:
                for k in s.keys():
                    m[k] = f
        return m
    Hm, Cm, Tm = tmap(head_dir), tmap(cand_dir), tmap(train_dir)
    if not (set(Hm) == set(Cm) == set(Tm)):
        raise ValueError("tensor key sets differ")
    stats = dict(tensors=0, changed=0, max_abs_delta=0.0)
    for shard in sorted(head_dir.glob("*" + WEIGHT_SUFFIX)):
        out, out32 = {}, {}
        with safe_open(str(shard), "pt") as sh:
            meta = sh.metadata()
            for k in sh.keys():
                h = sh.get_tensor(k)
                with safe_open(str(Cm[k]), "pt") as sc:
                    c = sc.get_tensor(k)
                with safe_open(str(Tm[k]), "pt") as stt:
                    t = stt.get_tensor(k)
                if not (h.shape == c.shape == t.shape):
                    raise ValueError("shape mismatch %s" % k)
                d = c.float() - t.float()
                m = h.float() + d
                stats["tensors"] += 1
                if bool((d != 0).any()):
                    stats["changed"] += 1
                    stats["max_abs_delta"] = max(stats["max_abs_delta"], float(d.abs().max()))
                out[k] = m.to(h.dtype).contiguous()
                if ftmp is not None:
                    out32[k] = m.contiguous()
        save_file(out, str(tmp / shard.name), metadata=meta)
        if ftmp is not None:
            save_file(out32, str(ftmp / shard.name), metadata=meta)
    for f in cand_dir.iterdir():
        if f.is_file() and not f.name.endswith(WEIGHT_SUFFIX) and not f.name.startswith("."):
            shutil.copy2(f, tmp / f.name)
    files = {f.name: _sha(f) for f in sorted(tmp.iterdir()) if f.is_file()}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    if ftmp is not None:
        if Path(fp32_out_dir).exists():
            shutil.rmtree(fp32_out_dir)
        os.replace(ftmp, fp32_out_dir)
    return dict(files=files, stats=stats)


def staleness_ok(training_head):   # back-compat
    s = staleness(training_head)
    return (s is not None and s <= STALENESS_MAX), s


if __name__ == "__main__":
    if sys.argv[1:2] == ["staleness"]:
        print(json.dumps(dict(training_head=sys.argv[2][:16], staleness=staleness(sys.argv[2]), max=STALENESS_MAX)))
    elif sys.argv[1:2] == ["merge"]:
        print(json.dumps(merge_dirs(*sys.argv[2:6]), indent=1))
