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
def _st():
    """PE_STATE root — required via env (no host-path default)."""
    env = os.environ.get("PE_STATE") or os.environ.get("PE_ST")
    if not env:
        raise RuntimeError("PE_STATE is unset (no host-path default)")
    return Path(env)


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
        row = dict(epoch=epoch, candidate=candidate, training_head=training_head, green_at=float(green_at),
                   green_at_cr=cr(green_at), files=files, state="pending", **(extra or {}))
        # PIT may stamp source='pit' + merge_blocked; refuse unknown sources.
        if row.get("source") not in (None, "pe", "pit"):
            raise ValueError("green source must be pe|pit")
        # Option B / Jacob stranded: stale rows arrive merge_blocked/stranded —
        # mark resolved so next_green skips them (never HEAD).
        if row.get("merge_blocked") is True or row.get("stranded") is True:
            row["state"] = "merge_blocked"
            row["merge_blocked"] = True
            row["stranded"] = True
            row["resolved_at_cr"] = cr()
        q.append(row)
        q.sort(key=lambda x: (x["green_at"], x["epoch"]))
        wj(_st() / "GREENS.json", q)
        return True


def accept_pit_decision(item, *, shadow=False):
    """Additive PIT entry point (PR #96). Live hot-fix hash-sync is preserved.

    Shadow mode must not call this for durable HEAD-CHAIN mutation. Live PIT
    passes shadow=False after cutover. Stale/stranded GREENs set merge_blocked=True
    and are never selected by next_green / decide_apply.
    """
    if shadow:
        raise RuntimeError("pe_apply.accept_pit_decision refuses shadow writes")
    if (item or {}).get("source") != "pit":
        raise ValueError("source must be pit")
    stranded = bool(item.get("stranded") or item.get("merge_blocked"))
    extra = dict(
        source="pit",
        decision=item.get("decision"),
        merge_blocked=stranded,
        stranded=stranded,
    )
    if item.get("pending"):
        extra["pending"] = item["pending"]
    return add_green(
        item["epoch"], item["candidate"], item["training_head"], item.get("green_at") or time.time(),
        files=item.get("files"),
        extra=extra,
    )


def greens():
    return rj(_st() / "GREENS.json", []) or []


def _fp32_master_index_dir(index_dir=None):
    if index_dir is not None:
        return Path(index_dir)
    for key in ("PIT_FP32_MASTER_INDEX", "AFFINE_FP32_MASTER_INDEX"):
        v = os.environ.get(key)
        if v:
            return Path(v)
    return None


def master_index_is_complete(candidate_id, *, index_dir=None):
    """True when the published fp32 master index lists real weight shards (not digest-only).

    When PIT_FP32_MASTER_INDEX / AFFINE_FP32_MASTER_INDEX (or index_dir) is unset the
    completeness gate is inactive and this returns True (legacy PE apply path).
    When the gate is configured: missing index / digest-only / no shards ⇒ False
    (fail closed). Used by next_green, apply_tick, and record_applied.
    """
    root = _fp32_master_index_dir(index_dir)
    if root is None:
        return True
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        return False
    path = Path(root) / ("%s.json" % candidate_id)
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return False
    if not isinstance(doc, dict):
        return False
    digest = str(doc.get("master_digest") or "")
    if len(digest) != 64:
        return False
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        return False
    shards = []
    for name, meta in files.items():
        n = str(name)
        if n in ("master_digest.txt", "MASTER.json", "FP32-MASTER.json"):
            continue
        if n.endswith(".safetensors") or n.endswith(".bin"):
            if isinstance(meta, dict) and meta.get("sha256"):
                shards.append(n)
    return bool(shards)


APPLY_BLOCKED_INCOMPLETE_MASTER = "incomplete-fp32-master"
STATE_BLOCKED_INCOMPLETE_MASTER = "blocked_incomplete_master"


def _is_order_blocking_pending(row):
    """True for a GREEN that must still be applied before later GREENs.

    Terminal rows (blocked_incomplete_master, merge_blocked, unrestorable,
    applied, stale, …) and apply_blocked markers never block green-time order.
    """
    if not isinstance(row, dict):
        return False
    if row.get("state") != "pending":
        return False
    if row.get("merge_blocked") or row.get("stranded"):
        return False
    if row.get("apply_blocked"):
        return False
    return True


def older_order_blocking_pendings(item, *, rows=None):
    """Older green-time rows that would make record_applied refuse order."""
    rows = rows if rows is not None else greens()
    key = (float(item.get("green_at") or 0), int(item.get("epoch") or 0))
    out = []
    for x in rows:
        if not _is_order_blocking_pending(x):
            continue
        if (float(x.get("green_at") or 0), int(x.get("epoch") or 0)) < key:
            out.append(x)
    return out


def can_write_apply_ready(item, *, index_dir=None):
    """Preflight before writing APPLY ready: master complete + no older blockers.

    Lifecycle must never consume an apply file whose record_applied would then
    fail — call this (or rely on apply_tick) before creating APPLY ready.
    """
    if not isinstance(item, dict):
        return False, "item-invalid"
    cand = str(item.get("candidate") or "")
    if not cand:
        return False, "missing-candidate"
    if not master_index_is_complete(cand, index_dir=index_dir):
        return False, "incomplete-fp32-master"
    older = older_order_blocking_pendings(item)
    if older:
        return False, "older-pending-first:%d" % len(older)
    return True, "ok"


def next_green():
    """First pending GREEN in green-time order that is eligible to become APPLY ready.

    Skips terminal / blocked rows and incomplete fp32 masters. Incomplete masters
    are moved to state=blocked_incomplete_master so they never block later GREENs.
    """
    for x in sorted(greens(), key=lambda r: (r["green_at"], r["epoch"])):
        if x.get("state") != "pending" or x.get("merge_blocked"):
            continue
        if x.get("apply_blocked"):
            continue
        cand = str(x.get("candidate") or "")
        if not cand or not master_index_is_complete(cand):
            continue
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


def mark_apply_blocked_incomplete(candidate, *, epoch=None, alert_fn=None):
    """Move GREEN to terminal blocked_incomplete_master; alert at most once.

    ``apply_blocked_alerted`` is set only after the alert sink succeeds, so a
    failing sink can retry on a later tick. Returns True when the row was newly
    transitioned to the terminal blocked state (or alert newly delivered).
    """
    should_alert = False
    newly_blocked = False
    with locked():
        q = greens()
        row = next((x for x in q if x.get("candidate") == candidate), None)
        if row is None:
            return False
        if (
            row.get("state") == STATE_BLOCKED_INCOMPLETE_MASTER
            and row.get("apply_blocked") == APPLY_BLOCKED_INCOMPLETE_MASTER
            and row.get("apply_blocked_alerted")
        ):
            return False
        dirty = False
        if row.get("state") != STATE_BLOCKED_INCOMPLETE_MASTER or row.get("apply_blocked") != APPLY_BLOCKED_INCOMPLETE_MASTER:
            row["state"] = STATE_BLOCKED_INCOMPLETE_MASTER
            row["apply_blocked"] = APPLY_BLOCKED_INCOMPLETE_MASTER
            row["why"] = "fp32 master index incomplete (need weight shards, not digest-only)"
            row["blocked_at"] = time.time()
            row["resolved_at_cr"] = cr()
            newly_blocked = True
            dirty = True
        if not row.get("apply_blocked_alerted"):
            should_alert = True
        if dirty:
            wj(_st() / "GREENS.json", q)
    alerted_ok = False
    if should_alert and alert_fn is not None:
        try:
            alert_fn({
                "event": "green_apply_blocked_incomplete_master",
                "candidate": candidate,
                "epoch": epoch,
                "apply_blocked": APPLY_BLOCKED_INCOMPLETE_MASTER,
                "state": STATE_BLOCKED_INCOMPLETE_MASTER,
            })
            alerted_ok = True
        except Exception:
            alerted_ok = False
    elif should_alert and alert_fn is None:
        # No sink provided — still mark alerted so we do not churn GREENS forever
        # when the caller only wanted the terminal state transition.
        alerted_ok = True
    if should_alert and alerted_ok:
        with locked():
            q = greens()
            row = next((x for x in q if x.get("candidate") == candidate), None)
            if row is not None and not row.get("apply_blocked_alerted"):
                row["apply_blocked_alerted"] = True
                wj(_st() / "GREENS.json", q)
    return newly_blocked or (should_alert and alerted_ok)


def block_incomplete_pending_greens(*, alert_fn=None, index_dir=None):
    """Scan pending GREENs; mark+alert incomplete masters once. Returns newly blocked candidates."""
    newly_blocked = []
    for item in greens():
        if item.get("state") != "pending" or item.get("merge_blocked"):
            continue
        cand = str(item.get("candidate") or "")
        if not cand or master_index_is_complete(cand, index_dir=index_dir):
            continue
        if mark_apply_blocked_incomplete(cand, epoch=item.get("epoch"), alert_fn=alert_fn):
            newly_blocked.append(cand)
    return newly_blocked


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
    if item is None or not isinstance(item, dict):
        raise RuntimeError("refuse record_applied: item is None/invalid (fail closed)")
    cand = item.get("candidate")
    if not isinstance(cand, str) or not cand:
        raise RuntimeError("refuse record_applied: missing candidate (fail closed)")
    if not master_index_is_complete(cand):
        raise RuntimeError(
            "refuse record_applied: incomplete fp32 master for %s (fail closed)" % str(cand)[:12]
        )
    with locked():
        c = chain()
        if any(a["candidate"] == item["candidate"] for a in c["applied"]):
            return c
        if current_head_of(c) != base_head:
            raise RuntimeError("base head moved under apply (%s != %s)" % (str(current_head_of(c))[:12], base_head[:12]))
        older = older_order_blocking_pendings(item)
        if older:
            raise RuntimeError("green-time order: %d older pending GREEN(s) first" % len(older))
        # Defense in depth: re-check under the lock before HEAD-CHAIN append.
        if not master_index_is_complete(cand):
            raise RuntimeError(
                "refuse record_applied: incomplete fp32 master for %s (fail closed)" % str(cand)[:12]
            )
        c["applied"].append(dict(candidate=item["candidate"], epoch=item["epoch"], training_head=item["training_head"],
                                 base_head=base_head, new_head=new_head, kind=kind, staleness=s,
                                 green_at=item["green_at"], applied_at=time.time(), applied_at_cr=cr(), merge=merge_info))
        wj(_st() / "HEAD-CHAIN.json", c)
        _mark(item["candidate"], state="applied", kind=kind, staleness=s, new_head=new_head, applied_at_cr=cr(),
              apply_blocked=None, apply_blocked_alerted=None)
    # Ops hook (2026-11-09 CR, Jacob option A): sync fp32 expected-master hash after GREEN head-apply.
    # Paths are env-injected (PE_SYNC_EXPECTED_HASH*); no host defaults. Unset ⇒ no-op.
    # sync_expected_hash.py keeps fail-closed + refuses when a JOB has no RESULT.
    try:
        _sync_expected_hash_after_head_apply(new_head)
    except Exception as ex:
        try:
            alert = os.environ.get("PE_SYNC_EXPECTED_HASH_ALERT") or os.environ.get(
                "PIT_SYNC_EXPECTED_HASH_ALERT"
            )
            if alert:
                Path(alert).parent.mkdir(parents=True, exist_ok=True)
                Path(alert).write_text(
                    json.dumps({"at_cr": cr(), "new_head": new_head, "err": str(ex)[:300]}, indent=2) + "\n"
                )
        except Exception:
            pass
    return c


def _sync_expected_hash_config():
    """Resolve injectable sync script / python / log paths from env (no host defaults)."""
    sync = (
        os.environ.get("PE_SYNC_EXPECTED_HASH")
        or os.environ.get("PIT_SYNC_EXPECTED_HASH")
        or ""
    ).strip()
    if not sync:
        return None
    py = (
        os.environ.get("PE_SYNC_EXPECTED_HASH_PYTHON")
        or os.environ.get("PIT_SYNC_EXPECTED_HASH_PYTHON")
        or sys.executable
    )
    logp = (
        os.environ.get("PE_SYNC_EXPECTED_HASH_LOG")
        or os.environ.get("PIT_SYNC_EXPECTED_HASH_LOG")
        or ""
    ).strip() or None
    return dict(sync=sync, python=py, log=logp)


def _sync_expected_hash_after_head_apply(new_head):
    """Best-effort: run sync_expected_hash.py --head=<new_head>. Refuses unfinished JOB (exit 3).

    Script path must be provided via PE_SYNC_EXPECTED_HASH (or PIT_SYNC_EXPECTED_HASH).
    When unset the hook is a no-op so tests never reach a host install path.
    """
    import subprocess
    cfg = _sync_expected_hash_config()
    if cfg is None:
        return None
    sync, py, logp = cfg["sync"], cfg["python"], cfg["log"]
    p = subprocess.run(
        [py, "-B", sync, "--head=%s" % new_head],
        capture_output=True, text=True, timeout=180,
    )
    line = "%s head=%s rc=%s stdout=%s stderr=%s\n" % (
        cr(), str(new_head)[:16], p.returncode,
        (p.stdout or "").replace("\n", " ")[:500],
        (p.stderr or "").replace("\n", " ")[:300],
    )
    if logp:
        try:
            Path(logp).parent.mkdir(parents=True, exist_ok=True)
            with open(logp, "a") as f:
                f.write(line)
        except Exception:
            pass
    return p.returncode


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
