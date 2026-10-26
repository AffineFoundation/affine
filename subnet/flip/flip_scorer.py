"""SN120 flip-incentive SHADOW scorer.

Credits miners for fail→pass flips on DISPATCH-selected tasks, weighted by surprise
from probe p_hat (running EMA / bandit). Optional debit for pass→fail.

LIVE: writes scores-shadow.json mirror + scores.json. Hourly writer reads flip shares.
Trainer/evaluator/Tobias untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

VERSION = "sn120-flip-incentive-live-v1"
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
FAIL_STATUS = {"unresolved", "wrong", "error", "mixed"}
PASS_STATUS = {"correct"}


def ss58(pub_hex: str, prefix: int = 42) -> str:
    if not pub_hex or len(pub_hex) != 64:
        return pub_hex
    raw = bytes([prefix]) + bytes.fromhex(pub_hex)
    b = raw + hashlib.blake2b(b"SS58PRE" + raw, digest_size=64).digest()[:2]
    n = int.from_bytes(b, "big")
    s = ""
    while n:
        n, r = divmod(n, 58)
        s = B58[r] + s
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + s


def cr(ts=None):
    ts = time.time() if ts is None else ts
    return time.strftime("%Y-%m-%d %H:%M:%S CR (UTC-6)", time.gmtime(ts - 6 * 3600))


def _atomic(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tmp = path.with_name("." + path.name + ".tmp%d" % os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def status_of(cell: dict | None) -> str:
    if not cell or not cell.get("n"):
        return "missing"
    c = int(cell.get("c") or 0)
    w = int(cell.get("w") or 0)
    u = int(cell.get("u") or 0)
    x = int(cell.get("x") or 0)
    n = int(cell.get("n") or 0)
    if c == n and n > 0:
        return "correct"
    kinds = sum(1 for v in (c, w, u, x) if v > 0)
    if kinds > 1:
        return "mixed"
    if u > 0:
        return "unresolved"
    if w > 0:
        return "wrong"
    if x > 0:
        return "error"
    return "missing"


def raw_pass(cell: dict | None) -> float | None:
    if not cell or not cell.get("n"):
        return None
    return float(cell.get("c") or 0) / float(cell["n"])


def published_epochs(ledger: dict) -> list[dict]:
    """Return published probe epochs sorted by epoch_index."""
    out = []
    for ckpt, ent in (ledger or {}).items():
        if not isinstance(ent, dict):
            continue
        pub = ent.get("published")
        if not pub or ent.get("state") not in ("published", "published_partial") and not pub:
            # accept any entry that has published blob
            if not pub:
                continue
        if ent.get("state") == "superseded" and not pub:
            continue
        if not pub:
            continue
        ep = pub.get("epoch_index")
        if ep is None:
            ep = ent.get("epoch_index")
        if ep is None:
            continue
        out.append(
            dict(
                epoch=int(ep),
                checkpoint_id=pub.get("checkpoint_id") or ckpt,
                partial=bool(pub.get("partial") or ent.get("partial")),
                samples=pub.get("samples") or pub.get("n"),
                finished_at=pub.get("finished_at"),
                file=pub.get("file"),
                sha256=pub.get("sha256"),
            )
        )
    # de-dupe by epoch keeping latest finished_at
    best = {}
    for row in out:
        prev = best.get(row["epoch"])
        if prev is None or (row.get("finished_at") or 0) >= (prev.get("finished_at") or 0):
            best[row["epoch"]] = row
    return [best[e] for e in sorted(best)]


def load_private_per_task(results_dir: Path, checkpoint_id: str) -> dict:
    p = Path(results_dir) / f"{checkpoint_id}.private.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    am = (d.get("envs") or {}).get("affine_math") or {}
    pt = am.get("per_task") or {}
    # normalize keys to int
    out = {}
    for k, v in pt.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def load_dispatch(epochs_root: Path, epoch: int) -> list[dict]:
    p = Path(epochs_root) / str(epoch) / "coordinator" / "DISPATCH.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    sel = (d.get("payload") or d).get("selected") or []
    rows = []
    for s in sel:
        try:
            idx = int(s["index"])
        except Exception:
            continue
        miner = s.get("miner") or ""
        rewards = []
        ng = (s.get("native_grades") or {}).get("results") or []
        for r in ng:
            if isinstance(r, dict) and "reward" in r:
                try:
                    rewards.append(float(r["reward"]))
                except Exception:
                    pass
        mean_reward = sum(rewards) / len(rewards) if rewards else None
        rows.append(
            dict(
                task_index=idx,
                miner=miner,
                batch_sha256=s.get("batch_sha256"),
                mean_reward=mean_reward,
                n_rollouts=len(rewards) if rewards else None,
            )
        )
    return rows


def load_finalized_lineage(epochs_root: Path, lo: int, hi: int) -> dict[int, dict]:
    """epoch -> {input, output, auth_input, trained}. trained = FINALIZED result checkpoint was produced by
    training FROM this epoch's input checkpoint (authorization input == epoch input, output != input).
    No-update (pause) and rollback finalizations are trained=False."""
    out = {}
    for e in range(int(lo), int(hi) + 1):
        p = Path(epochs_root) / str(e) / "FINALIZED.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            pl = d.get("payload") or d
            res = pl.get("result") or {}
            inp = pl.get("input_checkpoint")
            outc = (res.get("checkpoint") or {}).get("id")
            auth = ((res.get("checkpoint_authorization") or {}).get("payload") or {})
            ain = auth.get("input_checkpoint")
            out[e] = dict(input=inp, output=outc, auth_input=ain,
                          trained=bool(inp and outc and ain == inp and outc != inp))
        except Exception:
            continue
    return out


def credit_epoch_for_pair(e_from: int, e_to: int, ckpt_from: str | None, lineage: dict) -> tuple[int | None, str]:
    """Epoch whose training started from e_from's probe checkpoint (skip no-update/rollback epochs)."""
    have = [e for e in range(e_from, e_to) if e in lineage]
    if not have:
        return e_from, "fallback_no_finalized"
    for e in range(e_from, e_to):
        f = lineage.get(e)
        if f and f["trained"] and (ckpt_from is None or f["input"] == ckpt_from):
            return e, "trained_from_probe_checkpoint" if e == e_from else "skipped_no_update_to_trained_epoch"
    return None, "no_training_from_probe_checkpoint"


CREDIT_MODES = ("surprise_asymmetric", "symmetric")


def flip_weights(p_hat: float, credit_mode: str, surprise_mode: str, eps: float,
                 symmetric_weight: float = 1.0) -> tuple[float, float]:
    """-> (gain weight for fail->pass, loss weight for pass->fail).

    symmetric (2026-10-26, Jacob pt667u): credit = change in estimated pass probability.
      A gain and a loss on the same task have the same magnitude w (default 1), so a k=1 probe
      gives +w for fail->pass and -w for pass->fail; pure noise nets 0 in expectation.
    surprise_asymmetric (legacy, 2026-10-23..26): gain -log(p_hat), loss -log(1-p_hat).
    """
    if credit_mode == "symmetric":
        w = float(symmetric_weight)
        if not (math.isfinite(w) and w > 0):
            raise ValueError("symmetric_weight must be finite and > 0")
        return w, w
    if credit_mode != "surprise_asymmetric":
        raise ValueError("unknown credit_mode %r" % credit_mode)
    return (surprise_weight(p_hat, surprise_mode, eps),
            surprise_weight(max(eps, 1.0 - p_hat), surprise_mode, eps))


def surprise_weight(p_hat: float, mode: str, eps: float) -> float:
    p = min(1.0 - eps, max(eps, float(p_hat)))
    if mode == "inv":
        return 1.0 / p
    # default neglog
    return -math.log(p)


def load_bandit_ema(db_path: Path, env: str) -> dict[int, float]:
    if not db_path or not Path(db_path).exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        rows = con.execute(
            "SELECT task_index, ema_p, n, s FROM arms WHERE env=?", (env,)
        ).fetchall()
        con.close()
    except Exception:
        return {}
    out = {}
    for idx, ema, n, s in rows:
        if ema is not None:
            out[int(idx)] = float(ema)
        elif n:
            out[int(idx)] = float(s) / float(n)
    return out



UID_BAN_PATH = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/ops/uid-ban.json")


def load_uid_bans(path: Path = UID_BAN_PATH) -> set:
    """v829c hard UID ban (offline audit TOPLOC/sampling fail). Banned miners earn nothing."""
    if not path.exists():
        return set()
    d = json.loads(path.read_text())
    out = set()
    for r in d.get("banned") or []:
        m = r.get("miner") if isinstance(r, dict) else r
        if not (isinstance(m, str) and len(m) == 64):
            raise ValueError("uid-ban miner public key")
        out.add(m)
    return out


def load_uid_ban_rows(path: Path = UID_BAN_PATH) -> dict:
    """Display-only (2026-10-25 pt616u): miner hex -> {ban_epoch, ban_reason}. Never raises."""
    try:
        d = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    except Exception:
        return {}
    out = {}
    for r in d.get("banned") or []:
        m = r.get("miner") if isinstance(r, dict) else r
        if not (isinstance(m, str) and len(m) == 64):
            continue
        r = r if isinstance(r, dict) else {}
        ep = r.get("epoch")
        ep_n = None
        if isinstance(ep, int):
            ep_n = ep
        elif isinstance(ep, str):
            digits = ep.rsplit("-e", 1)[-1] if "-e" in ep else ep
            try:
                ep_n = int(digits.lstrip("0") or "0")
            except ValueError:
                ep_n = None
        bits = [str(r.get("reason") or "banned")]
        if r.get("error_type"):
            bits.append("error_type=%s" % r["error_type"])
        if r.get("error"):
            bits.append("message=%s" % str(r["error"])[:200])
        if r.get("job_id"):
            bits.append("audit_job=%s" % r["job_id"])
        if r.get("banned_at_cr"):
            bits.append("banned_at=%s" % r["banned_at_cr"])
        out[m] = dict(ban_epoch=ep_n, ban_reason="; ".join(bits))
    return out


def apply_uid_bans(shares: dict, banned: set, fallback_miners: list) -> tuple:
    """Zero banned miners and renormalize; never return an all-zero vector."""
    if not banned:
        return shares, dict(banned=0)
    kept = {m: v for m, v in (shares or {}).items() if m not in banned and v > 0}
    tot = sum(kept.values())
    if tot > 0:
        return ({m: (kept.get(m, 0.0) / tot) for m in shares}, dict(banned=len(banned & set(shares)), fallback=False))
    pool = [m for m in fallback_miners if m not in banned]
    if pool:
        u = 1.0 / len(pool)
        return ({m: u for m in pool}, dict(banned=len(banned & set(shares)), fallback="uniform_non_banned"))
    return shares, dict(banned=len(banned & set(shares)), fallback="none_left_kept_floor")


def load_verify_ledger(path: Path) -> dict:
    """batch_sha256 / epoch:miner:index -> status (verified_valid|confirmed_invalid|pending|...)."""
    p = Path(path) if path else Path()
    if not path or not p.exists():
        return dict(by_batch={}, by_key={}, meta={})
    try:
        d = json.loads(p.read_text())
    except Exception:
        return dict(by_batch={}, by_key={}, meta={})
    return dict(
        by_batch=dict(d.get("by_batch_sha256") or {}),
        by_key=dict(d.get("by_epoch_miner_index") or {}),
        meta=dict(version=d.get("version"), stats=d.get("stats"), generated_at_cr=d.get("generated_at_cr")),
    )


def verify_status_for(ledger: dict, *, batch_sha256: str | None, epoch: int, miner: str, task_index: int) -> str:
    if batch_sha256 and batch_sha256 in ledger.get("by_batch", {}):
        return ledger["by_batch"][batch_sha256]
    key = f"{epoch}:{miner}:{task_index}"
    if key in ledger.get("by_key", {}):
        return ledger["by_key"][key]
    return "pending"


def apply_sampling_verify_gate(delta: float, kind: str, status: str, gate: dict) -> tuple[float, str, str]:
    """TOPLOC/sampling fail => no flip credit. Optional: unverified earns nothing.

    no_authenticity_guard stays True (no authenticity *bonus* in the formula).
    """
    if not gate or not gate.get("enabled", False):
        return delta, kind, "gate_off"
    fail_strips = bool(gate.get("fail_strips_credit", True))
    require_pass = bool(gate.get("credit_requires_verify_pass", True))
    unknown_policy = gate.get("unknown_policy") or "earn_nothing"
    if status == "confirmed_invalid" and fail_strips:
        return 0.0, kind, "stripped_verify_fail"
    if status == "structural_reject" and fail_strips:
        return 0.0, kind, "stripped_structural_reject"
    if require_pass and status != "verified_valid":
        if status == "pending" and unknown_policy == "pass_through":
            return delta, kind, "pending_pass_through"
        return 0.0, kind, "stripped_unverified"
    return delta, kind, "ok_verified" if status == "verified_valid" else "ok"



def _positive_share_map(shares: dict) -> dict[str, float]:
    out = {}
    for m, s in (shares or {}).items():
        try:
            fv = float(s)
        except (TypeError, ValueError):
            continue
        if isinstance(m, str) and len(m) == 64 and math.isfinite(fv) and fv > 0:
            out[m] = fv
    return out


def load_shares_from_scores_file(path: Path) -> tuple[dict[str, float], dict]:
    p = Path(path)
    if not p.exists():
        return {}, {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}, {}
    sh = {}
    for m in d.get("miners") or []:
        if not isinstance(m, dict):
            continue
        key = m.get("miner_hex")
        try:
            share = float(m.get("share_now"))
        except (TypeError, ValueError):
            continue
        if isinstance(key, str) and len(key) == 64 and math.isfinite(share) and share > 0:
            sh[key] = share
    meta = dict(
        path=str(p),
        score_sha256=d.get("score_sha256"),
        generated_at_cr=d.get("generated_at_cr"),
        version=d.get("version"),
        n=len(sh),
    )
    return sh, meta


def ensure_nonzero_shares(
    shares: dict,
    *,
    last_good_paths: list,
    dispatch_miners: list[str] | None = None,
    event_miners: list[str] | None = None,
) -> tuple[dict[str, float], dict]:
    """Hard invariant for scores.json: never emit an all-zero / empty positive share vector.

    Preference: last good scores shares > uniform over recent DISPATCH/event miners.
    Floor always wins over sampling_verify_gate / unknown_policy=earn_nothing.
    """
    pos = _positive_share_map(shares)
    if pos:
        tot = sum(pos.values())
        return {m: pos[m] / tot for m in pos}, dict(floor_applied=False, reason="live_positive", n=len(pos))

    for p in last_good_paths:
        cand, meta = load_shares_from_scores_file(Path(p))
        if cand:
            tot = sum(cand.values())
            return (
                {m: cand[m] / tot for m in cand},
                dict(
                    floor_applied=True,
                    reason="last_good_scores",
                    choice="previous_nonzero_shares",
                    source=meta,
                    n=len(cand),
                    note="Preserved last good shares so unknown_policy/gates cannot all-zero scores.",
                ),
            )

    pool: list[str] = []
    seen = set()
    for src in (dispatch_miners or []) + (event_miners or []):
        if isinstance(src, str) and len(src) == 64 and src not in seen:
            seen.add(src)
            pool.append(src)
    if pool:
        u = 1.0 / len(pool)
        return (
            {m: u for m in pool},
            dict(
                floor_applied=True,
                reason="uniform_dispatch_or_event_miners",
                choice="uniform_last_dispatch_uids",
                n=len(pool),
                note="No last-good scores; uniform over recent DISPATCH/event miners (never UID0-only).",
            ),
        )

    return {}, dict(floor_applied=False, reason="no_floor_source", n=0)


CHAIN_UID_HELPER = Path(__file__).resolve().parent / "chain_uid_map.py"


def chain_hotkey_uid_map(cache_path: Path, *, netuid: int = 120, ttl: float = 600.0,
                         max_stale: float = 6 * 3600.0, timeout: float = 120.0) -> tuple[dict[str, int], dict]:
    """hotkey(ss58) -> uid from chain metagraph (SubtensorModule.Keys, netuid 120), read-only.

    Fresh cache (< ttl) is reused; else query chain in a subprocess (bounded timeout) and refresh the
    cache. Chain failure -> cache if younger than max_stale. Returns ({}, info) if neither is available.
    """
    import subprocess
    import sys
    cache_path = Path(cache_path)
    cached = None
    try:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
    except Exception:
        cached = None
    now = time.time()
    if cached and cached.get("hotkey_to_uid") and now - float(cached.get("taken_at") or 0) < ttl:
        return dict(cached["hotkey_to_uid"]), dict(source="chain_cache", block=cached.get("block"),
                                                    age_seconds=round(now - cached["taken_at"], 1), n=len(cached["hotkey_to_uid"]))
    err = None
    try:
        r = subprocess.run([sys.executable, "-B", str(CHAIN_UID_HELPER), str(netuid)], capture_output=True,
                           text=True, timeout=timeout, check=False)
        if r.returncode == 0:
            doc = json.loads(r.stdout.strip().splitlines()[-1])
            m = {str(k): int(v) for k, v in (doc.get("hotkey_to_uid") or {}).items()}
            if m:
                _atomic(cache_path, doc)
                return m, dict(source="chain", block=doc.get("block"), age_seconds=0.0, n=len(m))
            err = "empty"
        else:
            err = (r.stderr or "").strip().splitlines()[-1][:200] if r.stderr else "rc=%d" % r.returncode
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:160])
    if cached and cached.get("hotkey_to_uid") and now - float(cached.get("taken_at") or 0) < max_stale:
        return dict(cached["hotkey_to_uid"]), dict(source="chain_cache_stale", block=cached.get("block"),
                                                    age_seconds=round(now - cached["taken_at"], 1),
                                                    n=len(cached["hotkey_to_uid"]), chain_error=err)
    return {}, dict(source="unavailable", chain_error=err, n=0)


def hotkey_uid_map(live_scores_path: Path) -> dict[str, int]:
    """LEGACY (self-referential: previous scores.json). Last-resort fallback only."""
    p = Path(live_scores_path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    out = {}
    for m in d.get("miners") or []:
        hk = m.get("hotkey")
        uid = m.get("uid")
        if hk is not None and uid is not None:
            out[hk] = int(uid)
    return out


def live_meta(live_scores_path: Path) -> dict:
    p = Path(live_scores_path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    return dict(
        version=d.get("version"),
        score_epoch=d.get("score_epoch"),
        score_sha256=d.get("score_sha256"),
        formula_sequence=(d.get("formula") or {}).get("config_sequence"),
        formula_version=(d.get("formula") or {}).get("config_version"),
        weights_module=(d.get("formula") or {}).get("weights_module"),
        generated_at_cr=d.get("generated_at_cr"),
        url="https://affine.io/scores.json",
    )


def compute(cfg: dict) -> dict:
    paths = cfg["paths"]
    form = cfg["formula"]
    gate = dict(form.get("sampling_verify_gate") or {})
    verify_ledger = load_verify_ledger(Path((cfg.get("paths") or {}).get("verify_ledger") or gate.get("ledger_path") or ""))
    env = cfg.get("env") or "affine_math"
    eps = float(form.get("eps", 1e-4))
    surprise_mode = form.get("surprise") or "neglog"
    # Missing key = legacy behaviour, so deleting the key or setting "surprise_asymmetric" reverts.
    credit_mode = form.get("credit_mode") or "surprise_asymmetric"
    symmetric_weight = float(form.get("symmetric_weight", 1.0))
    flip_weights(0.5, credit_mode, surprise_mode, eps, symmetric_weight)  # validate config early
    min_epoch = int(form.get("min_epoch", 93))
    alpha = float(form.get("ema_alpha", 0.3))
    debit = bool(form.get("debit_pass_to_fail", True))
    credit = bool(form.get("credit_fail_to_pass", True))
    eligibility = form.get("eligibility") or "hard_fail"
    reward_gate = float(form.get("reward_gate_min", 0.0))

    ledger = json.loads(Path(paths["probe_ledger"]).read_text())
    pubs = [p for p in published_epochs(ledger) if p["epoch"] >= min_epoch]
    # optional window_pairs: keep last N pairs => need N+1 epochs
    wp = form.get("window_pairs")
    if wp:
        need = int(wp) + 1
        pubs = pubs[-need:] if len(pubs) >= need else pubs

    # load per_task for each published epoch
    per_epoch_pt = {}
    gaps = []
    for p in pubs:
        pt = load_private_per_task(Path(paths["probe_results"]), p["checkpoint_id"])
        if not pt:
            gaps.append(dict(epoch=p["epoch"], reason="missing_private", checkpoint_id=p["checkpoint_id"]))
            continue
        per_epoch_pt[p["epoch"]] = pt
        p["n_per_task"] = len(pt)

    bandit_ema = load_bandit_ema(Path(paths.get("bandit_db") or ""), env)
    uid_cfg = dict(cfg.get("uid_map") or {})
    if uid_cfg.get("source", "chain") == "chain":
        hk2uid, uid_info = chain_hotkey_uid_map(
            Path(uid_cfg.get("cache_path") or (Path(paths["audit_out"]) / "chain-uid-map.json")),
            netuid=int(uid_cfg.get("netuid", 120)), ttl=float(uid_cfg.get("ttl_seconds", 600)))
        if not hk2uid:
            hk2uid = hotkey_uid_map(Path(paths["live_scores"]))
            uid_info = dict(uid_info, source="legacy_previous_scores_fallback", n=len(hk2uid))
    else:
        hk2uid = hotkey_uid_map(Path(paths["live_scores"]))
        uid_info = dict(source="legacy_previous_scores", n=len(hk2uid))
    credit_rule = form.get("credit_epoch_rule") or "training_from_probe_checkpoint"
    ckpt_of = {p["epoch"]: p.get("checkpoint_id") for p in pubs}
    lineage = (load_finalized_lineage(Path(paths["epochs_root"]), min(ckpt_of), max(ckpt_of))
               if ckpt_of and credit_rule == "training_from_probe_checkpoint" else {})

    # running EMA of pass rate per task, updated after each published epoch (chronological)
    ema: dict[int, float] = {}
    pairs = []
    epochs_sorted = [p["epoch"] for p in pubs if p["epoch"] in per_epoch_pt]
    for i in range(len(epochs_sorted) - 1):
        e_from = epochs_sorted[i]
        e_to = epochs_sorted[i + 1]
        pt_from = per_epoch_pt[e_from]
        pt_to = per_epoch_pt[e_to]
        # p_hat snapshot BEFORE updating with e_from outcomes? 
        # Probe at e_from measures model before training on e_from batches;
        # flips observed at e_to are after training. Surprise uses p_hat at e_from.
        # Use EMA state *before* ingesting e_from, then ingest e_from after scoring the pair.
        p_hat_at_E = dict(ema)  # copy before ingest

        # 2026-10-25 (pt616u): flips between probe(e_from) and probe(e_to) are caused by the epoch that
        # TRAINED from probe(e_from)'s checkpoint; no-update / rollback epochs earn nothing.
        if credit_rule == "training_from_probe_checkpoint":
            e_credit, credit_reason = credit_epoch_for_pair(e_from, e_to, ckpt_of.get(e_from), lineage)
        else:
            e_credit, credit_reason = e_from, "legacy_pair_first_epoch"
        disp = load_dispatch(Path(paths["epochs_root"]), e_credit) if e_credit is not None else []
        if not disp:
            gaps.append(dict(epoch=e_from, epoch_to=e_to, credit_epoch=e_credit,
                             reason="missing_dispatch" if e_credit is not None else credit_reason))
            # still ingest probe into EMA
            for idx, cell in pt_from.items():
                rp = raw_pass(cell)
                if rp is None:
                    continue
                if idx not in ema:
                    ema[idx] = rp
                else:
                    ema[idx] = (1 - alpha) * ema[idx] + alpha * rp
            continue

        pair_events = []
        for row in disp:
            idx = row["task_index"]
            miner = row["miner"]
            if row.get("mean_reward") is not None and row["mean_reward"] < reward_gate:
                continue
            cell_e = pt_from.get(idx)
            cell_n = pt_to.get(idx)
            if cell_e is None or cell_n is None:
                continue
            st_e = status_of(cell_e)
            st_n = status_of(cell_n)
            # p_hat: prefer running EMA before E; else bandit; else raw pass at E; else 0.5
            if idx in p_hat_at_E:
                p_hat = p_hat_at_E[idx]
            elif idx in bandit_ema:
                p_hat = bandit_ema[idx]
            else:
                rp = raw_pass(cell_e)
                p_hat = rp if rp is not None else 0.5

            # credit_mode "symmetric": w == w_debit (gain and loss same magnitude); legacy: surprisal pair.
            w, w_debit = flip_weights(p_hat, credit_mode, surprise_mode, eps, symmetric_weight)

            delta = 0.0
            kind = "none"
            eligible = False

            if eligibility == "hard_fail":
                if st_e in FAIL_STATUS:
                    eligible = True
                    if credit and st_n in PASS_STATUS:
                        delta = w
                        kind = "fail_to_pass"
                    elif st_n in FAIL_STATUS:
                        kind = "fail_to_fail"
                elif debit and st_e in PASS_STATUS and st_n in FAIL_STATUS:
                    # optional debit even if not in fail-eligibility set
                    eligible = True
                    delta = -w_debit
                    kind = "pass_to_fail"
            else:
                # soft: always weight by (1-p_hat) style via surprise on outcome change
                eligible = True
                if credit and st_e in FAIL_STATUS and st_n in PASS_STATUS:
                    delta = w
                    kind = "fail_to_pass"
                elif debit and st_e in PASS_STATUS and st_n in FAIL_STATUS:
                    delta = -w_debit
                    kind = "pass_to_fail"

            if not eligible and kind == "none":
                continue

            vstatus = verify_status_for(
                verify_ledger,
                batch_sha256=row.get("batch_sha256"),
                epoch=e_credit,
                miner=miner,
                task_index=idx,
            )
            delta, kind, gate_reason = apply_sampling_verify_gate(delta, kind, vstatus, gate)

            pair_events.append(
                dict(
                    epoch_from=e_from,
                    epoch_to=e_to,
                    credit_epoch=e_credit,
                    task_index=idx,
                    miner=miner,
                    hotkey=ss58(miner),
                    mean_reward=row.get("mean_reward"),
                    status_e=st_e,
                    status_n=st_n,
                    p_hat=p_hat,
                    weight=w if kind == "fail_to_pass" else (w_debit if kind == "pass_to_fail" else 0.0),
                    delta=delta,
                    kind=kind,
                    transition=f"{st_e}->{st_n}",
                    batch_sha256=row.get("batch_sha256"),
                    verify_status=vstatus,
                    verify_gate_reason=gate_reason,
                )
            )

        pairs.append(
            dict(
                epoch_from=e_from,
                epoch_to=e_to,
                credit_epoch=e_credit,
                credit_reason=credit_reason,
                n_dispatch=len(disp),
                n_events=len(pair_events),
                n_fail_to_pass=sum(1 for e in pair_events if e["kind"] == "fail_to_pass"),
                n_pass_to_fail=sum(1 for e in pair_events if e["kind"] == "pass_to_fail"),
                events=pair_events,
            )
        )

        # ingest e_from into EMA after scoring
        for idx, cell in pt_from.items():
            rp = raw_pass(cell)
            if rp is None:
                continue
            if idx not in ema:
                ema[idx] = rp
            else:
                ema[idx] = (1 - alpha) * ema[idx] + alpha * rp

    # aggregate per miner
    agg = defaultdict(lambda: dict(
        flip_score=0.0,
        n_fail_to_pass=0,
        n_pass_to_fail=0,
        n_eligible=0,
        n_events=0,
        by_epoch={},
        mean_reward_sum=0.0,
        mean_reward_n=0,
    ))
    for pair in pairs:
        for ev in pair["events"]:
            m = ev["miner"]
            a = agg[m]
            a["flip_score"] += ev["delta"]
            a["n_events"] += 1
            if ev["kind"] == "fail_to_pass":
                a["n_fail_to_pass"] += 1
                a["n_eligible"] += 1
            elif ev["kind"] == "pass_to_fail":
                a["n_pass_to_fail"] += 1
                a["n_eligible"] += 1
            elif ev["status_e"] in FAIL_STATUS:
                a["n_eligible"] += 1
            ek = str(ev.get("credit_epoch", ev["epoch_from"]))
            be = a["by_epoch"].setdefault(ek, dict(credit=0.0, fail_to_pass=0, pass_to_fail=0, events=0))
            be["credit"] += ev["delta"]
            be["events"] += 1
            if ev["kind"] == "fail_to_pass":
                be["fail_to_pass"] += 1
            elif ev["kind"] == "pass_to_fail":
                be["pass_to_fail"] += 1
            if ev.get("mean_reward") is not None:
                a["mean_reward_sum"] += ev["mean_reward"]
                a["mean_reward_n"] += 1

    # shares
    share_mode = form.get("share_mode") or "relu_normalize"
    scores = {m: a["flip_score"] for m, a in agg.items()}
    if share_mode == "relu_normalize":
        pos = {m: max(0.0, s) for m, s in scores.items()}
        tot = sum(pos.values())
        shares = {m: (pos[m] / tot if tot > 0 else 0.0) for m in scores}
    elif share_mode == "softmax":
        # temperature 1 on raw scores
        mx = max(scores.values()) if scores else 0.0
        exps = {m: math.exp(s - mx) for m, s in scores.items()}
        tot = sum(exps.values()) or 1.0
        shares = {m: exps[m] / tot for m in scores}
    else:
        tot = sum(abs(s) for s in scores.values()) or 1.0
        shares = {m: scores[m] / tot for m in scores}

    # Hard nonzero floor (Jacob): gates / unknown_policy must never all-zero scores.json.
    dispatch_miners = sorted({ev.get("miner") for p in pairs for ev in p.get("events") or [] if ev.get("miner")})
    event_miners = sorted(agg.keys())
    floor_paths = []
    for key in ("live_scores",):
        if paths.get(key):
            floor_paths.append(paths[key])
    # audit previous shadow copy (may equal live after cutover)
    floor_paths.append(str(Path(paths["audit_out"]) / "scores-shadow.json"))
    shares, floor_info = ensure_nonzero_shares(
        shares,
        last_good_paths=floor_paths,
        dispatch_miners=dispatch_miners,
        event_miners=event_miners,
    )
    banned = load_uid_bans(Path(gate.get("uid_ban_path") or UID_BAN_PATH))
    ban_rows = load_uid_ban_rows(Path(gate.get("uid_ban_path") or UID_BAN_PATH))
    shares, ban_info = apply_uid_bans(shares, banned, dispatch_miners + event_miners)
    floor_info["uid_ban"] = ban_info
    if floor_info.get("floor_applied") or ban_info.get("fallback"):
        # ensure agg has stub rows for floor-only miners so they appear in miners[]
        for m in shares:
            if m not in agg:
                agg[m] = dict(
                    flip_score=0.0,
                    n_fail_to_pass=0,
                    n_pass_to_fail=0,
                    n_eligible=0,
                    n_events=0,
                    mean_reward_sum=0.0,
                    mean_reward_n=0,
                    by_epoch={},
                )

    miners = []
    for m, a in agg.items():
        hk = ss58(m)
        miners.append(
            dict(
                uid=hk2uid.get(hk),
                hotkey=hk,
                miner_hex=m,
                flip_score=a["flip_score"],
                share_now=shares.get(m, 0.0),
                n_fail_to_pass=a["n_fail_to_pass"],
                n_pass_to_fail=a["n_pass_to_fail"],
                n_eligible=a["n_eligible"],
                n_events=a["n_events"],
                mean_reward=(a["mean_reward_sum"] / a["mean_reward_n"] if a["mean_reward_n"] else None),
                by_epoch=a["by_epoch"],
                floor_share=bool(floor_info.get("floor_applied") and shares.get(m, 0.0) > 0 and a["flip_score"] == 0.0 and a["n_events"] == 0),
                # display only (pt616u): banned miners keep flip_score but share_now is forced to 0
                banned=m in banned,
                ban_epoch=(ban_rows.get(m) or {}).get("ban_epoch") if m in banned else None,
                ban_reason=(ban_rows.get(m) or {}).get("ban_reason") if m in banned else None,
            )
        )
    # include only share keys if somehow agg empty
    for m, s in shares.items():
        if m not in agg:
            continue
    miners.sort(key=lambda r: (-r["share_now"], -r["flip_score"], r["uid"] if r["uid"] is not None else 1 << 30))

    # top summary without full event dumps in public file
    pair_summary = [
        dict(
            epoch_from=p["epoch_from"],
            epoch_to=p["epoch_to"],
            credit_epoch=p.get("credit_epoch"),
            credit_reason=p.get("credit_reason"),
            n_dispatch=p["n_dispatch"],
            n_events=p["n_events"],
            n_fail_to_pass=p["n_fail_to_pass"],
            n_pass_to_fail=p["n_pass_to_fail"],
        )
        for p in pairs
    ]

    live = live_meta(Path(paths["live_scores"]))
    now = time.time()
    doc = dict(
        version=VERSION,
        mode="live",
        live_weights=True,
        enabled=bool(cfg.get("enabled", True)),
        generated_at=now,
        generated_at_cr=cr(now),
        formula=dict(
            version=VERSION,
            eligibility=eligibility,
            eligibility_note=form.get("eligibility_note"),
            credit_fail_to_pass=credit,
            debit_pass_to_fail=debit,
            surprise=surprise_mode,
            credit_mode=credit_mode,
            symmetric_weight=symmetric_weight if credit_mode == "symmetric" else None,
            credit_mode_effective=form.get("credit_mode_effective"),
            eps=eps,
            p_hat_source=form.get("p_hat_source"),
            ema_alpha=alpha,
            min_epoch=min_epoch,
            window_pairs=wp,
            credit_epoch_rule=credit_rule,
            share_mode=share_mode,
            no_authenticity_guard=True,
            sampling_verify_gate=dict(
                enabled=bool(gate.get("enabled", False)),
                fail_strips_credit=bool(gate.get("fail_strips_credit", True)),
                credit_requires_verify_pass=bool(gate.get("credit_requires_verify_pass", True)),
                unknown_policy=gate.get("unknown_policy") or "earn_nothing",
                ledger=verify_ledger.get("meta") or {},
                nonzero_floor=floor_info,
            ),
            nonzero_weight_floor=floor_info,
            text=((
                "LIVE flip incentive (credit_mode=symmetric): for each published probe pair (E→E_next), join "
                "DISPATCH-selected (miner, task_index) with private probe per_task at E and E_next. Per-task "
                "credit = change in estimated pass probability: fail→pass = +w, pass→fail = −w, same w (=%g) "
                "both ways, so pure noise nets 0 in expectation. A miner's score is the NET sum over the whole "
                "window; a negative net is kept and carries forward (it offsets later gains). Shares = "
                "max(0, net)/sum, clipped only at this final normalization step. TOPLOC/sampling verify fail "
                "strips flip credit; bans unchanged. Applied to on-chain mech-0 weights. Hard nonzero floor: if "
                "gates would all-zero shares, preserve last good scores shares (else uniform over recent DISPATCH miners)."
                % symmetric_weight) if credit_mode == "symmetric" else (
                "LIVE flip incentive: for each published probe pair (E→E_next), join DISPATCH-selected "
                "(miner, task_index) at E with private probe per_task at E and E_next. "
                "Eligibility default=hard_fail (baseline status in unresolved|wrong|error). "
                "Credit fail→pass by surprise w=-log(p_hat) (or 1/p_hat); optional debit pass→fail by "
                "surprise on (1-p_hat). p_hat from running EMA of prior probe pass rates (alpha), "
                "else bandit ema_p, else raw pass at E. Shares = relu(flip_score)/sum. "
                "No authenticity *bonus* (no_authenticity_guard). TOPLOC/sampling verify fail strips "
                "flip credit; unverified earns nothing when credit_requires_verify_pass. "
                "Applied to on-chain mech-0 weights. Hard nonzero floor: if gates would all-zero shares, preserve last good scores shares (else uniform over recent DISPATCH miners)."
            )),
        ),
        window=dict(
            min_epoch=min_epoch,
            published_epochs=epochs_sorted,
            pairs=pair_summary,
            n_pairs=len(pairs),
            n_miners=len(miners),
        ),
        compare_to_live=live,
        uid_map=uid_info,
        uid_ban=dict(n_banned=len(banned), n_banned_in_miners=sum(1 for r in miners if r["banned"]),
                     note="banned=true rows: flip_score kept for transparency, share_now forced to 0 (ops/uid-ban.json)."),
        gaps=gaps,
        miners=miners,
        reference=dict(
            how_to_compare=(
                "Open https://affine.io/scores-shadow.json next to https://affine.io/scores.json. "
                "Compare miners[].share_now / uid ranking. This document is LIVE."
            ),
            how_to_flip_live_later=(
                "ONE path/config change when ready: point the LIVE weight reader / hourly submit "
                "from scores.json (v9/seq7) to scores-shadow.json (or set flip-shadow config "
                "live_weights=true AND wire weights_v9 to consume shadow shares). Do NOT flip until "
                "Jacob signs off. Rollback: set enabled=false or restore LIVE path to scores.json."
            ),
            task_join_report="/workspace/probe-miner-analysis/TASK-JOIN-REPORT.md",
        ),
    )
    # sha over canonical body without sha field
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    doc["score_sha256"] = hashlib.sha256(payload).hexdigest()

    # detailed events for audit (not website)
    audit_detail = dict(
        version=VERSION,
        generated_at_cr=cr(now),
        pairs=[
            dict(
                epoch_from=p["epoch_from"],
                epoch_to=p["epoch_to"],
                credit_epoch=p.get("credit_epoch"),
                credit_reason=p.get("credit_reason"),
                n_dispatch=p["n_dispatch"],
                n_fail_to_pass=p["n_fail_to_pass"],
                n_pass_to_fail=p["n_pass_to_fail"],
                events=p["events"],
            )
            for p in pairs
        ],
        published=pubs,
        gaps=gaps,
    )
    return doc, audit_detail


def publish(cfg: dict, doc: dict, audit_detail: dict) -> dict:
    paths = cfg["paths"]
    pub = cfg.get("publish") or {}
    out = {}
    audit_dir = Path(paths["audit_out"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    # always write audit copies
    out["audit_scores"] = str(audit_dir / "scores-shadow.json")
    out["audit_sha"] = _atomic(audit_dir / "scores-shadow.json", doc)
    out["audit_detail"] = str(audit_dir / "scores-shadow-detail.json")
    _atomic(audit_dir / "scores-shadow-detail.json", audit_detail)
    # top UIDs snapshot
    top = [
        dict(rank=i + 1, uid=m.get("uid"), hotkey=m.get("hotkey"), share_now=m.get("share_now"),
             flip_score=m.get("flip_score"), n_fail_to_pass=m.get("n_fail_to_pass"),
             n_pass_to_fail=m.get("n_pass_to_fail"))
        for i, m in enumerate((doc.get("miners") or [])[:30])
    ]
    _atomic(audit_dir / "top30.json", dict(generated_at_cr=doc.get("generated_at_cr"), top=top,
                                           score_sha256=doc.get("score_sha256")))

    if not cfg.get("enabled", True):
        out["website"] = "skipped:enabled=false"
        return out

    if pub.get("website_scores_shadow", True):
        site = Path(paths["website"])
        fname = pub.get("filename") or "scores-shadow.json"
        out["website_scores"] = str(site / fname)
        out["website_sha"] = _atomic(site / fname, doc)
        # per-epoch mirror keyed by newest pair's epoch_to
        pairs = (doc.get("window") or {}).get("pairs") or []
        if pairs and pub.get("per_epoch_dir"):
            edir = site / pub["per_epoch_dir"]
            latest_e = pairs[-1]["epoch_to"]
            _atomic(edir / f"e{latest_e}.json", doc)
            idx_path = edir / "index.json"
            try:
                idx = json.loads(idx_path.read_text()) if idx_path.exists() else dict(version=VERSION, epochs=[])
            except Exception:
                idx = dict(version=VERSION, epochs=[])
            epochs = list(idx.get("epochs") or [])
            if latest_e not in epochs:
                epochs.append(latest_e)
            idx = dict(
                version=VERSION,
                epochs=sorted(set(int(x) for x in epochs)),
                latest="https://affine.io/scores-shadow.json",
                live_weights=True,
                updated_at_cr=doc.get("generated_at_cr"),
            )
            _atomic(idx_path, idx)
            out["website_epoch"] = str(edir / f"e{latest_e}.json")
            out["website_index"] = str(idx_path)
        # LIVE cutover: also promote to scores.json (hourly writer + public contract)
        if pub.get("also_write_scores_json"):
            live_name = pub.get("live_filename") or "scores.json"
            live_doc = dict(doc)
            live_doc["mode"] = "live"
            live_doc["live_weights"] = True
            out["website_scores_live"] = str(site / live_name)
            out["website_scores_live_sha"] = _atomic(site / live_name, live_doc)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="SN120 flip-incentive SHADOW scorer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="compute only; no website write")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.dry_run:
        cfg = dict(cfg)
        cfg["publish"] = dict(cfg.get("publish") or {}, website_scores_shadow=False)
    doc, detail = compute(cfg)
    info = publish(cfg, doc, detail)
    top = (doc.get("miners") or [])[:10]
    print(json.dumps(dict(
        ok=True,
        enabled=cfg.get("enabled"),
        live_weights=False,
        score_sha256=doc.get("score_sha256"),
        n_miners=len(doc.get("miners") or []),
        n_pairs=(doc.get("window") or {}).get("n_pairs"),
        published_epochs=(doc.get("window") or {}).get("published_epochs"),
        gaps=doc.get("gaps"),
        top=[dict(uid=m.get("uid"), share=round(m.get("share_now") or 0, 6),
                  flip=round(m.get("flip_score") or 0, 4),
                  f2p=m.get("n_fail_to_pass"), p2f=m.get("n_pass_to_fail")) for m in top],
        publish=info,
        generated_at_cr=doc.get("generated_at_cr"),
    ), indent=2))


if __name__ == "__main__":
    main()
