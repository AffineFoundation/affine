#!/usr/bin/env python3
"""SN120 mechanism-1 (Tobias / VEDA sparse-attention track) weight submitter.

Independent of the mechanism-0 hourly writer: own directory, own state, own timer.
Signs commit_timelocked_mechanism_weights(netuid=120, mecid=1) with the UID0 owner
hotkey (wallet default/default). Never touches mecid 0.
Safety: refuses to submit while the mech-0 writer unit is running, or before the
mech-0 commit for the current hour has landed (avoids nonce races with mech 0).
"""
import argparse, json, math, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

NETUID = 120
MECHID = 1
TRACK_ID = "veda-sn120-v2-r4"
# Track spec digest of the live track.json (veda-sn120-v2-r4, start 9323901, recomputed with repo parser @bffe446 2026-10-20 07:1x CR; was d1d4ced3... v2-r2).
TRACK_SPEC_DIGEST = "0d12378760b83639115ef2274978f2f099fd95501db9c937871016d15820b4e0"
SOURCE_URL = "https://iblosevwkuiootbbgnsy.supabase.co/storage/v1/object/public/affine_video/tracks/main/weights.json"
STATE = Path(__file__).resolve().parent / "state"
MAX_SOURCE_AGE_BLOCKS = 1800          # ~6 h at 12 s/block, judged by source_block vs chain head
MAX_SOURCE_AGE_S = 48 * 3600          # secondary guard on the HTTP Last-Modified header
MAX_SOURCE_BYTES = 1_000_000
WRITER_UNIT = "affine-live-reward-writer.service"


def log(rec, fname="runs.jsonl"):
    STATE.mkdir(parents=True, exist_ok=True)
    rec = dict(ts=datetime.now(timezone.utc).isoformat(), **rec)
    with open(STATE / fname, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps(rec, sort_keys=True))


def writer_active():
    r = subprocess.run(["systemctl", "--user", "is-active", WRITER_UNIT], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating", "reloading")


def _no_dup_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate JSON key {k!r}")
        d[k] = v
    return d


def parse_source(body):
    if len(body) > MAX_SOURCE_BYTES:
        raise ValueError("source too large")
    return json.loads(body, object_pairs_hook=_no_dup_keys)


def fetch_source():
    req = urllib.request.Request(SOURCE_URL, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read(MAX_SOURCE_BYTES + 1)
        lm = r.headers.get("Last-Modified")
    return parse_source(body), lm


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def resolve(doc, head, uid_of, last_modified=None, now=None):
    """Validate Tobias's file against the chain. Mirrors his validate_weight_artifact, except that
    entries whose hotkey is not registered on SN120 at the stated uid are dropped (not fatal).

    uid_of(hotkey) -> current SN120 uid or None.
    Returns (uids, weights, reasons, dropped). uids is None => caller must use the placeholder;
    reasons then says why. dropped lists entries removed from an otherwise valid file."""
    reasons, dropped = [], []
    if not isinstance(doc, dict) or not isinstance(doc.get("weights"), list) or not doc["weights"]:
        return None, None, ["bad-schema:weights must be a non-empty list"], dropped
    netuid, mechid = doc.get("netuid"), doc.get("mechid")
    if not _is_int(netuid) or netuid != NETUID:
        reasons.append(f"netuid={netuid!r}!=120")
    if not _is_int(mechid) or mechid != MECHID:
        reasons.append(f"mechid={mechid!r}!=1")
    if doc.get("track_id") != TRACK_ID:
        reasons.append(f"track_id={doc.get('track_id')!r}!={TRACK_ID}")
    dg = doc.get("track_spec_digest")
    if not isinstance(dg, str) or dg.lower() != TRACK_SPEC_DIGEST:
        reasons.append(f"track_spec_digest={str(dg)[:16]!r}...!=expected")
    sb = doc.get("source_block")
    if not _is_int(sb) or sb < 0:
        reasons.append(f"source_block-missing-or-invalid:{sb!r}")
    elif not _is_int(head) or head <= 0:
        reasons.append(f"chain-head-unknown:{head!r}")
    elif sb > head:
        reasons.append(f"source_block-future:{sb}>{head}")
    elif head - sb > MAX_SOURCE_AGE_BLOCKS:
        reasons.append(f"stale:source_block {sb} is {head - sb} blocks old (>{MAX_SOURCE_AGE_BLOCKS})")
    if last_modified:
        try:
            lm = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc).timestamp()
            age = (now if now is not None else time.time()) - lm
            if age > MAX_SOURCE_AGE_S:
                reasons.append(f"stale-last-modified:{int(age)}s")
        except ValueError:
            pass  # header is informational only; freshness is judged by source_block
    if reasons:
        return None, None, reasons, dropped

    seen_hk, seen_uid, entries = set(), set(), []
    for i, e in enumerate(doc["weights"]):
        if not isinstance(e, dict):
            return None, None, [f"bad-entry[{i}]:not an object"], dropped
        hk, uid, w = e.get("hotkey"), e.get("uid"), e.get("weight")
        if not isinstance(hk, str) or not hk:
            return None, None, [f"bad-entry[{i}]:hotkey"], dropped
        if not _is_int(uid) or not 0 <= uid <= 0xFFFF:
            return None, None, [f"bad-entry[{i}]:uid={uid!r}"], dropped
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
            return None, None, [f"bad-entry[{i}]:weight={w!r} (must be finite and non-negative)"], dropped
        if hk in seen_hk or uid in seen_uid:
            return None, None, [f"bad-entry[{i}]:duplicate hotkey/uid"], dropped
        seen_hk.add(hk); seen_uid.add(uid)
        entries.append((hk, uid, float(w)))

    pairs = {}
    for hk, uid, w in entries:
        chain_uid = uid_of(hk)
        chain_uid = None if chain_uid is None else int(chain_uid)
        if chain_uid != uid:
            dropped.append(f"{hk[:10]}@uid{uid}:chain_uid={chain_uid}")
            continue
        if w == 0.0:
            dropped.append(f"{hk[:10]}@uid{uid}:zero-weight")
            continue
        pairs[uid] = w
    total = sum(pairs.values())
    if not pairs or not math.isfinite(total) or total <= 0:
        return None, None, ["no-valid-registered-entries"] + [f"dropped:{d}" for d in dropped], dropped
    uids = sorted(pairs)
    return uids, [pairs[u] / total for u in uids], [], dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--ignore-writer-gate", action="store_true", help="only for the first manual run")
    ap.add_argument("--check-only", action="store_true",
                    help="fetch + validate + log to state/checks.jsonl; no plan, no submit, no gates")
    ap.add_argument("--source-file", help="validate a local JSON file instead of the URL (requires --check-only)")
    a = ap.parse_args()
    if a.source_file and not a.check_only:
        ap.error("--source-file is only allowed with --check-only")
    if a.check_only and a.execute:
        ap.error("--check-only and --execute are mutually exclusive")

    import bittensor as bt
    from bittensor.intents.weights import _preflight  # noqa: F401  (pinned SDK 11.0.2)

    sub = bt.Subtensor("finney")
    raw = sub._client._substrate if hasattr(sub, "_client") else None

    def q(name, params):
        return sub._call(raw.query("SubtensorModule", name, params))
    sub.substrate_query_uid = lambda hk: q("Uids", [NETUID, hk])

    wallet = bt.Wallet(name="default", hotkey="default", path=str(Path.home() / ".bittensor/wallets"))
    hk = wallet.hotkey.ss58_address
    owner_uid = q("Uids", [NETUID, hk])
    if owner_uid != 0:
        log(dict(status="abort", reason=f"hotkey uid {owner_uid} != 0")); return 2
    if int(q("MechanismCountCurrent", [NETUID])) < 2:
        log(dict(status="abort", reason="mechanism 1 does not exist")); return 2

    head = int(sub._call(raw.block_number()))
    try:
        if a.source_file:
            doc, lm = parse_source(Path(a.source_file).read_bytes()), None
        else:
            doc, lm = fetch_source()
        src = dict(track_id=doc.get("track_id"), netuid=doc.get("netuid"), mechid=doc.get("mechid"),
                   source_block=doc.get("source_block"), digest=str(doc.get("track_spec_digest"))[:16],
                   n=len(doc.get("weights") or []), last_modified=lm, file=a.source_file)
        uids, weights, reasons, dropped = resolve(doc, head, sub.substrate_query_uid, lm)
    except Exception as e:  # unreachable / unparsable source => placeholder
        uids, weights, reasons, dropped, src = None, None, [f"fetch-or-parse-error:{type(e).__name__}:{str(e)[:120]}"], [], {}
    mode = "tobias"
    if uids is None:
        mode = "placeholder-burn-uid0"
        uids, weights = [owner_uid], [1.0]   # owner hotkey incentive is withheld/burned (RecycleOrBurn=Burn)
        print(f"FALLBACK to placeholder (100% uid {owner_uid}): " + "; ".join(reasons), file=sys.stderr)

    lu0 = q("LastUpdate", [NETUID])
    lu0_uid0 = int(lu0[0]) if lu0 else 0
    lu1 = q("LastUpdate", [MECHID * 4096 + NETUID]) or []
    rec = dict(mode=mode, reasons=reasons, dropped=dropped, source=src, uids=uids, weights=weights, head=head,
               mech0_last_update=lu0_uid0, mech1_last_update=(int(lu1[0]) if lu1 else 0))
    if mode != "tobias":
        rec["fallback_reason"] = "; ".join(reasons)
    if a.check_only:
        log(dict(rec, status="check-only"), "checks.jsonl"); return 0

    # Gate: never race the mech-0 writer.
    if not a.ignore_writer_gate:
        if writer_active():
            log(dict(rec, status="skip", reason="mech0 writer running")); return 0
        minute = datetime.now().minute
        if minute < 25:
            log(dict(rec, status="skip", reason="inside mech0 writer window (:00-:25)")); return 0
        if head - lu0_uid0 > 330:
            log(dict(rec, status="skip", reason="mech0 commit for this hour not landed yet")); return 0
    if lu1 and int(lu1[0]) and head - int(lu1[0]) < 240:
        log(dict(rec, status="skip", reason="mech1 already committed within 240 blocks")); return 0

    intent = bt.SetWeights(netuid=NETUID, uids=uids, weights=weights, mechid=MECHID, version_key=0)
    if not a.execute:
        plan = sub.plan(intent, wallet)
        log(dict(rec, status="plan", plan=str(plan)[:2000])); return 0
    res = sub.execute(intent, wallet)
    log(dict(rec, status="submitted", result=str(res)[:3000]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
