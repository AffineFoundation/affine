#!/usr/bin/env python3
"""Read-only builder for affine.io/bans.json (Banned miners table).

Reads (never writes): the live UID ban list used by the train gate / flip scorer,
the verify-worker audit reports (roles.sqlite3, opened read-only), and the SN120
metagraph Keys map from chain. Writes only <website>/bans.json atomically.
"""
import hashlib, json, os, sqlite3, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BAN = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/ops/uid-ban.json")
ROLES = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/roles.sqlite3")
OUT = Path("/home/const/subnet120/affine/website/bans.json")
NETUID = 120
CR = timezone(timedelta(hours=-6))
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def ss58(pk: bytes, fmt: int = 42) -> str:
    d = bytes([fmt]) + pk
    d += hashlib.blake2b(b"SS58PRE" + d, digest_size=64).digest()[:2]
    n, s = int.from_bytes(d, "big"), ""
    while n:
        n, r = divmod(n, 58)
        s = B58[r] + s
    return s

def epoch_num(ep):
    try:
        return int(str(ep).rsplit("-e", 1)[1])
    except Exception:
        return None

def cr(ts):
    return datetime.fromtimestamp(ts, CR).strftime("%Y-%m-%d %H:%M CR (UTC-6)")

def invalid_sample_epochs(miners):
    out = {m: set() for m in miners}
    con = sqlite3.connect(f"file:{ROLES}?mode=ro", uri=True)
    try:
        for (rep,) in con.execute("select report from jobs where role='verify' and status='complete' and report is not null"):
            try:
                doc = json.loads(rep)
            except Exception:
                continue
            p = doc.get("payload", doc) if isinstance(doc, dict) else None
            if not isinstance(p, dict):
                continue
            for a in p.get("audits") or []:
                if not isinstance(a, dict):
                    continue
                for o in a.get("outcomes") or []:
                    if not isinstance(o, dict):
                        continue
                    m = o.get("miner") or a.get("miner")
                    if m in out and (o.get("error_type") == "InvalidSample" or o.get("failure_kind") == "invalid_sample"):
                        n = epoch_num(o.get("epoch") or a.get("epoch"))
                        if n is not None:
                            out[m].add(n)
    finally:
        con.close()
    return {m: sorted(v) for m, v in out.items()}

def main():
    ban = json.loads(BAN.read_text())
    rows = [b for b in ban.get("banned") or [] if isinstance(b, dict) and isinstance(b.get("miner"), str) and len(b["miner"]) == 64]
    import bittensor as bt
    st = bt.Subtensor("finney")
    keys = st.query_map(("SubtensorModule", "Keys"), [NETUID])
    hk2uid = {hk: int(uid) for uid, hk in (keys.items() if isinstance(keys, dict) else keys)}
    if len(hk2uid) < 64:
        sys.exit("metagraph lookup returned too few keys; keeping previous bans.json")
    block = st.block() if callable(st.block) else st.block
    fails = invalid_sample_epochs({b["miner"] for b in rows})
    entries = []
    for b in rows:
        hk = ss58(bytes.fromhex(b["miner"]))
        ts = b.get("banned_at")
        entries.append(dict(
            uid=hk2uid.get(hk), hotkey=hk, hotkey_short=hk[:4] + "…" + hk[-4:],
            banned_since_epoch=epoch_num(b.get("epoch")),
            banned_at=ts, banned_at_cr=cr(ts) if isinstance(ts, (int, float)) else b.get("banned_at_cr"),
            audits_failed=fails.get(b["miner"]) or ([epoch_num(b.get("epoch"))] if epoch_num(b.get("epoch")) is not None else []),
            reason="TOPLOC audit: InvalidSample" if b.get("reason") in ("invalid_sample", "InvalidSample") else str(b.get("reason") or ""),
        ))
    entries.sort(key=lambda e: (e["banned_at"] or 0, e["uid"] if e["uid"] is not None else 1e9))
    now = time.time()
    doc = dict(version="affine-site-bans-v1", generated_at=now, generated_at_cr=cr(now), chain_block=int(block),
               netuid=NETUID, ban_list_updated_at_cr=ban.get("updated_at_cr"), count=len(entries), banned=entries)
    tmp = OUT.with_name(".bans.json.new")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.chmod(tmp, 0o644)
    tmp.replace(OUT)
    print(f"bans.json: {len(entries)} entries, block {block}")

if __name__ == "__main__":
    main()
