#!/usr/bin/env python3
"""Green-only reveal cleanup (Jacob 2026-10-31 13:07 CR). Back up then delete RED epoch reveals, non-GREEN round reveals,
strip RED rows from GREEN round reveals, null gate_reveal links in RED turn.json, drop the two indexes for a rebuild.
Backups: <D>/backup/<TS>/deleted/<relpath> (0600). Writes <D>/backup/<TS>/DELETED.txt and MODIFIED.txt."""
import json, os, pathlib, shutil, sys
D = pathlib.Path(sys.argv[1]); TS = sys.argv[2]
W = pathlib.Path("/home/const/subnet120/affine/website"); G = pathlib.Path("/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs")
BK = D / "backup" / TS / "deleted"; os.umask(0o077)
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
def backup(p):
    q = BK / p.relative_to(W); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q); os.chmod(q, 0o600)
def write_public(p, obj):
    t = p.with_name("." + p.name + ".tmp"); t.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str)); os.chmod(t, 0o644); os.replace(t, p)
RULE = ('commitment = sha256(gate_version + "|" + seed_hex + "|" + comma-joined sorted task ids) (gate_core.sample_commitment); '
        'the seed is os.urandom(32). A round is revealed only if a GREEN closed it, and then only the GREEN turn: seed, task ids, '
        'the head baseline and the per-task grades of the GREEN candidate. RED turns and their probe results are never published; '
        'a round closed by a head change or by max_round_turns decided turns without a GREEN is never revealed. Score: '
        'inverse-variance-weighted per-task gain vs the head p_i on the kept tasks; GREEN iff gain > 0 and z > bar_z and '
        'estimated cap-hit < 0.20.')
dec = lambda e: (rj(G / str(e) / "DECISION.json") or {}).get("decision")
deleted, modified = [], []
for p in sorted((W / "gate/reveal").glob("*.json")):
    if p.stem.isdigit() and dec(int(p.stem)) != "green":
        backup(p); p.unlink(); deleted.append(str(p.relative_to(W)))
for p in sorted((W / "gate/heldout").glob("r*.json")):
    x = rj(p) or {}; ce = (x.get("closed") or {}).get("closing_epoch")
    g = [t for t in x.get("turns") or [] if t.get("epoch") == ce and t.get("decision") == "green"]
    if len(g) != 1:
        backup(p); p.unlink(); deleted.append(str(p.relative_to(W)))
    elif len(x["turns"]) > 1:
        backup(p); x["turns"] = g; x["rule"] = RULE
        write_public(p, x); modified.append(str(p.relative_to(W)) + " (kept only turn %d)" % ce)
for p in sorted((W / "k1/turns").glob("*/turn.json")):
    if not p.parent.name.isdigit(): continue
    x = rj(p)
    if isinstance(x, dict) and x.get("gate_reveal") and dec(int(p.parent.name)) != "green":
        backup(p); x["gate_reveal"] = None; write_public(p, x); modified.append(str(p.relative_to(W)) + " (gate_reveal -> null)")
for p in (W / "gate/reveal/index.json", W / "k1/turns/index.json"):
    if p.exists(): backup(p); p.unlink(); modified.append(str(p.relative_to(W)) + " (removed for sidecar rebuild)")
(D / "backup" / TS / "DELETED.txt").write_text("\n".join(deleted) + "\n"); (D / "backup" / TS / "MODIFIED.txt").write_text("\n".join(modified) + "\n")
print(len(deleted), "deleted;", len(modified), "modified")
