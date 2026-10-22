"""Sign writer time budget v2 (2026-10-20): v7b->v7d (evidence 1000->1400, maximum 1200->1600; entry+parallel_module
repinned to copies differing only in those bounds) and v7c->v7e (same measured-impact step, same blend module/config,
base repointed to v7d, same bounds). Seed never printed. Weights logic unchanged."""
import base64, hashlib, json, os
from pathlib import Path
from nacl.signing import SigningKey, VerifyKey
R=Path("/home/const/subnet120-rewrite/state/root-audits"); B=R/"20261018-writer-time-budget-v1"; M=R/"20261019-measured-impact-v1"; N=R/"20261020-writer-time-budget-v2"
AUTH="3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd"
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
sha=lambda b: hashlib.sha256(b).hexdigest(); pin=lambda f: dict(path=str(f),sha256=sha(Path(f).read_bytes()))
life=json.loads((R/"20261015-mimo-v8.2-release-succession/release-v822-signed/lifecycle.v8.2.2.ACTIVE.private.json").read_text())
key=SigningKey(bytes.fromhex(Path(life["authority_seed_file"]).read_text().strip())); assert key.verify_key.encode().hex()==AUTH,"authority mismatch"
def sign(p):
    out=dict(payload=p,signer=AUTH,signature=base64.b64encode(key.sign(canon(p)).signature).decode())
    VerifyKey(bytes.fromhex(AUTH)).verify(canon(p),base64.b64decode(out["signature"])); return out
def write(dst,doc):
    fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,json.dumps(doc,indent=1,sort_keys=True).encode()); os.close(fd)
def diff(a,b): return sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
src=json.loads((B/"HOST-COMPOSITION.v8.2.2-accel-window-v7b-budget.ROOT-SIGNED.private.json").read_text())
VerifyKey(bytes.fromhex(AUTH)).verify(canon(src["payload"]),base64.b64decode(src["signature"]))
p=dict(src["payload"]); p.update(entry=pin(N/"hourly_population_index_entry_v7d.py"),parallel_module=pin(N/"parallel_evidence_readers.py"),evidence_timeout_seconds=1400,maximum_seconds=1600)
d1=N/"HOST-COMPOSITION.v8.2.2-accel-window-v7d-budget2.ROOT-SIGNED.private.json"; write(d1,sign(p))
src2=json.loads((M/"HOST-COMPOSITION.v7c-impact.ROOT-SIGNED.private.json").read_text())
VerifyKey(bytes.fromhex(AUTH)).verify(canon(src2["payload"]),base64.b64decode(src2["signature"]))
q=dict(src2["payload"]); q.update(entry=pin(N/"hourly_entry_v7e_impact.py"),base_entry=pin(N/"hourly_population_index_entry_v7d.py"),base_composition=pin(d1),evidence_timeout_seconds=1400,maximum_seconds=1600)
d2=N/"HOST-COMPOSITION.v7e-impact-budget2.ROOT-SIGNED.private.json"; write(d2,sign(q))
print(json.dumps(dict(base=dict(path=str(d1),sha256=sha(d1.read_bytes()),diff=diff(src["payload"],p)),top=dict(path=str(d2),sha256=sha(d2.read_bytes()),diff=diff(src2["payload"],q)))))
