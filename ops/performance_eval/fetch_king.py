#!/usr/bin/env python3
"""Fetch one published king checkpoint to the PERFORMANCE-EVAL pod (calm-orbit-ef) via the gate's own presign.py (2 h read-only R2 GET URLs,
piped over ssh stdin, written 0600 on the pod, deleted by gate_fetch.py after the sha256-verified download; never logged).
usage: fetch_king.py CKPT_ID  -> prints JSON status line"""
import json, subprocess, sys
from pathlib import Path
LIFE = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1")
GC = Path("/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code")
_H = Path(__file__).resolve().parent
PP = str(_H / "pp.sh" if (_H / "pp.sh").exists() else _H.parent / "pp.sh")
D = "/data/perf"
def descriptor(ck):
    for d in sorted((LIFE / "epochs").glob("*"), key=lambda p: -int(p.name) if p.name.isdigit() else 1):
        if not d.name.isdigit(): continue
        try: r = json.loads((d / "TRAINING-RESULT.json").read_text())
        except Exception: continue
        desc = (((r.get("payload") or r).get("report") or {}).get("publication_receipt") or {}).get("payload", {}).get("descriptor") or {}
        if desc.get("id") == ck: return desc["files"]
    return None
def fetch(ck):
    files = descriptor(ck)
    if files is None: return dict(ckpt=ck, status="no-descriptor")
    blob = subprocess.run(["/home/const/subnet120-rewrite/.venv/bin/python", "-B", str(GC / "presign.py"), ck], input=json.dumps(files),
                          capture_output=True, text=True, timeout=120)
    if blob.returncode != 0: return dict(ckpt=ck, status="presign-failed", err=blob.stderr[-200:])
    cmd = ("set -e; umask 077; mkdir -p {D}/ckpt {D}/work; cd {D}/work && /usr/bin/python3 -c \"import json,sys;d=json.load(sys.stdin);"
           "json.dump(d['descriptor'],open('{D}/ckpt/{c}.descriptor.json','w'));json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
           "(nice -n 10 /usr/bin/python3 {D}/gate_fetch.py {D}/ckpt/{c}.descriptor.json {D}/ckpt {c}.urls.private.json --threads 32 >> {D}/ckpt/{c}.fetch.log 2>&1; "
           "rm -f {c}.urls.private.json; tail -1 {D}/ckpt/{c}.fetch.log)").format(D=D, c=ck)
    p = subprocess.run([PP, cmd], input=blob.stdout, capture_output=True, text=True, timeout=3600)
    return dict(ckpt=ck, status="rc%d" % p.returncode, out=p.stdout.strip()[-300:], err=p.stderr.strip()[-200:])
if __name__ == "__main__":
    for ck in sys.argv[1:]: print(json.dumps(fetch(ck)), flush=True)
