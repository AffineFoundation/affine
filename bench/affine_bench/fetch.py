"""Host side: materialize one published king checkpoint on the bench node.

Same mechanism as the SN120 perf-eval service (ops/performance_eval/fetch_king.py): the gate's presign.py makes 2 h read-only
R2 GET URLs on the host, they are piped over ssh stdin into a 0600 file on the node, gate_fetch.py downloads with parallel
ranged GETs and checks every file's sha256 against the signed descriptor, and the URL file is deleted right after.
No long-lived credential ever reaches the node."""
import json, subprocess
from pathlib import Path
from . import config as C


def descriptor(ck):
    """files{} of a published checkpoint, from the lifecycle record (TRAINING-RESULT publication receipt)."""
    for d in sorted(Path(C.LIFECYCLE_EPOCHS).glob("*"), key=lambda p: -int(p.name) if p.name.isdigit() else 1):
        if not d.name.isdigit():
            continue
        try:
            r = json.loads((d / "TRAINING-RESULT.json").read_text())
        except Exception:
            continue
        desc = (((r.get("payload") or r).get("report") or {}).get("publication_receipt") or {}).get("payload", {}).get("descriptor") or {}
        if desc.get("id") == ck:
            return desc["files"]
    return None


def fetch(ck, node):
    files = descriptor(ck)
    if files is None:
        return dict(ckpt=ck, status="no-descriptor")
    blob = subprocess.run([C.HOST_PYTHON, "-B", C.PRESIGN, ck], input=json.dumps(files), capture_output=True, text=True, timeout=120)
    if blob.returncode != 0:
        return dict(ckpt=ck, status="presign-failed", err=blob.stderr[-200:])
    D = C.NODE_KINGS
    cmd = (f"set -e; umask 077; mkdir -p {D} {D}/.work; cd {D}/.work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
           f"json.dump(d['descriptor'],open('{D}/{ck}.descriptor.json','w'));json.dump(d['urls'],open('{ck}.urls.private.json','w'))\" && "
           f"(nice -n 10 python3 {C.NODE_REPO}/bench/deploy/gate_fetch.py {D}/{ck}.descriptor.json {D} {ck}.urls.private.json --threads 32 "
           f">> {D}/{ck}.fetch.log 2>&1; rm -f {ck}.urls.private.json; tail -1 {D}/{ck}.fetch.log)")
    p = node.run(cmd, inp=blob.stdout, timeout=3600)
    return dict(ckpt=ck, status="rc%d" % p.returncode, out=p.stdout.strip()[-300:], err=p.stderr.strip()[-200:])
