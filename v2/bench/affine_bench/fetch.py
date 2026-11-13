"""Host side: materialize one published king checkpoint on the bench node.

The host makes 2 h read-only GET links (source.presigned). They are piped over ssh stdin into a 0600 file on the node,
gate_fetch.py downloads with parallel ranged GETs and checks every file's sha256 against the head manifest, and the
URL file is deleted right after. No long-lived credential ever reaches the node."""
import json
from . import config as C, source


def fetch(ck, node):
    blob = json.dumps(source.presigned(ck))
    D = C.NODE_KINGS
    cmd = (f"set -e; umask 077; mkdir -p {D} {D}/.work; cd {D}/.work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
           f"json.dump(d['descriptor'],open('{D}/{ck}.descriptor.json','w'));json.dump(d['urls'],open('{ck}.urls.private.json','w'))\" && "
           f"(nice -n 10 python3 {C.NODE_REPO}/bench/deploy/gate_fetch.py {D}/{ck}.descriptor.json {D} {ck}.urls.private.json --threads 32 "
           f">> {D}/{ck}.fetch.log 2>&1; rm -f {ck}.urls.private.json; tail -1 {D}/{ck}.fetch.log)")
    p = node.run(cmd, inp=blob, timeout=3600)
    return dict(ckpt=ck, status="rc%d" % p.returncode, out=p.stdout.strip()[-300:], err=p.stderr.strip()[-200:])
