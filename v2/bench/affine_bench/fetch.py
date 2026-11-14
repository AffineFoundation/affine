"""Host side: materialize the weights of one row on its bench node.

The host makes 2 h read-only GET links (source.presigned). They are piped over ssh stdin into a 0600 file on the node,
gate_fetch.py downloads with parallel ranged GETs and checks every file's sha256 against the head manifest, and the
URL file is deleted right after. No long-lived credential ever reaches the node."""
import hashlib, json, pathlib
from . import config as C, source

REBUILD = "bench/deploy/chain_rebuild.py"  # the node's script of a rebuild, under the checkout's v2 directory


def fetch(m, node):
    ck = m["base_hash"]; blob = json.dumps(source.presigned(m))
    D = C.NODE_KINGS
    cmd = (f"set -e; umask 077; mkdir -p {D} {D}/.work; cd {D}/.work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
           f"json.dump(d['descriptor'],open('{D}/{ck}.descriptor.json','w'));json.dump(d['urls'],open('{ck}.urls.private.json','w'))\" && "
           f"(nice -n 10 python3 {C.NODE_REPO}/bench/deploy/gate_fetch.py {D}/{ck}.descriptor.json {D} {ck}.urls.private.json --threads 32 "
           f">> {D}/{ck}.fetch.log 2>&1; rm -f {ck}.urls.private.json; tail -1 {D}/{ck}.fetch.log)")
    p = node.run(cmd, inp=blob, timeout=3600)
    return dict(ckpt=ck, status="rc%d" % p.returncode, out=p.stdout.strip()[-300:], err=p.stderr.strip()[-200:])


def rebuild(m, node, order):
    """Make the weights of one row on its node from the head chain (chain.order made `order`).

    The node's copy of the script must be, byte for byte, the script of the code that this host runs: else nothing
    runs. The script proves that the chain code on the node is at the commit `order["ref"]` before it runs the tool,
    and it writes the mark .verified only when each file of the result has the sha256 of the king's head manifest.
    The links go to the node on ssh stdin, as for a fetch; the script deletes its links file."""
    ck = m["base_hash"]; D = C.NODE_KINGS; B = C.NODE_ROOT; script = f"{C.NODE_REPO}/{REBUILD}"
    want = hashlib.sha256((pathlib.Path(__file__).resolve().parents[1] / "deploy" / "chain_rebuild.py").read_bytes()).hexdigest()
    cmd = (f"set -e; umask 077; mkdir -p {D}/.chain; cd {B} && . ./env.sh; "
           f"echo '{want}  {script}' | sha256sum -c --status || {{ echo '{{\"status\": \"refused\", \"why\": \"the rebuild script on the node is not the script of the host\"}}'; exit 3; }}; "
           f"nice -n 10 python3 {script} {D} {D}/.chain --python {B}/venv-vllm/bin/python --repo-url {C.REPO_URL} --ref {order['ref']} "
           f"--clone {B}/chain-repo --main-clone {C.NODE_CLONE} --deps {B}/chain-deps --device {order['device']}")
    p = node.run(cmd, inp=json.dumps(order["blob"]), timeout=order["limit_s"])
    return dict(ckpt=ck, status="rc%d" % p.returncode, out=p.stdout.strip()[-400:], err=p.stderr.strip()[-200:])
