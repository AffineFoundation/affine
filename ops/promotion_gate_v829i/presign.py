"""Promotion gate: presigned R2 GET URLs (read-only, 2 h) for a published checkpoint's model files.
Same bucket/key layout and credentials path as the env-probe daemon's _r2_presigned_fallback. Output is
written to stdout for piping straight to the probe pod; never logged.
usage: presign.py CKPT_ID < descriptor-files.json"""
import json, sys
from pathlib import Path
SRC = "/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822"
LIFE_CFG = "/home/const/subnet120-rewrite/state/root-audits/20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json"
def presign(ckpt_id, files, seconds=7200):
    if SRC not in sys.path: sys.path.insert(0, SRC)
    from subnet.storage import Bucket
    b = Bucket(json.loads(Path(LIFE_CFG).read_text())["bucket"])
    return {n: b.presign("mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/model/%s" % (ckpt_id, n), "get_object", seconds) for n in sorted(files)}
if __name__ == "__main__":
    ck = sys.argv[1]; files = json.load(sys.stdin)
    sys.stdout.write(json.dumps(dict(descriptor=dict(id=ck, files=files), urls=presign(ck, files))))
