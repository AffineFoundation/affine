#!/usr/bin/env python3
"""Read-only R2 listing: every checkpoint under the MiMo run prefix with its model files present (count, bytes, newest LastModified).
Uses the gate presign.py's bucket config (credentials stay on the host; nothing printed but ids/sizes/times). Writes r2_checkpoints.json."""
import json, sys, collections, os
from pathlib import Path
sys.path.insert(0, "/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822")
from subnet.storage import Bucket
LIFE_CFG = "/home/const/subnet120-rewrite/state/root-audits/20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json"
b = Bucket(json.loads(Path(LIFE_CFG).read_text())["bucket"])
P = "mimo/affine-mimo9b-grpo-20261013/checkpoints/"
agg = collections.defaultdict(lambda: dict(files=0, bytes=0, model_files=0, model_bytes=0, last=None, first=None))
pg = b.client.get_paginator("list_objects_v2")
for page in pg.paginate(Bucket=b.name, Prefix=P):
    for o in page.get("Contents", []):
        rest = o["Key"][len(P):]; ck = rest.split("/")[0]; a = agg[ck]
        a["files"] += 1; a["bytes"] += o["Size"]; t = o["LastModified"].timestamp()
        if rest.split("/")[1:2] == ["model"]: a["model_files"] += 1; a["model_bytes"] += o["Size"]
        a["last"] = max(a["last"] or t, t); a["first"] = min(a["first"] or t, t)
out = {k: v for k, v in agg.items()}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_checkpoints.json"), "w"), indent=1)
full = {k: v for k, v in out.items() if v["model_bytes"] > 15e9}
print("checkpoints:", len(out), "with full bf16 model (>15 GB):", len(full))
