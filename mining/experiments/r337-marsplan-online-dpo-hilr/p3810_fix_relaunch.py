#!/usr/bin/env python3
"""p3810: fix dead marsplan rev + live king, relaunch R337 bootstrap."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

NEW_BASE = "8b365bdcd8f270c61fe633fcc95d536a93516e02"
OLD_BASE = "556d02a2adfa9bd42a02de3c766f98be7e44ca46"
KING_REPO = "tammyfritz/Affine-5hmwhnfbix-tammy2"
KING_REV = "7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5"

envp = Path("/root/mine.env")
bak = Path("/root/mine.env.bak.p3810")
if envp.is_file() and not bak.is_file():
    bak.write_text(envp.read_text())

t = envp.read_text().replace(OLD_BASE, NEW_BASE)
t = re.sub(r"export KING_REPO=.*", f"export KING_REPO={KING_REPO}", t)
t = re.sub(r"export KING_REV=.*", f"export KING_REV={KING_REV}", t)
t = re.sub(
    r"export KING_LOCAL=.*",
    f"export KING_LOCAL=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/{KING_REV}",
    t,
)
t = re.sub(
    r"export BASE=.*",
    f"export BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/{NEW_BASE}",
    t,
)
envp.write_text(t)
print("mine.env keys:")
for line in t.splitlines():
    if any(k in line for k in ("KING_", "export BASE=", "HF_BASE", "R337_LR", "R337_MAX")):
        print(line)

for rel in (
    "s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh",
    "s4-h139-f44-tok-online-dpo-l2/start_h139.sh",
    "s4-h139-f44-tok-online-dpo-l2/post_train_pipeline.sh",
):
    p = Path("/root/mining_src") / rel
    if p.is_file():
        txt = p.read_text()
        if OLD_BASE in txt:
            p.write_text(txt.replace(OLD_BASE, NEW_BASE))
            print("patched", rel)

for f in Path("/root/logs").glob("*pipeline*.pid"):
    f.unlink(missing_ok=True)
    print("rm", f)

log = Path("/root/logs/r337_pipeline.p3810.nohup")
pidf = Path("/root/logs/r337_pipeline.p3810.pid")
with open(log, "a") as lf:
    lf.write("\n=== p3810 relaunch ===\n")
cmd = (
    "nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh "
    ">>/root/logs/r337_pipeline.p3810.nohup 2>&1 & echo $!"
)
out = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
print("launch_out", out)
pids = [x for x in out.split() if x.isdigit()]
if not pids:
    raise SystemExit("no pid")
pidf.write_text(pids[-1] + "\n")
print("pid", pids[-1])
time.sleep(5)
print("log_tail:")
print(log.read_text()[-2000:])
