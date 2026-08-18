#!/usr/bin/env python3
"""p3810b: refresh HF token from host .env and retry marsplan download + bootstrap."""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

HOST_ENV = Path("/tmp/mining_host.env")
MINE = Path("/root/mine.env")
NEW_BASE = "8b365bdcd8f270c61fe633fcc95d536a93516e02"

raw = HOST_ENV.read_text()
m = re.search(r"^(?:export\s+)?HF_TOKEN=(.*)$", raw, re.M)
if not m:
    raise SystemExit("HF_TOKEN missing in /tmp/mining_host.env")
token = m.group(1).strip().strip('"').strip("'")
print("host_token_len", len(token), "prefix", token[:8])

t = MINE.read_text()
t2, n = re.subn(r"export HF_TOKEN=.*", f"export HF_TOKEN={token}", t)
if n != 1:
    raise SystemExit(f"HF_TOKEN replace n={n}")
MINE.write_text(t2)
print("mine.env token refreshed")

os.environ["HF_TOKEN"] = token
# Use pod venv (system python has no huggingface_hub).
venv_py = "/root/venv/bin/python3"
probe = subprocess.check_output(
    [
        venv_py,
        "-c",
        "from huggingface_hub import hf_hub_download; import os; "
        f"p=hf_hub_download('marsplan0624/affine-5gedzafcvg-queen','config.json',"
        f"revision='{NEW_BASE}',token=os.environ['HF_TOKEN']); print('probe_ok', p)",
    ],
    text=True,
    env={**os.environ, "HF_TOKEN": token},
)
print(probe.strip())

# Full snapshot (long). Launch in background via bootstrap instead.
log = Path("/root/logs/r337_pipeline.p3810b.nohup")
pidf = Path("/root/logs/r337_pipeline.p3810b.pid")
with open(log, "a") as lf:
    lf.write("\n=== p3810b relaunch after token refresh ===\n")
cmd = (
    "nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh "
    ">>/root/logs/r337_pipeline.p3810b.nohup 2>&1 & echo $!"
)
out = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
print("launch_out", out)
pids = [x for x in out.split() if x.isdigit()]
if not pids:
    raise SystemExit("no pid")
pidf.write_text(pids[-1] + "\n")
print("pid", pids[-1])
time.sleep(12)
print("log_tail:")
print(log.read_text()[-2200:])
