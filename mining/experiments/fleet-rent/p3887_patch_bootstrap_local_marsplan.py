#!/usr/bin/env python3
"""p3887: patch bootstrap to use local marsplan BASE (HF gated on pod IP)."""
from pathlib import Path
import re
import sys

p = Path("/root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh")
t = p.read_text()
if "p3810_LOCAL_CACHE_SKIP" in t or "p3887_LOCAL_CACHE_SKIP" in t:
    print("already patched")
    raise SystemExit(0)

new = r'''python - <<'PY'
# p3887_LOCAL_CACHE_SKIP: marsplan gated on pod IP — use local BASE cache
import os
from pathlib import Path
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
base = Path(os.environ.get(
    "BASE",
    "/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46",
))
if not (base / "config.json").is_file():
    raise SystemExit(f"local marsplan BASE missing: {base}")
path = str(base)
print("[bootstrap] p3887_LOCAL_CACHE_SKIP using local BASE", path, flush=True)
open("/root/logs/marsplan_init.done", "w").write(path + "\n")
open("/root/logs/tok_init.done", "w").write(path + "\n")
print("[bootstrap] DOWNLOAD teacher start", flush=True)
tpath = snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print(f"[bootstrap] DOWNLOAD teacher done -> {tpath}", flush=True)
open("/root/logs/teacher.done", "w").write(tpath + "\n")
PY'''

# Match the full python heredoc that downloads marsplan + teacher
pat = re.compile(
    r"python - <<'PY'\nimport os\nfrom huggingface_hub import snapshot_download\n"
    r"token = os\.environ\[\"HF_TOKEN\"\]\n"
    r"repo = \"marsplan0624/affine-5gedzafcvg-queen\"\n"
    r"rev = \"[^\"]+\"\n"
    r"print\(\"[^\"]*DOWNLOAD marsplan-init start\".*?open\(\"/root/logs/teacher\.done\".*?\)\nPY",
    re.S,
)
m = pat.search(t)
if not m:
    idx = t.find("DOWNLOAD marsplan-init")
    print("marker idx", idx, file=sys.stderr)
    if idx != -1:
        print(repr(t[max(0, idx - 200) : idx + 600]), file=sys.stderr)
    raise SystemExit("download block not found")

p.write_text(t[: m.start()] + new + t[m.end() :])
print("patched ok", m.start(), m.end())
