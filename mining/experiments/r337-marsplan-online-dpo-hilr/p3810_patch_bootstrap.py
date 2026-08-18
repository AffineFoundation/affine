#!/usr/bin/env python3
"""Patch R337 bootstrap to use local marsplan BASE (pod IP gated on HF)."""
from pathlib import Path

p = Path("/root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh")
t = p.read_text()
if "p3810_LOCAL_CACHE_SKIP" in t:
    print("already patched")
    raise SystemExit(0)

candidates = [
    '''python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
repo = "marsplan0624/affine-5gedzafcvg-queen"
rev = "556d02a2adfa9bd42a02de3c766f98be7e44ca46"
print("[bootstrap-r337] DOWNLOAD marsplan-init start", repo, rev, flush=True)
path = snapshot_download(repo, revision=rev, token=token)
print(f"[bootstrap-r337] DOWNLOAD marsplan-init done -> {path}", flush=True)
open("/root/logs/marsplan_init.done", "w").write(path + "\\n")
open("/root/logs/tok_init.done", "w").write(path + "\\n")
print("[bootstrap-r337] DOWNLOAD teacher start", flush=True)
tpath = snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print(f"[bootstrap-r337] DOWNLOAD teacher done -> {tpath}", flush=True)
open("/root/logs/teacher.done", "w").write(tpath + "\\n")
PY''',
    '''python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
repo = "marsplan0624/affine-5gedzafcvg-queen"
rev = "8b365bdcd8f270c61fe633fcc95d536a93516e02"
print("[bootstrap-r337] DOWNLOAD marsplan-init start", repo, rev, flush=True)
path = snapshot_download(repo, revision=rev, token=token)
print(f"[bootstrap-r337] DOWNLOAD marsplan-init done -> {path}", flush=True)
open("/root/logs/marsplan_init.done", "w").write(path + "\\n")
open("/root/logs/tok_init.done", "w").write(path + "\\n")
print("[bootstrap-r337] DOWNLOAD teacher start", flush=True)
tpath = snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print(f"[bootstrap-r337] DOWNLOAD teacher done -> {tpath}", flush=True)
open("/root/logs/teacher.done", "w").write(tpath + "\\n")
PY''',
]

new = '''python - <<'PY'
# p3810_LOCAL_CACHE_SKIP: marsplan gated on pod IP — use local BASE cache
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
print("[bootstrap-r337] p3810_LOCAL_CACHE_SKIP using local BASE", path, flush=True)
open("/root/logs/marsplan_init.done", "w").write(path + "\\n")
open("/root/logs/tok_init.done", "w").write(path + "\\n")
print("[bootstrap-r337] DOWNLOAD teacher start", flush=True)
tpath = snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print(f"[bootstrap-r337] DOWNLOAD teacher done -> {tpath}", flush=True)
open("/root/logs/teacher.done", "w").write(tpath + "\\n")
PY'''

for old in candidates:
    if old in t:
        p.write_text(t.replace(old, new, 1))
        print("patched ok")
        raise SystemExit(0)

# Debug: show nearby lines
idx = t.find("DOWNLOAD marsplan-init")
print("marker idx", idx)
if idx != -1:
    print(repr(t[idx - 80 : idx + 400]))
raise SystemExit("download block not found")
