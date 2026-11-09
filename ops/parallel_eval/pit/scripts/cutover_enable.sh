#!/usr/bin/env bash
# Epoch-boundary PIT live cutover (CUTOVER-PLAN step 6).
# Preconditions: trainer idle (no mid-turn TRAINING-JOB without RESULT).
# Never embeds key material. Operator supplies PE_FLAG + delegated host certs.
set -euo pipefail

PE_FLAG="${PE_FLAG:-}"
PE_STATE="${PE_STATE:-}"
PIT_STATE="${PIT_STATE:-${PE_STATE:+$PE_STATE/pit}}"
LIFE_ROOT="${LIFE_ROOT:-/home/const/subnet120-rewrite/state/mimo-mainnet-v1}"

die() { echo "cutover_enable: $*" >&2; exit 2; }

[[ -n "$PE_FLAG" && -f "$PE_FLAG" ]] || die "set PE_FLAG to the live PROMOTION-GATE / PE flag JSON"
[[ -n "${AFFINE_PIT_HOST_KEY_FILE:-}" && -f "${AFFINE_PIT_HOST_KEY_FILE}" ]] || die "AFFINE_PIT_HOST_KEY_FILE required (0600)"
[[ -n "${AFFINE_PIT_HOST_CERT_FILE:-}" && -f "${AFFINE_PIT_HOST_CERT_FILE}" ]] || die "AFFINE_PIT_HOST_CERT_FILE required"
[[ -n "${AFFINE_PIT_ROOT_PUBKEY_HEX:-}" ]] || die "AFFINE_PIT_ROOT_PUBKEY_HEX required"

# Refuse mid-turn trainer.
python3 - <<'PY' "$LIFE_ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "epochs"
latest = None
for p in sorted(root.glob("*"), key=lambda x: int(x.name) if x.name.isdigit() else -1):
    if p.name.isdigit():
        latest = p
if latest is None:
    print("cutover_enable: no lifecycle epochs found; continuing", flush=True)
    raise SystemExit(0)
job = latest / "TRAINING-JOB.json"
res = latest / "TRAINING-RESULT.json"
if job.exists() and not res.exists():
    print("cutover_enable: REFUSED — trainer mid-turn at e%s (JOB without RESULT)" % latest.name, flush=True)
    raise SystemExit(3)
print("cutover_enable: trainer idle at e%s" % latest.name, flush=True)
PY

python3 - <<'PY' "$PE_FLAG"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
pe = doc.setdefault("parallel_eval", {})
pe["enabled"] = True
pit = pe.setdefault("pit", {})
pit.update({
    "enabled": True,
    "mode": "live",
    "cutover": True,
    "revert": False,
    "max_boxes": int(pit.get("max_boxes") or 5),
    "master_upload": "after_green",
    "root_mode": "delegated_host",
})
tmp = p.with_suffix(p.suffix + ".tmp")
tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
tmp.replace(p)
print("cutover_enable: wrote pit live cutover into", p)
PY

mkdir -p "${PIT_STATE:-/tmp/pit-state}"
echo "cutover_enable: start affine-pit-host (systemd --user) after this script"
echo "cutover_enable: one-command revert -> ops/parallel_eval/pit/scripts/cutover_revert.sh"
