#!/usr/bin/env bash
# One-command PIT live revert (CUTOVER-PLAN step 6 revert).
# Restores trainer dequeue / warm / standing lanes via pe_live + pe_lifecycle.
# Stops affine-pit-host when the unit is present. Does not touch weight nonces.
set -euo pipefail

PE_FLAG="${PE_FLAG:-}"
die() { echo "cutover_revert: $*" >&2; exit 2; }
[[ -n "$PE_FLAG" && -f "$PE_FLAG" ]] || die "set PE_FLAG to the live PROMOTION-GATE / PE flag JSON"

# Stop host daemon if running (best-effort).
if systemctl --user is-active --quiet affine-pit-host.service 2>/dev/null; then
  systemctl --user stop affine-pit-host.service || true
  echo "cutover_revert: stopped affine-pit-host"
fi

python3 - <<'PY' "$PE_FLAG"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
pe = doc.setdefault("parallel_eval", {})
pit = pe.setdefault("pit", {})
pit.update({
    "enabled": False,
    "mode": pit.get("mode") or "live",
    "cutover": False,
    "revert": True,
})
tmp = p.with_suffix(p.suffix + ".tmp")
tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
tmp.replace(p)
print("cutover_revert: pit.revert=true, enabled=false in", p)
print("cutover_revert: pe_live.dispatch_tick resumes; pe_lifecycle trainer path resumes")
PY
