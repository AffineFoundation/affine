#!/usr/bin/env bash
# Epoch-boundary PIT live cutover (CUTOVER-PLAN step 6).
# Preconditions: trainer idle (no mid-turn TRAINING-JOB without RESULT).
# Never embeds key material. Operator supplies PE_FLAG + delegated host certs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

PE_FLAG="${PE_FLAG:-}"
PE_STATE="${PE_STATE:-}"
PIT_STATE="${PIT_STATE:-${PE_STATE:+$PE_STATE/pit}}"
LIFE_ROOT="${LIFE_ROOT:-/home/const/subnet120-rewrite/state/mimo-mainnet-v1}"
PE_CONTROLLER_UNIT="${PE_CONTROLLER_UNIT:-}"

die() { echo "cutover_enable: $*" >&2; exit 2; }

[[ -n "$PE_FLAG" && -f "$PE_FLAG" ]] || die "set PE_FLAG to the live PROMOTION-GATE / PE flag JSON"
[[ -n "${AFFINE_PIT_HOST_KEY_FILE:-}" && -f "${AFFINE_PIT_HOST_KEY_FILE}" ]] || die "AFFINE_PIT_HOST_KEY_FILE required (0600)"
[[ -n "${AFFINE_PIT_HOST_CERT_FILE:-}" && -f "${AFFINE_PIT_HOST_CERT_FILE}" ]] || die "AFFINE_PIT_HOST_CERT_FILE required"
[[ -n "${AFFINE_PIT_ROOT_PUBKEY_HEX:-}" ]] || die "AFFINE_PIT_ROOT_PUBKEY_HEX required"
# On-box train (drive_item / run5090): required host-local paths — never defaulted in git.
[[ -n "${PIT_BUNDLE_JSON:-}" && -f "${PIT_BUNDLE_JSON}" ]] || die "PIT_BUNDLE_JSON required (train-cap-64k BUNDLE5090.json with trainer entry)"
[[ -n "${PIT_FP32_MASTER_INDEX:-}" && -d "${PIT_FP32_MASTER_INDEX}" ]] || die "PIT_FP32_MASTER_INDEX required (public k1/fp32-master index dir)"
# Fail closed if BUNDLE lacks trainer (PE_STATE/BUNDLE.json is probe-only).
export PIT_BUNDLE_JSON PIT_FP32_MASTER_INDEX
python3 - <<'PY' || die "PIT_BUNDLE_JSON missing trainer {key,sha256,bytes}"
import json, os, sys
b = json.load(open(os.environ["PIT_BUNDLE_JSON"]))
t = b.get("trainer") or {}
sys.exit(0 if t.get("key") and t.get("sha256") and "bytes" in t else 1)
PY

if [[ -z "$PE_STATE" ]]; then
  PE_STATE="$(dirname "$PE_FLAG")/state"
fi

LOCK="${CUT_OVER_LOCK:-}"
if [[ -z "$LOCK" ]]; then
  if [[ -n "$PE_STATE" ]]; then
    LOCK="$PE_STATE/cutover.lock"
  else
    LOCK="${PE_FLAG}.lock"
  fi
fi
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -x 9; then
  die "could not acquire cutover lock $LOCK"
fi

# TOCTOU: re-check trainer idle after lock.
export LIFE_ROOT
set +e
out="$(python3 -m ops.parallel_eval.pit.cutover_ops trainer-idle 2>&1)"
rc=$?
set -e
echo "$out"
if [[ "$rc" -eq 3 ]]; then
  echo "cutover_enable: REFUSED — $out" >&2
  exit 3
fi
[[ "$rc" -eq 0 ]] || die "trainer-idle check failed (rc=$rc)"

BAK="$(python3 -m ops.parallel_eval.pit.cutover_ops backup-flag)"
echo "cutover_enable: backed up flag -> $BAK"

export PE_FLAG
python3 -m ops.parallel_eval.pit.cutover_ops enable-patch

mkdir -p "$PE_STATE"
python3 - <<'PY' "$PE_STATE"
import sys
from pathlib import Path
from ops.parallel_eval.pit import cutover_ops
cutover_ops.write_warm_off(Path(sys.argv[1]))
print("cutover_enable: wrote WARM-OFF under", sys.argv[1])
PY


stop_pe_controller_process() {
  # Stop live pe_controller even when it runs as a bare process (not only systemd).
  local pidfile="${PE_CONTROLLER_PIDFILE:-${PE_STATE}/pe-controller.pid}"
  local pid=""
  if [[ -f "$pidfile" ]]; then
    pid="$(tr -d "[:space:]" < "$pidfile" || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      local cmd
      cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if echo "$cmd" | grep -Eq "pe_controller|pe-controller"; then
        echo "cutover_enable: stopping pe_controller pid=$pid ($cmd)"
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        return 0
      fi
      echo "cutover_enable: pidfile $pidfile pid=$pid cmdline mismatch; not killing" >&2
    fi
  fi
  # Cmdline match fallback (exact pe_controller module).
  while read -r pid cmd; do
    [[ -z "$pid" ]] && continue
    if echo "$cmd" | grep -Eq "(^|[[:space:]])pe_controller(\.py)?|[[:space:]]-m[[:space:]]+.*pe_controller"; then
      echo "cutover_enable: stopping pe_controller pid=$pid via cmdline match"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
  done < <(ps -eo pid=,args= 2>/dev/null | grep -E "pe_controller" | grep -v grep || true)
}

if [[ -n "$PE_CONTROLLER_UNIT" ]] && command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop "$PE_CONTROLLER_UNIT" 2>/dev/null || true
  echo "cutover_enable: stopped $PE_CONTROLLER_UNIT (best-effort)"
elif [[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/pe-controller.service" ]]; then
  systemctl --user stop pe-controller.service 2>/dev/null || true
  echo "cutover_enable: stopped pe-controller.service (best-effort)"
fi
stop_pe_controller_process || true

echo "cutover_enable: next step -> systemctl --user start affine-pit-host"
echo "cutover_enable: install unit template ops/parallel_eval/pit/scripts/affine-pit-host.service (placeholders)"
echo "cutover_enable: one-command revert -> ops/parallel_eval/pit/scripts/cutover_revert.sh"
