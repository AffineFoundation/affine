#!/usr/bin/env bash
# Epoch-boundary PIT live cutover (CUTOVER-PLAN step 6).
# Preconditions: trainer idle (no mid-turn TRAINING-JOB without RESULT);
# no live lane mid-probe; PE_CONTROLLER_PIDFILE points at the exact pe_controller PID.
# Never embeds key material. Operator supplies PE_FLAG + delegated host certs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
# PYTHONPATH must include the repo (for `ops`) AND resolve `subnet` without
# shadowing the release's ops/ package. Prefer an explicit operator PYTHONPATH
# that already contains subnet; otherwise prepend REPO_ROOT only.
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}"
elif [[ ":${PYTHONPATH}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"
fi

PE_FLAG="${PE_FLAG:-}"
PE_STATE="${PE_STATE:-}"
PIT_STATE="${PIT_STATE:-${PE_STATE:+$PE_STATE/pit}}"
LIFE_ROOT="${LIFE_ROOT:-}"
PE_CONTROLLER_UNIT="${PE_CONTROLLER_UNIT:-}"
# Exact PID only — never pattern-kill pe_controller by cmdline match.
PE_CONTROLLER_PIDFILE="${PE_CONTROLLER_PIDFILE:-}"

die() { echo "cutover_enable: $*" >&2; exit 2; }

[[ -n "$PE_FLAG" && -f "$PE_FLAG" ]] || die "set PE_FLAG to the live PROMOTION-GATE / PE flag JSON"
[[ -n "${AFFINE_PIT_HOST_KEY_FILE:-}" && -f "${AFFINE_PIT_HOST_KEY_FILE}" ]] || die "AFFINE_PIT_HOST_KEY_FILE required (0600)"
[[ -n "${AFFINE_PIT_HOST_CERT_FILE:-}" && -f "${AFFINE_PIT_HOST_CERT_FILE}" ]] || die "AFFINE_PIT_HOST_CERT_FILE required"
[[ -n "${AFFINE_PIT_ROOT_PUBKEY_HEX:-}" ]] || die "AFFINE_PIT_ROOT_PUBKEY_HEX required"
# On-box train (drive_item / run5090): required host-local paths — never defaulted in git.
[[ -n "${PIT_BUNDLE_JSON:-}" && -f "${PIT_BUNDLE_JSON}" ]] || die "PIT_BUNDLE_JSON required (train-cap-64k BUNDLE5090.json with trainer entry)"
[[ -n "${PIT_FP32_MASTER_INDEX:-}" && -d "${PIT_FP32_MASTER_INDEX}" ]] || die "PIT_FP32_MASTER_INDEX required (public k1/fp32-master index dir)"
# GREEN upload (box → R2 scoped PUT) needs the lifecycle bucket config + subnet package.
[[ -n "${PIT_LIFE_CFG:-}" && -f "${PIT_LIFE_CFG}" ]] || die "PIT_LIFE_CFG required (lifecycle ACTIVE.private.json with bucket; never commit host paths)"
[[ -n "${PE_CONTROLLER_PIDFILE}" ]] || die "PE_CONTROLLER_PIDFILE required (exact pe_controller PID; no cmdline pattern kill)"
[[ -f "${PE_CONTROLLER_PIDFILE}" ]] || die "PE_CONTROLLER_PIDFILE not found: ${PE_CONTROLLER_PIDFILE}"

# Fail closed if BUNDLE lacks trainer (PE_STATE/BUNDLE.json is probe-only).
export PIT_BUNDLE_JSON PIT_FP32_MASTER_INDEX PIT_LIFE_CFG PE_CONTROLLER_PIDFILE
python3 - <<'PY' || die "PIT_BUNDLE_JSON missing trainer {key,sha256,bytes}"
import json, os, sys
b = json.load(open(os.environ["PIT_BUNDLE_JSON"]))
t = b.get("trainer") or {}
sys.exit(0 if t.get("key") and t.get("sha256") and "bytes" in t else 1)
PY

# Preflight: subnet importable + PIT_LIFE_CFG readable; ops must still resolve from release.
python3 - <<'PY' || die "preflight failed: import subnet / ops.parallel_eval.pit (check PYTHONPATH; do not shadow release ops/)"
import json, os, sys
cfg = os.environ["PIT_LIFE_CFG"]
doc = json.load(open(cfg))
if not isinstance(doc.get("bucket"), (dict, str)):
    print("PIT_LIFE_CFG missing bucket", file=sys.stderr); sys.exit(1)
try:
    import subnet  # noqa: F401
except Exception as ex:
    print("import subnet failed: %s" % ex, file=sys.stderr); sys.exit(1)
try:
    import ops.parallel_eval.pit  # noqa: F401
except Exception as ex:
    print("import ops.parallel_eval.pit failed: %s" % ex, file=sys.stderr); sys.exit(1)
print("cutover_enable: preflight ok (subnet + ops + PIT_LIFE_CFG)")
PY

if [[ -z "$PE_STATE" ]]; then
  PE_STATE="$(dirname "$PE_FLAG")/state"
fi
if [[ -z "$LIFE_ROOT" ]]; then
  LIFE_ROOT="${PE_LIFE:-${PE_LIFE_STATE:-}}"
fi
[[ -n "$LIFE_ROOT" ]] || die "set LIFE_ROOT or PE_LIFE / PE_LIFE_STATE (lifecycle root; never hard-code host paths in git)"

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
export LIFE_ROOT PE_STATE
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

# Refuse stopping pe_controller while a live lane is mid-probe / mid-serve.
set +e
out="$(python3 -m ops.parallel_eval.pit.cutover_ops mid-probe-check 2>&1)"
rc=$?
set -e
echo "$out"
if [[ "$rc" -eq 3 ]]; then
  echo "cutover_enable: REFUSED — mid-probe lane; wait or abort ($out)" >&2
  exit 3
fi
[[ "$rc" -eq 0 ]] || die "mid-probe-check failed (rc=$rc)"

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
  # Exact PID from PE_CONTROLLER_PIDFILE only — never cmdline pattern kill.
  local pidfile="$PE_CONTROLLER_PIDFILE"
  local pid=""
  pid="$(tr -d '[:space:]' < "$pidfile" || true)"
  [[ -n "$pid" ]] || die "PE_CONTROLLER_PIDFILE empty: $pidfile"
  [[ "$pid" =~ ^[0-9]+$ ]] || die "PE_CONTROLLER_PIDFILE not a PID: $pid"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "cutover_enable: pe_controller pid=$pid already gone (ok)"
    return 0
  fi
  local cmd
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if ! echo "$cmd" | grep -Eq "pe_controller|pe-controller"; then
    die "PE_CONTROLLER_PIDFILE pid=$pid cmdline mismatch (got: $cmd); refuse kill"
  fi
  echo "cutover_enable: stopping pe_controller pid=$pid ($cmd)"
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  return 0
}

if [[ -n "$PE_CONTROLLER_UNIT" ]] && command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop "$PE_CONTROLLER_UNIT" 2>/dev/null || true
  echo "cutover_enable: stopped $PE_CONTROLLER_UNIT (best-effort)"
elif [[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/pe-controller.service" ]]; then
  systemctl --user stop pe-controller.service 2>/dev/null || true
  echo "cutover_enable: stopped pe-controller.service (best-effort)"
fi
stop_pe_controller_process

echo "cutover_enable: next step -> systemctl --user start affine-pit-host"
echo "cutover_enable: install unit template ops/parallel_eval/pit/scripts/affine-pit-host.service (placeholders)"
echo "cutover_enable: required env: PIT_BUNDLE_JSON PIT_FP32_MASTER_INDEX PIT_LIFE_CFG PE_CONTROLLER_PIDFILE"
echo "cutover_enable: PYTHONPATH must resolve subnet + ops (do not shadow release ops/)"
echo "cutover_enable: one-command revert -> ops/parallel_eval/pit/scripts/cutover_revert.sh"
