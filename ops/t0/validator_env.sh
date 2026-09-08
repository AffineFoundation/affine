# Source this to run operator scripts under the SAME secrets the live
# validator has (~/.affine-validator.env, see ops/run_validator.sh).
#
# Why not `.env`: the repo .env carries the R2 / Cloudflare keys but NOT
# HF_TOKEN or AFFINE_EVAL_TOKEN. provisioner._env_file_contents() writes the
# pod .eval_env from whatever is in the process env, so a redeploy driven
# from `.env` alone would ship a pod env without HF_TOKEN / AFFINE_EVAL_TOKEN
# and bootstrap would fail closed. Snapshot first, .env only fills gaps.
#
#     source ops/t0/validator_env.sh
_AFFINE_ENV_FILE="${AFFINE_VALIDATOR_ENV:-/home/const/.affine-validator.env}"
if [[ ! -r "$_AFFINE_ENV_FILE" ]]; then
  echo "validator_env.sh: missing readable env snapshot: $_AFFINE_ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi
eval "$(/home/const/subnet120/.venv/bin/python - "$_AFFINE_ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path
valid = re.compile(r"^[A-Z][A-Z0-9_]*$")   # uppercase only: skips pm2 metadata
# Shell-owned variables the snapshot also carries. Exporting PATH from it
# drops the caller's `.venv/bin` (found 2026-09-02: `python` vanished right
# after sourcing); the rest would re-home the shell. Secrets only.
skip = {"PATH", "HOME", "PWD", "OLDPWD", "SHELL", "SHLVL", "TERM", "USER",
        "LOGNAME", "LANG", "LC_ALL", "VIRTUAL_ENV", "PYTHONPATH",
        "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK"}
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    if valid.match(key) and key not in skip:
        print(f"export {key}={shlex.quote(val)}")
PY
)"
# .env fills anything the snapshot lacks (e.g. DISCORD_*); never overrides.
if [[ -r /home/const/subnet120/.env ]]; then
  while IFS= read -r _l; do
    [[ "$_l" =~ ^[A-Z][A-Z0-9_]*= ]] || continue
    _k="${_l%%=*}"
    [[ -n "${!_k:-}" ]] || export "$_l"
  done < /home/const/subnet120/.env
fi
unset _AFFINE_ENV_FILE _l _k
