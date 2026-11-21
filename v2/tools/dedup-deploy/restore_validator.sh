#!/usr/bin/env bash
# Stage a complete c5252865 package with the reviewed physical compatibility
# overlay. No production path is read or written by this wrapper.
set -euo pipefail
if [[ $# -eq 2 || ( $# -eq 4 && "$3" == "--tar" ) ]]; then
  :
else
  printf 'usage: %s OLD_C5252865_SN120_V2 NEW_STAGING_DIR [--tar NEW_TARBALL]\n' "$0" >&2
  exit 2
fi
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
exec "${DEDUP_RESTORE_PY:-python3}" -I "$KIT/restore_validator.py" "$@" --kit "$KIT"
