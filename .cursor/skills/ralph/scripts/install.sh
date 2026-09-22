#!/usr/bin/env bash
# Install the ralph runner to ~/.ralph/ralph.sh (machine-wide, not per-repo).
# Never overwrites an existing runner — a live loop may be using it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/ralph.sh"
DEST="${HOME}/.ralph/ralph.sh"

[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }

mkdir -p "${HOME}/.ralph"
if [[ -f "$DEST" ]]; then
  if cmp -s "$SRC" "$DEST"; then
    echo "ralph.sh already current at $DEST"
  else
    echo "ralph.sh exists at $DEST and differs from this skill — not overwriting"
    echo "to update: stop all loops, rm $DEST, re-run this script"
  fi
  chmod +x "$DEST"
  exit 0
fi

cp "$SRC" "$DEST"
chmod +x "$DEST"
echo "installed $DEST"
