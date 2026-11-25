#!/usr/bin/env bash
# Old-code loop after the reviewed dedup rollback.
# Usage: DEDUP_OLD_V2=OLD-CHECKOUT/v2 rollback-loop.sh --state-dir DIR --stack MODULE:FUNCTION
# Keep this kit outside the checkout that switches commits. Every CLI entry
# point, including status/end-round/run, goes through the hashed overlay.
set -euo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
: "${DEDUP_OLD_V2:?set DEDUP_OLD_V2 to OLD-CHECKOUT/v2}"
export DEDUP_OLD_V2
LOOP=${DEDUP_OLD_LOOP_SCRIPT:-$DEDUP_OLD_V2/systemd/sn120-v2-run-loop.sh}
exec bash "$LOOP" "$KIT/rollback-python" "$@"
