#!/bin/bash
# pm2 entry for the dedicated Cursor king box (pm2 `affine-kingcursor-tunnel`
# on the operator box). Same script as the /king box, different pod, local
# port, public path, key file and state file:
#   Cursor -> https://sn120.arbos.life/king-cursor/v1 -> CF tunnel
#   -> 127.0.0.1:9013 (this ssh -L loop) -> pod caddy :8080 -> fold_proxy -> vLLM
# Re-provision after a crown with the same env:
#   env $(grep -v '^#' tunnel-cursor.sh | grep '^export' | cut -d' ' -f2-) \
#     bash kingchat.sh provision
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export KINGCHAT_POD_NAME=affine-king-cursor
export KINGCHAT_LOCAL_PORT=9013
export KINGCHAT_PUBLIC_PATH=/king-cursor
export KINGCHAT_SECRETS="${HOME}/.affine-king-cursor.env"
export KINGCHAT_STATE="$HERE/state-cursor.json"
exec bash "$HERE/kingchat.sh" tunnel
