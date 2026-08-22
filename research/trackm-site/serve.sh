#!/usr/bin/env bash
# (Re)start the Track M site: local http.server + cloudflared quick tunnel.
# Prints the public trycloudflare URL. Idempotent — kills old instances first.
set -euo pipefail
PORT=8613
DIR="$(cd "$(dirname "$0")" && pwd)"

pkill -f "http.server $PORT" 2>/dev/null || true
pkill -f "cloudflared tunnel --url http://localhost:$PORT" 2>/dev/null || true
pkill -f "trackm-site/refresh_data.sh" 2>/dev/null || true
sleep 1

cd "$DIR"
python3 generate_data.py >/dev/null   # immediate refresh (log or mock)
nohup python3 -m http.server "$PORT" --bind 127.0.0.1 > /tmp/trackm_http.log 2>&1 &
disown
nohup cloudflared tunnel --url "http://localhost:$PORT" > /tmp/trackm_tunnel.log 2>&1 &
disown
nohup bash "$DIR/refresh_data.sh" > /dev/null 2>&1 &
disown

echo "waiting for tunnel…"
for _ in $(seq 1 30); do
  URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" /tmp/trackm_tunnel.log | head -1 || true)
  [ -n "${URL:-}" ] && break
  sleep 1
done
echo "local:  http://127.0.0.1:$PORT"
echo "public: ${URL:-<tunnel failed — see /tmp/trackm_tunnel.log>}"
