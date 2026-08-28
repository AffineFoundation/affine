#!/usr/bin/env bash
# Regenerate data.json from trackM_status.log every 2 minutes.
# Started by serve.sh via nohup; safe to run standalone too.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
while true; do
  python3 "$DIR/generate_data.py" -o "$DIR/data.json" >> /tmp/trackm_refresh.log 2>&1
  sleep 120
done
