#!/bin/bash
# Operator-side control for the PRIVATE king-chat pod (operator's own Cursor
# model; not part of scoring, not the public affine.io chat).
#
#   kingchat.sh provision   # upload pod_bootstrap.sh + secrets, (re)launch it.
#                           # Re-run after a crown: it reads the current king
#                           # digest from the public snapshot every time.
#   kingchat.sh tunnel      # foreground ssh -L loop (pm2 runs this)
#   kingchat.sh route       # add the /king/* ingress to the affine-sn120
#                           # Cloudflare tunnel (idempotent)
#   kingchat.sh status      # pod health + end-to-end check through Cloudflare
#   kingchat.sh logs        # tail the pod's vllm log
#
# Path: Cursor -> https://sn120.arbos.life/king/v1 -> Cloudflare tunnel
#   (pm2 affine-tunnel on this box) -> 127.0.0.1:LOCAL_PORT (pm2 ssh -L loop)
#   -> pod 127.0.0.1:8080 caddy (strips /king) -> pod vLLM :8000 (--api-key).
#
# State:  ops/king-chat/state.json   {"ssh": "root@HOST -p PORT", ...}
# Secret: ~/.affine-king-chat.env    KING_API_KEY=... (0600; the Cursor key)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STATE=$HERE/state.json
SECRETS=$HOME/.affine-king-chat.env
POD_NAME=${KINGCHAT_POD_NAME:-affine-chat-const}
LOCAL_PORT=${KINGCHAT_LOCAL_PORT:-9012}
PUBLIC_HOST=${KINGCHAT_PUBLIC_HOST:-sn120.arbos.life}
CF_TUNNEL_ID=${KINGCHAT_CF_TUNNEL_ID:-6cfb1dc0-2ef8-41ba-8245-2b581abbbd7f}
SNAPSHOT_URL=https://affine.io/api/v1/snapshot
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/known_hosts"
          -o ConnectTimeout=15 -o LogLevel=ERROR)

pod_ssh() {  # -> "root@HOST -p PORT"; resolves via lium and caches in state.json
  local ssh
  ssh=$(lium ps --format json 2>/dev/null | python3 -c "
import json, sys
for r in json.load(sys.stdin):
    if r.get('name') == '$POD_NAME':
        print(r['ssh_cmd'].removeprefix('ssh ').strip()); break") || true
  if [ -z "$ssh" ] && [ -f "$STATE" ]; then
    ssh=$(python3 -c "import json; print(json.load(open('$STATE')).get('ssh',''))")
  fi
  [ -n "$ssh" ] || { echo "no pod named $POD_NAME (lium ps)"; exit 1; }
  python3 - "$STATE" "$ssh" "$POD_NAME" <<'PY'
import json, sys, os, time
p, ssh, pod = sys.argv[1:4]
st = json.load(open(p)) if os.path.exists(p) else {}
st.update(ssh=ssh, pod=pod, updated_at=time.time())
json.dump(st, open(p, "w"), indent=1)
PY
  echo "$ssh"
}
ssh_pod() {  # ssh_pod "cmd"
  local ssh; ssh=$(pod_ssh)
  # shellcheck disable=SC2086
  ssh "${SSH_OPTS[@]}" $ssh "$@"
}
host_port() {  # -> "HOST PORT" (user stripped)
  local ssh target; ssh=$(pod_ssh); target=${ssh%% *}
  echo "${target#*@} ${ssh##*-p }"
}

king_digest() {
  curl -fsS "$SNAPSHOT_URL" | python3 -c "import sys,json; print(json.load(sys.stdin)['king']['revision'])"
}

cmd_provision() {
  if [ ! -f "$SECRETS" ]; then
    (umask 077; echo "KING_API_KEY=$(openssl rand -hex 24)" > "$SECRETS")
    echo "generated new API key in $SECRETS"
  fi
  # shellcheck disable=SC1090
  source "$SECRETS"
  local digest; digest=$(king_digest)
  echo "current king digest: $digest"
  read -r host port <<<"$(host_port)"
  scp "${SSH_OPTS[@]}" -P "$port" "$HERE/pod_bootstrap.sh" "root@$host:/tmp/pod_bootstrap.sh"
  # Secrets over stdin into a 0600 file — never on a command line.
  printf 'KING_API_KEY=%s\nKING_DIGEST=%s\n' "$KING_API_KEY" "$digest" | ssh_pod \
    'mkdir -p /root/king-chat /root/logs && umask 077 && cat > /root/king-chat/.env && \
     mv /tmp/pod_bootstrap.sh /root/king-chat/pod_bootstrap.sh && \
     chmod +x /root/king-chat/pod_bootstrap.sh && \
     { pkill -xf "bash pod_bootstrap.sh" || true; } && \
     cd /root/king-chat && (nohup bash pod_bootstrap.sh </dev/null >> /root/logs/bootstrap.log 2>&1 &) && \
     echo LAUNCHED'
  echo "bootstrap running; follow with: $0 logs  (or ssh ... tail -f /root/logs/bootstrap.log)"
}

cmd_tunnel() {
  # Loop forever; ssh exits on link death and we redial (pm2 keeps the loop).
  while true; do
    read -r host port <<<"$(host_port 2>/dev/null || echo "")"
    if [ -z "${host:-}" ]; then
      echo "$(date -u +%FT%TZ) no pod; retry in 30s" >&2; sleep 30; continue
    fi
    ssh -N "${SSH_OPTS[@]}" -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:8080" -p "$port" "root@$host"
    echo "$(date -u +%FT%TZ) tunnel dropped (exit $?), redialing in 5s" >&2
    sleep 5
  done
}

cmd_route() {
  # Insert {hostname: PUBLIC_HOST, path: ^/king(/|$)} -> LOCAL_PORT ahead of the
  # existing rules of the affine-sn120 tunnel. Reads CLOUDFLARE_* from the
  # validator env snapshot (the same token that runs the tunnel).
  # (grep, not source: the snapshot carries non-shell lines.)
  CLOUDFLARE_ACCOUNT_ID=$(grep -E '^(export )?CLOUDFLARE_ACCOUNT_ID=' "$HOME/.affine-validator.env" | tail -1 | cut -d= -f2-)
  CLOUDFLARE_API_TOKEN=$(grep -E '^(export )?CLOUDFLARE_API_TOKEN=' "$HOME/.affine-validator.env" | tail -1 | cut -d= -f2-)
  export CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_TOKEN
  python3 - "$CF_TUNNEL_ID" "$PUBLIC_HOST" "$LOCAL_PORT" <<'PY'
import json, os, sys, urllib.request
tunnel, host, port = sys.argv[1], sys.argv[2], sys.argv[3]
acct, tok = os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"]
url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/cfd_tunnel/{tunnel}/configurations"
hdr = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
cur = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr)))["result"]["config"]
rule = {"hostname": host, "path": "^/king(/|$)", "service": f"http://127.0.0.1:{port}"}
ingress = [r for r in cur["ingress"] if not (r.get("hostname") == host and r.get("path"))]
ingress.insert(0, rule)
cur["ingress"] = ingress
req = urllib.request.Request(url, data=json.dumps({"config": cur}).encode(), headers=hdr, method="PUT")
res = json.load(urllib.request.urlopen(req))
print("ok" if res["success"] else res["errors"], json.dumps(res["result"]["config"]["ingress"], indent=1))
PY
}

cmd_status() {
  # shellcheck disable=SC1090
  source "$SECRETS"
  echo "pod: $(pod_ssh)"
  ssh_pod 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; \
           curl -s -m 5 http://127.0.0.1:8000/health -o /dev/null -w "vllm /health %{http_code}\n"; \
           curl -s -m 5 http://127.0.0.1:8080/king/v1/models -H "Authorization: Bearer '"$KING_API_KEY"'" | head -c 300; echo; \
           tail -2 /root/logs/bootstrap.log 2>/dev/null; tail -2 /root/logs/vllm.log 2>/dev/null | cut -c1-200'
  echo "local forward 127.0.0.1:$LOCAL_PORT: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$LOCAL_PORT/king/v1/models -H "Authorization: Bearer $KING_API_KEY")"
  echo "public https://$PUBLIC_HOST/king/v1/models: $(curl -s -m 15 -o /dev/null -w '%{http_code}' https://$PUBLIC_HOST/king/v1/models -H "Authorization: Bearer $KING_API_KEY")"
}

cmd_logs() { ssh_pod 'tail -n 40 -f /root/logs/vllm.log'; }

case "${1:-}" in
  provision) cmd_provision ;;
  tunnel) cmd_tunnel ;;
  route) cmd_route ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
