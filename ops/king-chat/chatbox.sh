#!/bin/bash
# Operator-side control for the PUBLIC king chat box — the validator-managed
# `affine-chat` pod (AFFINE_ROLE=chat, evalsrv/chatsrv.py) published at
# https://chat.affine.io/v1 for Cursor and the affine.io chat page.
#
#   chatbox.sh status      pod, king served vs king in state.json, public URL
#   chatbox.sh deploy      push this tree's affine/ + .chat_env to the pod and
#                          restart chatsrv (code change; ~1 min dark, weights
#                          stay on disk; the validator's 5-strike terminate
#                          threshold is 5 min — do NOT use redeploy_pods.py
#                          for this pod, its tar upload happens while dark)
#   chatbox.sh overrides   push .chat_env only + restart chatsrv
#   chatbox.sh route       Cloudflare: tunnel ingress for CHATBOX_PUBLIC_HOST
#                          -> 127.0.0.1:CHATBOX_LOCAL_PORT (box token) and the
#                          proxied CNAME to the tunnel (temporary zone-scoped
#                          DNS token minted from the box token, deleted after)
#   chatbox.sh smoke [URL] end-to-end checks (ops/king-chat/smoke.py)
#   chatbox.sh watch       pm2 loop: re-push .chat_env after a re-rent, kick a
#                          chatsrv stuck in state=error, log king/state changes
#   chatbox.sh cursor      print the Cursor setup snippet
#   chatbox.sh logs        tail the pod's chatsrv log
#
# Who owns what: the VALIDATOR rents/keeps the pod and runs the ssh -L forward
# on 127.0.0.1:9002 (affine/affine/provisioner.py ChatMachineManager). The
# POD follows the king by itself (chatsrv polls the public snapshot and swaps
# the vLLM slot). This script only ships code/overrides, publishes the
# hostname and watches. Nothing here touches the validator, the toml, or
# any other pod.
#
# Config: ops/king-chat/chatbox.env (tracked, no secrets).
# pm2:    pm2 start ops/king-chat/chatbox.sh --name affine-chatbox-watch \
#             --interpreter bash -- watch && pm2 save
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${CHATBOX_PYTHON:-$REPO/.venv/bin/python}
STATE_JSON=$REPO/affine/state/state.json
ENV_FILE=${CHATBOX_ENV:-$HERE/chatbox.env}
VALIDATOR_ENV=${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
: "${CHATBOX_PUBLIC_HOST:?}" "${CHATBOX_LOCAL_PORT:?}" "${CHATBOX_CF_TUNNEL_ID:?}"
LOCAL_URL="http://127.0.0.1:${CHATBOX_LOCAL_PORT}"
PUBLIC_URL="https://${CHATBOX_PUBLIC_HOST}"
REMOTE_DIR=/root/affine
# Host keys churn on re-rent (same host:port, new pod) — same policy as the
# provisioner's _SSH_BASE.
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15 -o ServerAliveInterval=30 -o LogLevel=ERROR)

log() { echo "[chatbox] $(date -u +%FT%TZ) $*"; }

pod_ssh() {  # -> "root@HOST -p PORT" from the validator's state (read-only)
  "$PY" - "$STATE_JSON" <<'PY'
import json, sys
print((json.load(open(sys.argv[1])).get("chat_machine") or {}).get("ssh", ""))
PY
}
king_revision() {
  "$PY" - "$STATE_JSON" <<'PY'
import json, sys
print((json.load(open(sys.argv[1])).get("king") or {}).get("revision", ""))
PY
}
ssh_pod() {  # ssh_pod "remote command" [stdin]
  local ssh; ssh=$(pod_ssh)
  [ -n "$ssh" ] || { echo "no chat pod in $STATE_JSON" >&2; return 1; }
  # shellcheck disable=SC2086
  ssh "${SSH_OPTS[@]}" $ssh "$1"
}
scp_pod() {  # scp_pod local remote
  local ssh host port; ssh=$(pod_ssh); host=${ssh%% *}; port=${ssh##*-p }
  scp "${SSH_OPTS[@]}" -P "$port" "$1" "$host:$2"
}
health() { curl -s -m 10 "$LOCAL_URL/health"; }

# Only the AFFINE_CHAT_* lines travel to the pod.
chat_env_contents() { grep -E '^AFFINE_CHAT_[A-Z_]+=' "$ENV_FILE" | sed 's/^/export /'; }

push_overrides() {
  chat_env_contents | ssh_pod "umask 077 && cat > $REMOTE_DIR/.chat_env && echo CHAT_ENV_OK"
}

restart_chatsrv() {
  # Same stop sequence as MachineManager.redeploy; bootstrap.sh is the
  # supervisor, so it is relaunched too (it re-sources .eval_env/.chat_env).
  ssh_pod "pkill -f '[e]valsrv/bootstrap.sh' || true; pkill -f '[p]ython -m evalsrv' || true; sleep 3; pkill -9 -f '[v]llm serve' || true; cd $REMOTE_DIR && (nohup bash evalsrv/bootstrap.sh </dev/null >> /root/bootstrap.log 2>&1 &) && echo RESTARTED"
}

cmd_deploy() {
  local ssh; ssh=$(pod_ssh); [ -n "$ssh" ] || { echo "no chat pod"; exit 1; }
  log "packing affine/ from $REPO"
  local tar; tar=$("$PY" - "$REPO/affine" <<'PY'
import sys
from pathlib import Path
from affine.provisioner import _pack_affine_tar
print(_pack_affine_tar(Path(sys.argv[1])))
PY
)
  log "uploading $(du -h "$tar" | cut -f1) to $ssh (old server keeps serving meanwhile)"
  scp_pod "$tar" /tmp/affine-src.tar.gz; rm -f "$tar"
  ssh_pod "mkdir -p $REMOTE_DIR && tar xzf /tmp/affine-src.tar.gz -C $REMOTE_DIR && rm -f /tmp/affine-src.tar.gz && test \$(find $REMOTE_DIR/affine $REMOTE_DIR/evalsrv -name '*.py' | wc -l) -ge 10 && echo EXTRACT_OK"
  cmd_overrides
}

cmd_overrides() {
  log "writing .chat_env"; push_overrides
  log "restarting chatsrv (vLLM reloads the king from disk; /health stays ok, state=loading)"
  restart_chatsrv
  for _ in $(seq 1 30); do
    sleep 5
    if h=$(health) && [ -n "$h" ]; then log "chatsrv up: $h"; return 0; fi
  done
  log "WARNING chatsrv not answering on $LOCAL_URL after 150 s"; return 1
}

cf_env() {  # exports CLOUDFLARE_ACCOUNT_ID/API_TOKEN from the validator env snapshot
  # (grep, not source: the snapshot carries non-shell lines.)
  CLOUDFLARE_ACCOUNT_ID=$(grep -E '^(export )?CLOUDFLARE_ACCOUNT_ID=' "$VALIDATOR_ENV" | tail -1 | cut -d= -f2- | tr -d "\"'")
  CLOUDFLARE_API_TOKEN=$(grep -E '^(export )?CLOUDFLARE_API_TOKEN=' "$VALIDATOR_ENV" | tail -1 | cut -d= -f2- | tr -d "\"'")
  export CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_TOKEN
}

cmd_route() {
  cf_env
  "$PY" - "$CHATBOX_CF_TUNNEL_ID" "$CHATBOX_PUBLIC_HOST" "$CHATBOX_LOCAL_PORT" "${CHATBOX_ZONE_NAME:-affine.io}" <<'PY'
import json, os, sys, time, urllib.request, urllib.error
tunnel, host, port, zone_name = sys.argv[1:5]
acct, tok = os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"]
API = "https://api.cloudflare.com/client/v4"

def call(method, path, body=None, token=tok):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                                 method=method)
    try:
        res = json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:300]}")
    if not res.get("success"):
        raise SystemExit(f"{method} {path} -> {res.get('errors')}")
    return res["result"]

# 1. Tunnel ingress (box token has tunnel config rights). Idempotent.
cfg_path = f"/accounts/{acct}/cfd_tunnel/{tunnel}/configurations"
cfg = call("GET", cfg_path)["config"]
rule = {"hostname": host, "service": f"http://127.0.0.1:{port}"}
ingress = [r for r in cfg["ingress"] if r.get("hostname") != host]
ingress.insert(0, rule)
cfg["ingress"] = ingress
call("PUT", cfg_path, {"config": cfg})
print("ingress ok:", json.dumps(rule))

# 2. DNS. The box token cannot edit DNS; mint a zone-scoped DNS token that
#    lives one hour, use it for the CNAME, delete it right away.
zone = call("GET", f"/zones?name={zone_name}")[0]["id"]
perm = {g["name"]: g["id"] for g in call("GET", f"/accounts/{acct}/tokens/permission_groups")}
tmp = call("POST", f"/accounts/{acct}/tokens", {
    "name": f"chatbox-dns-temp-{int(time.time())}",
    "policies": [{"effect": "allow",
                  "resources": {f"com.cloudflare.api.account.zone.{zone}": "*"},
                  "permission_groups": [{"id": perm["DNS Write"]}]}],
    "expires_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)),
})
tmp_id, tmp_val = tmp["id"], tmp["value"]
try:
    target = f"{tunnel}.cfargotunnel.com"
    recs = call("GET", f"/zones/{zone}/dns_records?name={host}", token=tmp_val)
    want = {"type": "CNAME", "name": host, "content": target, "proxied": True, "ttl": 1}
    if not recs:
        call("POST", f"/zones/{zone}/dns_records", want, token=tmp_val)
        print("dns created:", host, "->", target)
    elif recs[0]["type"] != "CNAME" or recs[0]["content"] != target or not recs[0]["proxied"]:
        call("PUT", f"/zones/{zone}/dns_records/{recs[0]['id']}", want, token=tmp_val)
        print("dns updated:", host, "->", target)
    else:
        print("dns already correct:", host, "->", target)
finally:
    call("DELETE", f"/accounts/{acct}/tokens/{tmp_id}")
    print("temporary DNS token deleted")
PY
}

cmd_status() {
  local ssh king h
  ssh=$(pod_ssh); king=$(king_revision)
  echo "pod (state.json chat_machine): ${ssh:-none}"
  echo "king (state.json):             ${king:0:12}"
  h=$(health || true)
  echo "local $LOCAL_URL/health:     ${h:-unreachable}"
  echo "public $PUBLIC_URL/health:   $(curl -s -m 20 "$PUBLIC_URL/health" || echo unreachable)"
  echo "public /v1/models (demo key): $(curl -s -m 20 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${AFFINE_CHAT_PUBLIC_KEY:-}" "$PUBLIC_URL/v1/models")"
  if [ -n "$ssh" ]; then
    ssh_pod 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; test -f /root/affine/.chat_env && echo ".chat_env present" || echo ".chat_env MISSING"; tail -1 /root/logs/evalsrv.log | cut -c1-160' 2>/dev/null || true
  fi
}

cmd_smoke() { "$PY" "$HERE/smoke.py" --base "${1:-$PUBLIC_URL/v1}" --key "${AFFINE_CHAT_PUBLIC_KEY:-}"; }

cmd_logs() { ssh_pod 'tail -n 40 -f /root/logs/evalsrv.log'; }

cmd_cursor() {
  cat <<EOF
Cursor -> Settings -> Models:
  1. Turn every built-in model off, click "Add model", name it:  affine-king
  2. OpenAI API Key:            ${AFFINE_CHAT_PUBLIC_KEY:-<key>}
  3. Override OpenAI Base URL:  ${PUBLIC_URL}/v1
  4. Click Verify, then pick "affine-king" in the chat model menu.
Same thing as curl:
  curl ${PUBLIC_URL}/v1/chat/completions \\
    -H "Authorization: Bearer ${AFFINE_CHAT_PUBLIC_KEY:-<key>}" -H "Content-Type: application/json" \\
    -d '{"model":"affine-king","messages":[{"role":"user","content":"hello"}]}'
EOF
}

cmd_watch() {
  # State machine kept in shell vars; one log line per change. Actions:
  #  - new pod ssh (validator re-rented) -> once /health answers, push
  #    .chat_env + restart if the file is missing (fresh rental has none)
  #  - state=error for ERR_TICKS consecutive ticks -> restart chatsrv, at most
  #    once per RESTART_COOLDOWN_S (stale-code loops like 2026-09-11, where
  #    the in-memory server predated r2store and re-failed every poll)
  local interval=${CHATBOX_WATCH_INTERVAL_S:-60} ERR_TICKS=5 RESTART_COOLDOWN_S=1800
  local last_line="" err_ticks=0 last_restart=0 checked_env_for=""
  log "watch start public=$PUBLIC_URL local=$LOCAL_URL"
  while true; do
    local ssh king h state served line now
    ssh=$(pod_ssh 2>/dev/null || true); king=$(king_revision 2>/dev/null || true)
    h=$(health || true)
    state=$("$PY" -c 'import json,sys; d=json.loads(sys.argv[1]) if sys.argv[1] else {}; print(d.get("state","unreachable"))' "$h" 2>/dev/null || echo unreachable)
    served=$("$PY" -c 'import json,sys; d=json.loads(sys.argv[1]) if sys.argv[1] else {}; print(((d.get("king") or {}).get("revision") or "")[:12])' "$h" 2>/dev/null || true)
    line="pod=${ssh:-none} state=$state served=${served:-?} king=${king:0:12}"
    [ "$line" != "$last_line" ] && { log "$line"; last_line=$line; }
    now=$(date +%s)
    if [ -n "$ssh" ] && [ "$state" != "unreachable" ] && [ "$checked_env_for" != "$ssh" ]; then
      if ssh_pod "test -f $REMOTE_DIR/.chat_env" 2>/dev/null; then
        log ".chat_env present on $ssh"
      else
        log ".chat_env missing on $ssh (fresh rental?) — pushing overrides"
        cmd_overrides || log "WARNING overrides push failed"
        last_restart=$now
      fi
      checked_env_for=$ssh
    fi
    if [ "$state" = "error" ]; then
      err_ticks=$((err_ticks + 1))
      if [ "$err_ticks" -ge "$ERR_TICKS" ] && [ $((now - last_restart)) -ge "$RESTART_COOLDOWN_S" ]; then
        log "state=error for $err_ticks ticks — restarting chatsrv"
        restart_chatsrv >/dev/null 2>&1 && last_restart=$now || log "WARNING restart failed"
        err_ticks=0
      fi
    else
      err_ticks=0
    fi
    sleep "$interval"
  done
}

case "${1:-}" in
  status) cmd_status ;;
  deploy) cmd_deploy ;;
  overrides) cmd_overrides ;;
  route) cmd_route ;;
  smoke) shift; cmd_smoke "$@" ;;
  watch) cmd_watch ;;
  cursor) cmd_cursor ;;
  logs) cmd_logs ;;
  *) sed -n '2,32p' "$0"; exit 1 ;;
esac
