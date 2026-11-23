#!/usr/bin/env bash
# Poll the Generator inbox; write only the live reserve named by reserve-live.path.
# Run from the v2 checkout: generator-ingestd.sh PYTHON [PRIOR_INDEX]
set -u
PY=${1:?python interpreter required}
PRIOR_INDEX=${2:-${SN120_GENERATOR_PRIOR_INDEX:-/root/sn120-run/generator/prior-index.json}}
PATH_FILE=${SN120_RESERVE_PATH_FILE:-/root/sn120-run/reserve-live.path}
INBOX=${SN120_GENERATOR_INBOX:-/root/sn120-run/generator/inbox}
LOG=${SN120_GENERATOR_LOG:-/root/sn120-run/generator/ingestd.log}
PUBLIC=${SN120_PUBLIC_URL:-https://affine.io}
POLL_S=${SN120_GENERATOR_POLL_S:-120}

IFS= read -r SN120_RESERVE_DIR < "$PATH_FILE" || exit 64
case $SN120_RESERVE_DIR in /*) export SN120_RESERVE_DIR ;; *) exit 64 ;; esac
mkdir -p "$INBOX" "$(dirname "$LOG")" || exit 1
exec 9> "${SN120_GENERATOR_LOCK:-/root/sn120-run/generator/ingestd.lock}"
flock -n 9 || exit 0

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }
say "started inbox=$INBOX reserve=$SN120_RESERVE_DIR"
while :; do
    for input in "$INBOX"/accepted-*.jsonl; do
        [ -f "$input" ] || continue
        name=${input##*/}
        epoch=${name#accepted-}
        epoch=${epoch%.jsonl}
        case $epoch in [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9]Z) ;; *) say "skip $epoch invalid epoch name"; continue ;; esac
        if grep -q "\"epoch\":\"$epoch\"" "$SN120_RESERVE_DIR/generator-ingested.jsonl" 2> /dev/null; then
            continue
        fi
        results="$INBOX/results-$epoch.json"
        done_file="$INBOX/accepted-$epoch.done"
        if [ ! -s "$done_file" ] && [ ! -s "$results" ]; then
            if ! curl -fLsS --retry 3 --connect-timeout 15 --max-time 90 "$PUBLIC/generator/epochs/$epoch/results.json" -o "$results.tmp"; then
                rm -f "$results.tmp"
                say "waiting $epoch .done or results.json"
                continue
            fi
            mv "$results.tmp" "$results"
        fi
        if [ ! -s "$done_file" ] && [ ! -s "$results" ]; then
            continue
        fi
        if "$PY" -m sn120_v2.envs.reserve ingest-generator "$input" --epoch "$epoch" --results "$results" --prior-index "$PRIOR_INDEX" > "$INBOX/ingest-$epoch.last.json" 2> "$INBOX/ingest-$epoch.last.err"; then
            say "$epoch $(python3 -I -c 'import json,sys; raw=sys.stdin.read(); print(json.dumps(json.loads(raw[raw.index("{"):]),separators=(",",":")))' < "$INBOX/ingest-$epoch.last.json")"
        else
            rc=$?
            say "$epoch failed exit=$rc $(tr -d '\n' < "$INBOX/ingest-$epoch.last.err")"
        fi
    done
    sleep "$POLL_S"
done
