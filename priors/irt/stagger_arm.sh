#!/usr/bin/bash
# Stagger the irt-info draw one round after the 12K max_tokens switch (both were set to land at the next round open).
# Now: weighting off (the next round = 12K, prior-v1 weighting). Once that 12K round has opened: weighting on -> the round after is irt-info.
set -u; export TZ=America/Costa_Rica XDG_RUNTIME_DIR=/run/user/$(id -u)
RA=/home/const/subnet120-rewrite/state/root-audits; I=$RA/20261102-irt-live; C=$I/code; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
H=$RA/20261026-promotion-gate/state/heldout; L=$I/state/stagger.log
say(){ echo "$(date "+%F %T CR") $*" | tee -a $L; echo "- $(date "+%H:%M CR"): stagger: $*" >> $I/PROGRESS.md; }
OPEN=$(python3 -c "import json;print(json.load(open(\"$H/CURRENT.private.json\"))[\"id\"])")
flock -w 60 $FLAG.lock python3 $C/setflag_irt.py $FLAG off "stagger: let the 12K switch round open alone (prior-v1 weighting); irt-info from the round after"
say "weighting off while round $OPEN is open; waiting for the 12K round to open"
while :; do
  [[ -f $I/state/REVERTED ]] && { say "REVERTED marker: stop"; exit 0; }
  N=$(python3 -c "import json;print(json.load(open(\"$H/CURRENT.private.json\"))[\"id\"])" 2>/dev/null || echo "")
  if [[ -n $N && $N != $OPEN ]]; then
    MT=$(python3 -c "import json;print(json.load(open(\"$H/$N/ROUND.private.json\"))[\"cfg\"].get(\"max_tokens\"))")
    W=$(python3 -c "import json;print(json.load(open(\"$H/$N/ROUND.private.json\")).get(\"weighting\"))")
    say "round $N opened (max_tokens=$MT weighting=$W)"
    if [[ $MT == 12288 ]]; then
      flock -w 60 $FLAG.lock python3 $C/setflag_irt.py $FLAG on "stagger: 12K round $N open; irt-info from the next round"
      say "weighting on: the round after $N is the first irt-info round"; exit 0
    fi
    OPEN=$N
  fi
  sleep 15
done
