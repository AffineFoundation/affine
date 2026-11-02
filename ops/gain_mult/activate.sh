#!/bin/bash
# Activate the king gain multiplier (Jacob pt1136u/pt1137u). activation_epoch = the last gate epoch with a DECISION right now.
set -euo pipefail
G=/home/const/subnet120-rewrite/state/root-audits/20261102-gain-mult
RA=/home/const/subnet120-rewrite/state/root-audits
EP=$RA/20261026-promotion-gate/state/epochs
A=$(python3 -c "
import json,pathlib
E=pathlib.Path('$EP'); best=None
for p in E.iterdir():
    if p.name.isdigit():
        try: d=json.loads((p/'DECISION.json').read_text())
        except Exception: continue
        if d.get('decision'): best=max(best or 0,int(p.name))
print(best)")
[ -n "$A" ] && [ "$A" != None ] || { echo "no activation epoch"; exit 1; }
NOW=$(TZ=America/Costa_Rica date "+%Y-%m-%d %H:%M CR (UTC-6)")
python3 - "$A" "$NOW" <<PY
import json,sys; p="$G/code/config.json"; c=json.load(open(p)); c["activation_epoch"]=int(sys.argv[1]); c["activated_at_cr"]=sys.argv[2]; c["enabled"]=True
json.dump(c,open(p,"w"),indent=1)
PY
python3 -B $G/code/gain_mult.py
python3 $G/code/patch_k1_gainmult.py $RA/20261027-k1-per-uid/code/k1_service.py
python3 -m py_compile $RA/20261027-k1-per-uid/code/k1_service.py
python3 $G/code/patch_tp_gainmult.py $RA/20261026-promotion-gate/code/training_params.py
python3 -m py_compile $RA/20261026-promotion-gate/code/training_params.py
python3 $G/code/patch_controller_hold.py /home/const/subnet120-rewrite/priors/controller.py
python3 -m py_compile /home/const/subnet120-rewrite/priors/controller.py
python3 - <<PY
import json; p="$RA/20261101-prior-table/controller.json"; c=json.load(open(p))
c["parent_hold"]=dict(enabled=True, min_tasks=7000, max_s=1800, cap_from_t=1793644742.0,
  note="RA/20261102-gain-mult: on a head change keep sampling the parent until it has >= 7000 distinct tasks at the current cap (max 30 min), so king-vs-parent pairs reach >= 2000 shared tasks")
json.dump(c,open(p,"w"),indent=1)
PY
cp $G/code/affine-gain-mult.service $G/code/affine-gain-mult.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart affine-k1.service affine-training-params.service affine-prior-table.service
systemctl --user enable --now affine-gain-mult.timer
echo "- $NOW: ACTIVATED (code/activate.sh): activation_epoch=$A (kings crowned at E$((A+1))+ get m; E$A and earlier m=1); lambda=1; k1_service/training_params/controller patched; parent_hold on (7000 tasks, 30 min); timer affine-gain-mult every 5 min. Revert: bash $G/revert.sh" >> $G/PROGRESS.md
echo "activation_epoch=$A"
