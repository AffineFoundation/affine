#!/usr/bin/bash
# Revert the SN120 variance levers (code + flag + public texts; closes a still-open NEW-rules round so the next round opens under the
# old rules: n 1000, k_cand 4, per-round sigma_run). NOCLOSE=1 keeps that round open. Dry run (copies only): MODE=dryrun bash revert.sh
export XDG_RUNTIME_DIR=/run/user/$(id -u)
[[ ${MODE:-revert} == revert ]] && systemctl --user stop affine-vl-watchdog 2>/dev/null
MODE=${MODE:-revert} WHY="${WHY:-manual revert.sh}" exec bash /home/const/subnet120-rewrite/state/root-audits/20261031-variance-levers/activate_levers.sh
