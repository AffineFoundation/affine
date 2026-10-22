#!/usr/bin/env bash
# Roll back v8.2.3 -> v8.2.2: remove the two v8.2.3 drop-ins (v8.2.2 ones beneath stay), restart lifecycle + mining observer.
# Safe at any time: an already-created TRAINING-JOB is reloaded from disk with the config it was signed with.
set -euo pipefail
U=/home/const/.config/systemd/user
rm -f "$U/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~v823-release.conf" "$U/affine-mining-discovery.service.d/zzzz~~~mimo-observer-v823.conf"
systemctl --user daemon-reload && systemctl --user restart affine-mimo-lifecycle.service && sleep 20 && systemctl --user restart affine-mining-discovery.service
echo "$(date -u +%FT%TZ) rolled back to v8.2.2 drop-ins" | tee -a /home/const/subnet120-rewrite/state/root-audits/20261016-v823/ROLLBACK.log
