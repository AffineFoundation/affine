# Rollback writer time budget v2 (2026-10-20)
1. rm ~/.config/systemd/user/affine-live-reward-writer.service.d/zzzz~~~~~~~~~mimo-v822-v7e-budget2.conf
2. systemctl --user daemon-reload   (next minute uses the unchanged v7c drop-in / composition 2d78a747 over v7b e0dacce0; nothing else to undo)
Backups: /home/const/subnet120-rewrite/state/root-audits/20261020-writer-time-budget-v2/backup/ (v7c + v7b compositions, v7c drop-in as installed, effective unit before).
Changed: only evidence_timeout_seconds 1000->1400, maximum_seconds 1200->1600 (entry + parallel_module + v7c entry repinned to copies whose only diff is those bounds), unit TimeoutStartSec 1380->1780.
Blend config / measured_impact_blend.py untouched (blend 0.0; its evidence guard still references 1000 s -> at most forces the exact current vector).
