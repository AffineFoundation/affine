# R924 — in flight (p4044)

- TRAIN_DONE 900 steps; merge fixed p4043 (`--adapter …/train/adapter`).
- **p4044:** host-relay cosmic-orbit-55 → crown brave-comet-f4; chall GPUs 6,7 :8002 + v4 n80 vs reign36.
- Decision rule unchanged: margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30.
- Poll: `ssh -p 40298 root@95.133.252.28 'tail -40 /root/logs/p4044_r924_chall_n80_wvk7.log; cat /root/affine_data/r924_decision_reign36_wvk7.json 2>/dev/null'`
