# R709 — marsplan Soft MidRank LoBeta SoftCtx UltraExtra ep3×LoLR

## Status (p3747)
**TRAIN LIVE** on zesty GPUs **4,5** (idle post-R703 REFUTE; R708 TRAIN on 6,7).

- train pid **841466** · wait→merge armed
- kept=**604**/604 · β=0.02 r=32 @12288 steps=**7200** ep=3 lr=1e-6
- BASE pinned marsplan0624@`556d02a2` (after mine.env)
- Log: `/root/logs/r709_train.nohup`
- Wait: `/root/logs/r709_wait_merge.nohup`

## Axis
Transfer R637 Soft MidRank LoBeta SoftCtx SIGNAL ~1.45× onto marsplan UltraExtra.
≠ R701 HyperExtra LoBeta · ≠ R708 MidBeta UltraExtra · ≠ R703 HiBeta HyperExtra REFUTE.

## Check
`lium exec zesty-comet-da 'tail -30 /root/logs/r709_train.nohup; cat /root/logs/r709_train.pid; nvidia-smi -i 4,5 --query-gpu=index,memory.used --format=csv'`
