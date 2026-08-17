# R708 — marsplan Soft MidRank MidBeta SoftCtx UltraExtra ep3×LoLR

## Status (p3746)
**TRAIN LIVE** on zesty GPUs **6,7** (idle post-R702 REFUTE; R703 n80 owns 4,5).

- train pid **839998** · wait→merge pid (outer wait armed)
- kept=**259**/604 · β=0.1 r=32 @12288 steps=**7200** ep=3 lr=1e-6
- BASE pinned marsplan0624@`556d02a2` (after mine.env)
- Log: `/root/logs/r708_train.nohup`
- Wait: `/root/logs/r708_wait_merge.nohup`

## Axis
Transfer R675 Soft MidRank MidBeta SoftCtx UltraExtra near-miss ~0.97× onto marsplan.
≠ R702 HyperExtra REFUTE (−0.29×).

## Check
`lium exec zesty-comet-da 'tail -30 /root/logs/r708_train.nohup; cat /root/logs/r708_train.pid; nvidia-smi -i 6,7 --query-gpu=index,memory.used --format=csv'`
