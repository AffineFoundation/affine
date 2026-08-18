# R835 — marsplan MidCtx HiRank HiBeta UltraLoLR

Axis: Offline-DPO HiAlpha HiRank HiBeta MidCtx MegaSuperExtraSteps ep=4 UltraLoLR
Base: marsplan0624/affine-5gedzafcvg-queen@556d02a2
β=0.3 α=128 r=64 lr=5e-7 @8192 max_steps=19200 ep=4
Signal: R818 MidCtx Mid Hi n80 in flight + R813 r252 MidCtx Hi Hi REFUTE → marsplan MidCtx Hi Hi UltraLoLR; ≠ R818 MidRank / ≠ R807 MidCtx Hi Mid / ≠ R817 MidCtx Hi Lo / ≠ Online / ≠ GRPO
Decision: Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign35
Pod: mine-r337 GPUs 6,7 (idle after online-DPO merge); merge-only + host-relay→lunar
