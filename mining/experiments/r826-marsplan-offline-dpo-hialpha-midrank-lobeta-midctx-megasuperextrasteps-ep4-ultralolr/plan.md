# R826 — marsplan MidRank×LoBeta×MidCtx UltraLoLR

## Claim
After R817 (marsplan MidCtx HiRank LoBeta UltraLoLR) REFUTE v4 m=+0.000677~0.11× (thought✓210 B✓0.515 k=3),
try MidRank (r=32) sibling at same LoBeta MidCtx UltraLoLR.

## Knobs
- base: marsplan0624/affine-5gedzafcvg-queen@556d02a2 (local cache)
- Offline-DPO β=0.02 α=128 r=32 lr=5e-7 @8192 max_steps=19200 ep=4
- ≠ R817 HiRank / ≠ R818 MidRank HiBeta / ≠ R809 tammy Mid Mid Lo / ≠ Online / ≠ GRPO

## Decision rule
Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign35.
