# R1032 — ShortCtx MidRank Hiβ Ultra MidLR

| field | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| method | Offline DPO (Reason-pair prefs) |
| β / α / r / lr | **0.3** / 128 / 32 / **1e-6** |
| max_len / steps | **6144** ShortCtx / **28800** Ultra |
| parent | R1015 ShortCtx MidRank Hiβ Mega MidLR REFUTE m=−0.002353 ~−0.94× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.

## p4177 — CROWN_OK → SUBMITTED

| item | value |
|---|---|
| n80 | m=**+0.005461** SE=**0.002603** z=**2.098** n=**80** bar=**0.005206** (~**1.049×**) |
| floors | thought✓ **175.5** · B✓ **0.475** · k=**3** · τ=**0.03** |
| HF | `unconst/Affine-5czsc2fc98-r1032-vera-odpo-midrank-hibeta-shortctx-ultraextra-ep4-midlr-merged` @`62dfb322fdce5873543bd92692ab4ecc3e13f941` (65.4GB; purged LOST r861/938/959/1008 first) |
| hotkey | `r1032` · `5DyVW9mb6mbAkNdD9xjenzsVXnTUWosrKp9q3AetWJnDbXeC` · reg extrinsic **8887516-0019** |
| submit | block `0x64e13f98f2236a9c151479c9d4075d9a6071a77add47b0e08a7280b62e1bfb9f` · reveal **31481268** · slot **burned** |
| --check | pre-flight OK |
