# SUBMISSIONS — every hotkey we register / burn

**Check this file before every submit.** One eval slot per hotkey, ever. Slot burns at enqueue.

## Hotkeys

| hotkey name | ss58 | registered block | burn τ | repo | revision | submit check output | verdict | slot |
|---|---|---|---|---|---|---|---|---|
| default | `5G1sKqsDSMEktjGvXAt8BRyon8Lkug6eRt5ETmWxbgSPVQrj` | not on SN120 | — | — | — | — | — | unused |
| **r158** | `5Dw5qrFs3xGpy73YGcH1AeP4q6LwxZFM9cGiwvaDWgFZ7ePv` | extrinsic **8833614-0010** · uid**41** | ≈τ1.25 | `unconst/Affine-5czsc2fc98-r158-merged` | `87e23e3ac089bf803ea7ccaa25cea050af2afdc0` | pre-flight OK | **DEAD** — never RC; merged HF **purged** (only `-r158-lora` left); cannot re-reveal same rev | slot **not** enqueued |
| **r172** | `5DbtRsNVFv822so4imxGgzLTwzvSwbnxembfkWgrESA8BwVA` | extrinsic **8840482-0014** | ≈τ1.10 | `unconst/Affine-5czsc2fc98-r172-merged` | `27e2f083b35546e543b58305df164152439ab9ad` | pre-flight OK | **LOST** chal-**00659** m=+0.00104 z=0.61 (below crown bar) | slot **burned** |
| **r334** | `5CS7NkUZXyaSp8wyHqvNpijKD5pFq5xRpmHzJ2qpJJ4J3kia` | extrinsic **8849207-0006** | ≈τ1.23 | `unconst/Affine-5czsc2fc98-r334-online-dpo-merged` | `40d715805d8791478e704578c045fb60f415cc3e` | pre-flight OK | **LOST** chal-**00683** @**2026-08-15T21:41:29Z** m=**−0.00031** SE=0.00126 z=−0.25 n=2003 thought✓(med206) B✓(0.42) vs marsplan r24 (n80 was +0.01182 vs *prior* king) | slot **burned** |
| **r490** | `5DUBLbN8x2WG8LNdxgZm4PJpEFxGsgifT4uXfnsuYDejTKq5` | extrinsic **8854951-0006** | ≈τ0.95 | `unconst/Affine-5czsc2fc98-r490-offline-dpo-hialpha-midrank-lobeta-extrasteps-merged` | `17f4191af0eb2678bde2c25bff89e01ea92154a4` | pre-flight OK | **LOST chal-00735** @**2026-08-16T05:21:19Z** m=**−0.00326** SE=0.00141 z=−2.31 n=1804 thought✓(med233) B✓(0.44) vs sbs-v5 r29 (n80 was +0.01735) | slot **burned** |
| **r496** | `5FUW8WTt5GHXTaSXTcKMK8f8kjB6Xw3oXwt6Wzz6HS3nfuEe` | extrinsic **8855430-0010** | ≈τ1.35 | `unconst/Affine-5czsc2fc98-r496-sbsv5-offline-dpo-hialpha-midrank-lobeta-extrasteps-merged` | `935807e7b004e31b7affd5b9719eabaa51ff783f` | pre-flight OK | **LOST chal-00738** @**2026-08-16T07:00:28Z** m=**+0.00172** SE=0.00089 z=1.93 n=1857 bar=δ**0.002** thought✓(med194) B✓(0.39) vs sbs-v5 r29 (n80 was +0.01569) | slot **burned** |
| **r512** | `5GgcziW7vghRQzoapj3utXHbjjSWmzrNjRgkuZze5vKikE6q` | extrinsic **8855963-0007** | ≈τ1.21 | `unconst/Affine-5czsc2fc98-r512-offline-dpo-hialpha-hirank-lobeta-midctx-ultraextrasteps-merged` | `660618710710f9a2cd513900731923f51ee43317` | pre-flight OK | **LOST chal-00741** @**2026-08-16T08:45:04Z** m=**−0.00294** SE=0.00128 z=−2.29 n=1815 bar=0.00257 thought✓(med243) B✓(0.43) vs sbs-v5 r29 (n80 was +0.01180) · reveal **31355049** | slot **burned** |
| **r252** | `5Fxumb3kHWht1ZEKybi23UYJMZRsQiDFn9A7sCpMKUAFJkB7` | extrinsic **8857818-0008** | ≈τ1.03 | `unconst/Affine-5czsc2fc98-r252-merged` | `b42d6245d77fe30885ea8a90387771e1bc465e0f` | pre-flight OK | **CROWNED reign33** @**2026-08-16T15:23:18Z** block **8857842** score≈0.00276 · n80 was +0.00475 vs loveaffine r32 · earning uid**90** | slot **burned** at enqueue |
| **r596** | `5CoeGCs6tTb2docZh88w5N3AEpcmskfdM4fTGCQwRCz3mJHq` | extrinsic **8863872-0017** | ≈τ2.00 | `unconst/Affine-5czsc2fc98-r596-r252-odpo-hirank-midbeta-softctx-megaextra-merged` | `3a778f1fd066e0018d962c82829c73ada6f5cc91` | pre-flight OK | **SUBMITTED** @**2026-08-17** blockhash `0xe902d1e6…` · reveal round **31386688** · **chal-00822** · n80 vs reign34 m=+0.00620 ~1.31× bar · pending duel | slot **burned** at enqueue |
| **r637** | `5DMRyV3UQXDJR4gBwoXng26bqLT6XfcxYmEN3JWxmCcCS4Ln` | extrinsic **8864188-0005** | ≈τ2.21 | `unconst/Affine-5czsc2fc98-r637-r252-odpo-midrank-lobeta-softctx-ep3-lolr-merged` | `7aded176a1f13873f8138f38b7e92f0ff0c6a304` | pre-flight OK | **QUEUED chal-00829** @**2026-08-17T11:48:46Z** · reveal **31387948** · n80 vs reign34 m=+0.005735 ~1.45× bar · pending duel | slot **burned** at enqueue |
| **r683** | `5DRydxjU1Vr6ANNjnKNJvHdbmABLcVNhgojJRHDbgBRskccm` | extrinsic **8867097-0018** | ≈τ1.53 | `unconst/Affine-5czsc2fc98-r683-r252-odpo-midrank-hibeta-shortctx-ultraextra-ep3-lolr-merged` | `f3314c7cdc174cc4b0331bc6125c21e9b45ddbe1` | pre-flight OK | **LOST chal-00860** @**2026-08-17T21:53:43Z** m=**+4.7e-5** SE=0.000716 z=0.066 n=1221 bar=δ**0.002** thought✓(med159) B✓(0.403) vs reign34 (n80 was +0.002137 ~1.07× knife-edge) | slot **burned** |
| **r861** | `5FBwqMmqnj2uGFK3UU8XWTxaf9M5VVH5v6YniKAeisKJ69aA` | extrinsic **8875721-0018** | ≈τ2.83 | `unconst/Affine-5czsc2fc98-r861-vera-odpo-midrank-midbeta-softctx-megaextra-ep4-ultralolr-merged` | `f20753584e79f874e3c1984211a8653bc8785fc2` | pre-flight OK | **LOST chal-00934** @**2026-08-19T02:39:36Z** m=**+0.001182** SE=0.000646 z=1.83 n=1288 bar=δ**0.002** (~**0.59×**) thought✓(med169) B✓(0.416) vs reign36 (n80 was +0.003665 ~1.088×) | slot **burned** |
| **r938** | `5DZFzeYow7jq9woPwDkM1nLfjJPBiBf6hrrQmzQi1ZTnKzAu` | extrinsic **8882722-0013** | ≈τ2.20 | `unconst/Affine-5czsc2fc98-r938-vera-odpo-midrank-hibeta-softctx-megaextra-ep4-ultralolr-merged` | `8ef1b06acfe849cc179fbc6e21c2db7561ce4092` | pre-flight OK | **LOST chal-00949** @**2026-08-20T03:40:02Z** m=**−0.000615** SE=0.000547 z=−1.12 n=1290 bar=δ**0.002** (~**−0.31×**) thought✓(med167) B✓(0.417) vs reign36 (n80 was +0.004951 ~1.20×) · reveal **31462190** | slot **burned** |
| **r959** | `5DkVMypUxctB5wAbNPhL1K5LsEafSseywCoGJFHDHurxzoXJ` | extrinsic **8884228-0003** | ≈τ2.61 | `unconst/Affine-5czsc2fc98-r959-vera-odpo-hirank-midbeta-softctx-megaextra-ep4-ultralolr-merged` | `3b427b1678d53890655c27f0e43166a4cc6a6d65` | pre-flight OK | **LOST chal-00957** @**2026-08-20T11:16:49Z** m=**−0.000659** SE=0.000569 z=−1.16 n=1294 bar=δ**0.002** (~**−0.33×**) thought✓(med167) B✓(0.410) vs reign36 (n80 was +0.006384 ~1.23×) · reveal **31468125** | slot **burned** |
| **r1008** | `5GvDN949S96Qgs6eRatzzHd141JRFnPqJkSwqW626Udjj3K3` | extrinsic **8886082-0013** | ≈τ2+ | `unconst/Affine-5czsc2fc98-r1008-vera-odpo-hirank-midbeta-midctx-megaextra-ep4-midlr-merged` | `ff9153704c9d7c94d58aecf211e93bdb5c4238bb` | pre-flight OK | **LOST chal-00961** @**2026-08-20T15:08:14Z** m=**+0.000468** SE=0.000750 z=0.624 n=1291 bar=δ**0.002** (~**0.23×**) vs reign36 (n80 was +0.005917 ~1.325×) | slot **burned** |
| **r1032** | `5DyVW9mb6mbAkNdD9xjenzsVXnTUWosrKp9q3AetWJnDbXeC` | extrinsic **8887516-0019** | ≈τ2+ | `unconst/Affine-5czsc2fc98-r1032-vera-odpo-midrank-hibeta-shortctx-ultraextra-ep4-midlr-merged` | `62dfb322fdce5873543bd92692ab4ecc3e13f941` | pre-flight OK | **LOST chal-00967** @**2026-08-20T20:57:06Z** m=**+0.001428** SE=0.000516 z=2.77 n=1292 bar=δ**0.002** (~**0.71×**) thought✓(med177) B✓(0.463) vs reign36 (cleared 2·SE; failed δ) · n80 was +0.005461 ~1.049× | slot **burned** |
| **r1064** | `5EkHrq6nrQd8VoYEoMVKWu3M2BaHSfKd1MqXh1WNANQKmRW8` | extrinsic **8888667-0007** | ≈τ2+ | `unconst/Affine-5czsc2fc98-r1064-vera-odpo-midrank-midbeta-midctx-ultraextra-ep4-hilr-merged` | `03affd41ac7c256eadc8376da828b5dab033a2c0` | pre-flight OK | **SUBMITTED** @**2026-08-20T21:20:27Z** blockhash `0x04922d41…` · reveal round **31485871** · n80 vs reign36 m=+0.006632 ~1.021× bar thought✓201 B✓0.521 · pending duel | slot **burned** at enqueue |

## Identity tokens (repo naming)

- coldkey `5CZscRf3nZmGspyqs2ZvFXSjnnondpjNU5QbJWVFFT92FC98` → token `5czsc2fc98`
- hotkey `5G1sKqsDSMEktjGvXAt8BRyon8Lkug6eRt5ETmWxbgSPVQrj` → token `5g1skpvqrj`
- Example repo: `unconst/Affine-5czsc2fc98-<name>`

## Rules (Reason v4 / wvk=7)

- Fresh registered hotkey per submission.
- Simulated paired **Reason v4** margin over current king
  **> max(k_sigma · SE, min_margin)** **and** thought-len ≥80 **and** B pass
  ≥0.30 before submit. Live knobs from `api/v1/contract` (wvk=**7**, k=**3**,
  tau=**0.03**, n_turns=**1300**, k_sigma=**2.0**, δ=**0.002**). **No 1.5×
  headroom.** Pre-v4 / k=1 sims do **not** license submit.
- Always `submit.py --check` first; paste output into the experiment log.
- Never resubmit a content revision that has ever been submitted by anyone.
- Confirm live `weight_version_key` on contract before enqueue (now **7**).
- Model card must document training (operator 2026-08-16).