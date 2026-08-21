#!/usr/bin/env bash
# Host poller: rent many mine-* 8×B300 (else 8×B200) with distinct axes.
# Keeps going until TARGET live mine pods or CAP, not one-and-done.
# Never touches non-mine pods. Always --ttl. No bulk rm.
set -euo pipefail

ROOT=/home/const/subnet120
EXP="$ROOT/mining/experiments/fleet-rent"
LOG="$EXP/logs/wait_fleet_b300.log"
PIDF="$EXP/logs/wait_fleet_b300.pid"
STAMP_DIR="$EXP/artifacts"
TTL=${TTL:-24h}
CAP=${MINE_CAP:-25}
# Burn floor ≈ $833/h ÷ ~$64/h ≈ 13 boxes; keep renting to CAP (floor≠ceiling).
TARGET=${TARGET_MINES:-25}
# Empty-stock sleep between ls polls. 0 → 0.25s (avoid CPU spin; ls≈0.6s anyway).
POLL_S=${POLL_S:-0}
# Max distinct axes to claim per stock sighting (one node → one name).
PARALLEL_N=${PARALLEL_N:-22}
MAX_ITERS=${MAX_ITERS:-86400}
PASS=${PASS:-3196}

# Distinct experimental axes (one pod each). Skip names already live.
# Format: name|axis_id|short_note
# R24–R32 (structural) sit after R3b — ahead of cosmetic parent-swap GRPO.
QUEUE=(
  # p4281: R1158 HEAD — re-rent after 1/8 GPU dud tear (brave-matrix-2a); vera×Reason-GRPO
  "mine-r1158-vera-reason-grpo-1|R1158|vera×Reason-GRPO base vera6@8e3f1695 (≠ Offline-DPO UltraLoLR fleet; re-rent after 1/8 B200 dud p4281)"
  # p3379: R252 RENTED gentle-wolf-8c 8×B300 — do not re-rent. Sync bash↔API.
  # p3374/p3379: quarantine R259 — executor 7b9ee272… SSH-refused badhost; demote from bash HEAD.
  # p3196: R227 TRAIN salvage on idle brave-raven (ex-R226 host) — do not re-rent. HEAD→R259.
  # p3192: crown → sbs-v5 reign27 @22:14Z; fleet KING retarget; n80 must vs sbs-v5.
  # p3190: R227 → genesis×FullFT-HiLR (r26; keep pod name). Sync bash↔API.
  # p3189: R226 HEAD → genesis×FullFT (live r26); keep pod name for bootstrap case.
  # p3178: R225 lean on R262 4,5 — do not re-rent. HEAD → R226 FullFT.
  # p3175: promote R225 REINFORCE method axis to HEAD (≠ more nonking×HiAlpha-GRPO
  # clones R259/R252; live R262/R260 already rented). Sync bash↔API.
  # "mine-r225-marsplan-reinforce-1|R225|marsplan×REINFORCE … LIVE lean R262 4,5 p3178"
  # "mine-r227-marsplan-fullft-hilr-1|R227|LIVE salvage brave-raven p3196 — do not re-rent"
  # "mine-r259-michael-h2-nonking-grpo-1|R259|QUARANTINE p3374 badhost 7b9ee272 — do not HEAD"
  # "mine-r252-vera-t4-nonking-grpo-1|R252|LIVE gentle-wolf-8c p3379 — do not re-rent"
  # R337 LIVE noble-hawk-1f — skip if present
  "mine-r337-marsplan-online-dpo-hilr-1|R337|marsplan×Online-DPO×HiLR lr=2e-5 β=0.1 α=32 r=16 G=4 @6144 max_steps=300 (≠ R334)"
  # R4/R4b/R5/R6/R6b/R8 REFUTED. R7 live on warm mine-r4-fullft-1 (p2158) — do not re-rent R7/R8.
  # R3b live on mine-r3-grpo-1 (p2127 retarget after R3 REFUTE) — do not re-rent.
  # R24 live on mine-r3-grpo-1 (p2205 warm-arm after R15 REFUTE) — do not re-rent.
  # R24 live on mine-r3; R25=mine-r25-hitemp-1; R26=mine-crown-1 — do not re-rent.
  # p2224: sbs-v2 still GATED (index 403) — demote R10/R18.
  # "mine-r10-merge-rl-1|R10|Tok×sbs-v2 α-merge → Reason-GRPO (BLOCKED Hub gated)"
  # "mine-r18-sbs-grpo-1|R18|pure sbs-v2-init Reason-GRPO (BLOCKED Hub gated)"
  # p2235: R5b warm on mine-r4-fullft-1; R33 warm on mine-crown-1 — do not re-rent.
  # "mine-r5-nonking-2|R5b|Talent/kevin non-king base FT"
  # p2241: R5b SIGNAL_POS_BELOW; R19 warm on mine-r4-fullft-1 — do not re-rent.
  # "mine-r19-talent-grpo-1|R19|TalentPigs-init Reason-GRPO (≠ R3/R5b; sbs gated)"
  # p2249: R22 warm on mine-crown-1 — do not re-rent.
  # "mine-r22-golden-grpo-1|R22|golden-crown-init Reason-GRPO (≠ R3/R16/R19–R21)"
  # p2252: R23 warm on mine-r3-grpo-1 — do not re-rent.
  # "mine-r23-diane-grpo-1|R23|diane613-init Reason-GRPO (≠ R3/R16/R19–R22)"
  # p2253: R27 warm on mine-r4-fullft-1 — do not re-rent.
  # "mine-r27-bigg-1|R27|Tok GRPO group_size=16 (≠ R3 G=4 / R3b G=8+alt-lr)"
  # p2264: R28 warm-armed on mine-crown-1 after R22 REFUTE — do not re-rent.
  # "mine-r28-hilr-1|R28|Tok GRPO lr=2e-5 (≠ R3 5e-6; isolates LR vs R3b)"
  # p2288: R29 warm on mine-r3-grpo-1 — do not re-rent.
  # "mine-r29-hirank-1|R29|Tok GRPO lora_r=64 (≠ R3 r=16; isolates rank vs R3b)"
  # p2294: R30 warm on mine-crown-1 after R28 REFUTE — do not re-rent.
  # "mine-r30-hialpha-1|R30|Tok GRPO lora_alpha=128 r=16 (≠ R3 α=32; isolates α vs R29)"
  # p3174: demote stale Tok ladder — was wrongly HEAD before R259 sync
  # "mine-r31-nodrop-1|R31|Tok GRPO lora_dropout=0.0 (≠ R3 0.05; isolates dropout)"
  # "mine-r32-kl-1|R32|Tok GRPO kl_coef=0.02 vs base (≠ R3 kl=0; isolates KL)"
  # "mine-r34-longctx-hilr-1|R34|Tok LongCtx×HiLR max_len=16384 lr=2e-5 (R24×R28 compound)"
  # p2608: awesome reign13 → awesome×HiAlpha HEAD (also HEAD in API QUEUE)
  # live as mine-r165 — skip if present
  "mine-r165-awesome-hialpha-1|R165|awesome×HiAlpha α=128 r=16 G=4 lr=5e-6 @6144 from live reign-13 0pentensor/Affine-5dflhtkufw-awesome-v11@450bdfc3 (≠ R161–R164 guass demoted / R160 thermo / R158 guass×HiAlpha)"
  # p2617–p2619: awesome axis isolates ahead of Talent filler (mirror API QUEUE)
  # p2636: R166 warm-armed on mine-r160 (brave-fox-8e) — do not re-rent
  # "mine-r166-awesome-hialpha-bigg-1|R166|awesome×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 from live reign-13 (≠ R165 G=4)"
    # p2639: R167 warm-armed on mine-crown-1 (gentle-orbit-bd) after R79 REFUTE — do not re-rent.
#   "mine-r167-awesome-hialpha-hilr-1|R167|awesome×HiAlpha×HiLR α=128 r=16 G=4 lr=2e-5 @6144 from live reign-13 (≠ R165@5e-6 / R166 BigG)"
  # p2640: R168 warm-armed on mine-r165 (lunar) — do not re-rent
  # "mine-r168-awesome-hialpha-longctx-1|R168|awesome×HiAlpha×LongCtx α=128 r=16 G=4 lr=5e-6 16384/1024 from live reign-13 (≠ R165@6144 / R166 BigG / R167 HiLR)"
  # p2641: R169 warm-armed on mine-crown-1 — REFUTED p2664 — do not re-rent
  # "mine-r169-awesome-hialpha-hirank-1|R169|awesome×HiAlpha×HiRank α=128 r=64 G=4 lr=5e-6 @6144 from live reign-13 (≠ R165 r=16 / R166 BigG / R167 HiLR / R168 LongCtx)"
  # p2642: R170 warm-armed on mine-r160 — do not re-rent
  # "mine-r170-awesome-hialpha-bigg-hilr-1|R170|awesome×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 from live reign-13 (≠ R165/R166@5e-6 / R167 G4 / R77 Tok / R169 HiRank)"
  # p2668: R171 warm-armed on mine-crown-1 GPUs 4–5 — do not re-rent
  # "mine-r171-awesome-hialpha-hirank-bigg-1|R171|awesome×HiAlpha×HiRank×BigG α=128 r=64 G=16 lr=5e-6 @6144 from live reign-13 (≠ R165 r16 G4 / R166 BigG r16 / R169 HiRank G4 / R170 BigG×HiLR / R29 Tok HiRank)"
  # p2669: R172 warm-armed on mine-crown-1 GPUs 6–7 — do not re-rent
  # "mine-r172-awesome-hialpha-hirank-hilr-1|R172|awesome×HiAlpha×HiRank×HiLR α=128 r=64 G=4 lr=2e-5 @6144 from live reign-13 (≠ R165 r16@5e-6 / R167 HiLR r16 / R169 HiRank@5e-6 / R170 BigG×HiLR / R171 HiRank×BigG / R29 Tok HiRank)"
  # p2677: R173 warm-armed on mine-r165 lunar GPUs 6–7 after R168 REFUTE — do not re-rent
  # "mine-r173-awesome-hialpha-hirank-longctx-1|R173|awesome×HiAlpha×HiRank×LongCtx α=128 r=64 G=4 lr=5e-6 16384/1024 from live reign-13 (≠ R165 r16@6144 / R168 LongCtx r16 / R169 HiRank@6144 / R172 HiRank×HiLR / R171 HiRank×BigG / R67 Tok)"
  # p2685: R175 warm-armed on mine-r165 lunar GPUs 4–5 after R173 post→6,7 — do not re-rent
  # p2684: R174 warm-armed on mine-crown-1 GPUs 6–7 after R172 chall purge — do not re-rent
  # "mine-r174-awesome-hialpha-hirank-bigg-hilr-1|R174|awesome×HiAlpha×HiRank×BigG×HiLR α=128 r=64 G=16 lr=2e-5 @6144 from live reign-13 (≠ R165/R170/R171/R172/R173)"
  "mine-r175-awesome-hialpha-bigg-longctx-1|R175|awesome×HiAlpha×BigG×LongCtx α=128 r=16 G=16 lr=5e-6 16384/1024 from live reign-13 (≠ R165/R166/R168/R170/R173/R174)"
  "mine-r176-awesome-hialpha-hilr-longctx-1|R176|awesome×HiAlpha×HiLR×LongCtx α=128 r=16 G=4 lr=2e-5 16384/1024 from live reign-13 (≠ R165/R167@6144 / R168@5e-6 / R170/R172/R173/R175 / R76 Tok)"
  "mine-r177-awesome-hialpha-bigg-hilr-longctx-1|R177|awesome×HiAlpha×BigG×HiLR×LongCtx α=128 r=16 G=16 lr=2e-5 16384/1024 from live reign-13 (≠ R165/R170@6144 / R175@5e-6 / R176 G4 / R174 HiRank / R79 Tok)"
  "mine-r178-awesome-hialpha-hirank-bigg-longctx-1|R178|awesome×HiAlpha×HiRank×BigG×LongCtx α=128 r=64 G=16 lr=5e-6 16384/1024 from live reign-13 (≠ R165/R171@6144 / R173 G4 / R175 r16 / R174 HiRank×BigG×HiLR / R177 r16)"
  "mine-r179-awesome-hialpha-hirank-hilr-longctx-1|R179|awesome×HiAlpha×HiRank×HiLR×LongCtx α=128 r=64 G=4 lr=2e-5 16384/1024 from live reign-13 (≠ R165/R172@6144 / R173@5e-6 / R176 r16 / R178 BigG G16@5e-6 / R174)"
  "mine-r180-awesome-hialpha-hirank-bigg-hilr-longctx-1|R180|awesome×HiAlpha×HiRank×BigG×HiLR×LongCtx α=128 r=64 G=16 lr=2e-5 16384/1024 from live reign-13 (≠ R165@6144 / R174@6144 / R178@5e-6 / R179 G4 / R177 r16)"
  # p2749: R181 lean on crown 4,5 — do not re-rent
  # "mine-r181-awesome-reason-sft-1|R181|awesome×Reason-SFT thought-mask α=128 r=16 lr=2e-5 @8192 from live reign-13 (≠ R165–R180 GRPO / ≠ R1 Tok SFT)"
  # p2753: R182 lean on crown 4,5 — do not re-rent
  # "mine-r182-awesome-datafilt-sft-1|R182|awesome×DataFilt-SFT top150 Reason EP=2 α=128 r=16 lr=2e-5 @8192 from live reign-13 (≠ R181 n460 EP1 / ≠ R165–R180 GRPO / ≠ R7 Tok datafilt)"
  # p2756: R182 REFUTE m=-0.0084 — purge; R35 lean on crown 4,5 — do not re-rent
  # "mine-r35-talent-longctx-1|R35|Talent×LongCtx max_len=16384 (R19 SIGNAL_POS × R24 LongCtx)"
  # demoted p2608: "mine-r161-guass-hialpha-bigg-1|R161|guass×HiAlpha×BigG …"
  "mine-r36-talent-hilr-1|R36|Talent×HiLR lr=2e-5 (R19 SIGNAL_POS × R28 HiLR; ≠ R35 LongCtx)"
  # p2759 lean on brave 6,7 — do not re-rent
# "mine-r37-golden-longctx-1|R37|Golden×LongCtx max_len=16384 (R22 golden × R24 LongCtx SIGNAL; ≠ R22@6144)"
  "mine-r38-diane-longctx-1|R38|Diane×LongCtx max_len=16384 (R23 diane × R24 LongCtx SIGNAL; ≠ R23@6144)"
  "mine-r39-ckp333-1|R39|ckp333-init Reason-GRPO (reign-5 parent; ≠ Tok/Talent/golden/diane/guass)"
  "mine-r40-ckp333-longctx-1|R40|ckp333×LongCtx max_len=16384 (R39 parent × R24 LongCtx SIGNAL; ≠ R39@6144)"
  "mine-r41-talent-bigg-1|R41|Talent×BigG G=16 (R19 SIGNAL_POS × R27 G isolate; ≠ R35 LongCtx / R36 HiLR / R27 Tok)"
  "mine-r42-golden-hilr-1|R42|Golden×HiLR lr=2e-5 (R22 golden × R28 HiLR; ≠ R37 LongCtx / R28 Tok / R36 Talent×HiLR)"
  "mine-r43-diane-hilr-1|R43|Diane×HiLR lr=2e-5 (R23 diane × R28 HiLR; ≠ R38 LongCtx / R42 Golden×HiLR / R36 Talent×HiLR)"
  "mine-r44-ckp333-hilr-1|R44|ckp333×HiLR lr=2e-5 (R39 ckp333 × R28 HiLR; ≠ R39@5e-6 / R40 LongCtx / R42 Golden×HiLR)"
  "mine-r45-diane-bigg-1|R45|Diane×BigG G=16 (R23 diane × R27 BigG; ≠ R41 Talent×BigG / R43 Diane×HiLR / R38 LongCtx / R27 Tok)"
  "mine-r46-golden-bigg-1|R46|Golden×BigG G=16 (R22 golden × R27 BigG; ≠ R41 Talent×BigG / R45 Diane×BigG / R42 Golden×HiLR / R37 LongCtx / R27 Tok)"
  # p2801: crown flipped fqb→marsplan queen reign16 → QUEUE HEAD was marsplan-init
  # p2814: R210 lean-warm on mine-r165 GPUs 4–5 (R204 recipe) — do not re-rent R204
  # "mine-r204-marsplan-hialpha-1|R204|marsplan×HiAlpha α=128 r=16 G=4 lr=5e-6 @6144 from live reign-16 marsplan0624/affine-5gedzafcvg-queen@556d02a2 (≠ R165 awesome / R39 ckp333 / R158 guass; n80 king=marsplan)"
  # "mine-r210-marsplan-hialpha-1|R210|marsplan×HiAlpha α=128 r=16 G=4 lr=5e-6 @6144 lean on lunar (R204 recipe; ≠ R209 HiRank / R205 BigG; n80 king=marsplan)"
  # p2824: R205 lean-warm on mine-crown-1 GPUs 6–7 — do not re-rent
  # "mine-r205-marsplan-hialpha-bigg-1|R205|marsplan×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 from live reign-16 (≠ R204/R210 G=4; n80 king=marsplan)"
  # p2815: R206 lean-warm on mine-crown-1 GPUs 4–5 — do not re-rent
  # "mine-r206-marsplan-hialpha-hilr-1|R206|marsplan×HiAlpha×HiLR α=128 r=16 G=4 lr=2e-5 @6144 from live reign-16 (≠ R204 lr=5e-6 / R205 G=16; n80 king=marsplan)"
  "mine-r207-marsplan-hialpha-hirank-hilr-1|R207|marsplan×HiAlpha×HiRank×HiLR α=128 r=64 G=4 lr=2e-5 @6144 from live reign-16 (R172 recipe; n80 king=marsplan)"
  "mine-r208-marsplan-hialpha-longctx-1|R208|marsplan×HiAlpha×LongCtx α=128 r=16 G=4 lr=5e-6 16384/1024 from live reign-16 (≠ R204@6144; n80 king=marsplan)"
  # p2816: QUEUE marsplan×HiRank×BigG (R171 recipe on live-king)
  # p2835: R211 lean-warm on mine-r165 GPUs 4–5 — do not re-rent
  # "mine-r211-marsplan-hialpha-hirank-bigg-1|R211|marsplan×HiAlpha×HiRank×BigG α=128 r=64 G=16 lr=5e-6 @6144 from live reign-16 (R171 recipe; ≠ R205/R209/R207; n80 king=marsplan)"
  # p2817: QUEUE→HEAD marsplan×BigG×LongCtx (R175 recipe on live-king)
  # p2841: R212 lean-warm on mine-r165 GPUs 6–7 after R178 purge — do not re-rent
  # "mine-r212-marsplan-hialpha-bigg-longctx-1|R212|marsplan×HiAlpha×BigG×LongCtx α=128 r=16 G=16 lr=5e-6 16384/1024 from live reign-16 (R175 recipe; ≠ R208/R205/R211; n80 king=marsplan)"
  # p2818: QUEUE marsplan×HiRank×LongCtx (R173 recipe on live-king)
  # p2844: R213 lean-warm on mine-r160 GPUs 6–7 after R216 REFUTE+purge — do not re-rent
  # "mine-r213-marsplan-hialpha-hirank-longctx-1|R213|marsplan×HiAlpha×HiRank×LongCtx α=128 r=64 G=4 lr=5e-6 16384/1024 from live reign-16 (R173 recipe; ≠ R208/R209/R211/R212; n80 king=marsplan)"
  # p2819: QUEUE→HEAD marsplan×HiRank×BigG×HiLR (R174/R172×BigG on live-king)
  # p2846: R214 lean-warm on mine-r160 GPUs 4–5 after R207 REFUTE — do not re-rent.
  # "mine-r214-marsplan-hialpha-hirank-bigg-hilr-1|R214|marsplan×HiAlpha×HiRank×BigG×HiLR α=128 r=64 G=16 lr=2e-5 @6144 from live reign-16 (R174/R172×BigG; ≠ R207 G4 / R211@5e-6; n80 king=marsplan)"
  # p2820: QUEUE→HEAD marsplan×BigG×HiLR (R170 recipe on live-king)
  # p2847: R215 lean-warm on mine-crown-1 GPUs 4–5 after R208 SIGNAL — do not re-rent.
  # "mine-r215-marsplan-hialpha-bigg-hilr-1|R215|marsplan×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 from live reign-16 (R170 recipe; ≠ R205@5e-6 / R206 G4 / R214 r64; n80 king=marsplan)"
  # p2825: QUEUE→HEAD marsplan×HiRank×BigG×LongCtx (R178 recipe on live-king)
  "mine-r217-marsplan-hialpha-hirank-bigg-longctx-1|R217|marsplan×HiAlpha×HiRank×BigG×LongCtx α=128 r=64 G=16 lr=5e-6 16384/1024 from live reign-16 (R178 recipe; ≠ R212/R211/R213/R214; n80 king=marsplan)"
  # p2849: QUEUE marsplan×Teacher-ZC (R9 method on live-king)
  "mine-r232-marsplan-teacher-zc-1|R232|marsplan×Teacher-ZC lr=1e-5 r=32 α=64 EP=3 @16384 teacher z_C from live reign-16 (R9 method; ≠ R221–R231 / ≠ R9 Tok; n80 king=marsplan)"
  # p2850: QUEUE Genesis-nonking×HiAlpha-GRPO (R204 knobs on Genesis base; n80 king=marsplan)
  "mine-r233-genesis-nonking-grpo-1|R233|Genesis-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 from Genesis@abe89194 (≠ R204–R232 marsplan / ≠ R5 FullFT / ≠ R231 KL; n80 king=marsplan)"
  # p2851: QUEUE Tok-af10-nonking×HiAlpha-GRPO (R204 knobs on Tok af10 reign-4; n80 king=marsplan)
  "mine-r234-tok-nonking-grpo-1|R234|Tok-af10-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 from Tok@eb8bf9a (≠ R204–R233 marsplan/Genesis / ≠ Tok ladder R3–R133 / ≠ R5 FullFT / ≠ R233 Genesis; n80 king=marsplan)"
  # p2852: QUEUE Talent-nonking×HiAlpha-GRPO (R204 knobs on TalentPigs reign-3; n80 king=marsplan)
  "mine-r235-talent-nonking-grpo-1|R235|Talent-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 from Talent@dbfbb3e2 (≠ R204–R234 marsplan/Genesis/Tok / ≠ Talent ladder R19–R133 / ≠ R5 FullFT / ≠ R233 Genesis / ≠ R234 Tok; n80 king=marsplan)"
  # p2853: QUEUE Kevin-nonking×HiAlpha-GRPO (R204 knobs on kevin954 reign-2; n80 king=marsplan)
  "mine-r236-kevin-nonking-grpo-1|R236|Kevin-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 from kevin@6a5815fa (≠ R204–R235 marsplan/Genesis/Tok/Talent / ≠ R5 FullFT / ≠ R233 Genesis / ≠ R234 Tok / ≠ R235 Talent; n80 king=marsplan)"
  # p2854: QUEUE Pandora-nonking×HiAlpha-GRPO (R204 knobs on pandora-box reign-1; n80 king=marsplan)
  "mine-r237-pandora-nonking-grpo-1|R237|Pandora-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 from pandora@5218b138 (≠ R204–R236 marsplan/Genesis/Tok/Talent/Kevin / ≠ R21 α=32 / ≠ R5 FullFT / ≠ R233–R236; n80 king=marsplan)"
  "mine-r238-ckp333-nonking-grpo-1|R238|Ckp333-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 from tolegend@24c137e8 (≠ R204–R237 marsplan/Genesis/Tok/Talent/Kevin/Pandora / ≠ R39/R89 / ≠ R5 FullFT / ≠ R233–R237; n80 king=marsplan)"
  # p2826: QUEUE marsplan×HiRank×HiLR×LongCtx (R179 recipe on live-king)
  "mine-r218-marsplan-hialpha-hirank-hilr-longctx-1|R218|marsplan×HiAlpha×HiRank×HiLR×LongCtx α=128 r=64 G=4 lr=2e-5 16384/1024 from live reign-16 (R179 recipe; ≠ R207@6144 / R213@5e-6 / R216 r16 / R217 G16; n80 king=marsplan)"
  # p2807–p2810/p2816–p2826: R205–R208+R211–R218 also in API QUEUE (shell lag OK — API waiter is live)
  # p2813: R209 lean-warm on mine-r160 GPUs 4–5 after R39 SIGNAL_POS_BELOW+purge — do not re-rent
  # "mine-r209-marsplan-hialpha-hirank-1|R209|marsplan×HiAlpha×HiRank α=128 r=64 G=4 lr=5e-6 @6144 from live reign-16 (≠ R204 r=16 / R207 HiRank×HiLR; n80 king=marsplan)"
  "mine-r47-ckp333-bigg-1|R47|ckp333×BigG G=16 demoted after reign16 (R39 ckp333 × R27 BigG; ≠ R41 Talent×BigG / R45 Diane×BigG / R46 Golden×BigG / R44 HiLR / R40 LongCtx / R39 G=4 / R27 Tok)"
  "mine-r48-bigg-hilr-1|R48|Tok BigG×HiLR G=16 lr=2e-5 (R27×R28 compound; ≠ R27 G-only / R28 lr-only / R34 LongCtx×HiLR / R3b)"
  "mine-r49-talent-bigg-hilr-1|R49|Talent×BigG×HiLR G=16 lr=2e-5 (R41×R36/R28 compound; ≠ R41 G-only / R36 HiLR@G=4 / R48 Tok BigG×HiLR / R35 LongCtx)"
  "mine-r50-diane-bigg-hilr-1|R50|Diane×BigG×HiLR G=16 lr=2e-5 (R45×R43/R28 compound; ≠ R45 G-only / R43 HiLR@G=4 / R49 Talent BigG×HiLR / R48 Tok / R38 LongCtx)"
  "mine-r51-golden-bigg-hilr-1|R51|Golden×BigG×HiLR G=16 lr=2e-5 (R46×R42/R28 compound; ≠ R46 G-only / R42 HiLR@G=4 / R50 Diane BigG×HiLR / R49 Talent / R48 Tok / R37 LongCtx)"
  "mine-r52-ckp333-bigg-hilr-1|R52|ckp333×BigG×HiLR G=16 lr=2e-5 (R47×R44/R28 compound; ≠ R47 G-only / R44 HiLR@G=4 / R51 Golden BigG×HiLR / R50 Diane / R49 Talent / R48 Tok / R40 LongCtx)"
  "mine-r53-talent-longctx-hilr-1|R53|Talent×LongCtx×HiLR 16384/1024 lr=2e-5 (R35×R36/R28 compound; ≠ R35 LongCtx@5e-6 / R36 HiLR@6144 / R34 Tok LongCtx×HiLR / R49 Talent×BigG×HiLR)"
  "mine-r54-diane-longctx-hilr-1|R54|Diane×LongCtx×HiLR 16384/1024 lr=2e-5 (R38×R43/R28 compound; ≠ R38 LongCtx@5e-6 / R43 HiLR@6144 / R34 Tok LongCtx×HiLR / R50 Diane×BigG×HiLR / R53 Talent×LongCtx×HiLR)"
  "mine-r55-golden-longctx-hilr-1|R55|Golden×LongCtx×HiLR 16384/1024 lr=2e-5 (R37×R42/R28 compound; ≠ R37 LongCtx@5e-6 / R42 HiLR@6144 / R34 Tok LongCtx×HiLR / R51 Golden×BigG×HiLR / R53 Talent×LongCtx×HiLR / R54 Diane×LongCtx×HiLR)"
  "mine-r56-ckp333-longctx-hilr-1|R56|ckp333×LongCtx×HiLR 16384/1024 lr=2e-5 (R40×R44/R28 compound; ≠ R40 LongCtx@5e-6 / R44 HiLR@6144 / R34 Tok LongCtx×HiLR / R52 ckp333×BigG×HiLR / R53–R55 LongCtx×HiLR parents)"
  "mine-r57-longctx-bigg-1|R57|Tok LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R24×R27 compound; ≠ R34 LongCtx×HiLR / R48 BigG×HiLR / R24 / R27)"
  "mine-r58-talent-longctx-bigg-1|R58|Talent×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R19×R24×R27; ≠ R57 Tok / R53 Talent×LongCtx×HiLR / R41 Talent×BigG@6144 / R35 LongCtx / R49 BigG×HiLR)"
  "mine-r59-diane-longctx-bigg-1|R59|Diane×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R23×R24×R27; ≠ R58 Talent / R57 Tok / R54 Diane×LongCtx×HiLR / R45 Diane×BigG@6144 / R38 LongCtx / R50 Diane×BigG×HiLR)"
  "mine-r60-golden-longctx-bigg-1|R60|Golden×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R22×R24×R27; ≠ R59 Diane / R58 Talent / R57 Tok / R55 Golden×LongCtx×HiLR / R46 Golden×BigG@6144 / R37 LongCtx / R51 BigG×HiLR)"
  "mine-r61-ckp333-longctx-bigg-1|R61|ckp333×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R39/R40×R24×R27; ≠ R60 Golden / R59 Diane / R58 Talent / R57 Tok / R56 ckp333×LongCtx×HiLR / R47 ckp333×BigG@6144 / R40 LongCtx@G=4 / R52 BigG×HiLR)"
  "mine-r62-longctx-bigg-hilr-1|R62|Tok LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R24×R27×R28; ≠ R57@5e-6 / R34 LongCtx×HiLR@G=4 / R48 BigG×HiLR@6144 / R24 / R27 / R28)"
  "mine-r63-talent-longctx-bigg-hilr-1|R63|Talent×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R19×R24×R27×R28; ≠ R62 Tok triple / R58@5e-6 / R53 LongCtx×HiLR@G=4 / R49 BigG×HiLR@6144 / R41 BigG / R35 LongCtx)"
  "mine-r64-diane-longctx-bigg-hilr-1|R64|Diane×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R23×R24×R27×R28; ≠ R63 Talent triple / R62 Tok / R59@5e-6 / R54 LongCtx×HiLR@G=4 / R50 BigG×HiLR@6144 / R45 BigG / R38 LongCtx)"
  "mine-r65-golden-longctx-bigg-hilr-1|R65|Golden×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R22×R24×R27×R28; ≠ R64 Diane triple / R63 Talent / R62 Tok / R60@5e-6 / R55 LongCtx×HiLR@G=4 / R51 BigG×HiLR@6144 / R46 BigG / R37 LongCtx)"
  "mine-r66-ckp333-longctx-bigg-hilr-1|R66|ckp333×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R39/R40×R24×R27×R28; ≠ R65 Golden triple / R64 Diane / R63 Talent / R62 Tok / R61@5e-6 / R56 LongCtx×HiLR@G=4 / R52 BigG×HiLR@6144 / R47 BigG / R40 LongCtx)"
  # p2349: R67 warm on mine-r3 — skip
  # "mine-r67-hirank-longctx-1|R67|Tok HiRank×LongCtx r=64 16384/1024 (R29×R24; ≠ R29@6144 / R24@r=16 / R34 LongCtx×HiLR / R62 LongCtx×BigG×HiLR / R28 HiLR)"
  # p2398: reign-7 → fjq-init head of queue (also HEAD in API QUEUE)
  # p2411: crown=guass reign8 → R158 guass×HiAlpha HEAD; R156 already warm on mine-r3
  # p2425: R158 warm-armed on mine-r3-grpo-1 after R156 REFUTE vs guass — skip
  # "mine-r158-guass-hialpha-1|R158|guass×HiAlpha α=128 r=16 G=4 @6144 from live reign-8 ttttxxxxsada/Affine-5guassq3tu@e86758f5 (≠ R33 α=32 REFUTED / R157 fjq×HiAlpha / R30 Tok×HiAlpha / R75 Tok×HiAlpha×BigG; n80 king=guass)"
  # "mine-r156-fjq-grpo-1|R156|fjq-init Reason-GRPO from live reign-7 dent1s2/Affine-5FjqRq3dGA-v1@cfd789c9 (≠ R3 Tok / R33 guass-init REFUTED / R19–R23; n80 king=same fjq)"
  # p2426: r31 REBOOT_FAILED → HF rematch HEAD
  # p2428: rematch warm on mine-r3 GPUs4–5 — skip re-rent
  # "mine-r157-rematch-1|R157|HF rematch fjq×HiAlpha LoRA (unconst/Affine-5czsc2fc98-r157-lora@step-150) merge+n80 vs guass after r31 REBOOT_FAILED"
  # p2434: reign-9 legend; R69 REFUTE → legend-init HEAD (also HEAD in API QUEUE; warm on crown)
  # "mine-r159-legend-grpo-1|R159|legend-init Reason-GRPO α=128 r=16 G=4 @6144 from live reign-9 diceofgod/affine-5fjgc5jhxq-legend@d259cb38 (≠ R69 Tok REFUTED vs legend / R158 guass×HiAlpha / R156·R157 fjq / R30·R75 Tok×HiAlpha; n80 king=legend)"
  # p2448: R159 REFUTE; R75 warm on mine-r4 — do not re-rent
  # "mine-r75-hialpha-bigg-1|R75|Tok HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R30×R27; ≠ R30@G=4 / R27@α32 / R74@HiLR+G=4 / R73@16384 / R69@r64)"
  # p2450: R68 warm-armed on mine-r3-grpo-1 — do not re-rent.
  # "mine-r68-hirank-hilr-1|R68|Tok HiRank×HiLR r=64 lr=2e-5 (R29×R28; ≠ R29@5e-6 / R28@r=16 / R67 HiRank×LongCtx / R34 LongCtx×HiLR@r=16 / R3b G=8)"
  # p2356: R69 warm on mine-crown-1 after R32 REFUTE — skip
  # "mine-r69-hirank-bigg-1|R69|Tok HiRank×BigG r=64 G=16 (R29×R27; ≠ R29@G=4 / R27@r=16 / R68 HiRank×HiLR / R67 HiRank×LongCtx / R48 BigG×HiLR@r=16 / R3b G=8)"
  # p2364: R71 warm on mine-r4 after R27 REFUTE — skip
  # "mine-r71-hirank-longctx-bigg-1|R71|Tok HiRank×LongCtx×BigG r=64 16384/1024 G=16 lr=5e-6 (R29×R24×R27; ≠ R67@G=4 / R69@6144 / R70@G=4/2e-5 / R57@r=16 / R62@r=16)"
  # p2459: reign-10 thermopylae → live-king-init HEAD (Tok ladder demoted)
  "mine-r160-thermopylae-grpo-1|R160|thermopylae-init Reason-GRPO α=128 r=16 G=4 @6144 from live reign-10 thermopylae-777/Affine-5eptsnvsre-v1@b5f748bf (≠ R159 legend REFUTED / R158 guass×HiAlpha / R69 Tok vs legend / R70 Tok ladder; n80 king=thermopylae)"
  "mine-r70-hirank-longctx-hilr-1|R70|Tok HiRank×LongCtx×HiLR r=64 16384/1024 lr=2e-5 (R29×R24×R28; ≠ R67@5e-6 / R68@6144 / R34@r=16 / R69 BigG / R62@r=16)"
  # p2502: R72 warm on mine-r160 — skip
  # "mine-r72-hirank-longctx-bigg-hilr-1|R72|Tok HiRank×LongCtx×BigG×HiLR r=64 16384/1024 G=16 lr=2e-5 (R29×R24×R27×R28; ≠ R71@5e-6 / R70@G=4 / R62@r=16 / R69@6144 / R68@6144 / R67@G=4/5e-6)"
  "mine-r73-hialpha-longctx-1|R73|Tok HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R30×R24; ≠ R30@6144 / R24@α32 / R34@α32+HiLR / R67@r64)"
  # p2530: R74 warm-armed on mine-crown-1 after R78 SIGNAL_POS_BELOW — do not re-rent while warm.
  # "mine-r74-hialpha-hilr-1|R74|Tok HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R30×R28; ≠ R30@5e-6 / R28@α32 / R73@16384+5e-6 / R34@α32 / R68@r64)"
  # duplicate removed p2425 — kept earlier QUEUE head entry
  # "mine-r75-hialpha-bigg-1|R75|Tok HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R30×R27; ≠ R30@G=4 / R27@α32 / R74@HiLR+G=4 / R73@16384 / R69@r64)"
  # p2550: R74 FINAL REFUTE → R76 warm-armed on mine-crown-1 — do not re-rent while warm.
  # "mine-r76-hialpha-longctx-hilr-1|R76|Tok HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R30×R24×R28; ≠ R73@5e-6 / R74@6144 / R34@α32 / R70@r64 / R75@G16)"
  # p2565: R77 warm on mine-r160 — skip
  # "mine-r77-hialpha-bigg-hilr-1|R77|Tok HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R30×R27×R28; ≠ R75@5e-6 / R74@G4 / R48@α32 / R76@16384 / R69@r64)"
  "mine-r78-hialpha-longctx-bigg-1|R78|Tok HiAlpha×LongCtx×BigG α=128 r=16 16384/1024 G=16 lr=5e-6 (R30×R24×R27; ≠ R73@G4 / R75@6144 / R76@2e-5 / R77@6144+HiLR / R57@α32 / R71@r64)"
  # p2567: R79 warm on mine-crown-1 — skip
  # "mine-r79-hialpha-longctx-bigg-hilr-1|R79|Tok HiAlpha×LongCtx×BigG×HiLR α=128 r=16 16384/1024 G=16 lr=2e-5 (R30×R24×R27×R28; ≠ R78@5e-6 / R76@G4 / R77@6144 / R72@r64 / R62@α32)"
  "mine-r80-talent-hialpha-1|R80|Talent×HiAlpha α=128 r=16 lr=5e-6 @6144 (R19×R30; ≠ R19@α32 / R30 Tok / R36 Talent×HiLR / R35 LongCtx / R74 Tok×HiAlpha×HiLR)"
  "mine-r81-talent-hialpha-longctx-1|R81|Talent×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R80×R24; ≠ R80@6144 / R35@α32 / R73 Tok×HiAlpha×LongCtx / R53 Talent×LongCtx×HiLR / R19@α32)"
  "mine-r82-talent-hialpha-hilr-1|R82|Talent×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R80×R36/R28; ≠ R80@5e-6 / R36@α32 / R74 Tok×HiAlpha×HiLR / R81 LongCtx / R30 Tok)"
  "mine-r83-talent-hialpha-bigg-1|R83|Talent×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R80×R41/R27; ≠ R80@G4 / R41@α32 / R75 Tok×HiAlpha×BigG / R82 HiLR / R81 LongCtx / R30 Tok)"
  "mine-r84-talent-hialpha-longctx-hilr-1|R84|Talent×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R81×R82/R28; ≠ R81@5e-6 / R82@6144 / R76 Tok×HiAlpha×LongCtx×HiLR / R53@α32 / R80@6144)"
  "mine-r85-talent-hialpha-longctx-bigg-1|R85|Talent×HiAlpha×LongCtx×BigG α=128 r=16 16384/1024 G=16 lr=5e-6 (R81×R83/R27; ≠ R81@G4 / R83@6144 / R78 Tok×HiAlpha×LongCtx×BigG / R58@α32 / R84 HiLR)"
  "mine-r86-talent-hialpha-longctx-bigg-hilr-1|R86|Talent×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 16384/1024 G=16 lr=2e-5 (R85×R84/R28; ≠ R85@5e-6 / R84@G4 / R79 Tok quintuple / R63@α32 / R83@6144)"
  "mine-r87-diane-hialpha-1|R87|Diane×HiAlpha α=128 r=16 lr=5e-6 @6144 (R23×R30; ≠ R23@α32 / R80 Talent×HiAlpha / R30 Tok / R43 Diane×HiLR / R38 LongCtx)"
  "mine-r88-golden-hialpha-1|R88|Golden×HiAlpha α=128 r=16 lr=5e-6 @6144 (R22×R30; ≠ R22@α32 / R87 Diane×HiAlpha / R80 Talent×HiAlpha / R30 Tok / R42 Golden×HiLR / R37 LongCtx)"
  "mine-r89-ckp333-hialpha-1|R89|ckp333×HiAlpha α=128 r=16 lr=5e-6 @6144 (R39×R30; ≠ R39@α32 / R88 Golden×HiAlpha / R87 Diane×HiAlpha / R80 Talent×HiAlpha / R30 Tok / R44 ckp333×HiLR / R40 LongCtx)"
  "mine-r90-diane-hialpha-longctx-1|R90|Diane×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R87×R24; ≠ R87@6144 / R81 Talent×HiAlpha×LongCtx / R73 Tok×HiAlpha×LongCtx / R38 Diane×LongCtx@α32 / R88 Golden×HiAlpha / R89 ckp333×HiAlpha / R30 Tok)"
  "mine-r91-golden-hialpha-longctx-1|R91|Golden×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R88×R24; ≠ R88@6144 / R90 Diane×HiAlpha×LongCtx / R81 Talent×HiAlpha×LongCtx / R73 Tok×HiAlpha×LongCtx / R37 Golden×LongCtx@α32 / R89 ckp333×HiAlpha / R30 Tok)"
  "mine-r92-ckp333-hialpha-longctx-1|R92|ckp333×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R89×R24; ≠ R89@6144 / R91 Golden×HiAlpha×LongCtx / R90 Diane×HiAlpha×LongCtx / R81 Talent×HiAlpha×LongCtx / R73 Tok×HiAlpha×LongCtx / R40 ckp333×LongCtx@α32 / R30 Tok)"
  "mine-r93-diane-hialpha-hilr-1|R93|Diane×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R87×R43/R28; ≠ R87@5e-6 / R43@α32 / R90 LongCtx / R82 Talent×HiAlpha×HiLR / R74 Tok×HiAlpha×HiLR / R30 Tok)"
  "mine-r94-golden-hialpha-hilr-1|R94|Golden×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R88×R42/R28; ≠ R88@5e-6 / R42@α32 / R91 LongCtx / R93 Diane×HiAlpha×HiLR / R82 Talent×HiAlpha×HiLR / R74 Tok×HiAlpha×HiLR / R30 Tok)"
  "mine-r95-ckp333-hialpha-hilr-1|R95|ckp333×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R89×R44/R28; ≠ R89@5e-6 / R44@α32 / R92 LongCtx / R94 Golden×HiAlpha×HiLR / R93 Diane×HiAlpha×HiLR / R82 Talent×HiAlpha×HiLR / R74 Tok×HiAlpha×HiLR / R30 Tok)"
  "mine-r96-diane-hialpha-bigg-1|R96|Diane×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R87×R45/R27; ≠ R87@G4 / R45@α32 / R93 HiLR / R90 LongCtx / R83 Talent×HiAlpha×BigG / R75 Tok×HiAlpha×BigG / R50 Diane×BigG×HiLR / R30 Tok)"
  "mine-r97-golden-hialpha-bigg-1|R97|Golden×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R88×R46/R27; ≠ R88@G4 / R46@α32 / R94 HiLR / R91 LongCtx / R96 Diane×HiAlpha×BigG / R75 Tok×HiAlpha×BigG / R51 Golden×BigG×HiLR / R30 Tok)"
  "mine-r98-ckp333-hialpha-bigg-1|R98|ckp333×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R89×R47/R27; ≠ R89@G4 / R47@α32 / R95 HiLR / R92 LongCtx / R97 Golden×HiAlpha×BigG / R96 Diane×HiAlpha×BigG / R75 Tok×HiAlpha×BigG / R52 ckp333×BigG×HiLR / R30 Tok)"
  "mine-r99-talent-hialpha-bigg-hilr-1|R99|Talent×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R83×R82/R28; ≠ R83@5e-6 / R82@G4 / R49@α32 / R86 LongCtx / R77 Tok×HiAlpha×BigG×HiLR / R98 ckp333×HiAlpha×BigG / R30 Tok)"
  "mine-r100-diane-hialpha-bigg-hilr-1|R100|Diane×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R96×R93/R28; ≠ R96@5e-6 / R93@G4 / R50@α32 / R90 LongCtx / R99 Talent×HiAlpha×BigG×HiLR / R77 Tok×HiAlpha×BigG×HiLR / R30 Tok)"
  "mine-r101-golden-hialpha-bigg-hilr-1|R101|Golden×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R97×R94/R28; ≠ R97@5e-6 / R94@G4 / R51@α32 / R91 LongCtx / R100 Diane×HiAlpha×BigG×HiLR / R99 Talent×HiAlpha×BigG×HiLR / R77 Tok×HiAlpha×BigG×HiLR / R30 Tok)"
  "mine-r102-ckp333-hialpha-bigg-hilr-1|R102|ckp333×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R98×R95/R28; ≠ R98@5e-6 / R95@G4 / R52@α32 / R92 LongCtx / R101 Golden×HiAlpha×BigG×HiLR / R100 Diane×HiAlpha×BigG×HiLR / R99 Talent×HiAlpha×BigG×HiLR / R77 Tok×HiAlpha×BigG×HiLR / R30 Tok)"
  "mine-r103-diane-hialpha-longctx-hilr-1|R103|Diane×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R90×R93/R28; ≠ R90@5e-6 / R93@6144 / R100 BigG×HiLR@6144 / R84 Talent×HiAlpha×LongCtx×HiLR / R76 Tok×HiAlpha×LongCtx×HiLR / R54@α32 / R30 Tok)"
  "mine-r104-golden-hialpha-longctx-hilr-1|R104|Golden×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R91×R94/R28; ≠ R91@5e-6 / R94@6144 / R101 BigG×HiLR@6144 / R103 Diane×HiAlpha×LongCtx×HiLR / R84 Talent×HiAlpha×LongCtx×HiLR / R76 Tok×HiAlpha×LongCtx×HiLR / R55@α32 / R30 Tok)"
  "mine-r105-ckp333-hialpha-longctx-hilr-1|R105|ckp333×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R92×R95/R28; ≠ R92@5e-6 / R95@6144 / R102 BigG×HiLR@6144 / R104 Golden×HiAlpha×LongCtx×HiLR / R103 Diane×HiAlpha×LongCtx×HiLR / R84 Talent×HiAlpha×LongCtx×HiLR / R76 Tok×HiAlpha×LongCtx×HiLR / R56@α32 / R30 Tok)"
  "mine-r106-diane-hialpha-longctx-bigg-hilr-1|R106|Diane×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=16 16384/1024 lr=2e-5 (R103×R100/R27; ≠ R103@G4 / R100@6144 / R86 Talent×HiAlpha×LongCtx×BigG×HiLR / R79 Tok×HiAlpha×LongCtx×BigG×HiLR / R64@α32 / R105 ckp333×HiAlpha×LongCtx×HiLR / R104 Golden×HiAlpha×LongCtx×HiLR)"
  "mine-r107-golden-hialpha-longctx-bigg-hilr-1|R107|Golden×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=16 16384/1024 lr=2e-5 (R104×R101/R27)"
  "mine-r108-ckp333-hialpha-longctx-bigg-hilr-1|R108|ckp333×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=16 16384/1024 lr=2e-5 (R105×R102/R27; ≠ R105@G4 / R102@6144 / R107 Golden / R106 Diane / R66@α32 / R86 Talent / R79 Tok)"
  "mine-r109-nodrop-longctx-1|R109|Tok NoDrop×LongCtx drop=0.0 16384/1024 (R31×R24; ≠ R31@6144 / R24 drop=0.05 / R67 HiRank×LongCtx)"
  "mine-r113-nodrop-hialpha-1|R113|Tok NoDrop×HiAlpha drop=0.0 α=128 r=16 lr=5e-6 @6144/512 G=4 (R31×R30; ≠ R31@α32 / R30 drop=0.05 / R112 HiRank / R109–R111)"
  "mine-r114-nodrop-kl-1|R114|Tok NoDrop×KL drop=0.0 kl_coef=0.02 r=16 α=32 lr=5e-6 @6144/512 G=4 (R31×R32; ≠ R31 kl=0 / R32 drop=0.05 / R109–R113)"
  "mine-r115-nodrop-longctx-hilr-1|R115|Tok NoDrop×LongCtx×HiLR drop=0.0 16384/1024 lr=2e-5 r=16 α=32 G=4 (R109×R110; ≠ R109@5e-6 / R110@6144 / R34 drop=0.05 / R70 HiRank / R114 KL)"
  "mine-r116-nodrop-longctx-bigg-1|R116|Tok NoDrop×LongCtx×BigG drop=0.0 16384/1024 G=16 lr=5e-6 r=16 α=32 (R109×R111; ≠ R109@G=4 / R111@6144 / R115@G=4/2e-5 / R57 drop=0.05 / R114 KL)"
  "mine-r117-nodrop-longctx-hirank-1|R117|Tok NoDrop×LongCtx×HiRank drop=0.0 r=64 α=128 16384/1024 G=4 lr=5e-6 (R109×R112; ≠ R109@r16 / R112@6144 / R67 drop=0.05 / R115–R116)"
  "mine-r118-nodrop-longctx-hialpha-1|R118|Tok NoDrop×LongCtx×HiAlpha drop=0.0 α=128 r=16 16384/1024 G=4 lr=5e-6 (R109×R113; ≠ R109@α32 / R113@6144 / R73 drop=0.05 / R115–R117)"
  "mine-r119-nodrop-longctx-kl-1|R119|Tok NoDrop×LongCtx×KL drop=0.0 kl=0.02 r=16 α=32 16384/1024 G=4 lr=5e-6 (R109×R114; ≠ R109@kl=0 / R114@6144 / R32 drop=0.05 / R115–R118)"
  "mine-r120-nodrop-longctx-bigg-hilr-1|R120|Tok NoDrop×LongCtx×BigG×HiLR drop=0.0 G=16 lr=2e-5 r=16 α=32 16384/1024 (R115×R116; ≠ R115@G=4 / R116@5e-6 / R62 drop=0.05 / R119 KL)"
  "mine-r121-nodrop-longctx-hirank-hilr-1|R121|Tok NoDrop×LongCtx×HiRank×HiLR drop=0.0 r=64 α=128 16384/1024 lr=2e-5 G=4 (R117×R115; ≠ R117@5e-6 / R115@r16 / R120@r16G16 / R70 drop=0.05)"
  "mine-r122-nodrop-longctx-hialpha-hilr-1|R122|Tok NoDrop×LongCtx×HiAlpha×HiLR drop=0.0 α=128 r=16 16384/1024 lr=2e-5 G=4 (R118×R115; ≠ R118@5e-6 / R115@α32 / R121@r64 / R120@r16G16 / R76 drop=0.05)"
  "mine-r123-nodrop-longctx-kl-hilr-1|R123|Tok NoDrop×LongCtx×KL×HiLR drop=0.0 kl=0.02 r=16 α=32 16384/1024 lr=2e-5 G=4 (R119×R115; ≠ R119@5e-6 / R115@kl=0 / R114@6144 / R122@α128 / R32 drop=0.05)"
  "mine-r124-nodrop-longctx-kl-bigg-1|R124|Tok NoDrop×LongCtx×KL×BigG drop=0.0 kl=0.02 r=16 α=32 16384/1024 G=16 lr=5e-6 (R119×R116; ≠ R119@G=4 / R116@kl=0 / R120@kl=0+HiLR / R123@G=4+HiLR / R32 drop=0.05)"
  "mine-r125-nodrop-longctx-kl-bigg-hilr-1|R125|Tok NoDrop×LongCtx×KL×BigG×HiLR drop=0.0 kl=0.02 r=16 α=32 16384/1024 G=16 lr=2e-5 (R124×R123; ≠ R124@5e-6 / R123@G=4 / R120@kl=0 / R119@G=4 / R32 drop=0.05)"
  "mine-r126-nodrop-longctx-hirank-bigg-1|R126|Tok NoDrop×LongCtx×HiRank×BigG drop=0.0 r=64 α=128 16384/1024 G=16 lr=5e-6 (R117×R116; ≠ R117@G=4 / R116@r16 / R121@HiLR / R71 drop=0.05 / R125 KL)"
  "mine-r127-nodrop-longctx-hialpha-bigg-1|R127|Tok NoDrop×LongCtx×HiAlpha×BigG drop=0.0 α=128 r=16 16384/1024 G=16 lr=5e-6 (R118×R116; ≠ R118@G=4 / R116@α32 / R122@HiLR / R75 drop=0.05 / R126 HiRank×BigG)"
  "mine-r128-nodrop-longctx-hirank-bigg-hilr-1|R128|Tok NoDrop×LongCtx×HiRank×BigG×HiLR drop=0.0 r=64 α=128 16384/1024 G=16 lr=2e-5 (R126×R121; ≠ R126@5e-6 / R121@G=4 / R71 drop=0.05 / R125 KL / R127 HiAlpha×BigG)"
  "mine-r129-nodrop-longctx-hialpha-bigg-hilr-1|R129|Tok NoDrop×LongCtx×HiAlpha×BigG×HiLR drop=0.0 α=128 r=16 16384/1024 G=16 lr=2e-5 (R127×R122; ≠ R127@5e-6 / R122@G=4 / R128 HiRank×BigG×HiLR / R75 drop=0.05 / R125 KL)"
  "mine-r130-nodrop-longctx-kl-hirank-bigg-hilr-1|R130|Tok NoDrop×LongCtx×KL×HiRank×BigG×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=2e-5 (R125×R128; ≠ R125@α32 / R128@kl=0 / R129 HiAlpha / R124@5e-6 / R126 no-KL)"
  "mine-r131-nodrop-longctx-kl-hialpha-bigg-hilr-1|R131|Tok NoDrop×LongCtx×KL×HiAlpha×BigG×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=2e-5 (R125×R129; ≠ R125@α32 / R129@kl=0 / R130 HiRank / R122@G=4 / R127@5e-6)"
  "mine-r132-nodrop-longctx-kl-hirank-bigg-1|R132|Tok NoDrop×LongCtx×KL×HiRank×BigG drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=5e-6 (R126×R124; ≠ R130@2e-5 HiLR / R126@kl=0 / R131 HiAlpha / R124@r16 / R128@kl=0+HiLR)"
  "mine-r133-nodrop-longctx-kl-hialpha-bigg-1|R133|Tok NoDrop×LongCtx×KL×HiAlpha×BigG drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=5e-6 (R127×R124; ≠ R131@2e-5 HiLR / R127@kl=0 / R132 HiRank / R124@α32 / R129@kl=0+HiLR)"
  "mine-r134-nodrop-longctx-kl-hirank-1|R134|Tok NoDrop×LongCtx×KL×HiRank drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=5e-6 (R117×R119; ≠ R132@G16 BigG / R130@G16+HiLR / R117@kl=0 / R121@kl=0+HiLR / R119@r16 / R133 HiAlpha×BigG)"
  "mine-r135-nodrop-longctx-kl-hialpha-1|R135|Tok NoDrop×LongCtx×KL×HiAlpha drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=5e-6 (R118×R119; ≠ R133@G16 BigG / R131@G16+HiLR / R118@kl=0 / R122@kl=0+HiLR / R134 HiRank@G4 / R119@α32)"
  "mine-r136-nodrop-longctx-kl-hirank-hilr-1|R136|Tok NoDrop×LongCtx×KL×HiRank×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=2e-5 (R134×HiLR; ≠ R134@5e-6 / R130@G16 BigG+HiLR / R121@kl=0 / R123@α32 / R132@G16@5e-6 / R135 HiAlpha@G4)"
  "mine-r137-nodrop-longctx-kl-hialpha-hilr-1|R137|Tok NoDrop×LongCtx×KL×HiAlpha×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=2e-5 (R135×HiLR; ≠ R135@5e-6 / R131@G16 BigG+HiLR / R122@kl=0 / R123@α32 / R133@G16@5e-6 / R136 HiRank@G4)"
  "mine-r138-nodrop-longctx-kl-hialpha-hitemp-1|R138|Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=5e-6 temp=1.2 (R135×HiTemp; ≠ R135@temp0.8 / R137@HiLR@temp0.8 / R25@drop0.05 short / R131@G16 BigG+HiLR / R122@kl=0 / R136 HiRank×HiLR)"
  "mine-r139-nodrop-longctx-kl-hirank-hitemp-1|R139|Tok NoDrop×LongCtx×KL×HiRank×HiTemp drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=5e-6 temp=1.2 (R134×HiTemp; ≠ R134@temp0.8 / R136@HiLR@temp0.8 / R138 HiAlpha×HiTemp / R25@drop0.05 short / R130@G16 BigG+HiLR / R121@kl=0)"
  "mine-r140-nodrop-longctx-kl-hialpha-hitemp-hilr-1|R140|Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=2e-5 temp=1.2 (R138×HiLR; ≠ R138@lr5e-6@temp1.2 / R137@HiLR@temp0.8 / R139 HiRank×HiTemp / R25@drop0.05 short / R131@G16 BigG+HiLR / R122@kl=0)"
  "mine-r141-nodrop-longctx-kl-hirank-hitemp-hilr-1|R141|Tok NoDrop×LongCtx×KL×HiRank×HiTemp×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=2e-5 temp=1.2 (R139×HiLR; ≠ R139@lr5e-6@temp1.2 / R136@HiLR@temp0.8 / R140 HiAlpha×HiTemp×HiLR / R138 HiAlpha×HiTemp / R25@drop0.05 short / R130@G16 BigG+HiLR)"
  "mine-r142-nodrop-longctx-kl-hialpha-hitemp-bigg-1|R142|Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp×BigG drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=5e-6 temp=1.2 (R138×BigG; ≠ R138@G4@temp1.2 / R140@HiLR@G4 / R133@BigG@temp0.8 / R131@BigG+HiLR / R141 HiRank×HiTemp×HiLR / R25@drop0.05 short)"
  "mine-r143-nodrop-longctx-kl-hirank-hitemp-bigg-1|R143|Tok NoDrop×LongCtx×KL×HiRank×HiTemp×BigG drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=5e-6 temp=1.2 (R139×BigG; ≠ R139@G4@temp1.2 / R141@HiLR@G4 / R142 HiAlpha×HiTemp×BigG / R132@BigG@temp0.8 / R130@BigG+HiLR / R25@drop0.05 short)"
  "mine-r144-nodrop-longctx-kl-hialpha-hitemp-bigg-hilr-1|R144|Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp×BigG×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=2e-5 temp=1.2 (R142×HiLR; ≠ R142@5e-6@G16@temp1.2 / R140@HiLR@G4 / R143 HiRank×HiTemp×BigG / R141 HiRank×HiTemp×HiLR / R131@BigG+HiLR@temp0.8 / R25@drop0.05 short)"
  "mine-r145-nodrop-longctx-kl-hirank-hitemp-bigg-hilr-1|R145|Tok NoDrop×LongCtx×KL×HiRank×HiTemp×BigG×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=2e-5 temp=1.2 (R143×HiLR; ≠ R143@5e-6@G16@temp1.2 / R141@HiLR@G4 / R144 HiAlpha×HiTemp×BigG×HiLR / R130@BigG+HiLR@temp0.8 / R132@BigG@temp0.8 / R25@drop0.05 short)"
  "mine-r146-nodrop-longctx-kl-megarank-hitemp-bigg-hilr-1|R146|Tok NoDrop×LongCtx×KL×MegaRank×HiTemp×BigG×HiLR drop=0.0 kl=0.02 r=128 α=256 16384/1024 G=16 lr=2e-5 temp=1.2 (R145×MegaRank; ≠ R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R29@r64@6144 / R25@drop0.05 short)"
  "mine-r147-nodrop-longctx-kl-megarank-ultratemp-bigg-hilr-1|R147|Tok NoDrop×LongCtx×KL×MegaRank×UltraTemp×BigG×HiLR drop=0.0 kl=0.02 r=128 α=256 16384/1024 G=16 lr=2e-5 temp=1.5 (R146×UltraTemp; ≠ R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"
  "mine-r148-nodrop-longctx-kl-megarank-supertemp-bigg-hilr-1|R148|Tok NoDrop×LongCtx×KL×MegaRank×SuperTemp×BigG×HiLR drop=0.0 kl=0.02 r=128 α=256 16384/1024 G=16 lr=2e-5 temp=2.0 (R147×SuperTemp; ≠ R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"
  "mine-r149-nodrop-longctx-kl-ultramegarank-supertemp-bigg-hilr-1|R149|Tok NoDrop×LongCtx×KL×UltraMegaRank×SuperTemp×BigG×HiLR drop=0.0 kl=0.02 r=256 α=512 16384/1024 G=16 lr=2e-5 temp=2.0 (R148×UltraMegaRank; ≠ R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"
  "mine-r150-nodrop-longctx-kl-ultramegarank-extremetemp-bigg-hilr-1|R150|Tok NoDrop×LongCtx×KL×UltraMegaRank×ExtremeTemp×BigG×HiLR drop=0.0 kl=0.02 r=256 α=512 16384/1024 G=16 lr=2e-5 temp=2.5 (R149×ExtremeTemp; ≠ R149@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"
  "mine-r151-nodrop-longctx-kl-hypermegarank-extremetemp-bigg-hilr-1|R151|Tok NoDrop×LongCtx×KL×HyperMegaRank×ExtremeTemp×BigG×HiLR drop=0.0 kl=0.02 r=512 α=1024 16384/1024 G=16 lr=2e-5 temp=2.5 (R150×HyperMegaRank; ≠ R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"
  "mine-r152-nodrop-longctx-kl-hypermegarank-infernotemp-bigg-hilr-1|R152|Tok NoDrop×LongCtx×KL×HyperMegaRank×InfernoTemp×BigG×HiLR drop=0.0 kl=0.02 r=512 α=1024 16384/1024 G=16 lr=2e-5 temp=3.0 (R151×InfernoTemp; ≠ R151@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"
  "mine-r153-nodrop-longctx-kl-gigarank-infernotemp-bigg-hilr-1|R153|Tok NoDrop×LongCtx×KL×GigaRank×InfernoTemp×BigG×HiLR drop=0.0 kl=0.02 r=1024 α=2048 16384/1024 G=16 lr=2e-5 temp=3.0 (R152×GigaRank; ≠ R152@r512@temp3.0 / R151@r512@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"
  "mine-r154-nodrop-longctx-kl-gigarank-plasmatemp-bigg-hilr-1|R154|Tok NoDrop×LongCtx×KL×GigaRank×PlasmaTemp×BigG×HiLR drop=0.0 kl=0.02 r=1024 α=2048 16384/1024 G=16 lr=2e-5 temp=3.5 (R153×PlasmaTemp; ≠ R153@r1024@temp3.0 / R152@r512@temp3.0 / R151@r512@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"
  "mine-r155-nodrop-longctx-kl-terarank-plasmatemp-bigg-hilr-1|R155|Tok NoDrop×LongCtx×KL×TeraRank×PlasmaTemp×BigG×HiLR drop=0.0 kl=0.02 r=2048 α=4096 16384/1024 G=16 lr=2e-5 temp=3.5 (R154×TeraRank; ≠ R154@r1024@temp3.5 / R153@r1024@temp3.0 / R152@r512@temp3.0 / R151@r512@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"
  # "mine-r33-guass-grpo-1|R33|guass-init Reason-GRPO (≠ R3 Tok / R19–R23; LoRA from live king)"
  # p3074: R389 Offline-Long×LongCtx (after R388; rent after HEAD nonkings via API; shell lag OK)
  "mine-r389-marsplan-offline-dpo-long-longctx-1|R389|marsplan×Offline-DPO×Long×LongCtx β=0.1 α=32 r=16 lr=5e-6 @16384@600 (R368×LongCtx isolate)"
  # p3075: R390 Offline-Long×LongCtx×HiLR (after R389; rent after HEAD nonkings via API; shell lag OK)
  "mine-r390-marsplan-offline-dpo-long-longctx-hilr-1|R390|marsplan×Offline-DPO×Long×LongCtx×HiLR β=0.1 α=32 r=16 lr=2e-5 @16384@600 (R389×HiLR isolate)"
  # p3076: R391 Offline-HiAlpha×HiLR (after R390; rent after HEAD nonkings via API; shell lag OK)
  "mine-r391-marsplan-offline-dpo-hialpha-hilr-1|R391|marsplan×Offline-DPO×HiAlpha×HiLR β=0.1 α=128 r=16 lr=2e-5 @6144@600 (R369×HiLR isolate)"
  # p3077: R392 Offline-Long×HiLR (after R391; rent after HEAD nonkings via API; shell lag OK)
  "mine-r392-marsplan-offline-dpo-long-hilr-1|R392|marsplan×Offline-DPO×Long×HiLR β=0.1 α=32 r=16 lr=2e-5 @6144@600 (R368×HiLR isolate)"
  # p3078: R393 Offline-Long×HiBeta (after R392; rent after HEAD nonkings via API; shell lag OK)
  "mine-r393-marsplan-offline-dpo-long-hibeta-1|R393|marsplan×Offline-DPO×Long×HiBeta β=0.5 α=32 r=16 lr=5e-6 @6144@600 (R368×HiBeta isolate)"
  # p3079: R394 Offline-Long×LoBeta (after R393; rent after HEAD nonkings via API; shell lag OK)
  "mine-r394-marsplan-offline-dpo-long-lobeta-1|R394|marsplan×Offline-DPO×Long×LoBeta β=0.02 α=32 r=16 lr=5e-6 @6144@600 (R368×LoBeta isolate)"
  # p3080: R395 Offline-Long×ExtraLong (after R394; rent after HEAD nonkings via API; shell lag OK)
  "mine-r395-marsplan-offline-dpo-long-extralong-1|R395|marsplan×Offline-DPO×Long×ExtraLong β=0.1 α=32 r=16 lr=5e-6 @6144@900 (R368×ExtraLong isolate)"
  # p3081: R396 Offline-Long×LoBeta×HiLR (after R395; rent after HEAD nonkings via API; shell lag OK)
  "mine-r396-marsplan-offline-dpo-long-lobeta-hilr-1|R396|marsplan×Offline-DPO×Long×LoBeta×HiLR β=0.02 α=32 r=16 lr=2e-5 @6144@600 (R394×HiLR compound)"
  # p3082: R397 Offline-Long×ExtraLong×HiLR (after R396; rent after HEAD nonkings via API; shell lag OK)
  "mine-r397-marsplan-offline-dpo-long-extralong-hilr-1|R397|marsplan×Offline-DPO×Long×ExtraLong×HiLR β=0.1 α=32 r=16 lr=2e-5 @6144@900 (R395×HiLR compound)"
  # p3083: R398 Offline-Long×LoBeta×HiRank (after R397; rent after HEAD nonkings via API; shell lag OK)
  "mine-r398-marsplan-offline-dpo-long-lobeta-hirank-1|R398|marsplan×Offline-DPO×Long×LoBeta×HiRank β=0.02 α=32 r=64 lr=5e-6 @6144@600 (R394×HiRank compound)"
  # p3084: R399 Offline-Long×HiBeta×HiLR (after R398; rent after HEAD nonkings via API; shell lag OK)
  "mine-r399-marsplan-offline-dpo-long-hibeta-hilr-1|R399|marsplan×Offline-DPO×Long×HiBeta×HiLR β=0.5 α=32 r=16 lr=2e-5 @6144@600 (R393×HiLR compound)"
  # p3085: R400 Offline-Long×LoBeta×ExtraLong (after R399; rent after HEAD nonkings via API; shell lag OK)
  "mine-r400-marsplan-offline-dpo-long-lobeta-extralong-1|R400|marsplan×Offline-DPO×Long×LoBeta×ExtraLong β=0.02 α=32 r=16 lr=5e-6 @6144@900 (R394×ExtraLong compound)"
  # p3086: R401 Offline-Long×HiBeta×HiRank (after R400; rent after HEAD nonkings via API; shell lag OK)
  "mine-r401-marsplan-offline-dpo-long-hibeta-hirank-1|R401|marsplan×Offline-DPO×Long×HiBeta×HiRank β=0.5 α=32 r=64 lr=5e-6 @6144@600 (R393×HiRank compound)"
  # p3087: R402 Offline-Long×HiBeta×ExtraLong (after R401; rent after HEAD nonkings via API; shell lag OK)
  "mine-r402-marsplan-offline-dpo-long-hibeta-extralong-1|R402|marsplan×Offline-DPO×Long×HiBeta×ExtraLong β=0.5 α=32 r=16 lr=5e-6 @6144@900 (R393×ExtraLong compound)"
  # p3088: R403 Offline-Long×HiBeta×HiRank×HiLR (after R402; rent after HEAD nonkings via API; shell lag OK)
  "mine-r403-marsplan-offline-dpo-long-hibeta-hirank-hilr-1|R403|marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR β=0.5 α=32 r=64 lr=2e-5 @6144@600 (R401×HiLR compound)"
  # p3089: R404 Offline-Long×HiBeta×HiLR×ExtraLong (after R403; rent after HEAD nonkings via API; shell lag OK)
  "mine-r404-marsplan-offline-dpo-long-hibeta-hilr-extralong-1|R404|marsplan×Offline-DPO×Long×HiBeta×HiLR×ExtraLong β=0.5 α=32 r=16 lr=2e-5 @6144@900 (R399×ExtraLong compound)"
  # p3090: R405 Offline-Long×HiBeta×HiRank×ExtraLong (after R404; rent after HEAD nonkings via API; shell lag OK)
  "mine-r405-marsplan-offline-dpo-long-hibeta-hirank-extralong-1|R405|marsplan×Offline-DPO×Long×HiBeta×HiRank×ExtraLong β=0.5 α=32 r=64 lr=5e-6 @6144@900 (R401×ExtraLong compound)"
  # p3091: R406 Offline-Long×HiBeta×HiRank×HiLR×ExtraLong (after R405; rent after HEAD nonkings via API; shell lag OK)
  "mine-r406-marsplan-offline-dpo-long-hibeta-hirank-hilr-extralong-1|R406|marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×ExtraLong β=0.5 α=32 r=64 lr=2e-5 @6144@900 (R403×ExtraLong compound)"
  # p3092: R407 Offline-Long×LoBeta×HiRank×ExtraLong (after R406; rent after HEAD nonkings via API; shell lag OK)
  "mine-r407-marsplan-offline-dpo-long-lobeta-hirank-extralong-1|R407|marsplan×Offline-DPO×Long×LoBeta×HiRank×ExtraLong β=0.02 α=32 r=64 lr=5e-6 @6144@900 (R398×ExtraLong compound)"
  # p3093: R408 Offline-Long×LoBeta×HiRank×HiLR (after R407; rent after HEAD nonkings via API; shell lag OK)
  "mine-r408-marsplan-offline-dpo-long-lobeta-hirank-hilr-1|R408|marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR β=0.02 α=32 r=64 lr=2e-5 @6144@600 (R398×HiLR compound)"
  # p3094: R409 Offline-Long×LoBeta×HiRank×HiLR×ExtraLong (after R408; rent after HEAD nonkings via API; shell lag OK)
  "mine-r409-marsplan-offline-dpo-long-lobeta-hirank-hilr-extralong-1|R409|marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×ExtraLong β=0.02 α=32 r=64 lr=2e-5 @6144@900 (R408×ExtraLong compound)"
  # p3095: R410 Offline-Long×LoBeta×HiLR×ExtraLong (after R409; rent after HEAD nonkings via API; shell lag OK)
  "mine-r410-marsplan-offline-dpo-long-lobeta-hilr-extralong-1|R410|marsplan×Offline-DPO×Long×LoBeta×HiLR×ExtraLong β=0.02 α=32 r=16 lr=2e-5 @6144@900 (R396×ExtraLong / R400×HiLR compound)"
  # p3096: R411 Offline-Long×LoBeta×HiRank×LongCtx (after R410; rent after HEAD nonkings via API; shell lag OK)
  "mine-r411-marsplan-offline-dpo-long-lobeta-hirank-longctx-1|R411|marsplan×Offline-DPO×Long×LoBeta×HiRank×LongCtx β=0.02 α=32 r=64 lr=5e-6 @16384@600 (R398×LongCtx / R387×LoBeta compound)"
  # p3097: R412 Offline-Long×LoBeta×HiLR×LongCtx (after R411; rent after HEAD nonkings via API; shell lag OK)
  "mine-r412-marsplan-offline-dpo-long-lobeta-hilr-longctx-1|R412|marsplan×Offline-DPO×Long×LoBeta×HiLR×LongCtx β=0.02 α=32 r=16 lr=2e-5 @16384@600 (R396×LongCtx / R390×LoBeta compound)"
  # p3098: R413 Offline-Long×LoBeta×HiRank×HiLR×LongCtx (after R412; rent after HEAD nonkings via API; shell lag OK)
  "mine-r413-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-1|R413|marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×LongCtx β=0.02 α=32 r=64 lr=2e-5 @16384@600 (R412×HiRank / R411×HiLR / R408×LongCtx compound)"
  # p3099: R414 Offline-Long×HiBeta×LongCtx (after R413; rent after HEAD nonkings via API; shell lag OK)
  "mine-r414-marsplan-offline-dpo-long-hibeta-longctx-1|R414|marsplan×Offline-DPO×Long×HiBeta×LongCtx β=0.5 α=32 r=16 lr=5e-6 @16384@600 (R393×LongCtx compound)"
  # p3100: R415 Offline-Long×HiBeta×HiLR×LongCtx (after R414; rent after HEAD nonkings via API; shell lag OK)
  "mine-r415-marsplan-offline-dpo-long-hibeta-hilr-longctx-1|R415|marsplan×Offline-DPO×Long×HiBeta×HiLR×LongCtx β=0.5 α=32 r=16 lr=2e-5 @16384@600 (R414×HiLR / R399×LongCtx compound)"
  # p3101: R416 Offline-Long×HiBeta×HiRank×LongCtx (after R415; rent after HEAD nonkings via API; shell lag OK)
  "mine-r416-marsplan-offline-dpo-long-hibeta-hirank-longctx-1|R416|marsplan×Offline-DPO×Long×HiBeta×HiRank×LongCtx β=0.5 α=32 r=64 lr=5e-6 @16384@600 (R414×HiRank / R401×LongCtx compound)"
  # p3102: R417 Offline-Long×HiBeta×HiRank×HiLR×LongCtx (after R416; rent after HEAD nonkings via API; shell lag OK)
  "mine-r417-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-1|R417|marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×LongCtx β=0.5 α=32 r=64 lr=2e-5 @16384@600 (R416×HiLR / R403×LongCtx / R415×HiRank compound)"
  # p3103: R418 Offline-Long×HiBeta×LongCtx×ExtraLong (after R417; rent after HEAD nonkings via API; shell lag OK)
  "mine-r418-marsplan-offline-dpo-long-hibeta-longctx-extralong-1|R418|marsplan×Offline-DPO×Long×HiBeta×LongCtx×ExtraLong β=0.5 α=32 r=16 lr=5e-6 @16384@900 (R414×ExtraLong / R402×LongCtx compound)"
  # p3104: R419 Offline-Long×HiBeta×HiLR×LongCtx×ExtraLong (after R418; rent after HEAD nonkings via API; shell lag OK)
  "mine-r419-marsplan-offline-dpo-long-hibeta-hilr-longctx-extralong-1|R419|marsplan×Offline-DPO×Long×HiBeta×HiLR×LongCtx×ExtraLong β=0.5 α=32 r=16 lr=2e-5 @16384@900 (R415×ExtraLong / R404×LongCtx / R418×HiLR compound)"
  # p3105: R420 Offline-Long×HiBeta×HiRank×LongCtx×ExtraLong (after R419; rent after HEAD nonkings via API; shell lag OK)
  "mine-r420-marsplan-offline-dpo-long-hibeta-hirank-longctx-extralong-1|R420|marsplan×Offline-DPO×Long×HiBeta×HiRank×LongCtx×ExtraLong β=0.5 α=32 r=64 lr=5e-6 @16384@900 (R416×ExtraLong / R405×LongCtx / R418×HiRank compound)"
  # p3106: R421 Offline-Long×HiBeta×HiRank×HiLR×LongCtx×ExtraLong (after R420; rent after HEAD nonkings via API; shell lag OK)
  "mine-r421-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-extralong-1|R421|marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×LongCtx×ExtraLong β=0.5 α=32 r=64 lr=2e-5 @16384@900 (R417×ExtraLong / R406×LongCtx / R420×HiLR compound)"
  # p3107: R422 Offline-Long×LoBeta×LongCtx×ExtraLong (after R421; rent after HEAD nonkings via API; shell lag OK)
  "mine-r422-marsplan-offline-dpo-long-lobeta-longctx-extralong-1|R422|marsplan×Offline-DPO×Long×LoBeta×LongCtx×ExtraLong β=0.02 α=32 r=16 lr=5e-6 @16384@900 (R418×LoBeta / R400×LongCtx / R414×LoBeta×ExtraLong compound)"
  # p3108: R423 Offline-Long×LoBeta×HiRank×LongCtx×ExtraLong (after R422; rent after HEAD nonkings via API; shell lag OK)
  "mine-r423-marsplan-offline-dpo-long-lobeta-hirank-longctx-extralong-1|R423|marsplan×Offline-DPO×Long×LoBeta×HiRank×LongCtx×ExtraLong β=0.02 α=32 r=64 lr=5e-6 @16384@900 (R411×ExtraLong / R422×HiRank / R407×LongCtx compound)"
  # p3109: R424 Offline-Long×LoBeta×HiLR×LongCtx×ExtraLong (after R423; rent after HEAD nonkings via API; shell lag OK)
  "mine-r424-marsplan-offline-dpo-long-lobeta-hilr-longctx-extralong-1|R424|marsplan×Offline-DPO×Long×LoBeta×HiLR×LongCtx×ExtraLong β=0.02 α=32 r=16 lr=2e-5 @16384@900 (R412×ExtraLong / R422×HiLR / R410×LongCtx compound)"
  # p3110: R425 Offline-Long×LoBeta×HiRank×HiLR×LongCtx×ExtraLong (after R424; rent after HEAD nonkings via API; shell lag OK)
  "mine-r425-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-extralong-1|R425|marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×LongCtx×ExtraLong β=0.02 α=32 r=64 lr=2e-5 @16384@900 (R413×ExtraLong / R423×HiLR / R424×HiRank / R409×LongCtx compound)"
  # p3111: R426 Offline-HiAlpha×LongCtx×ExtraLong (after R425; rent after HEAD nonkings via API; shell lag OK)
  "mine-r426-marsplan-offline-dpo-hialpha-longctx-extralong-1|R426|marsplan×Offline-DPO×HiAlpha×LongCtx×ExtraLong β=0.1 α=128 r=16 lr=5e-6 @16384@900 (R371×ExtraLong compound)"
  # p3112: R427 Offline-HiAlpha×LongCtx×HiRank×ExtraLong (after R426; rent after HEAD nonkings via API; shell lag OK)
  "mine-r427-marsplan-offline-dpo-hialpha-longctx-hirank-extralong-1|R427|marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×ExtraLong β=0.1 α=128 r=64 lr=5e-6 @16384@900 (R373×ExtraLong / R426×HiRank)"
  # p3113: R428 Offline-HiAlpha×LongCtx×HiLR×ExtraLong (after R427; rent after HEAD nonkings via API; shell lag OK)
  "mine-r428-marsplan-offline-dpo-hialpha-longctx-hilr-extralong-1|R428|marsplan×Offline-DPO×HiAlpha×LongCtx×HiLR×ExtraLong β=0.1 α=128 r=16 lr=2e-5 @16384@900 (R380×ExtraLong / R426×HiLR)"
  # p3114: R429 Offline-HiAlpha×LongCtx×HiRank×HiLR×ExtraLong (after R428; rent after HEAD nonkings via API; shell lag OK)
  "mine-r429-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-extralong-1|R429|marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×HiLR×ExtraLong β=0.1 α=128 r=64 lr=2e-5 @16384@900 (R381×ExtraLong / R427×HiLR / R428×HiRank)"
  # p3115: R430 Offline-HiAlpha×LoBeta×LongCtx×ExtraLong (after R429; rent after HEAD nonkings via API; shell lag OK)
  "mine-r430-marsplan-offline-dpo-hialpha-lobeta-longctx-extralong-1|R430|marsplan×Offline-DPO×HiAlpha×LoBeta×LongCtx×ExtraLong β=0.02 α=128 r=16 lr=5e-6 @16384@900 (R426×LoBeta)"
  # p3116: R431 Offline-HiAlpha×HiBeta×LongCtx×ExtraLong (after R430; rent after HEAD nonkings via API; shell lag OK)
  "mine-r431-marsplan-offline-dpo-hialpha-hibeta-longctx-extralong-1|R431|marsplan×Offline-DPO×HiAlpha×HiBeta×LongCtx×ExtraLong β=0.5 α=128 r=16 lr=5e-6 @16384@900 (R426×HiBeta)"

  # p3117: R432 Offline-HiAlpha×LoBeta×HiRank×LongCtx×ExtraLong (after R431; rent after HEAD nonkings via API; shell lag OK)
  "mine-r432-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-extralong-1|R432|marsplan×Offline-DPO×HiAlpha×LoBeta×HiRank×LongCtx×ExtraLong β=0.02 α=128 r=64 lr=5e-6 @16384@900 (R430×HiRank / R427×LoBeta)"
  # p3118: R433 Offline-HiAlpha×HiBeta×HiRank×LongCtx×ExtraLong (after R432; rent after HEAD nonkings via API; shell lag OK)
  "mine-r433-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-extralong-1|R433|marsplan×Offline-DPO×HiAlpha×HiBeta×HiRank×LongCtx×ExtraLong β=0.5 α=128 r=64 lr=5e-6 @16384@900 (R431×HiRank / R432×HiBeta)"
  # p3119: R434 Offline-HiAlpha×LoBeta×LongCtx×HiLR×ExtraLong (after R433; rent after HEAD nonkings via API; shell lag OK)
  "mine-r434-marsplan-offline-dpo-hialpha-lobeta-longctx-hilr-extralong-1|R434|marsplan×Offline-DPO×HiAlpha×LoBeta×LongCtx×HiLR×ExtraLong β=0.02 α=128 r=16 lr=2e-5 @16384@900 (R430×HiLR / R428×LoBeta)"
  # p3120: R435 Offline-HiAlpha×HiBeta×HiLR×LongCtx×ExtraLong (after R434; rent after HEAD nonkings via API; shell lag OK)
  "mine-r435-marsplan-offline-dpo-hialpha-hibeta-longctx-hilr-extralong-1|R435|marsplan×Offline-DPO×HiAlpha×HiBeta×HiLR×LongCtx×ExtraLong β=0.5 α=128 r=16 lr=2e-5 @16384@900 (R431×HiLR / R428×HiBeta)"
  # p3121: R436 Offline-HiAlpha×LoBeta×HiRank×LongCtx×HiLR×ExtraLong (after R435; rent after HEAD nonkings via API; shell lag OK)
  "mine-r436-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-hilr-extralong-1|R436|marsplan×Offline-DPO×HiAlpha×LoBeta×HiRank×LongCtx×HiLR×ExtraLong β=0.02 α=128 r=64 lr=2e-5 @16384@900 (R432×HiLR / R434×HiRank)"
  # p3122: R437 Offline-HiAlpha×HiBeta×HiRank×LongCtx×HiLR×ExtraLong (after R436; rent after HEAD nonkings via API; shell lag OK)
  "mine-r437-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-hilr-extralong-1|R437|marsplan×Offline-DPO×HiAlpha×HiBeta×HiRank×LongCtx×HiLR×ExtraLong β=0.5 α=128 r=64 lr=2e-5 @16384@900 (R433×HiLR / R435×HiRank)"
  # p3123: R438 Online-DPO×ExtraLong (after R437; rent after HEAD nonkings via API; shell lag OK)
  "mine-r438-marsplan-online-dpo-extralong-1|R438|marsplan×Online-DPO×ExtraLong β=0.1 α=32 r=16 G=4 lr=5e-6 @6144@900 (R351×ExtraLong isolate)"
  # p3124: R439 Online-DPO×ExtraLong×HiLR (after R438; lean-warm on lunar 4,5; rent after HEAD nonkings via API; shell lag OK)
  "mine-r439-marsplan-online-dpo-extralong-hilr-1|R439|marsplan×Online-DPO×ExtraLong×HiLR β=0.1 α=32 r=16 G=4 lr=2e-5 @6144@900 (R438×R337 compound)"
  # p3125: R440 Online-DPO×ExtraLong×BigG (after R439; rent after HEAD nonkings via API; shell lag OK)
  "mine-r440-marsplan-online-dpo-extralong-bigg-1|R440|marsplan×Online-DPO×ExtraLong×BigG β=0.1 α=32 r=16 G=8 lr=5e-6 @6144@900 (R438×R336 compound)"
  # p3128: R443 Online-DPO×ExtraLong×HiRank×HiLR (after R442; rent after HEAD nonkings via API; shell lag OK)
  "mine-r443-marsplan-online-dpo-extralong-hirank-hilr-1|R443|marsplan×Online-DPO×ExtraLong×HiRank×HiLR β=0.1 α=32 r=64 G=4 lr=2e-5 @6144@900 (R442×R439 compound)"
  # p3129: R444 Online-DPO×ExtraLong×HiRank×BigG (after R443; rent after HEAD nonkings via API; shell lag OK)
  "mine-r444-marsplan-online-dpo-extralong-hirank-bigg-1|R444|marsplan×Online-DPO×ExtraLong×HiRank×BigG β=0.1 α=32 r=64 G=8 lr=5e-6 @6144@900 (R442×R440 compound)"
  # p3130: R445 Online-DPO×ExtraLong×HiRank×BigG×HiLR (after R444; rent after HEAD nonkings via API; shell lag OK)
  "mine-r445-marsplan-online-dpo-extralong-hirank-bigg-hilr-1|R445|marsplan×Online-DPO×ExtraLong×HiRank×BigG×HiLR β=0.1 α=32 r=64 G=8 lr=2e-5 @6144@900 (R444×R443 compound)"
  # p3131: R446 Online-DPO×ExtraLong×HiTemp (after R445; lean-warm crown 6,7; rent after HEAD nonkings via API; shell lag OK)
  "mine-r446-marsplan-online-dpo-extralong-hitemp-1|R446|marsplan×Online-DPO×ExtraLong×HiTemp β=0.1 α=32 r=16 G=4 lr=5e-6 temp=1.5 @6144@900 (R438×HiTemp isolate)"
  # p3132: R447 Online-DPO×ExtraLong×UltraTemp (after R446; lean-warm crown 4,5; rent after HEAD nonkings via API; shell lag OK)
  "mine-r447-marsplan-online-dpo-extralong-ultratemp-1|R447|marsplan×Online-DPO×ExtraLong×UltraTemp β=0.1 α=32 r=16 G=4 lr=5e-6 temp=2.0 @6144@900 (R446×UltraTemp isolate)"
  # p3127: R442 Online-DPO×ExtraLong×HiRank (after R441; rent after HEAD nonkings via API; shell lag OK)
  "mine-r442-marsplan-online-dpo-extralong-hirank-1|R442|marsplan×Online-DPO×ExtraLong×HiRank β=0.1 α=32 r=64 G=4 lr=5e-6 @6144@900 (R438×R339 compound)"
  # p3126: R441 Online-DPO×ExtraLong×BigG×HiLR (after R440; rent after HEAD nonkings via API; shell lag OK)
  "mine-r441-marsplan-online-dpo-extralong-bigg-hilr-1|R441|marsplan×Online-DPO×ExtraLong×BigG×HiLR β=0.1 α=32 r=16 G=8 lr=2e-5 @6144@900 (R440×R439 compound)"

)

mkdir -p "$EXP/logs" "$STAMP_DIR"
echo $$ >"$PIDF"
exec >>"$LOG" 2>&1

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

log() { echo "[fleet-rent] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

mine_names() {
  python3 - <<'PY'
import json, subprocess, sys
try:
    raw = subprocess.check_output(["lium", "ps", "--format", "json"], text=True, timeout=60)
except Exception as e:
    print(f"PS_FAIL {e}", file=sys.stderr)
    sys.exit(0)
try:
    data = json.loads(raw)
except Exception:
    sys.exit(0)
pods = data if isinstance(data, list) else data.get("pods") or data.get("data") or []
for p in pods:
    if not isinstance(p, dict):
        continue
    name = p.get("name") or p.get("Name") or p.get("pod_name") or ""
    if isinstance(name, str) and name.startswith("mine-"):
        print(name)
PY
}

mine_count() { mine_names | wc -l; }

name_live() {
  local n=$1
  mine_names | grep -qx "$n"
}

# Print available 8×$gpu node ids (prefer huid, else uuid), one per line.
# p2586: skip executor UUIDs in LIUM_EXECUTOR_BLACKLIST / artifacts/executor_blacklist.txt
# (CLI previously preferred huid of 8f34559f… and would re-rent the BAD_HOST).
# Env must be on the *python* side of the pipe (VAR= cmd1 | cmd2 only exports to cmd1).
list_nodes() {
  local gpu=$1
  local bl_env="${LIUM_EXECUTOR_BLACKLIST:-8f34559f-30e7-40d2-bc80-629b0903742a}"
  local bl_file="${LIUM_EXECUTOR_BLACKLIST_FILE:-$STAMP_DIR/executor_blacklist.txt}"
  # stderr discarded: empty stock prints "All … rented out" tips.
  lium ls --gpu "$gpu" --count 8 --format json 2>/dev/null \
    | LIUM_EXECUTOR_BLACKLIST="$bl_env" LIUM_EXECUTOR_BLACKLIST_FILE="$bl_file" \
      python3 -c 'import json,os,sys
from pathlib import Path
bl=set()
for x in os.environ.get("LIUM_EXECUTOR_BLACKLIST","").split(","):
    x=x.strip()
    if x: bl.add(x)
bf=os.environ.get("LIUM_EXECUTOR_BLACKLIST_FILE","")
if bf and Path(bf).is_file():
    for line in Path(bf).read_text().splitlines():
        line=line.strip()
        if line and not line.startswith("#"):
            bl.add(line.split()[0])
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
nodes = d if isinstance(d, list) else (d.get("nodes") or d.get("data") or [])
for n in nodes:
    if not isinstance(n, dict):
        continue
    eid = str(n.get("id") or "")
    if eid and eid in bl:
        continue
    nid = n.get("huid") or n.get("id") or n.get("name")
    if nid:
        print(nid)
'
}

stock_ok() {
  local gpu=$1
  list_nodes "$gpu" | grep -q .
}

next_slot() {
  local live
  live=$(mine_names | tr '\n' ' ')
  local ent name
  for ent in "${QUEUE[@]}"; do
    name=${ent%%|*}
    if ! echo " $live " | grep -q " $name "; then
      echo "$ent"
      return 0
    fi
  done
  return 1
}

# Emit up to $1 distinct non-live queue entries (one per line).
next_slots() {
  local want=${1:-1}
  local live ent name
  local -a out=()
  live=$(mine_names | tr '\n' ' ')
  for ent in "${QUEUE[@]}"; do
    name=${ent%%|*}
    if ! echo " $live " | grep -q " $name "; then
      out+=("$ent")
      if (( ${#out[@]} >= want )); then
        break
      fi
    fi
  done
  if (( ${#out[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${out[@]}"
}

# Rent a concrete node id/huid (fast path once ls shows stock).
try_rent_node() {
  local node=$1 name=$2
  log "attempting lium up node=$node name=$name ttl=$TTL"
  set +e
  lium up "$node" --name "$name" --ttl "$TTL" --no-ssh -y
  local rc=$?
  set -e
  return "$rc"
}

# Legacy auto-select fallback (slower when empty; keep for rare ls/up races).
try_rent() {
  local gpu=$1 name=$2
  log "attempting lium up gpu=$gpu name=$name ttl=$TTL"
  set +e
  lium up --gpu "$gpu" -c 8 --name "$name" --ttl "$TTL" --no-ssh -y --ports 12
  local rc=$?
  set -e
  return "$rc"
}

write_rent_stamp() {
  local name=$1 axis=$2 gpu=$3 note=$4
  RENT_NAME="$name" RENT_AXIS="$axis" RENT_GPU="$gpu" RENT_NOTE="$note" \
  RENT_TTL="$TTL" RENT_PASS="$PASS" RENT_STAMP_DIR="$STAMP_DIR" \
  python3 - <<'PY'
import json, os, subprocess, time
from pathlib import Path
name = os.environ["RENT_NAME"]
try:
    ps = subprocess.check_output(["lium", "ps", "--format", "json"], text=True, timeout=60)
except Exception as e:
    ps = f"err:{e}"
path = Path(os.environ["RENT_STAMP_DIR"]) / f"rented_{name.replace('/', '_')}.json"
path.write_text(json.dumps({
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pass": int(os.environ["RENT_PASS"]),
    "name": name,
    "axis": os.environ["RENT_AXIS"],
    "gpu": os.environ["RENT_GPU"],
    "note": os.environ["RENT_NOTE"],
    "ttl": os.environ["RENT_TTL"],
    "ps_json": ps[:4000],
}, indent=2) + "\n")
print("STAMP_OK", path)
PY
}

bal_ok() {
  python3 - <<'PY'
import subprocess, re, sys
try:
    raw = subprocess.check_output(["lium", "balance"], text=True, timeout=30)
except Exception:
    sys.exit(0)  # don't block rent on balance parse failure
m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw.replace(",", ""))
if not m:
    sys.exit(0)
bal = float(m.group(1))
# Hard floor $10k — refuse new rents below that.
sys.exit(0 if bal >= 10000 else 1)
PY
}

log "start target=$TARGET cap=$CAP poll=${POLL_S}s parallel=$PARALLEL_N max_iters=$MAX_ITERS pass=$PASS mode=ls-then-node-id"
log "live_mines=$(mine_names | tr '\n' ' ')|count=$(mine_count)"

empty_sleep() {
  # POLL_S=0 → brief pause so empty-stock loops don't peg a core (ls≈0.6s).
  if (( POLL_S > 0 )); then
    sleep "$POLL_S"
  else
    sleep 0.25
  fi
}

live_now=""
n=0
names=()
declare -A slot_axis=() slot_note=()

refresh_queue() {
  live_now=$(mine_names | tr '\n' ' ')
  n=$(echo "$live_now" | awk '{print NF}')
  if (( n >= TARGET )); then
    log "TARGET reached mine_count=$n >= $TARGET — exit"
    exit 0
  fi
  if (( n >= CAP )); then
    log "ABORT at cap mine_count=$n >= $CAP"
    exit 3
  fi
  local remain=$(( CAP - n ))
  local local_n=$PARALLEL_N
  if (( remain < local_n )); then
    local_n=$remain
  fi
  if (( local_n < 1 )); then
    log "ABORT at cap mine_count=$n >= $CAP"
    exit 3
  fi
  names=()
  slot_axis=()
  slot_note=()
  local ent name rest axis note
  for ent in "${QUEUE[@]}"; do
    name=${ent%%|*}
    rest=${ent#*|}
    axis=${rest%%|*}
    note=${rest#*|}
    if echo " $live_now " | grep -q " $name "; then
      continue
    fi
    names+=("$name")
    slot_axis["$name"]=$axis
    slot_note["$name"]=$note
    if (( ${#names[@]} >= local_n )); then
      break
    fi
  done
  if (( ${#names[@]} == 0 )); then
    log "QUEUE exhausted with mine_count=$n < target=$TARGET — exit"
    exit 0
  fi
}

refresh_queue

for i in $(seq 1 "$MAX_ITERS"); do
  # bal / ps only every 40 empty iters — keep the hunt on ls latency.
  if (( i == 1 || i % 40 == 0 )); then
    if ! bal_ok; then
      log "ABORT balance below \$10k floor"
      exit 4
    fi
    refresh_queue
  fi

  # Fast path (p2134): ls B300 (~0.6s) then rent concrete node ids.
  # Blind parallel `lium up --gpu` when empty burned ~20s/round and missed flickers.
  # B200 probed every 10th empty iter only — keep the B300 flicker window tight.
  mapfile -t b300_nodes < <(list_nodes B300 2>/dev/null || true)
  gpu_label=B300
  nodes=("${b300_nodes[@]}")
  if (( ${#nodes[@]} == 0 )) && (( i % 10 == 0 )); then
    mapfile -t b200_nodes < <(list_nodes B200 2>/dev/null || true)
    if (( ${#b200_nodes[@]} > 0 )); then
      gpu_label=B200
      nodes=("${b200_nodes[@]}")
      log "B300×8 empty — claiming B200×8 stock (${#nodes[@]} nodes)"
    fi
  fi

  if (( ${#nodes[@]} == 0 )); then
    if (( i % 40 == 1 )); then
      bal=$(lium balance 2>/dev/null | tr -d '\n' | head -c 80 || true)
      log "iter=$i ls-empty B300/B200×8; mine=$n/$TARGET (cap $CAP) next=${names[*]:0:6}… bal=$bal"
    fi
    empty_sleep
    continue
  fi

  # Stock appeared — refresh live names before claiming.
  refresh_queue
  if (( ${#names[@]} == 0 )); then
    continue
  fi

  # Pair each available node with next axis name; rent in parallel by node id.
  n_claim=${#nodes[@]}
  if (( n_claim > ${#names[@]} )); then
    n_claim=${#names[@]}
  fi
  log "STOCK ${gpu_label}×8 n=${#nodes[@]} — claiming $n_claim axes: ${names[*]:0:n_claim}"

  declare -A rented_gpu=()
  pids=()
  for ((j=0; j<n_claim; j++)); do
    name=${names[$j]}
    node=${nodes[$j]}
    (
      if try_rent_node "$node" "$name"; then
        echo "$gpu_label" >"/tmp/fleet_rent_${name}.gpu"
        exit 0
      fi
      # Race: node vanished between ls and up — one auto-select retry.
      # p2586: never auto-select B200 (sole ls hit is often blacklisted 8f34559f…).
      if [[ "$gpu_label" == "B300" ]] && try_rent "$gpu_label" "$name"; then
        echo "$gpu_label" >"/tmp/fleet_rent_${name}.gpu"
        exit 0
      fi
      exit 1
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done

  for ((j=0; j<n_claim; j++)); do
    name=${names[$j]}
    if [[ -f "/tmp/fleet_rent_${name}.gpu" ]]; then
      rented_gpu["$name"]=$(cat "/tmp/fleet_rent_${name}.gpu")
      rm -f "/tmp/fleet_rent_${name}.gpu"
    fi
  done

  n_rented=${#rented_gpu[@]}
  if (( n_rented == 0 )); then
    log "iter=$i STOCK sighting but 0 rents (${gpu_label} n=${#nodes[@]}) — keep polling"
    empty_sleep
    continue
  fi

  sleep 8
  for name in "${!rented_gpu[@]}"; do
    gpu=${rented_gpu[$name]}
    axis=${slot_axis[$name]}
    note=${slot_note[$name]}
    if name_live "$name"; then
      log "RENTED ok gpu=$gpu name=$name axis=$axis"
      write_rent_stamp "$name" "$axis" "$gpu" "$note" || true
    else
      log "up rc=0 but $name not in ps — keep polling"
    fi
  done
  refresh_queue
  # Immediately re-ls if more stock (no long sleep).
done

log "TIMEOUT after $MAX_ITERS iters — mine_count=$(mine_count)"
exit 2
