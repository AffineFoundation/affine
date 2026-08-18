#!/usr/bin/env python3
"""Fast fleet snatcher via Lium HTTP API (session-reused /executors).

One unfiltered GET /executors?gpu_count=8 per poll, then client-filter B300
(preferred) else B200. Avoids dual machine_names queries that 429 the key.

On stock: POST /executors/{id}/rent (fast path) then schedule-removal for TTL;
falls back to `lium up <executor_id> --name mine-* --ttl …` if API rent fails.
Never touches non-mine pods. Hard-stops if balance < $10k.

Critical: /pods API failure must NOT look like mine_count=0 (overshoots CAP).
"""
from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path("/home/const/subnet120")
EXP = ROOT / "mining/experiments/fleet-rent"
LOG = EXP / "logs/wait_fleet_b300.log"
PIDF = EXP / "logs/wait_fleet_b300.pid"
STAMP_DIR = EXP / "artifacts"

TTL = os.environ.get("TTL", "24h")
CAP = int(os.environ.get("MINE_CAP", "25"))
TARGET = int(os.environ.get("TARGET_MINES", "25"))
PARALLEL_N = int(os.environ.get("PARALLEL_N", "22"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "86400"))
PASS = int(os.environ.get('PASS', '3196'))
# Stay under Lium 429 while still beating CLI ls (~0.63s). 0.5s ≈ 2 Hz.
EMPTY_SLEEP = float(os.environ.get("EMPTY_SLEEP", "0.5"))
PODS_FAIL_SLEEP = float(os.environ.get("PODS_FAIL_SLEEP", "2.0"))
# Pytorch (Cuda + DinD) cuda13.0.2 — same image as live mine-* pods.
# p2573/p2578 one-time/private templates 404'd or API-rent 400
# ("One-time and temporary templates can only be used by a single pod");
# CLI `lium up` still worked. p2585: switch to the *public catalog* id
# 345273fa… (identical daturaai/pytorch:2.12.0-py3.12-cuda13.0.2-…-dind)
# so POST /executors/{id}/rent succeeds without burning the flicker window
# on a doomed API attempt before CLI fallback.
TEMPLATE_ID = os.environ.get(
    "LIUM_TEMPLATE_ID", "345273fa-4818-46f7-a8fa-32f0e331713c"
)
SSH_PUBKEY_PATH = Path(
    os.environ.get("LIUM_SSH_PUBKEY", str(Path.home() / ".ssh/id_ed25519.pub"))
)
# p2581: executor 8f34559f… (GPU-eb8e642a…) hung MoE init twice even with
# --enforce-eager (shm_broadcast / workers 0% CPU / GPU1 stuck 100%). Crown
# B200 serves teacher fine without eager — host is bad, not the flag. Skip.
# p2582: also merge artifacts/executor_blacklist.txt (BAD_HOST watcher grows it).
def _load_executor_blacklist() -> frozenset[str]:
    """Load executor UUIDs to skip. Strip inline comments (uuid  # note).

    p3815: a mid-line `# comment` on fbb1135f… made the API set hold the
    whole string, so bare eid never matched and R337 bad-host could be
    re-rented. Match bash list_nodes: first whitespace token only.
    """
    ids: set[str] = set()
    env = os.environ.get(
        "LIUM_EXECUTOR_BLACKLIST",
        "8f34559f-30e7-40d2-bc80-629b0903742a",
    )
    for x in env.split(","):
        x = x.strip()
        if x and not x.startswith("#"):
            ids.add(x.split()[0])
    blf = STAMP_DIR / "executor_blacklist.txt"
    if blf.is_file():
        for line in blf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line.split()[0])
    return frozenset(ids)


EXECUTOR_BLACKLIST = _load_executor_blacklist()

# Distinct axes (one pod each). Skip names already live.
QUEUE = [
    # R7 live on warm mine-r4-fullft-1 (p2158 after R8 REFUTE) — do not re-rent.
    # R8 REFUTED p2158 n80 m=-0.027 z=-1.64 vs ckp333 — do not re-rent.
    # R24 live on mine-r3-grpo-1 (p2205 warm-arm after R15 REFUTE) — do not re-rent.
    # ("mine-r24-longctx-1", "R24", "Tok GRPO max_len=16384 max_new=1024 (≠ R3 6144/512)"),
    # R25 RENTED p2211 as mine-r25-hitemp-1 — do not re-rent.
    # ("mine-r25-hitemp-1", "R25", "Tok GRPO temperature=1.2 (≠ R3 temp=0.8)"),
    # R26 live on mine-crown-1 (p2213 warm-arm after R17 REFUTE) — do not re-rent.
    # ("mine-r26-lotemp-1", "R26", "Tok GRPO temperature=0.5 (≠ R3 0.8 / R25 1.2)"),
    # p2224: sbs-v2 still GATED (index 403) — p2223 repo_info≠weights false OK. Do not re-rent R10/R18.
    # ("mine-r10-merge-rl-1", "R10", "Tok×sbs-v2 α-merge → Reason-GRPO (BLOCKED Hub gated)"),
    # ("mine-r18-sbs-grpo-1", "R18", "pure sbs-v2-init Reason-GRPO (BLOCKED Hub gated)"),
    # p2235: R5b warm-armed on mine-r4-fullft-1 after R21 REFUTE — do not re-rent.
    # ("mine-r5-nonking-2", "R5b", "Talent/kevin non-king base FT"),
    # p2241: R5b SIGNAL_POS_BELOW vs guass; R19 warm-armed on mine-r4-fullft-1 — do not re-rent.
    # ("mine-r19-talent-grpo-1", "R19", "TalentPigs-init Reason-GRPO (≠ R3/R5b; sbs gated)"),
    # p2249: R22 warm-armed on mine-crown-1 after R33 REFUTE — do not re-rent.
    # ("mine-r22-golden-grpo-1", "R22", "golden-crown-init Reason-GRPO (≠ R3/R16/R19–R21)"),
    # p2252: R23 warm-armed on mine-r3-grpo-1 after R25 REFUTE — do not re-rent.
    # ("mine-r23-diane-grpo-1", "R23", "diane613-init Reason-GRPO (≠ R3/R16/R19–R22)"),
    # p2253: R27 warm-armed on mine-r4-fullft-1 after R19 SIGNAL_POS_BELOW — do not re-rent.
    # ("mine-r27-bigg-1", "R27", "Tok GRPO group_size=16 (≠ R3 G=4 / R3b G=8+alt-lr)"),
    # p2264: R28 warm-armed on mine-crown-1 after R22 REFUTE — do not re-rent.
    # ("mine-r28-hilr-1", "R28", "Tok GRPO lr=2e-5 (≠ R3 5e-6; isolates LR vs R3b)"),
    # p2288: R29 warm-armed on mine-r3-grpo-1 after R23 REFUTE — do not re-rent.
    # ("mine-r29-hirank-1", "R29", "Tok GRPO lora_r=64 (≠ R3 r=16; isolates rank vs R3b)"),
    # p2294: R30 warm-armed on mine-crown-1 after R28 REFUTE — do not re-rent.
    # ("mine-r30-hialpha-1", "R30", "Tok GRPO lora_alpha=128 r=16 (≠ R3 α=32; isolates α vs R29)"),
    # p2329: R31 live on mine-r31-nodrop-1 — do not re-rent.
    # ("mine-r31-nodrop-1", "R31", "Tok GRPO lora_dropout=0.0 (≠ R3 0.05; isolates dropout)"),
    # p2328: R32 warm-armed on mine-crown-1 after R30 SIGNAL_POS_BELOW — do not re-rent.
    # ("mine-r32-kl-1", "R32", "Tok GRPO kl_coef=0.02 vs base (≠ R3 kl=0; isolates KL)"),
    # p2349: R67 warm-armed on mine-r3-grpo-1 after R29 SIGNAL_POS_BELOW — do not re-rent.
    # ("mine-r67-hirank-longctx-1", "R67", "Tok HiRank×LongCtx r=64 16384/1024 (R29×R24; ≠ R29@6144 / R24@r=16 / R34 LongCtx×HiLR / R62 LongCtx×BigG×HiLR / R28 HiLR)"),
    # p2356: R69 warm-armed on mine-crown-1 after R32 REFUTE — do not re-rent.
    # ("mine-r69-hirank-bigg-1", "R69", "Tok HiRank×BigG r=64 G=16 (R29×R27; ≠ R29@G=4 / R27@r=16 / R68 HiRank×HiLR / R67 HiRank×LongCtx / R48 BigG×HiLR@r=16 / R3b G=8)"),
    # p2364: R71 warm-armed on mine-r4-fullft-1 after R27 REFUTE — do not re-rent.
    # ("mine-r71-hirank-longctx-bigg-1", "R71", "Tok HiRank×LongCtx×BigG r=64 16384/1024 G=16 lr=5e-6 (R29×R24×R27; ≠ R67@G=4 / R69@6144 / R70@G=4/2e-5 / R57@r=16 / R62@r=16)"),
    # p2366: R73 warm-armed on mine-r31-nodrop-1 after R31 REFUTE — do not re-rent.
    # ("mine-r73-hialpha-longctx-1", "R73", "Tok HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R30×R24; ≠ R30@6144 / R24@α32 / R34@α32+HiLR / R67@r64)"),
    # p2398: reign-7 crown flip → prioritize fjq-init (copy shamelessly) ahead of Tok ladder.
    # p2401: R156 warm-armed on mine-r3-grpo-1 after R67 REFUTE vs fjq — do not re-rent.
    # ("mine-r156-fjq-grpo-1", "R156", "fjq-init Reason-GRPO from live reign-7 dent1s2/Affine-5FjqRq3dGA-v1@cfd789c9 (≠ R3 Tok / R33 guass-init REFUTED / R19–R23; n80 king=same fjq)"),
    # p2350: R29/R24/R30 SIGNAL + R28 HiLR REFUTED → rent SIGNAL compounds before HiLR-primary R34.
    # Prefer no-HiLR first among HiRank/HiAlpha compounds.
    # p2411: crown=guass reign8 again → guass×HiAlpha HEAD (R33 α=32 REFUTED; ≠ Tok ladder / fjq R156/R157).
    # p2425: R158 warm-armed on mine-r3-grpo-1 after R156 REFUTE vs guass — do not re-rent.
    # ("mine-r158-guass-hialpha-1", "R158", "guass×HiAlpha α=128 r=16 G=4 @6144 from live reign-8 ttttxxxxsada/Affine-5guassq3tu@e86758f5 (≠ R33 α=32 REFUTED / R157 fjq×HiAlpha / R30 Tok×HiAlpha / R75 Tok×HiAlpha×BigG; n80 king=guass)"),
    # p2426: mine-r31 REBOOT_FAILED mid-R157 n80@65/80 — HF rematch LoRA→n80 before Tok ladder.
    # p2428: rematch running warm on mine-r3 GPUs4–5 (R158 keeps 6–7) — do not re-rent rematch.
    # ("mine-r157-rematch-1", "R157", "HF rematch fjq×HiAlpha LoRA (unconst/Affine-5czsc2fc98-r157-lora@step-150) merge+n80 vs guass after r31 REBOOT_FAILED (≠ R156 α32 REFUTE / R158 guass×HiAlpha / R75 Tok×HiAlpha×BigG)"),
    # p2364: R71 warm-armed on mine-r4-fullft-1 — SIGNAL_POS_BELOW vs legend p2441 m=+0.00145 z=0.13 — do not re-rent.
    # p2441: R75 warm-armed on mine-r4-fullft-1 after R71 SIGNAL — do not re-rent.
    # p2434: reign-9 legend; R69 REFUTE vs legend → QUEUE HEAD was legend-init (warm on crown).
    # p2448: R159 final REFUTE vs legend m=−0.00058 z=−1.18 n=79 empty-z both — warm R78 on crown; do not re-rent R159.
    # ("mine-r159-legend-grpo-1", "R159", "legend-init Reason-GRPO α=128 r=16 G=4 @6144 from live reign-9 diceofgod/affine-5fjgc5jhxq-legend@d259cb38 (≠ R69 Tok REFUTED vs legend / R158 guass×HiAlpha / R156·R157 fjq / R30·R75 Tok×HiAlpha; n80 king=legend)"),
    # ("mine-r75-hialpha-bigg-1", "R75", "Tok HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R30×R27; ≠ R30@G=4 / R27@α32 / R74@HiLR+G=4 / R73@16384 / R69@r64)"),
    # p2448: R78 warm-armed on mine-crown-1 after R159 REFUTE — do not re-rent while warm.
    # ("mine-r78-hialpha-longctx-bigg-1", "R78", "Tok HiAlpha×LongCtx×BigG α=128 r=16 16384/1024 G=16 lr=5e-6 (R30×R24×R27; ≠ R73@G4 / R75@6144 / R76@2e-5 / R77@6144+HiLR / R57@α32 / R71@r64)"),
    # p2450: R68 warm-armed on mine-r3-grpo-1 after R158 SUBMIT — do not re-rent.
    # ("mine-r68-hirank-hilr-1", "R68", "Tok HiRank×HiLR r=64 lr=2e-5 (R29×R28; ≠ R29@5e-6 / R28@r=16 / R67 HiRank×LongCtx / R34 LongCtx×HiLR@r=16 / R3b G=8)"),
    # p2459: reign-10 crown flip → thermopylae-init HEAD (copy shamelessly; Tok ladder demoted).
    # p2486: R160 rented on brave-fox-8e — do not re-rent.
    # ("mine-r160-thermopylae-grpo-1", "R160", "thermopylae-init Reason-GRPO α=128 r=16 G=4 @6144 from live reign-10 thermopylae-777/Affine-5eptsnvsre-v1@b5f748bf (≠ R159 legend REFUTED / R158 guass×HiAlpha / R69 Tok vs legend / R70 Tok ladder; n80 king=thermopylae)"),
    # p2487: R70 warm-armed on mine-r3-grpo-1 after R68 FINAL REFUTE — do not re-rent while warm.
    # ("mine-r70-hirank-longctx-hilr-1", "R70", "Tok HiRank×LongCtx×HiLR r=64 16384/1024 lr=2e-5 (R29×R24×R28; ≠ R67@5e-6 / R68@6144 / R34@r=16 / R69 BigG / R62@r=16)"),
    # p2502: R72 warm-armed on mine-r160-thermopylae-grpo-1 after R160 form SIGNAL_POS_BELOW — do not re-rent while warm.
    # ("mine-r72-hirank-longctx-bigg-hilr-1", "R72", "Tok HiRank×LongCtx×BigG×HiLR r=64 16384/1024 G=16 lr=2e-5 (R29×R24×R27×R28; ≠ R71@5e-6 / R70@G=4 / R62@r=16 / R69@6144 / R68@6144 / R67@G=4/5e-6)"),
    # p2530: R74 warm-armed on mine-crown-1 after R78 SIGNAL_POS_BELOW — do not re-rent while warm.
    # ("mine-r74-hialpha-hilr-1", "R74", "Tok HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R30×R28; ≠ R30@5e-6 / R28@α32 / R73@16384+5e-6 / R34@α32 / R68@r64)"),
    # p2550: R74 FINAL REFUTE → R76 warm-armed on mine-crown-1 — do not re-rent while warm.
    # ("mine-r76-hialpha-longctx-hilr-1", "R76", "Tok HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R30×R24×R28; ≠ R73@5e-6 / R74@6144 / R34@α32 / R70@r64 / R75@G16)"),
    # p2565: R72 FINAL SIGNAL_POS_BELOW → R77 warm-armed on mine-r160 — do not re-rent while warm.
    # ("mine-r77-hialpha-bigg-hilr-1", "R77", "Tok HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R30×R27×R28; ≠ R75@5e-6 / R74@G4 / R48@α32 / R76@16384 / R69@r64)"),
    # p2567: R76 FINAL SIGNAL_POS_BELOW → R79 warm-armed on mine-crown-1 — do not re-rent while warm.
    # ("mine-r79-hialpha-longctx-bigg-hilr-1", "R79", "Tok HiAlpha×LongCtx×BigG×HiLR α=128 r=16 16384/1024 G=16 lr=2e-5 (R30×R24×R27×R28; ≠ R78@5e-6 / R76@G4 / R77@6144 / R72@r64 / R62@α32)"),
    # p2608: live king=awesome-v11 reign13 — QUEUE HEAD = awesome-init (copy shamelessly).
    # Demote R161–R164 guass-init (king flipped @ 22:38Z).
    # p2694: R165 REFUTED (p2666); box still live as warm host for R173/R175 — do not re-rent if lunar dies.
    # ("mine-r165-awesome-hialpha-1", "R165", "awesome×HiAlpha α=128 r=16 G=4 lr=5e-6 @6144 from live reign-13 0pentensor/Affine-5dflhtkufw-awesome-v11@450bdfc3 (≠ R161–R164 guass demoted / R160 thermo / R158 guass×HiAlpha; n80 king=awesome)"),
    # p2617: QUEUE#2 awesome×BigG (isolate G vs R165) — prefer live-king init over Talent×LongCtx when stock flickers.
    # p2636: R166 warm-armed on mine-r160-thermopylae-grpo-1 (brave-fox-8e) after R77 FINAL SIGNAL_POS_BELOW — do not re-rent.
    # ("mine-r166-awesome-hialpha-bigg-1", "R166", "awesome×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 from live reign-13 (≠ R165 G=4 / R161 guass×BigG demoted / R77 Tok×HiAlpha×BigG×HiLR; n80 king=awesome)"),
    # p2618: QUEUE#3 awesome×HiLR (isolate lr vs R165) — prefer live-king init over Talent×LongCtx when stock flickers.
    # p2639: R167 warm-armed on mine-crown-1 (gentle-orbit-bd) after R79 REFUTE — do not re-rent.
#     ("mine-r167-awesome-hialpha-hilr-1", "R167", "awesome×HiAlpha×HiLR α=128 r=16 G=4 lr=2e-5 @6144 from live reign-13 (≠ R165@5e-6 / R166 BigG / R74 Tok×HiAlpha×HiLR REFUTED / R162 guass×HiLR demoted; n80 king=awesome)"),
    # p2619: QUEUE#4 awesome×LongCtx (isolate 16384/1024 vs R165@6144) — prefer live-king init over Talent×LongCtx filler.
    # p2640: R168 warm-armed on mine-r165 (lunar-wolf-be) GPUs 4–5 after API empty — do not re-rent.
    # ("mine-r168-awesome-hialpha-longctx-1", "R168", "awesome×HiAlpha×LongCtx α=128 r=16 G=4 lr=5e-6 16384/1024 from live reign-13 (≠ R165@6144 / R166 BigG / R167 HiLR / R163 guass×LongCtx demoted; n80 king=awesome)"),
    # p2620: QUEUE#5 awesome×HiRank (isolate r=64 vs R165 r=16) — prefer live-king init over Talent×LongCtx filler.
    # p2641: R169 warm-armed on mine-crown-1 GPUs 4–5 (isolated /root/r169; R167 keeps 6–7) — do not re-rent.
    # ("mine-r169-awesome-hialpha-hirank-1", "R169", "awesome×HiAlpha×HiRank α=128 r=64 G=4 lr=5e-6 @6144 from live reign-13 (≠ R165 r=16 / R166 BigG / R167 HiLR / R168 LongCtx / R29 Tok HiRank SIGNAL; n80 king=awesome)"),
    # p2621: QUEUE#6 awesome×BigG×HiLR compound (R166×R167; R77 Tok mid SIGNAL on live-king init).
    # p2642: R170 warm-armed on mine-r160 (brave-fox-8e) GPUs 4–5 after freeing R77 chall; R166 keeps 6–7 — do not re-rent.
    # ("mine-r170-awesome-hialpha-bigg-hilr-1", "R170", "awesome×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 from live reign-13 (≠ R165 G4@5e-6 / R166 BigG@5e-6 / R167 HiLR G4 / R77 Tok mid SIGNAL / R168 LongCtx / R169 HiRank; n80 king=awesome)"),
    # p2622: QUEUE#7 awesome×HiRank×BigG compound (R169×R166; Tok R29 HiRank SIGNAL).
    # p2668: R171 warm-armed on mine-crown-1 GPUs 4–5 (isolated /root/r171; T/K 0–3; 6–7 free for chall) — do not re-rent.
    # ("mine-r171-awesome-hialpha-hirank-bigg-1", "R171", "awesome×HiAlpha×HiRank×BigG α=128 r=64 G=16 lr=5e-6 @6144 from live reign-13 (≠ R165 r16 G4 / R166 BigG r16 / R169 HiRank G4 / R170 BigG×HiLR / R29 Tok HiRank SIGNAL; n80 king=awesome)"),
    # p2623: QUEUE#8 awesome×HiRank×HiLR compound (R169×R167; HiLR on HiRank without BigG).
    # p2669: R172 warm-armed on mine-crown-1 GPUs 6–7 (isolated /root/r172; R171 keeps 4–5; lunar R165 chall purged for R168 merge) — do not re-rent.
    # ("mine-r172-awesome-hialpha-hirank-hilr-1", "R172", "awesome×HiAlpha×HiRank×HiLR α=128 r=64 G=4 lr=2e-5 @6144 from live reign-13 (≠ R165 r16@5e-6 / R167 HiLR r16 / R169 HiRank@5e-6 / R170 BigG×HiLR r16 / R171 HiRank×BigG @5e-6 / R29 Tok HiRank SIGNAL; n80 king=awesome)"),
    # p2624: QUEUE#9 awesome×HiRank×LongCtx compound (R169×R168; Tok R29 HiRank SIGNAL + R24 LongCtx SIGNAL_POS).
    # p2677: R173 warm-armed on mine-r165 lunar GPUs 6–7 after R168 REFUTE (isolated /root/r173; T/K 0–3; post 4–5) — do not re-rent.
    # ("mine-r173-awesome-hialpha-hirank-longctx-1", "R173", "awesome×HiAlpha×HiRank×LongCtx α=128 r=64 G=4 lr=5e-6 16384/1024 from live reign-13 (≠ R165 r16@6144 / R168 LongCtx r16 / R169 HiRank@6144 / R172 HiRank×HiLR / R171 HiRank×BigG / R67 Tok HiRank×LongCtx REFUTED; n80 king=awesome)"),
    # p2625: QUEUE#10 awesome×HiRank×BigG×HiLR triple (R171×R172 / R169×R170; R77 Tok mid SIGNAL on live-king init).
    # p2684: R174 warm-armed on mine-crown-1 GPUs 6–7 after R172 chall purge (isolated /root/r174; R171 keeps 4–5) — do not re-rent.
    # ("mine-r174-awesome-hialpha-hirank-bigg-hilr-1", "R174", "awesome×HiAlpha×HiRank×BigG×HiLR α=128 r=64 G=16 lr=2e-5 @6144 from live reign-13 (≠ R165 r16 G4@5e-6 / R170 BigG×HiLR r16 / R171 HiRank×BigG @5e-6 / R172 HiRank×HiLR G4 / R173 HiRank×LongCtx / R77 Tok mid SIGNAL; n80 king=awesome)"),
    # p2626: QUEUE#11 awesome×BigG×LongCtx (R166×R168; Tok R24 LongCtx SIGNAL + BigG isolate; R78 Tok analogue on live-king).
    # p2685: R175 warm-armed on mine-r165 lunar GPUs 4–5 after R173 post→6,7 (isolated /root/r175; R173 keeps 6–7) — do not re-rent.
    # ("mine-r175-awesome-hialpha-bigg-longctx-1", "R175", "awesome×HiAlpha×BigG×LongCtx α=128 r=16 G=16 lr=5e-6 16384/1024 from live reign-13 (≠ R165 G4@6144 / R166 BigG@6144 / R168 LongCtx G4 / R170 BigG×HiLR / R173 HiRank×LongCtx / R174 HiRank×BigG×HiLR / R78 Tok; n80 king=awesome)"),
    # p2627: QUEUE#12 awesome×HiLR×LongCtx (R167×R168; Tok R76 HiAlpha×LongCtx×HiLR SIGNAL_POS_BELOW analogue on live-king).
    # p2693: R176 warm-armed on mine-r160 (brave-fox-8e) GPUs 6–7 after R166 chall purge; R170 keeps 4–5 — do not re-rent.
    # ("mine-r176-awesome-hialpha-hilr-longctx-1", "R176", "awesome×HiAlpha×HiLR×LongCtx α=128 r=16 G=4 lr=2e-5 16384/1024 from live reign-13 (≠ R165 G4@5e-6@6144 / R167 HiLR@6144 / R168 LongCtx@5e-6 / R170 BigG×HiLR / R172 HiRank×HiLR / R173 HiRank×LongCtx / R175 BigG×LongCtx / R76 Tok; n80 king=awesome)"),
    # p2628: QUEUE#13 awesome×BigG×HiLR×LongCtx (R170×R168 / R175×R167 / R176×R166; Tok R79 HiAlpha×LongCtx×BigG×HiLR analogue on live-king).
    # p2698: R177 warm-armed on mine-r160 (brave-fox-8e) GPUs 4–5 after R170 REFUTE; R176 keeps 6–7 — do not re-rent.
    # ("mine-r177-awesome-hialpha-bigg-hilr-longctx-1", "R177", "awesome×HiAlpha×BigG×HiLR×LongCtx α=128 r=16 G=16 lr=2e-5 16384/1024 from live reign-13 (≠ R165 G4@5e-6@6144 / R170 BigG×HiLR@6144 / R175 BigG×LongCtx@5e-6 / R176 HiLR×LongCtx G4 / R174 HiRank×BigG×HiLR / R79 Tok; n80 king=awesome)"),
    # p2629: QUEUE#14 awesome×HiRank×BigG×LongCtx (R171×R168 / R173×R166 / R175×R169; Tok R29 HiRank + R24 LongCtx + BigG on live-king).
    # p2706: R178 warm-armed on mine-r165 lunar GPUs 6–7 after R173 REFUTE; R175 keeps 4–5 — do not re-rent.
    # ("mine-r178-awesome-hialpha-hirank-bigg-longctx-1", "R178", "awesome×HiAlpha×HiRank×BigG×LongCtx α=128 r=64 G=16 lr=5e-6 16384/1024 from live reign-13 (≠ R165 r16 G4@6144 / R171 HiRank×BigG@6144 / R173 HiRank×LongCtx G4 / R175 BigG×LongCtx r16 / R174 HiRank×BigG×HiLR / R177 BigG×HiLR×LongCtx r16; n80 king=awesome)"),
    # p2630: QUEUE#15 awesome×HiRank×HiLR×LongCtx (R172×R168 / R173×R167; Tok R29 HiRank + R28 HiLR + R24 LongCtx; no BigG).
    # p2721: R179 warm-armed on mine-crown-1 GPUs 4–5 after R171 purge; R174 keeps 6–7 — do not re-rent.
    # ("mine-r179-awesome-hialpha-hirank-hilr-longctx-1", "R179", "awesome×HiAlpha×HiRank×HiLR×LongCtx α=128 r=64 G=4 lr=2e-5 16384/1024 from live reign-13 (≠ R165 r16 G4@5e-6@6144 / R172 HiRank×HiLR@6144 / R173 HiRank×LongCtx@5e-6 / R176 HiLR×LongCtx r16 / R178 HiRank×BigG×LongCtx G16@5e-6 / R174 HiRank×BigG×HiLR@6144; n80 king=awesome)"),
    # p2631: QUEUE#16 awesome×HiRank×BigG×HiLR×LongCtx full 4-way (R174×R168 / R178×R167 / R179×R166; completes factorial).
    # p2742: R180 lean-warm on mine-crown-1 GPUs 6–7 after R174 REFUTE+purge; R179 keeps 4–5 — do not re-rent.
    # ("mine-r180-awesome-hialpha-hirank-bigg-hilr-longctx-1", "R180", "awesome×HiAlpha×HiRank×BigG×HiLR×LongCtx α=128 r=64 G=16 lr=2e-5 16384/1024 from live reign-13 (≠ R165@6144 / R174 HiRank×BigG×HiLR@6144 / R178 HiRank×BigG×LongCtx@5e-6 / R179 HiRank×HiLR×LongCtx G4 / R177 BigG×HiLR×LongCtx r16; n80 king=awesome)"),
    # p2632: QUEUE#17 awesome×Reason-SFT method axis (≠ GRPO R165–R180 factorial / ≠ R1 Tok-init SFT).
    # p2749: R181 lean-warm on mine-crown-1 GPUs 4–5 after R179 purge; R180 keeps 6–7 — do not re-rent.
    # ("mine-r181-awesome-reason-sft-1", "R181", "awesome×Reason-SFT thought-mask α=128 r=16 lr=2e-5 @8192 from live reign-13 (≠ R165–R180 GRPO ladder / ≠ R1 Tok-init SFT; n80 king=awesome)"),
    # p2753: R182 lean-warm on mine-crown-1 GPUs 4–5 after R181 SIGNAL_POS_BELOW+purge; R180 keeps 6–7 — do not re-rent.
    # ("mine-r182-awesome-datafilt-sft-1", "R182", "awesome×DataFilt-SFT top150 Reason EP=2 α=128 r=16 lr=2e-5 @8192 from live reign-13 (≠ R181 n460 EP1 / ≠ R165–R180 GRPO / ≠ R7 Tok datafilt; n80 king=awesome)"),
    # p2756: R182 REFUTE m=-0.0084 z=-1.83 thought✓ B✓ — purge chall; R35 Talent×LongCtx lean on crown 4,5 (download Talent; R180@6,7).
    # ("mine-r35-talent-longctx-1", "R35", "Talent×LongCtx max_len=16384 (R19 SIGNAL_POS × R24 LongCtx)"),
    # p2633 was QUEUE#18; next rent head after R35 lean-warm:
    # p2570–p2571 guass compounds — demoted after reign13 flip (kept for replay; do not promote while king=awesome).
    # p2778: crown flipped awesome→guass reign14 @12:30Z → QUEUE HEAD was guass-init.
    # p2782: R161 lean-warm on mine-crown-1 GPUs 4–5 after R37 REFUTE+purge (isolated /root/r161; R180 keeps 6–7) — do not re-rent.
    # ("mine-r161-guass-hialpha-bigg-1", "R161", "guass×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 from live reign-14 ttttxxxxsada/Affine-5guassq3tu@e86758f5 (≠ R158 G=4 submitted / R33 α=32 REFUTED / R77 Tok×HiAlpha×BigG×HiLR; n80 king=guass)"),
    # p2784: R162 lean-warm on mine-r160 GPUs 6–7 (isolated /root/r162; R177 keeps 4–5) — do not re-rent.
    # ("mine-r162-guass-hialpha-hilr-1", "R162", "guass×HiAlpha×HiLR α=128 r=16 G=4 lr=2e-5 @6144 from live reign-14 (≠ R158@5e-6 / R161 BigG G=16 / R33 α=32 REFUTED; n80 king=guass)"),
    # p2787: crown flipped guass→fqb/ckp333 reign15 @13:28Z → QUEUE HEAD = ckp333-init (demote guass R163/R164).
    # p2795: R39 lean-warm on mine-r160 GPUs 4–5 after R177 REFUTE+purge (isolated /root/r39; R162 keeps 6–7) — do not re-rent.
    # ("mine-r39-ckp333-1", "R39", "ckp333-init Reason-GRPO from live reign-15 tolegend/Affine-5fqbxvz29b-ckp333@24c137e8 (≠ Tok/Talent/golden/diane/guass; n80 king=fqb)"),
    # p2800: R162 SIGNAL_POS_BELOW vs fqb m=+0.00046 bar=0.0187 → purge; R40 lean-warm on mine-r160 GPUs 6–7 (isolated /root/r40; R39 keeps 4–5) — do not re-rent.
    # ("mine-r40-ckp333-longctx-1", "R40", "ckp333×LongCtx max_len=16384 from live reign-15 (R39 parent × R24 LongCtx; ≠ R39@6144; n80 king=fqb)"),
    # p2801: crown flipped fqb→marsplan queen reign16 @14:48Z → QUEUE HEAD = marsplan-init (copy shamelessly; demote ckp333 R47).
    # p2814: R210 lean-warm on mine-r165 GPUs 4–5 (isolated /root/r210; R178 keeps 6–7) — R204 recipe; do not re-rent.
    # ("mine-r204-marsplan-hialpha-1", "R204", "marsplan×HiAlpha α=128 r=16 G=4 lr=5e-6 @6144 from live reign-16 marsplan0624/affine-5gedzafcvg-queen@556d02a2 (≠ R165 awesome / R39 ckp333 / R158 guass; n80 king=marsplan)"),
    # ("mine-r210-marsplan-hialpha-1", "R210", "marsplan×HiAlpha α=128 r=16 G=4 lr=5e-6 @6144 lean on lunar (R204 recipe; ≠ R209 HiRank / R205 BigG; n80 king=marsplan)"),
    # p2807: QUEUE#2→HEAD marsplan×BigG (isolate G vs R204/R210) — prefer live-king init over demoted ckp333/guass.
    # p2824: R205 lean-warm on mine-crown-1 GPUs 6–7 after R180 REFUTE+purge (isolated /root/r205; R206 keeps 4–5) — do not re-rent.
    # ("mine-r205-marsplan-hialpha-bigg-1", "R205", "marsplan×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 from live reign-16 (≠ R204/R210 G=4 / R165 awesome / R47 ckp333×BigG demoted / R161 guass×BigG; n80 king=marsplan)"),
    # p2808: QUEUE#3 marsplan×HiLR (isolate lr=2e-5 vs R204@5e-6; G=4 ≠ R205 G=16) — prefer live-king init over demoted ckp333/guass.
    # p2815: R206 lean-warm on mine-crown-1 GPUs 4–5 after R161 demoted-guass stop (isolated /root/r206; R180 keeps 6–7) — do not re-rent.
    # ("mine-r206-marsplan-hialpha-hilr-1", "R206", "marsplan×HiAlpha×HiLR α=128 r=16 G=4 lr=2e-5 @6144 from live reign-16 (≠ R204 lr=5e-6 / R205 G=16 / R165 awesome / R44 ckp333×HiLR demoted / R162 guass×HiLR; n80 king=marsplan)"),
    # p2809: QUEUE#4→HEAD marsplan×HiRank×HiLR — port R172 CONFIRMED recipe onto live-king init (r=64 lr=2e-5; ≠ R206 r=16).
    # p2830: R207 lean-warm on mine-r160 GPUs 4–5 after R209 REFUTE+purge (isolated /root/r207; R216 keeps 6–7) — do not re-rent.
    # ("mine-r207-marsplan-hialpha-hirank-hilr-1", "R207", "marsplan×HiAlpha×HiRank×HiLR α=128 r=64 G=4 lr=2e-5 @6144 from live reign-16 (R172 recipe on marsplan; ≠ R204 r16@5e-6 / R206 r16@2e-5 / R172 awesome parent; n80 king=marsplan)"),
    # p2810: QUEUE#5 marsplan×LongCtx (isolate 16384/1024 vs R204@6144; awesome R168 analogue) — prefer live-king init over demoted ckp333/guass.
    # p2831: R208 lean-warm on mine-crown-1 GPUs 4–5 after R206 SIGNAL_POS_BELOW+purge (isolated /root/r208; R205 keeps 6–7) — do not re-rent.
    # ("mine-r208-marsplan-hialpha-longctx-1", "R208", "marsplan×HiAlpha×LongCtx α=128 r=16 G=4 lr=5e-6 16384/1024 from live reign-16 (≠ R204@6144 / R205 BigG / R206 HiLR / R207 HiRank×HiLR / R168 awesome parent; n80 king=marsplan)"),
    # p2816: QUEUE#4→HEAD marsplan×HiRank×BigG — port R171 SIGNAL recipe (r=64 G=16 lr=5e-6) onto live-king init.
    # p2835: R211 lean-warm on mine-r165 GPUs 4–5 after R210 SIGNAL_POS_BELOW+purge (isolated /root/r211; R178 keeps 6–7) — do not re-rent.
    # ("mine-r211-marsplan-hialpha-hirank-bigg-1", "R211", "marsplan×HiAlpha×HiRank×BigG α=128 r=64 G=16 lr=5e-6 @6144 from live reign-16 (R171 recipe on marsplan; ≠ R205 r16@G16 / R209 r64@G4 / R207 r64@G4@2e-5 / R171 awesome parent; n80 king=marsplan)"),
    # p2817: QUEUE#5→HEAD marsplan×BigG×LongCtx — port R175 SIGNAL recipe (G=16 16384/1024 lr=5e-6) onto live-king init.
    # p2841: R212 lean-warm on mine-r165 GPUs 6–7 after R178 SIGNAL_POS_BELOW+purge (isolated /root/r212; R211 keeps 4–5) — do not re-rent.
    # ("mine-r212-marsplan-hialpha-bigg-longctx-1", "R212", "marsplan×HiAlpha×BigG×LongCtx α=128 r=16 G=16 lr=5e-6 16384/1024 from live reign-16 (R175 recipe on marsplan; ≠ R208 G4@16384 / R205 BigG@6144 / R211 HiRank×BigG@6144 / R175 awesome parent; n80 king=marsplan)"),
    # p2818: QUEUE#6→HEAD marsplan×HiRank×LongCtx — port R173 recipe (r=64 G=4 lr=5e-6 16384/1024) onto live-king init.
    # p2844: R213 lean-warm on mine-r160 GPUs 6–7 after R216 REFUTE+purge (isolated /root/r213; R207 keeps 4–5) — do not re-rent.
    # ("mine-r213-marsplan-hialpha-hirank-longctx-1", "R213", "marsplan×HiAlpha×HiRank×LongCtx α=128 r=64 G=4 lr=5e-6 16384/1024 from live reign-16 (R173 recipe on marsplan; ≠ R208 LongCtx r16 / R209 HiRank@6144 / R211 HiRank×BigG / R212 BigG×LongCtx / R173 awesome parent; n80 king=marsplan)"),
    # p2846: R214 lean-warm on mine-r160 GPUs 4–5 after R207 REFUTE+purge (isolated /root/r214; R213 keeps 6–7) — do not re-rent.
    # ("mine-r214-marsplan-hialpha-hirank-bigg-hilr-1", "R214", "marsplan×HiAlpha×HiRank×BigG×HiLR α=128 r=64 G=16 lr=2e-5 @6144 from live reign-16 (R174/R172×BigG on marsplan; ≠ R207 G4@2e-5 / R211 HiRank×BigG@5e-6 / R205 BigG r16 / R174 awesome parent; n80 king=marsplan)"),
    # p2820: QUEUE#8→HEAD marsplan×BigG×HiLR — port R170 recipe (r=16 G=16 lr=2e-5) onto live-king init.
    # p2847: R215 lean-warm on mine-crown-1 GPUs 4–5 after R208 SIGNAL_POS_BELOW+purge (isolated /root/r215; R205 keeps 6–7) — do not re-rent.
    # ("mine-r215-marsplan-hialpha-bigg-hilr-1", "R215", "marsplan×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 from live reign-16 (R170 recipe on marsplan; ≠ R205 BigG@5e-6 / R206 HiLR G4 / R214 HiRank×BigG×HiLR r64 / R170 awesome parent; n80 king=marsplan)"),
    # p2858: crown flipped isomsom r17→marsplan queen reign18 @20:17Z → QUEUE HEAD back to marsplan-init (R217); demote R240→isomsom-nonking later.
    # p2825: QUEUE#9→HEAD marsplan×HiRank×BigG×LongCtx — port R178 recipe (r=64 G=16 16384/1024 lr=5e-6) onto live-king init.
    # p2866: R217 lean-warm on mine-r160 GPUs 6–7 after R213 SIGNAL_POS_BELOW+purge (isolated /root/r217; R214 keeps 4–5) — do not re-rent.
    # ("mine-r217-marsplan-hialpha-hirank-bigg-longctx-1", "R217", "marsplan×HiAlpha×HiRank×BigG×LongCtx α=128 r=64 G=16 lr=5e-6 16384/1024 from live reign-18 (R178 recipe on marsplan; ≠ R212 BigG×LongCtx r16 / R211 HiRank×BigG@6144 / R213 HiRank×LongCtx G4 / R214 HiRank×BigG×HiLR / R178 awesome parent; n80 king=marsplan)"),
    # p2826: QUEUE#10→HEAD marsplan×HiRank×HiLR×LongCtx — port R179 recipe (r=64 G=4 lr=2e-5 16384/1024) onto live-king init.
    # p2874: R218 lean-warm on mine-crown-1 GPUs 6–7 after R205 SIGNAL_POS_BELOW+purge (isolated /root/r218; R215 keeps 4–5) — do not re-rent.
    # ("mine-r218-marsplan-hialpha-hirank-hilr-longctx-1", "R218", "marsplan×HiAlpha×HiRank×HiLR×LongCtx α=128 r=64 G=4 lr=2e-5 16384/1024 from live reign-16 (R179 recipe on marsplan; ≠ R207 HiRank×HiLR@6144 / R213 HiRank×LongCtx@5e-6 / R216 HiLR×LongCtx r16 / R217 HiRank×BigG×LongCtx G16 / R179 awesome parent; n80 king=marsplan)"),
    # p2899: crown flipped marsplan r18→leary-criste t3 reign19 @23:10Z → QUEUE HEAD was leary-t3-init.
    # p2900: crown flipped leary r19→marsplan queen reign20 @23:12Z → QUEUE HEAD back to marsplan R219; demote R277→leary-nonking.
    # ("mine-r277-leary-t3-hialpha-1", "R277", "leary-t3-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from prior-crown leary-criste/affine-5g4yy75zuz-t3@1ee64fe6 (R204 knobs; ≠ R204–R276 / ≠ R251 leary-nonking-era / ≠ R5 FullFT; n80 king=marsplan-queen)"),
    # p2913: R219 lean-warm on mine-crown-1 GPUs 6–7 after R215 REFUTE+purge (isolated /root/r219; 4–5 free for R220) — do not re-rent.
    # ("mine-r219-marsplan-hialpha-hirank-bigg-hilr-longctx-1", "R219", "marsplan×HiAlpha×HiRank×BigG×HiLR×LongCtx α=128 r=64 G=16 lr=2e-5 16384/1024 from live reign-20 marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R180 recipe; ≠ R214@6144 / R217@5e-6 / R218 G4 / R215 r16 / R277 leary-prior; n80 king=marsplan-queen)"),
    # p2914: R220 lean-warm on mine-crown-1 GPUs 4–5 (R219 keeps 6–7; isolated /root/r220) — do not re-rent.
    # ("mine-r220-marsplan-hialpha-bigg-hilr-longctx-1", "R220", "marsplan×HiAlpha×BigG×HiLR×LongCtx α=128 r=16 G=16 lr=2e-5 16384/1024 from live reign-20 (R177 recipe on marsplan; ≠ R215 BigG×HiLR@6144 / R212 BigG×LongCtx@5e-6 / R216 HiLR×LongCtx G4 / R219 HiRank×BigG×HiLR×LongCtx r64 / R177 awesome parent; n80 king=marsplan)"),
    # p2917: crown flipped vera r21→marsplan queen reign22 @01:11Z → QUEUE HEAD = R221 marsplan×Reason-SFT; demote R252→vera-nonking.
    # p2918: R221 lean-warm on mine-r160 GPUs 4–5 after R214 SIGNAL re-serve purge (isolated /root/r221; R217 keeps 6–7) — do not re-rent.
    # ("mine-r221-marsplan-reason-sft-1", "R221", "marsplan×Reason-SFT thought-mask α=128 r=16 lr=2e-5 @8192 from live reign-22 marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R181 method; promoted p2917 after vera flip-back; ≠ R252 vera-prior / ≠ R204–R220 GRPO; n80 king=marsplan-queen)"),
    # p3175: rent HEAD → R225 REINFORCE (method axis) ahead of more nonking×HiAlpha-GRPO
    # clones (R259/R252). Live R262/R260 already rented — do not re-rent. R262 REFUTE p3168.
    # p3009: was HEAD R262 Kevin-v5 — live as mine-r262-kevin-v5-nonking-grpo-1.
    # ("mine-r262-kevin-v5-nonking-grpo-1", "R262", "Kevin-v5-nonking×HiAlpha-GRPO … LIVE — do not re-rent"),
    # p2987: was R260 Elonmasky — live as mine-r260-elonmasky-ckp777-nonking-grpo-1.
    # ("mine-r260-elonmasky-ckp777-nonking-grpo-1", "R260", "Elonmasky-ckp777-nonking×HiAlpha-GRPO … LIVE — do not re-rent"),
    # p3178: R225 lean TRAIN on mine-r262 golden GPUs 4,5 after R448 FALSE_PROBE — do not re-rent.
    # p2924/p3005: R225 was after R252; p3175 promote to HEAD (package+prebuilt 456K+HF lora OK).
    # p2836: QUEUE marsplan×REINFORCE — method axis (R8 recipe on live-king init; ≠ GRPO factorial).
    # ("mine-r225-marsplan-reinforce-1", "R225", "marsplan×REINFORCE … LIVE lean on R262 4,5 p3178 — do not re-rent"),
    # p3192: crown → sbs-v5 reign27; R226 LIVE on brave-raven — do not re-rent; HEAD → R227.
    # p3189: R226 HEAD retarget marsplan→genesis r26 (live king FullFT method axis; same pod name/bootstrap).
    # p3178: rent HEAD → R226 FullFT (next method axis).
    # p3029: R335 demoted from rent — lean TRAIN on lunar GPUs 4,5 (G=8 while R230 G=4 on 6,7).
    # p3006: was after R225; p3004 QUEUE#124 marsplan×BoN-CE×BigG G=8 isolate vs R230 G=4.
    # p2837: QUEUE#18 marsplan×FullFT — method axis (R4/H121 recipe on live-king init; ≠ R204–R225 LoRA GRPO/SFT/REINFORCE / ≠ R4 Tok parent).
    # p3007: R226 package+prebuilt+HF armed (was missing upload_and_launch; bootstrap would fail on rent).
    # ("mine-r226-marsplan-fullft-1", "R226", "LIVE brave-raven p3190 — do not re-rent; n80 king=sbs-v5 r27"),
    # p3196: R227 TRAIN salvage on idle brave-raven (ex-R226) — do not re-rent. HEAD→R259.
    # p3190: R227 FullFT-HiLR retarget marsplan→genesis r26 (keep pod name; package+prebuilt rebuilt).
    # p2838: was QUEUE#19 marsplan FullFT-HiLR.
    # ("mine-r227-marsplan-fullft-hilr-1", "R227", "LIVE salvage brave-raven p3196 — do not re-rent; n80 king=sbs-v5"),
    # p3379: R252 RENTED gentle-wolf-8c — comment out; n80 king=loveaffine r32.
    # ("mine-r252-vera-t4-nonking-grpo-1", "R252", "LIVE gentle-wolf-8c p3379 loveaffine n80 — do not re-rent"),
    # p3374: quarantine R259 — executor 7b9ee272… SSH-refused badhost (p3365); do not re-HEAD until host pool changes.
    # p3175: demote R259/R252 nonking×HiAlpha-GRPO clones behind method axes (same method as live R262/R260).
    # p2986: was HEAD R259 Michael-h2 (chal-00636) — demoted p2987 after chal flip; further demoted p3175.
    # ("mine-r259-michael-h2-nonking-grpo-1", "R259", "Michael-h2-nonking×HiAlpha-GRPO … QUARANTINE p3374 badhost 7b9ee272"),
    # p2918: was R252 vera-nonking — demoted p3175 behind method axes; LIVE p3379.
    # p2919: R222 lean TRAIN on mine-r165 lunar GPUs4–5 (purged idle R211 chall) — do not re-rent.
    # p2832: QUEUE#14 marsplan×DataFilt-SFT — method axis (R182 recipe on live-king init; ≠ R221 EP1 broader / ≠ R182 awesome parent).
    # ("mine-r222-marsplan-datafilt-sft-1", "R222", "marsplan×DataFilt-SFT top150 Reason EP=2 α=128 r=16 lr=2e-5 @8192 from live reign-16 (R182 method on marsplan; ≠ R221 Reason-SFT EP1 / ≠ R204–R221 GRPO+SFT ladder / ≠ R182 awesome-init DataFilt / ≠ R7 Tok datafilt; n80 king=marsplan)"),
    # p2922: R223 lean-warm on mine-r160 brave GPUs4–5 after R222 SIGNAL harvest (isolated /root/r223; R217 keeps 6–7) — do not re-rent.
    # p2833: QUEUE#15 marsplan×Thought-Format — method axis (R6 recipe on live-king init; ≠ R221/R222 SFT curricula / ≠ R6 Tok parent).
    # ("mine-r223-marsplan-thought-format-1", "R223", "marsplan×Thought-Format short-z≤180 non-listy EP=6 α=32 r=16 lr=5e-6 @16384 from live reign-16 (R6 method on marsplan; ≠ R221 Reason-SFT / ≠ R222 DataFilt-SFT / ≠ R204–R222 GRPO+SFT ladder / ≠ R6 Tok thought-format; n80 king=marsplan)"),
    # p2923: R224 lean-warm on mine-r165 lunar GPUs4–5 after R222 SIGNAL left slot free (isolated /root/r224; R212 keeps 6–7) — do not re-rent.
    # p2834: QUEUE#16 marsplan×Long-Thought — method axis (R6b recipe on live-king init; ≠ R223 short-z / ≠ R221/R222 SFT / ≠ R6b Tok parent).
    # ("mine-r224-marsplan-long-thought-1", "R224", "marsplan×Long-Thought long-z>180 non-listy EP=6 α=32 r=16 lr=5e-6 @16384 from live reign-16 (R6b method on marsplan; ≠ R223 short-z Thought-Format / ≠ R221 Reason-SFT / ≠ R222 DataFilt-SFT / ≠ R204–R223 GRPO+SFT ladder / ≠ R6b Tok long-thought; n80 king=marsplan)"),
    # p3034: R336 demoted from rent — lean TRAIN on lunar GPUs 6,7 (G=8 Online-DPO while R335 BoN-BigG on 4,5).
    # p3010: was after R227; R229 SIGNAL method isolate group-size; ≠ R334 G=4 / ≠ R335 BoN-BigG.
    # ("mine-r336-marsplan-online-dpo-bigg-1", "R336", "marsplan×Online-DPO×BigG G=8 β=0.1 α=32 r=16 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL method isolate G; ≠ R334 G=4@300 / ≠ R229 G=4@150 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3011: R337 marsplan×Online-DPO×HiLR lr=2e-5 after R336 (R229 SIGNAL method isolate LR; ≠ R334/R336/R229 @5e-6).
    ("mine-r337-marsplan-online-dpo-hilr-1", "R337", "marsplan×Online-DPO×HiLR lr=2e-5 β=0.1 α=32 r=16 G=4 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL method isolate LR; ≠ R334 G=4@5e-6 / ≠ R336 G=8@5e-6 / ≠ R229 G=4@150@5e-6 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3012: R338 marsplan×Online-DPO×BigG×HiLR G=8 lr=2e-5 after R337 (R229 SIGNAL G×LR compound; ≠ R336 G-only / ≠ R337 LR-only).
    ("mine-r338-marsplan-online-dpo-bigg-hilr-1", "R338", "marsplan×Online-DPO×BigG×HiLR G=8 lr=2e-5 β=0.1 α=32 r=16 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL G×LR compound; ≠ R336 G=8@5e-6 / ≠ R337 G=4@2e-5 / ≠ R334 G=4@5e-6@300 / ≠ R229 G=4@150 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3013: R339 marsplan×Online-DPO×HiRank r=64 after R338 (R229 SIGNAL method isolate rank; ≠ R334 r=16 / ≠ R336–R338 G/LR).
    ("mine-r339-marsplan-online-dpo-hirank-1", "R339", "marsplan×Online-DPO×HiRank r=64 β=0.1 α=32 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL method isolate rank; ≠ R334 r=16@G=4@5e-6 / ≠ R336 G=8@5e-6 / ≠ R337 G=4@2e-5 / ≠ R338 G=8@2e-5 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3014: R340 marsplan×Online-DPO×HiRank×BigG r=64 G=8 after R339 (R229 SIGNAL HiRank×BigG compound; ≠ R339 G=4 / ≠ R336 r=16).
    ("mine-r340-marsplan-online-dpo-hirank-bigg-1", "R340", "marsplan×Online-DPO×HiRank×BigG r=64 G=8 β=0.1 α=32 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL HiRank×BigG compound; ≠ R339 r=64@G=4 / ≠ R336 G=8@r=16 / ≠ R334 r=16@G=4 / ≠ R337–R338 HiLR / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3015: R341 marsplan×Online-DPO×HiRank×HiLR r=64 lr=2e-5 after R340 (R229 SIGNAL HiRank×HiLR compound; ≠ R339@5e-6 / ≠ R337 r=16 / ≠ R340 BigG).
    ("mine-r341-marsplan-online-dpo-hirank-hilr-1", "R341", "marsplan×Online-DPO×HiRank×HiLR r=64 lr=2e-5 β=0.1 α=32 G=4 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL HiRank×HiLR compound; ≠ R339 r=64@5e-6 / ≠ R337 r=16@2e-5 / ≠ R340 r=64@G=8@5e-6 / ≠ R338 G=8@2e-5 / ≠ R334 r=16@G=4@5e-6 / ≠ R336 G=8 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3016: R342 marsplan×Online-DPO×HiRank×BigG×HiLR r=64 G=8 lr=2e-5 after R341 (R229 SIGNAL HiRank×BigG×HiLR triple; ≠ R341 G=4 / ≠ R340@5e-6 / ≠ R338 r=16).
    ("mine-r342-marsplan-online-dpo-hirank-bigg-hilr-1", "R342", "marsplan×Online-DPO×HiRank×BigG×HiLR r=64 G=8 lr=2e-5 β=0.1 α=32 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL HiRank×BigG×HiLR triple; ≠ R341 r=64@G=4@2e-5 / ≠ R340 r=64@G=8@5e-6 / ≠ R339 r=64@G=4@5e-6 / ≠ R338 G=8@r=16@2e-5 / ≠ R337 r=16@2e-5 / ≠ R336 G=8@r=16 / ≠ R334 r=16@G=4 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3017: R343 marsplan×Online-DPO×LongCtx 16384/1024 after R342 (R229 SIGNAL context isolate vs R334@6144; baseline r=16 G=4 lr=5e-6).
    ("mine-r343-marsplan-online-dpo-longctx-1", "R343", "marsplan×Online-DPO×LongCtx 16384/1024 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL method isolate context; ≠ R334 @6144 / ≠ R342 HiRank×BigG×HiLR / ≠ R336–R341 G/LR/rank / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3018: R344 marsplan×Online-DPO×LongCtx×BigG 16384/1024 G=8 after R343 (R229 SIGNAL LongCtx×BigG compound; ≠ R343 G=4 / ≠ R336 G=8@6144).
    ("mine-r344-marsplan-online-dpo-longctx-bigg-1", "R344", "marsplan×Online-DPO×LongCtx×BigG 16384/1024 G=8 β=0.1 α=32 r=16 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×BigG compound; ≠ R343 G=4@16384 / ≠ R336 G=8@6144 / ≠ R334 G=4@6144 / ≠ R342 HiRank×BigG×HiLR / ≠ R337–R341 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3019: R345 marsplan×Online-DPO×LongCtx×HiLR 16384/1024 lr=2e-5 after R344 (R229 SIGNAL LongCtx×HiLR compound; ≠ R343@5e-6 / ≠ R337@6144 / ≠ R344 BigG).
    ("mine-r345-marsplan-online-dpo-longctx-hilr-1", "R345", "marsplan×Online-DPO×LongCtx×HiLR 16384/1024 lr=2e-5 β=0.1 α=32 r=16 G=4 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×HiLR compound; ≠ R343 lr=5e-6@16384 / ≠ R337 lr=2e-5@6144 / ≠ R344 LongCtx×BigG G=8 / ≠ R338 BigG×HiLR@6144 / ≠ R334 @6144 / ≠ R336–R342 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3020: R346 marsplan×Online-DPO×LongCtx×BigG×HiLR 16384/1024 G=8 lr=2e-5 after R345 (R229 SIGNAL LongCtx×BigG×HiLR triple; ≠ R345 G=4 / ≠ R344@5e-6 / ≠ R338@6144).
    ("mine-r346-marsplan-online-dpo-longctx-bigg-hilr-1", "R346", "marsplan×Online-DPO×LongCtx×BigG×HiLR 16384/1024 G=8 lr=2e-5 β=0.1 α=32 r=16 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×BigG×HiLR triple; ≠ R345 LongCtx×HiLR G=4@2e-5 / ≠ R344 LongCtx×BigG G=8@5e-6 / ≠ R343 LongCtx G=4@5e-6 / ≠ R338 BigG×HiLR@6144 / ≠ R336–R342 / ≠ R334 @6144 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3021: R347 marsplan×Online-DPO×LongCtx×HiRank 16384/1024 r=64 after R346 (R229 SIGNAL LongCtx×HiRank compound; ≠ R346 BigG×HiLR / ≠ R343 r=16 / ≠ R339@6144).
    ("mine-r347-marsplan-online-dpo-longctx-hirank-1", "R347", "marsplan×Online-DPO×LongCtx×HiRank 16384/1024 r=64 β=0.1 α=32 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×HiRank compound; ≠ R346 LongCtx×BigG×HiLR G=8@2e-5 / ≠ R345 LongCtx×HiLR / ≠ R344 LongCtx×BigG / ≠ R343 LongCtx r=16 / ≠ R339 HiRank@6144 / ≠ R336–R342 / ≠ R334 @6144 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),

    # p3022: R348 marsplan×Online-DPO×LongCtx×HiRank×BigG 16384/1024 r=64 G=8 after R347 (R229 SIGNAL LongCtx×HiRank×BigG compound; ≠ R347 G=4 / ≠ R344 r=16 / ≠ R340@6144).
    ("mine-r348-marsplan-online-dpo-longctx-hirank-bigg-1", "R348", "marsplan×Online-DPO×LongCtx×HiRank×BigG 16384/1024 r=64 G=8 β=0.1 α=32 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×HiRank×BigG compound; ≠ R347 LongCtx×HiRank G=4@5e-6 / ≠ R346 LongCtx×BigG×HiLR / ≠ R345 LongCtx×HiLR / ≠ R344 LongCtx×BigG r=16 / ≠ R343 LongCtx / ≠ R340 HiRank×BigG@6144 / ≠ R339 HiRank@6144 / ≠ R336–R342 / ≠ R334 @6144 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),    # p2940: R228 lean-warm on mine-r165 lunar GPUs4–5 after R224 REFUTE+purge (isolated /root/r228; R212 keeps 6–7) — do not re-rent.
    # p3024: R349 marsplan×Online-DPO×LongCtx×HiRank×HiLR 16384/1024 r=64 G=4 lr=2e-5 after R348 (R229 SIGNAL LongCtx×HiRank×HiLR compound; ≠ R348 BigG / ≠ R347@5e-6 / ≠ R341@6144).
    ("mine-r349-marsplan-online-dpo-longctx-hirank-hilr-1", "R349", "marsplan×Online-DPO×LongCtx×HiRank×HiLR 16384/1024 r=64 G=4 lr=2e-5 β=0.1 α=32 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×HiRank×HiLR compound; ≠ R348 LongCtx×HiRank×BigG G=8@5e-6 / ≠ R347 LongCtx×HiRank G=4@5e-6 / ≠ R346 LongCtx×BigG×HiLR / ≠ R345 LongCtx×HiLR r=16 / ≠ R341 HiRank×HiLR@6144 / ≠ R343–R344 / ≠ R336–R342 / ≠ R334 @6144 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3025: R350 marsplan×Online-DPO×LongCtx×HiRank×BigG×HiLR 16384/1024 r=64 G=8 lr=2e-5 after R349 (R229 SIGNAL full 4-way; ≠ R349 G=4 / ≠ R348@5e-6 / ≠ R346 r=16 / ≠ R342@6144).
    ("mine-r350-marsplan-online-dpo-longctx-hirank-bigg-hilr-1", "R350", "marsplan×Online-DPO×LongCtx×HiRank×BigG×HiLR 16384/1024 r=64 G=8 lr=2e-5 β=0.1 α=32 temp=1.2 FORCE_PREFIX min_gap=0 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R229 SIGNAL LongCtx×HiRank×BigG×HiLR full 4-way; ≠ R349 LongCtx×HiRank×HiLR G=4@2e-5 / ≠ R348 LongCtx×HiRank×BigG G=8@5e-6 / ≠ R347 LongCtx×HiRank / ≠ R346 LongCtx×BigG×HiLR r=16 / ≠ R342 HiRank×BigG×HiLR@6144 / ≠ R334–R345 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3035: R351 marsplan×Online-DPO×Long max_steps=600 after R350 (R334 SIGNAL 2× train length; ≠ R334@300).
    ("mine-r351-marsplan-online-dpo-long-1", "R351", "marsplan×Online-DPO×Long max_steps=600 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R334 SIGNAL 2× steps; ≠ R334@300 / ≠ R336 G=8@300 / ≠ R337 HiLR / ≠ R338–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3036: R352 marsplan×Online-DPO×Long×BigG max_steps=600 G=8 after R351 (R334 SIGNAL length × BigG; ≠ R351 G=4@600 / ≠ R336 G=8@300).
    ("mine-r352-marsplan-online-dpo-long-bigg-1", "R352", "marsplan×Online-DPO×Long×BigG max_steps=600 G=8 β=0.1 α=32 r=16 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R351×R336 compound on R334 SIGNAL; ≠ R351 G=4@600 / ≠ R336 G=8@300 / ≠ R334 G=4@300 / ≠ R337–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3037: R353 marsplan×Online-DPO×Long×HiLR max_steps=600 lr=2e-5 after R352 (R351×R337 compound on R334 SIGNAL; ≠ R351@5e-6 / ≠ R337@300 / ≠ R352 BigG).
    ("mine-r353-marsplan-online-dpo-long-hilr-1", "R353", "marsplan×Online-DPO×Long×HiLR max_steps=600 lr=2e-5 β=0.1 α=32 r=16 G=4 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R351×R337 compound on R334 SIGNAL; ≠ R351 G=4@600@5e-6 / ≠ R337 G=4@300@2e-5 / ≠ R352 G=8@600@5e-6 / ≠ R336 G=8@300 / ≠ R334 G=4@300 / ≠ R338–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3038: R354 marsplan×Online-DPO×Long×BigG×HiLR max_steps=600 G=8 lr=2e-5 after R353 (R352×R353 / R351×R338 compound on R334 SIGNAL).
    ("mine-r354-marsplan-online-dpo-long-bigg-hilr-1", "R354", "marsplan×Online-DPO×Long×BigG×HiLR max_steps=600 G=8 lr=2e-5 β=0.1 α=32 r=16 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R352×R353 compound on R334 SIGNAL; ≠ R353 G=4@600@2e-5 / ≠ R352 G=8@600@5e-6 / ≠ R351 G=4@600@5e-6 / ≠ R338 G=8@300@2e-5 / ≠ R337 G=4@300@2e-5 / ≠ R336 G=8@300 / ≠ R334 G=4@300 / ≠ R339–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3039: R355 marsplan×Online-DPO×Long×HiRank max_steps=600 r=64 after R354 (R351×R339 compound on R334 SIGNAL).
    ("mine-r355-marsplan-online-dpo-long-hirank-1", "R355", "marsplan×Online-DPO×Long×HiRank max_steps=600 r=64 β=0.1 α=32 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R351×R339 compound on R334 SIGNAL; ≠ R351 r=16@600 / ≠ R339 r=64@300 / ≠ R352–R354 Long×BigG/HiLR / ≠ R340–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3040: R356 marsplan×Online-DPO×Long×HiRank×BigG max_steps=600 r=64 G=8 after R355 (R355×R352 / R340×R351 compound on R334 SIGNAL).
    ("mine-r356-marsplan-online-dpo-long-hirank-bigg-1", "R356", "marsplan×Online-DPO×Long×HiRank×BigG max_steps=600 r=64 G=8 β=0.1 α=32 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R355×R352 compound on R334 SIGNAL; ≠ R355 r=64@G=4@600 / ≠ R352 G=8@r=16@600 / ≠ R340 r=64@G=8@300 / ≠ R351–R354 / ≠ R339–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3041: R357 marsplan×Online-DPO×Long×HiRank×HiLR max_steps=600 r=64 lr=2e-5 after R356 (R355×R353 / R341×R351 compound on R334 SIGNAL).
    ("mine-r357-marsplan-online-dpo-long-hirank-hilr-1", "R357", "marsplan×Online-DPO×Long×HiRank×HiLR max_steps=600 r=64 lr=2e-5 β=0.1 α=32 G=4 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R355×R353 compound on R334 SIGNAL; ≠ R355 r=64@G=4@600@5e-6 / ≠ R353 r=16@G=4@600@2e-5 / ≠ R341 r=64@G=4@300@2e-5 / ≠ R356 r=64@G=8@600 / ≠ R351–R354 / ≠ R339–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3042: R358 marsplan×Online-DPO×Long×HiRank×BigG×HiLR max_steps=600 r=64 G=8 lr=2e-5 after R357 (R356×R357 / R342×R351 compound on R334 SIGNAL).
    ("mine-r358-marsplan-online-dpo-long-hirank-bigg-hilr-1", "R358", "marsplan×Online-DPO×Long×HiRank×BigG×HiLR max_steps=600 r=64 G=8 lr=2e-5 β=0.1 α=32 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R356×R357 compound on R334 SIGNAL; ≠ R357 r=64@G=4@600@2e-5 / ≠ R356 r=64@G=8@600@5e-6 / ≠ R355 r=64@G=4@600@5e-6 / ≠ R354 G=8@r=16@600@2e-5 / ≠ R342 r=64@G=8@300@2e-5 / ≠ R351–R353 / ≠ R339–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3043: R359 marsplan×Online-DPO×Long×LongCtx max_steps=600 @16384/1024 after R358 (R351×R343; Long@600 factorial complete → Long×LongCtx@600).
    ("mine-r359-marsplan-online-dpo-long-longctx-1", "R359", "marsplan×Online-DPO×Long×LongCtx max_steps=600 16384/1024 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R351×R343 compound on R334 SIGNAL; ≠ R351 @6144@600 / ≠ R343 @16384@300 / ≠ R334 @6144@300 / ≠ R344–R350 LongCtx@300 / ≠ R352–R358 Long@600 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3044: R360 marsplan×Online-DPO×Long×LongCtx×BigG max_steps=600 @16384/1024 G=8 after R359 (R359×R344 / R352 on R334 SIGNAL).
    ("mine-r360-marsplan-online-dpo-long-longctx-bigg-1", "R360", "marsplan×Online-DPO×Long×LongCtx×BigG max_steps=600 16384/1024 G=8 β=0.1 α=32 r=16 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R359×R344/R352 compound on R334 SIGNAL; ≠ R359 G=4@16384@600 / ≠ R344 G=8@16384@300 / ≠ R352 G=8@6144@600 / ≠ R351 G=4@6144@600 / ≠ R343 G=4@16384@300 / ≠ R334–R358 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3045: R361 marsplan×Online-DPO×Long×LongCtx×HiLR max_steps=600 @16384/1024 lr=2e-5 after R360 (R359×R345 / R353 on R334 SIGNAL).
    ("mine-r361-marsplan-online-dpo-long-longctx-hilr-1", "R361", "marsplan×Online-DPO×Long×LongCtx×HiLR max_steps=600 16384/1024 lr=2e-5 β=0.1 α=32 r=16 G=4 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R359×R345/R353 compound on R334 SIGNAL; ≠ R359 G=4@16384@600@5e-6 / ≠ R360 G=8@16384@600@5e-6 / ≠ R345 G=4@16384@300@2e-5 / ≠ R353 G=4@6144@600@2e-5 / ≠ R351–R358 / ≠ R343–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3046: R362 marsplan×Online-DPO×Long×LongCtx×BigG×HiLR max_steps=600 @16384/1024 G=8 lr=2e-5 after R361 (R360×R361 / R346 / R354 on R334 SIGNAL).
    ("mine-r362-marsplan-online-dpo-long-longctx-bigg-hilr-1", "R362", "marsplan×Online-DPO×Long×LongCtx×BigG×HiLR max_steps=600 16384/1024 G=8 lr=2e-5 β=0.1 α=32 r=16 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R360×R361 / R346 / R354 compound on R334 SIGNAL; ≠ R360 G=8@16384@600@5e-6 / ≠ R361 G=4@16384@600@2e-5 / ≠ R359 G=4@16384@600@5e-6 / ≠ R346 G=8@16384@300@2e-5 / ≠ R354 G=8@6144@600@2e-5 / ≠ R351–R358 / ≠ R343–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3048: R363 marsplan×Online-DPO×Long×LongCtx×HiRank max_steps=600 @16384/1024 r=64 after R362 (R359×R355 / R339 on R334 SIGNAL).
    ("mine-r363-marsplan-online-dpo-long-longctx-hirank-1", "R363", "marsplan×Online-DPO×Long×LongCtx×HiRank max_steps=600 16384/1024 r=64 β=0.1 α=32 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R359×R355 / R339 compound on R334 SIGNAL; ≠ R359 r=16@16384@600 / ≠ R355 r=64@6144@600 / ≠ R362 G=8@16384@600@2e-5 / ≠ R361 G=4@16384@600@2e-5 / ≠ R360 G=8@16384@600@5e-6 / ≠ R351–R358 / ≠ R343–R350 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3049: R364 marsplan×Online-DPO×Long×LongCtx×HiRank×BigG max_steps=600 @16384/1024 r=64 G=8 after R363 (R363×R360 / R348 / R356 on R334 SIGNAL).
    ("mine-r364-marsplan-online-dpo-long-longctx-hirank-bigg-1", "R364", "marsplan×Online-DPO×Long×LongCtx×HiRank×BigG max_steps=600 16384/1024 r=64 G=8 β=0.1 α=32 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R363×R360 / R348 / R356 compound on R334 SIGNAL; ≠ R363 G=4@16384@600@r64 / ≠ R360 G=8@16384@600@r16 / ≠ R362 G=8@16384@600@2e-5 / ≠ R348 G=8@16384@300@r64 / ≠ R359–R361 / ≠ R355–R358 / ≠ R351–R354 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3050: R365 marsplan×Online-DPO×Long×LongCtx×HiRank×BigG×HiLR max_steps=600 @16384/1024 r=64 G=8 lr=2e-5 after R364 (R364×R361 / R350 / R358 on R334 SIGNAL; completes Long×LongCtx×HiRank@600 factorial).
    ("mine-r365-marsplan-online-dpo-long-longctx-hirank-bigg-hilr-1", "R365", "marsplan×Online-DPO×Long×LongCtx×HiRank×BigG×HiLR max_steps=600 16384/1024 r=64 G=8 lr=2e-5 β=0.1 α=32 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R364×R361 / R350 / R358 compound on R334 SIGNAL; ≠ R364 G=8@16384@600@r64@5e-6 / ≠ R363 G=4@16384@600@r64 / ≠ R362 G=8@16384@600@r16@2e-5 / ≠ R361 G=4@16384@600@2e-5 / ≠ R360 / ≠ R358@6144 / ≠ R350@300 / ≠ R351–R359 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3051: R366 marsplan×Online-DPO×Long×LongCtx×HiRank×HiLR max_steps=600 @16384/1024 r=64 G=4 lr=2e-5 after R365 (R363×R361 / R349 / R357 on R334 SIGNAL; fills missing HiLR cell no-BigG).
    ("mine-r366-marsplan-online-dpo-long-longctx-hirank-hilr-1", "R366", "marsplan×Online-DPO×Long×LongCtx×HiRank×HiLR max_steps=600 16384/1024 r=64 G=4 lr=2e-5 β=0.1 α=32 temp=1.2 FORCE_PREFIX min_gap=0 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R363×R361 / R349 / R357 compound on R334 SIGNAL; ≠ R365 G=8@16384@600@r64@2e-5 / ≠ R364 G=8@16384@600@r64@5e-6 / ≠ R363 G=4@16384@600@r64@5e-6 / ≠ R357@6144 / ≠ R349@300 / ≠ R361 r=16 / ≠ R351–R362 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3052: R367 marsplan×Online-DPO×HiAlpha α=128 after R366 (R334 SIGNAL method isolate α; Long×LongCtx×HiRank@600 factorial filled).
    ("mine-r367-marsplan-online-dpo-hialpha-1", "R367", "marsplan×Online-DPO×HiAlpha α=128 β=0.1 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=300 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R334 SIGNAL method isolate α; ≠ R334 α=32@300 / ≠ R337 HiLR α=32 lr=2e-5 / ≠ R339 HiRank r=64 / ≠ R336 G=8 / ≠ R351–R366 Long×LongCtx×HiRank@600 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227 / ≠ R228 Offline REFUTE; n80 king=marsplan-queen)"),
    # p3053: R368 marsplan×Offline-DPO×Long max_steps=600 after R367 (FORCE_PREFIX-era retry; R228@200 REFUTED).
    ("mine-r368-marsplan-offline-dpo-long-1", "R368", "marsplan×Offline-DPO×Long β=0.1 α=32 r=16 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (FORCE_PREFIX-era Offline retry 3× R228 steps; ≠ R228@200 REFUTE / ≠ R334–R367 Online-DPO / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3054: R369 marsplan×Offline-DPO×HiAlpha α=128 max_steps=600 after R368 (α isolate on Offline Long).
    ("mine-r369-marsplan-offline-dpo-hialpha-1", "R369", "marsplan×Offline-DPO×HiAlpha β=0.1 α=128 r=16 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (α isolate vs R368 α=32@600; Offline counterpart to R367 Online-HiAlpha; ≠ R368 α=32 / ≠ R228@200 REFUTE / ≠ R367 Online-HiAlpha / ≠ R334–R366 Online / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3055: R370 marsplan×Online-DPO×HiAlpha×Long α=128 max_steps=600 after R369 (R367×R351 compound).
    ("mine-r370-marsplan-online-dpo-hialpha-long-1", "R370", "marsplan×Online-DPO×HiAlpha×Long α=128 β=0.1 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R367×R351 compound; ≠ R367 α=128@300 / ≠ R351 α=32@600 / ≠ R369 Offline-HiAlpha / ≠ R368 Offline-Long / ≠ R334–R366 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227 / ≠ R228 Offline REFUTE; n80 king=marsplan-queen)"),

    # p3056: R371 marsplan×Offline-DPO×HiAlpha×LongCtx α=128 @16384 max_steps=600 after R370 (R369×LongCtx compound).
    ("mine-r371-marsplan-offline-dpo-hialpha-longctx-1", "R371", "marsplan×Offline-DPO×HiAlpha×LongCtx β=0.1 α=128 r=16 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R369 Offline-HiAlpha × LongCtx isolate; ≠ R369@6144 / ≠ R368 Offline-Long α=32@6144@600 / ≠ R370 Online-HiAlpha×Long / ≠ R228@200 REFUTE / ≠ R367 Online-HiAlpha / ≠ R334–R366 Online / ≠ R343–R366 LongCtx Online / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),    # p2839: QUEUE#20 marsplan×Offline-DPO — method axis (R13 recipe on live-king init; ≠ R204–R227 GRPO/SFT/REINFORCE/FullFT / ≠ R13 Tok / ≠ R11 online DPO).
    # p3057: R372 marsplan×Online-DPO×HiAlpha×LongCtx α=128 @16384/1024 max_steps=600 after R371 (R370×LongCtx compound).
    ("mine-r372-marsplan-online-dpo-hialpha-longctx-1", "R372", "marsplan×Online-DPO×HiAlpha×LongCtx α=128 β=0.1 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R370×LongCtx compound; ≠ R370@6144 / ≠ R371 Offline-HiAlpha×LongCtx / ≠ R367 α=128@300 / ≠ R359 Long×LongCtx α=32 / ≠ R343 LongCtx@300 / ≠ R334–R366 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227 / ≠ R228 Offline REFUTE; n80 king=marsplan-queen)"),
    # p3058: R373 marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank α=128 r=64 @16384 max_steps=600 after R372 (R371×HiRank compound).
    ("mine-r373-marsplan-offline-dpo-hialpha-longctx-hirank-1", "R373", "marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank β=0.1 α=128 r=64 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R371 Offline-HiAlpha×LongCtx × HiRank isolate; ≠ R371@r16 / ≠ R369@6144 / ≠ R368 Offline-Long α=32 / ≠ R370 Online-HiAlpha×Long / ≠ R372 Online-HiAlpha×LongCtx / ≠ R355 Online-Long×HiRank / ≠ R228@200 REFUTE / ≠ R367 Online-HiAlpha / ≠ R334–R366 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3059: R374 marsplan×Online-DPO×HiAlpha×LongCtx×HiRank α=128 r=64 @16384/1024 max_steps=600 after R373 (R372×HiRank compound).
    ("mine-r374-marsplan-online-dpo-hialpha-longctx-hirank-1", "R374", "marsplan×Online-DPO×HiAlpha×LongCtx×HiRank α=128 β=0.1 r=64 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R372 Online-HiAlpha×LongCtx × HiRank isolate; ≠ R372@r16 / ≠ R373 Offline-HiAlpha×LongCtx×HiRank / ≠ R370 Online-HiAlpha×Long@6144 / ≠ R371 Offline-HiAlpha×LongCtx / ≠ R363 Online-Long×LongCtx×HiRank α=32 / ≠ R355 Online-Long×HiRank / ≠ R367 Online-HiAlpha@300 / ≠ R334–R366 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3060: R375 marsplan×Online-DPO×HiAlpha×LongCtx×BigG α=128 G=8 @16384/1024 max_steps=600 after R374 (R372×BigG isolate).
    ("mine-r375-marsplan-online-dpo-hialpha-longctx-bigg-1", "R375", "marsplan×Online-DPO×HiAlpha×LongCtx×BigG α=128 β=0.1 r=16 G=8 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R372 Online-HiAlpha×LongCtx × BigG isolate; ≠ R372@G=4 / ≠ R374 Online-HiAlpha×LongCtx×HiRank / ≠ R373 Offline-HiAlpha×LongCtx×HiRank / ≠ R370 Online-HiAlpha×Long@6144 / ≠ R360 Online-Long×LongCtx×BigG α=32 / ≠ R352 Online-Long×BigG / ≠ R336 Online-BigG@300 / ≠ R367 Online-HiAlpha@300 / ≠ R334–R366 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3061: R376 marsplan×Online-DPO×HiAlpha×LongCtx×HiRank×BigG α=128 r=64 G=8 @16384/1024 max_steps=600 after R375 (R374×R375 compound).
    ("mine-r376-marsplan-online-dpo-hialpha-longctx-hirank-bigg-1", "R376", "marsplan×Online-DPO×HiAlpha×LongCtx×HiRank×BigG α=128 β=0.1 r=64 G=8 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R374 Online-HiAlpha×LongCtx×HiRank × BigG isolate / R375 Online-HiAlpha×LongCtx×BigG × HiRank isolate; ≠ R375@r16 / ≠ R374@G=4 / ≠ R372@r16@G=4 / ≠ R373 Offline×HiRank / ≠ R370 Online-HiAlpha×Long@6144 / ≠ R360 Online-Long×LongCtx×BigG α=32 / ≠ R348 Online-LongCtx×HiRank×BigG α=32 / ≠ R336 Online-BigG@300 / ≠ R367 Online-HiAlpha@300 / ≠ R334–R366 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3062: R377 marsplan×Online-DPO×HiAlpha×LongCtx×HiRank×BigG×HiLR α=128 r=64 G=8 lr=2e-5 @16384/1024 max_steps=600 after R376 (R376×HiLR isolate).
    ("mine-r377-marsplan-online-dpo-hialpha-longctx-hirank-bigg-hilr-1", "R377", "marsplan×Online-DPO×HiAlpha×LongCtx×HiRank×BigG×HiLR α=128 β=0.1 r=64 G=8 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R376 Online-HiAlpha×LongCtx×HiRank×BigG × HiLR isolate; ≠ R376@5e-6 / ≠ R374@G=4@5e-6 / ≠ R375@r16@5e-6 / ≠ R372@r16@G=4 / ≠ R373 Offline×HiRank / ≠ R79 Tok HiAlpha×LongCtx×BigG×HiLR / ≠ R370 Online-HiAlpha×Long@6144 / ≠ R336 Online-BigG@300 / ≠ R367 Online-HiAlpha@300 / ≠ R334–R375 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3063: R378 marsplan×Online-DPO×HiAlpha×LongCtx×HiRank×HiLR α=128 r=64 G=4 lr=2e-5 @16384/1024 max_steps=600 after R377 (R374×HiLR isolate; no BigG).
    ("mine-r378-marsplan-online-dpo-hialpha-longctx-hirank-hilr-1", "R378", "marsplan×Online-DPO×HiAlpha×LongCtx×HiRank×HiLR α=128 β=0.1 r=64 G=4 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R374 Online-HiAlpha×LongCtx×HiRank × HiLR isolate; ≠ R374@5e-6 / ≠ R377 G=8@2e-5 / ≠ R376 G=8@5e-6 / ≠ R341 HiRank×HiLR@6144 / ≠ R349 LongCtx×HiRank×HiLR α=32 / ≠ R372@r16 / ≠ R373 Offline×HiRank / ≠ R334–R377 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3064: R379 marsplan×Online-DPO×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=8 lr=2e-5 @16384/1024 max_steps=600 after R378 (R375×HiLR isolate; no HiRank).
    ("mine-r379-marsplan-online-dpo-hialpha-longctx-bigg-hilr-1", "R379", "marsplan×Online-DPO×HiAlpha×LongCtx×BigG×HiLR α=128 β=0.1 r=16 G=8 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R375 Online-HiAlpha×LongCtx×BigG × HiLR isolate; ≠ R375@5e-6 / ≠ R377 HiRank×BigG×HiLR / ≠ R376 HiRank×BigG@5e-6 / ≠ R378 HiRank×HiLR G=4 / ≠ R372@G=4 / ≠ R334–R378 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3065: R380 marsplan×Offline-DPO×HiAlpha×LongCtx×HiLR α=128 r=16 lr=2e-5 @16384 max_steps=600 after R379 (R371×HiLR isolate; Offline has no G — STATE \"BigG\" → HiLR).
    ("mine-r380-marsplan-offline-dpo-hialpha-longctx-hilr-1", "R380", "marsplan×Offline-DPO×HiAlpha×LongCtx×HiLR β=0.1 α=128 r=16 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R371 Offline-HiAlpha×LongCtx × HiLR isolate; ≠ R371@5e-6 / ≠ R373 Offline×HiRank@5e-6 / ≠ R369 Offline-HiAlpha@6144 / ≠ R368 Offline-Long α=32 / ≠ R379 Online×BigG×HiLR / ≠ R372 Online-HiAlpha×LongCtx / ≠ R228@200 REFUTE / ≠ R334–R379 Online / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3066: R381 marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×HiLR α=128 r=64 lr=2e-5 @16384 max_steps=600 after R380 (R373×HiLR / R380×HiRank compound).
    ("mine-r381-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-1", "R381", "marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×HiLR β=0.1 α=128 r=64 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R373 Offline-HiAlpha×LongCtx×HiRank × HiLR isolate / R380 Offline-HiAlpha×LongCtx×HiLR × HiRank isolate; ≠ R380@r16 / ≠ R373@5e-6 / ≠ R378 Online×HiRank×HiLR / ≠ R371@r16@5e-6 / ≠ R369 Offline-HiAlpha@6144 / ≠ R368 Offline-Long α=32 / ≠ R379 Online×BigG×HiLR / ≠ R228@200 REFUTE / ≠ R334–R380 Online / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3067: R382 marsplan×Online-DPO×HiAlpha×LongCtx×HiLR α=128 r=16 G=4 lr=2e-5 @16384/1024 max_steps=600 after R381 (R372×HiLR isolate; no BigG/HiRank — completes Online HiAlpha×LongCtx factorial missing cell).
    ("mine-r382-marsplan-online-dpo-hialpha-longctx-hilr-1", "R382", "marsplan×Online-DPO×HiAlpha×LongCtx×HiLR α=128 β=0.1 r=16 G=4 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @16384/1024 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R372 Online-HiAlpha×LongCtx × HiLR isolate; ≠ R372@5e-6 / ≠ R379 BigG×HiLR G=8 / ≠ R378 HiRank×HiLR / ≠ R377 HiRank×BigG×HiLR / ≠ R375 BigG@5e-6 / ≠ R380 Offline×HiLR / ≠ R381 Offline×HiRank×HiLR / ≠ R334–R381 / ≠ R335 BoN-BigG / ≠ R228 Offline REFUTE / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3068: R383 marsplan×Offline-DPO×Long×HiRank α=32 r=64 lr=5e-6 @6144 max_steps=600 after R382 (R368×HiRank isolate; no HiAlpha/LongCtx).
    ("mine-r383-marsplan-offline-dpo-long-hirank-1", "R383", "marsplan×Offline-DPO×Long×HiRank β=0.1 α=32 r=64 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R368 Offline-Long × HiRank isolate; ≠ R368@r16 / ≠ R373 Offline-HiAlpha×LongCtx×HiRank@16384 / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR / ≠ R369 Offline-HiAlpha@6144 / ≠ R355 Online-Long×HiRank / ≠ R228@200 REFUTE / ≠ R334–R382 Online / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3069: R384 marsplan×Offline-DPO×HiAlpha×HiRank α=128 r=64 lr=5e-6 @6144 max_steps=600 after R383 (R369×HiRank isolate; no LongCtx).
    ("mine-r384-marsplan-offline-dpo-hialpha-hirank-1", "R384", "marsplan×Offline-DPO×HiAlpha×HiRank β=0.1 α=128 r=64 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R369 Offline-HiAlpha × HiRank isolate; ≠ R369@r16 / ≠ R383 Offline-Long×HiRank@α32 / ≠ R373 Offline-HiAlpha×LongCtx×HiRank@16384 / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR / ≠ R368 Offline-Long / ≠ R355 Online-Long×HiRank / ≠ R228@200 REFUTE / ≠ R334–R383 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3070: R385 marsplan×Offline-DPO×Long×HiRank×HiLR α=32 r=64 lr=2e-5 @6144 max_steps=600 after R384 (R383×HiLR isolate; no HiAlpha/LongCtx).
    ("mine-r385-marsplan-offline-dpo-long-hirank-hilr-1", "R385", "marsplan×Offline-DPO×Long×HiRank×HiLR β=0.1 α=32 r=64 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R383 Offline-Long×HiRank × HiLR isolate; ≠ R383@5e-6 / ≠ R384 Offline-HiAlpha×HiRank / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR / ≠ R357 Online-Long×HiRank×HiLR / ≠ R368 Offline-Long@r16 / ≠ R228@200 REFUTE / ≠ R334–R384 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3071: R386 marsplan×Offline-DPO×HiAlpha×HiRank×HiLR α=128 r=64 lr=2e-5 @6144 max_steps=600 after R385 (R384×HiLR isolate; no LongCtx).
    ("mine-r386-marsplan-offline-dpo-hialpha-hirank-hilr-1", "R386", "marsplan×Offline-DPO×HiAlpha×HiRank×HiLR β=0.1 α=128 r=64 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R384 Offline-HiAlpha×HiRank × HiLR isolate; ≠ R384@5e-6 / ≠ R385 Offline-Long×HiRank×HiLR@α32 / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR@16384 / ≠ R383 Offline-Long×HiRank / ≠ R369 Offline-HiAlpha@r16 / ≠ R228@200 REFUTE / ≠ R334–R385 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3072: R387 marsplan×Offline-DPO×Long×HiRank×LongCtx α=32 r=64 lr=5e-6 @16384 max_steps=600 after R386 (R383×LongCtx isolate).
    ("mine-r387-marsplan-offline-dpo-long-hirank-longctx-1", "R387", "marsplan×Offline-DPO×Long×HiRank×LongCtx β=0.1 α=32 r=64 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R383 Offline-Long×HiRank × LongCtx isolate; ≠ R383@6144 / ≠ R385 Offline-Long×HiRank×HiLR / ≠ R373 Offline-HiAlpha×LongCtx×HiRank@α128 / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR / ≠ R386 Offline-HiAlpha×HiRank×HiLR / ≠ R368 Offline-Long@r16 / ≠ R228@200 REFUTE / ≠ R334–R386 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3073: R388 marsplan×Offline-DPO×Long×HiRank×LongCtx×HiLR α=32 r=64 lr=2e-5 @16384 max_steps=600 after R387 (R387×HiLR isolate).
    ("mine-r388-marsplan-offline-dpo-long-hirank-longctx-hilr-1", "R388", "marsplan×Offline-DPO×Long×HiRank×LongCtx×HiLR β=0.1 α=32 r=64 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R387 Offline-Long×HiRank×LongCtx × HiLR isolate; ≠ R387@5e-6 / ≠ R385 Offline-Long×HiRank×HiLR@6144 / ≠ R383 Offline-Long×HiRank@6144@5e-6 / ≠ R373 Offline-HiAlpha×LongCtx×HiRank@α128 / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR / ≠ R386 Offline-HiAlpha×HiRank×HiLR / ≠ R368 Offline-Long@r16 / ≠ R228@200 REFUTE / ≠ R334–R387 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3074: R389 marsplan×Offline-DPO×Long×LongCtx α=32 r=16 lr=5e-6 @16384 max_steps=600 after R388 (R368×LongCtx isolate; no HiRank/HiLR).
    ("mine-r389-marsplan-offline-dpo-long-longctx-1", "R389", "marsplan×Offline-DPO×Long×LongCtx β=0.1 α=32 r=16 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R368 Offline-Long × LongCtx isolate; ≠ R387 Offline-Long×HiRank×LongCtx@r64 / ≠ R388 Offline-Long×HiRank×LongCtx×HiLR / ≠ R368@6144 / ≠ R385 Offline-Long×HiRank×HiLR / ≠ R383 Offline-Long×HiRank / ≠ R373 Offline-HiAlpha×LongCtx×HiRank@α128 / ≠ R381 Offline-HiAlpha×LongCtx×HiRank×HiLR / ≠ R228@200 REFUTE / ≠ R334–R388 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3075: R390 marsplan×Offline-DPO×Long×LongCtx×HiLR α=32 r=16 lr=2e-5 @16384 max_steps=600 after R389 (R389×HiLR isolate).
    ("mine-r390-marsplan-offline-dpo-long-longctx-hilr-1", "R390", "marsplan×Offline-DPO×Long×LongCtx×HiLR β=0.1 α=32 r=16 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R389 Offline-Long×LongCtx × HiLR isolate; ≠ R389@5e-6 / ≠ R388 Offline-Long×HiRank×LongCtx×HiLR@r64 / ≠ R387 Offline-Long×HiRank×LongCtx / ≠ R385 Offline-Long×HiRank×HiLR@6144 / ≠ R380 Offline-HiAlpha×LongCtx×HiLR@α128 / ≠ R368 Offline-Long@6144 / ≠ R228@200 REFUTE / ≠ R334–R389 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3076: R391 marsplan×Offline-DPO×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 max_steps=600 after R390 (R369×HiLR isolate; no HiRank/LongCtx).
    ("mine-r391-marsplan-offline-dpo-hialpha-hilr-1", "R391", "marsplan×Offline-DPO×HiAlpha×HiLR β=0.1 α=128 r=16 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R369 Offline-HiAlpha × HiLR isolate; ≠ R369@5e-6 / ≠ R386 Offline-HiAlpha×HiRank×HiLR@r64 / ≠ R380 Offline-HiAlpha×LongCtx×HiLR@16384 / ≠ R368 Offline-Long α=32 / ≠ R390 Offline-Long×LongCtx×HiLR / ≠ R228@200 REFUTE / ≠ R334–R390 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3077: R392 marsplan×Offline-DPO×Long×HiLR α=32 r=16 lr=2e-5 @6144 max_steps=600 after R391 (R368×HiLR isolate; no HiRank/LongCtx/HiAlpha).
    ("mine-r392-marsplan-offline-dpo-long-hilr-1", "R392", "marsplan×Offline-DPO×Long×HiLR β=0.1 α=32 r=16 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R368 Offline-Long × HiLR isolate; ≠ R368@5e-6 / ≠ R385 Offline-Long×HiRank×HiLR@r64 / ≠ R390 Offline-Long×LongCtx×HiLR@16384 / ≠ R391 Offline-HiAlpha×HiLR@α128 / ≠ R383 Offline-Long×HiRank / ≠ R228@200 REFUTE / ≠ R334–R391 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3078: R393 marsplan×Offline-DPO×Long×HiBeta α=32 r=16 β=0.5 lr=5e-6 @6144 max_steps=600 after R392 (R368×HiBeta isolate; no HiLR/HiRank/LongCtx/HiAlpha).
    ("mine-r393-marsplan-offline-dpo-long-hibeta-1", "R393", "marsplan×Offline-DPO×Long×HiBeta β=0.5 α=32 r=16 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R368 Offline-Long × HiBeta isolate; ≠ R368@β0.1 / ≠ R392 Long×HiLR@2e-5 / ≠ R385 Offline-Long×HiRank×HiLR@r64 / ≠ R391 Offline-HiAlpha×HiLR@α128 / ≠ R383 Offline-Long×HiRank / ≠ R228@200 REFUTE / ≠ R334–R392 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3079: R394 marsplan×Offline-DPO×Long×LoBeta α=32 r=16 β=0.02 lr=5e-6 @6144 max_steps=600 after R393 (R368×LoBeta isolate; no HiLR/HiRank/LongCtx/HiAlpha).
    ("mine-r394-marsplan-offline-dpo-long-lobeta-1", "R394", "marsplan×Offline-DPO×Long×LoBeta β=0.02 α=32 r=16 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R368 Offline-Long × LoBeta isolate; ≠ R368@β0.1 / ≠ R393@β0.5 / ≠ R392 Long×HiLR@2e-5 / ≠ R385 Offline-Long×HiRank×HiLR@r64 / ≠ R391 Offline-HiAlpha×HiLR@α128 / ≠ R383 Offline-Long×HiRank / ≠ R228@200 REFUTE / ≠ R334–R393 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3080: R395 marsplan×Offline-DPO×Long×ExtraLong α=32 r=16 β=0.1 lr=5e-6 @6144 max_steps=900 after R394 (R368×ExtraLong isolate; no HiLR/HiRank/LongCtx/HiAlpha/LoBeta/HiBeta).
    ("mine-r395-marsplan-offline-dpo-long-extralong-1", "R395", "marsplan×Offline-DPO×Long×ExtraLong β=0.1 α=32 r=16 lr=5e-6 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R368 Offline-Long × ExtraLong steps isolate; ≠ R368@600 / ≠ R394 LoBeta@600 / ≠ R393 HiBeta@600 / ≠ R392 Long×HiLR@2e-5 / ≠ R385 Offline-Long×HiRank×HiLR@r64 / ≠ R391 Offline-HiAlpha×HiLR@α128 / ≠ R383 Offline-Long×HiRank / ≠ R228@200 REFUTE / ≠ R334–R394 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),

    # p3081: R396 marsplan×Offline-DPO×Long×LoBeta×HiLR α=32 r=16 β=0.02 lr=2e-5 @6144 max_steps=600 after R395 (R394×HiLR compound; no HiRank/LongCtx/HiAlpha/ExtraLong).
    ("mine-r396-marsplan-offline-dpo-long-lobeta-hilr-1", "R396", "marsplan×Offline-DPO×Long×LoBeta×HiLR β=0.02 α=32 r=16 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R394 Offline-Long×LoBeta × HiLR compound; ≠ R394@5e-6 / ≠ R392@β0.1@2e-5 / ≠ R395 ExtraLong@900 / ≠ R393 HiBeta@0.5 / ≠ R368@β0.1@5e-6 / ≠ R385 Offline-Long×HiRank×HiLR@r64 / ≠ R391 Offline-HiAlpha×HiLR@α128 / ≠ R228@200 REFUTE / ≠ R334–R395 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3082: R397 marsplan×Offline-DPO×Long×ExtraLong×HiLR α=32 r=16 β=0.1 lr=2e-5 @6144 max_steps=900 after R396 (R395×HiLR compound; no HiRank/LongCtx/HiAlpha/LoBeta/HiBeta).
    ("mine-r397-marsplan-offline-dpo-long-extralong-hilr-1", "R397", "marsplan×Offline-DPO×Long×ExtraLong×HiLR β=0.1 α=32 r=16 lr=2e-5 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R395 Offline-Long×ExtraLong × HiLR compound; ≠ R395@5e-6 / ≠ R392@600@2e-5 / ≠ R396 LoBeta×HiLR@600 / ≠ R394 LoBeta / ≠ R393 HiBeta / ≠ R368@600@5e-6 / ≠ R385 Offline-Long×HiRank×HiLR@r64 / ≠ R391 Offline-HiAlpha×HiLR@α128 / ≠ R228@200 REFUTE / ≠ R334–R396 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3083: R398 marsplan×Offline-DPO×Long×LoBeta×HiRank α=32 r=64 β=0.02 lr=5e-6 @6144 max_steps=600 after R397 (R394×HiRank compound; no HiLR/LongCtx/HiAlpha/ExtraLong/HiBeta).
    ("mine-r398-marsplan-offline-dpo-long-lobeta-hirank-1", "R398", "marsplan×Offline-DPO×Long×LoBeta×HiRank β=0.02 α=32 r=64 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R394 Offline-Long×LoBeta × HiRank compound; ≠ R394@r16 / ≠ R383@β0.1 / ≠ R396 LoBeta×HiLR@r16@2e-5 / ≠ R385 Long×HiRank×HiLR@2e-5 / ≠ R397 ExtraLong×HiLR / ≠ R395 ExtraLong / ≠ R393 HiBeta / ≠ R368@r16@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R397 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3084: R399 marsplan×Offline-DPO×Long×HiBeta×HiLR α=32 r=16 β=0.5 lr=2e-5 @6144 max_steps=600 after R398 (R393×HiLR compound; no HiRank/LongCtx/HiAlpha/ExtraLong/LoBeta).
    ("mine-r399-marsplan-offline-dpo-long-hibeta-hilr-1", "R399", "marsplan×Offline-DPO×Long×HiBeta×HiLR β=0.5 α=32 r=16 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R393 Offline-Long×HiBeta × HiLR compound; ≠ R393@5e-6 / ≠ R396 LoBeta×HiLR@β0.02 / ≠ R392@β0.1@2e-5 / ≠ R394 LoBeta / ≠ R397 ExtraLong×HiLR / ≠ R398 LoBeta×HiRank / ≠ R368@β0.1@5e-6 / ≠ R385 Long×HiRank×HiLR / ≠ R391 HiAlpha×HiLR / ≠ R228@200 REFUTE / ≠ R334–R398 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3085: R400 marsplan×Offline-DPO×Long×LoBeta×ExtraLong α=32 r=16 β=0.02 lr=5e-6 @6144 max_steps=900 after R399 (R394×ExtraLong compound; no HiLR/HiRank/LongCtx/HiAlpha/HiBeta).
    ("mine-r400-marsplan-offline-dpo-long-lobeta-extralong-1", "R400", "marsplan×Offline-DPO×Long×LoBeta×ExtraLong β=0.02 α=32 r=16 lr=5e-6 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R394 Offline-Long×LoBeta × ExtraLong compound; ≠ R394@600 / ≠ R395@β0.1@900 / ≠ R397 ExtraLong×HiLR / ≠ R396 LoBeta×HiLR / ≠ R398 LoBeta×HiRank / ≠ R399 HiBeta×HiLR / ≠ R368@β0.1@600 / ≠ R228@200 REFUTE / ≠ R334–R399 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3086: R401 marsplan×Offline-DPO×Long×HiBeta×HiRank α=32 r=64 β=0.5 lr=5e-6 @6144 max_steps=600 after R400 (R393×HiRank compound; no HiLR/ExtraLong/LongCtx/HiAlpha/LoBeta).
    ("mine-r401-marsplan-offline-dpo-long-hibeta-hirank-1", "R401", "marsplan×Offline-DPO×Long×HiBeta×HiRank β=0.5 α=32 r=64 lr=5e-6 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R393 Offline-Long×HiBeta × HiRank compound; ≠ R393@r16 / ≠ R399 HiBeta×HiLR@r16@2e-5 / ≠ R398 LoBeta×HiRank@β0.02 / ≠ R383 Long×HiRank@β0.1 / ≠ R400 LoBeta×ExtraLong / ≠ R394 LoBeta / ≠ R368@β0.1@r16 / ≠ R228@200 REFUTE / ≠ R334–R400 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3087: R402 marsplan×Offline-DPO×Long×HiBeta×ExtraLong α=32 r=16 β=0.5 lr=5e-6 @6144 max_steps=900 after R401 (R393×ExtraLong compound; no HiLR/HiRank/LongCtx/HiAlpha/LoBeta).
    ("mine-r402-marsplan-offline-dpo-long-hibeta-extralong-1", "R402", "marsplan×Offline-DPO×Long×HiBeta×ExtraLong β=0.5 α=32 r=16 lr=5e-6 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R393 Offline-Long×HiBeta × ExtraLong compound; ≠ R393@600 / ≠ R401 HiBeta×HiRank@r64@600 / ≠ R399 HiBeta×HiLR@r16@2e-5 / ≠ R400 LoBeta×ExtraLong@β0.02@900 / ≠ R395 ExtraLong@β0.1@900 / ≠ R397 ExtraLong×HiLR / ≠ R394 LoBeta / ≠ R368@β0.1@600 / ≠ R228@200 REFUTE / ≠ R334–R401 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3088: R403 marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR α=32 r=64 β=0.5 lr=2e-5 @6144 max_steps=600 after R402 (R401×HiLR compound; no ExtraLong/LongCtx/HiAlpha/LoBeta).
    ("mine-r403-marsplan-offline-dpo-long-hibeta-hirank-hilr-1", "R403", "marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR β=0.5 α=32 r=64 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R401 Offline-Long×HiBeta×HiRank × HiLR compound; ≠ R401@5e-6 / ≠ R399 HiBeta×HiLR@r16 / ≠ R398 LoBeta×HiRank / ≠ R385 Long×HiRank×HiLR@β0.1 / ≠ R393 HiBeta@r16 / ≠ R383 Long×HiRank@β0.1 / ≠ R402 HiBeta×ExtraLong / ≠ R400 LoBeta×ExtraLong / ≠ R368@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R402 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3089: R404 marsplan×Offline-DPO×Long×HiBeta×HiLR×ExtraLong α=32 r=16 β=0.5 lr=2e-5 @6144 max_steps=900 after R403 (R399×ExtraLong compound; no HiRank/LongCtx/HiAlpha/LoBeta).
    ("mine-r404-marsplan-offline-dpo-long-hibeta-hilr-extralong-1", "R404", "marsplan×Offline-DPO×Long×HiBeta×HiLR×ExtraLong β=0.5 α=32 r=16 lr=2e-5 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R399 Offline-Long×HiBeta×HiLR × ExtraLong compound; ≠ R399@600 / ≠ R402 HiBeta×ExtraLong@5e-6@900 / ≠ R403 HiBeta×HiRank×HiLR@r64@600 / ≠ R397 ExtraLong×HiLR@β0.1 / ≠ R393 HiBeta@5e-6 / ≠ R400 LoBeta×ExtraLong / ≠ R395 ExtraLong@β0.1 / ≠ R368@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R403 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3090: R405 marsplan×Offline-DPO×Long×HiBeta×HiRank×ExtraLong α=32 r=64 β=0.5 lr=5e-6 @6144 max_steps=900 after R404 (R401×ExtraLong compound; no HiLR/LongCtx/HiAlpha/LoBeta).
    ("mine-r405-marsplan-offline-dpo-long-hibeta-hirank-extralong-1", "R405", "marsplan×Offline-DPO×Long×HiBeta×HiRank×ExtraLong β=0.5 α=32 r=64 lr=5e-6 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R401 Offline-Long×HiBeta×HiRank × ExtraLong compound; ≠ R401@600 / ≠ R403 HiBeta×HiRank×HiLR@2e-5@600 / ≠ R404 HiBeta×HiLR×ExtraLong@r16@2e-5@900 / ≠ R402 HiBeta×ExtraLong@r16@5e-6@900 / ≠ R399 HiBeta×HiLR@r16 / ≠ R398 LoBeta×HiRank / ≠ R383 Long×HiRank@β0.1 / ≠ R400 LoBeta×ExtraLong / ≠ R393 HiBeta@r16 / ≠ R368@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R404 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3091: R406 marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×ExtraLong α=32 r=64 β=0.5 lr=2e-5 @6144 max_steps=900 after R405 (R403×ExtraLong / R405×HiLR compound; no LongCtx/HiAlpha/LoBeta).
    ("mine-r406-marsplan-offline-dpo-long-hibeta-hirank-hilr-extralong-1", "R406", "marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×ExtraLong β=0.5 α=32 r=64 lr=2e-5 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R403 Offline-Long×HiBeta×HiRank×HiLR × ExtraLong compound; ≠ R403@600 / ≠ R405 HiBeta×HiRank×ExtraLong@5e-6@900 / ≠ R404 HiBeta×HiLR×ExtraLong@r16@2e-5@900 / ≠ R401 HiBeta×HiRank@5e-6@600 / ≠ R402 HiBeta×ExtraLong@r16 / ≠ R399 HiBeta×HiLR@r16 / ≠ R398 LoBeta×HiRank / ≠ R385 Long×HiRank×HiLR@β0.1 / ≠ R400 LoBeta×ExtraLong / ≠ R393 HiBeta@r16 / ≠ R368@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R405 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3092: R407 marsplan×Offline-DPO×Long×LoBeta×HiRank×ExtraLong α=32 r=64 β=0.02 lr=5e-6 @6144 max_steps=900 after R406 (R398×ExtraLong / R400×HiRank compound; no HiLR/LongCtx/HiAlpha/HiBeta).
    ("mine-r407-marsplan-offline-dpo-long-lobeta-hirank-extralong-1", "R407", "marsplan×Offline-DPO×Long×LoBeta×HiRank×ExtraLong β=0.02 α=32 r=64 lr=5e-6 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R398 Offline-Long×LoBeta×HiRank × ExtraLong compound; ≠ R398@600 / ≠ R400 LoBeta×ExtraLong@r16@900 / ≠ R405 HiBeta×HiRank×ExtraLong@β0.5@900 / ≠ R406 HiBeta×HiRank×HiLR×ExtraLong@2e-5@900 / ≠ R401 HiBeta×HiRank@β0.5@600 / ≠ R403 HiBeta×HiRank×HiLR / ≠ R394 LoBeta@r16 / ≠ R383 Long×HiRank@β0.1 / ≠ R368@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R406 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3093: R408 marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR α=32 r=64 β=0.02 lr=2e-5 @6144 max_steps=600 after R407 (R398×HiLR compound; no ExtraLong/LongCtx/HiAlpha/HiBeta).
    ("mine-r408-marsplan-offline-dpo-long-lobeta-hirank-hilr-1", "R408", "marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR β=0.02 α=32 r=64 lr=2e-5 @6144 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R398 Offline-Long×LoBeta×HiRank × HiLR compound; ≠ R398@5e-6@600 / ≠ R407 LoBeta×HiRank×ExtraLong@900 / ≠ R396 LoBeta×HiLR@r16 / ≠ R403 HiBeta×HiRank×HiLR@β0.5 / ≠ R406 HiBeta×HiRank×HiLR×ExtraLong / ≠ R385 Long×HiRank×HiLR@β0.1 / ≠ R400 LoBeta×ExtraLong / ≠ R394 LoBeta@r16 / ≠ R383 Long×HiRank@β0.1 / ≠ R368@β0.1 / ≠ R228@200 REFUTE / ≠ R334–R407 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3094: R409 marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×ExtraLong α=32 r=64 β=0.02 lr=2e-5 @6144 max_steps=900 after R408 (R408×ExtraLong / R407×HiLR compound; no LongCtx/HiAlpha/HiBeta).
    ("mine-r409-marsplan-offline-dpo-long-lobeta-hirank-hilr-extralong-1", "R409", "marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×ExtraLong β=0.02 α=32 r=64 lr=2e-5 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R408 Offline-Long×LoBeta×HiRank×HiLR × ExtraLong compound; ≠ R408@600 / ≠ R407 LoBeta×HiRank×ExtraLong@5e-6@900 / ≠ R406 HiBeta×HiRank×HiLR×ExtraLong@β0.5 / ≠ R403 HiBeta×HiRank×HiLR@β0.5@600 / ≠ R396 LoBeta×HiLR@r16 / ≠ R398 LoBeta×HiRank@5e-6@600 / ≠ R400 LoBeta×ExtraLong@r16 / ≠ R385 Long×HiRank×HiLR@β0.1 / ≠ R334–R408 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3095: R410 marsplan×Offline-DPO×Long×LoBeta×HiLR×ExtraLong α=32 r=16 β=0.02 lr=2e-5 @6144 max_steps=900 after R409 (R396×ExtraLong / R400×HiLR compound; no HiRank/LongCtx/HiAlpha/HiBeta).
    ("mine-r410-marsplan-offline-dpo-long-lobeta-hilr-extralong-1", "R410", "marsplan×Offline-DPO×Long×LoBeta×HiLR×ExtraLong β=0.02 α=32 r=16 lr=2e-5 @6144 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R396 Offline-Long×LoBeta×HiLR × ExtraLong / R400 Offline-Long×LoBeta×ExtraLong × HiLR compound; ≠ R409@r64 / ≠ R396@600 / ≠ R397 ExtraLong×HiLR@β0.1 / ≠ R400@5e-6 / ≠ R408@r64@600 / ≠ R404 HiBeta×HiLR×ExtraLong@β0.5 / ≠ R334–R409 / ≠ R335 BoN-BigG / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),

    # p3096: R411 marsplan×Offline-DPO×Long×LoBeta×HiRank×LongCtx α=32 r=64 β=0.02 lr=5e-6 @16384 max_steps=600 after R410 (R398×LongCtx / R387×LoBeta compound; no HiLR/ExtraLong/HiAlpha/HiBeta).
    ("mine-r411-marsplan-offline-dpo-long-lobeta-hirank-longctx-1", "R411", "marsplan×Offline-DPO×Long×LoBeta×HiRank×LongCtx β=0.02 α=32 r=64 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R398 Offline-Long×LoBeta×HiRank × LongCtx / R387 Offline-Long×HiRank×LongCtx × LoBeta compound; ≠ R387@β0.1 / ≠ R398@6144 / ≠ R407 ExtraLong / ≠ R408 HiLR / ≠ R409 HiLR×ExtraLong / ≠ R410 LoBeta×HiLR×ExtraLong@r16 / ≠ R389@r16 / ≠ R388×HiLR / ≠ R334–R410 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),

    # p3097: R412 marsplan×Offline-DPO×Long×LoBeta×HiLR×LongCtx α=32 r=16 β=0.02 lr=2e-5 @16384 max_steps=600 after R411 (R396×LongCtx / R390×LoBeta compound; no HiRank/ExtraLong/HiAlpha/HiBeta).
    ("mine-r412-marsplan-offline-dpo-long-lobeta-hilr-longctx-1", "R412", "marsplan×Offline-DPO×Long×LoBeta×HiLR×LongCtx β=0.02 α=32 r=16 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R396 Offline-Long×LoBeta×HiLR × LongCtx / R390 Offline-Long×LongCtx×HiLR × LoBeta compound; ≠ R390@β0.1 / ≠ R396@6144 / ≠ R410 ExtraLong@900 / ≠ R411 HiRank×LongCtx@r64 / ≠ R388×HiRank×HiLR / ≠ R389@5e-6 / ≠ R334–R411 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),

    # p3098: R413 marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×LongCtx α=32 r=64 β=0.02 lr=2e-5 @16384 max_steps=600 after R412 (R412×HiRank / R411×HiLR / R408×LongCtx compound; no ExtraLong/HiAlpha/HiBeta).
    ("mine-r413-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-1", "R413", "marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×LongCtx β=0.02 α=32 r=64 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R412 Offline-Long×LoBeta×HiLR×LongCtx × HiRank / R411 Offline-Long×LoBeta×HiRank×LongCtx × HiLR / R408 Offline-Long×LoBeta×HiRank×HiLR × LongCtx compound; ≠ R412@r16 / ≠ R411@5e-6 / ≠ R408@6144 / ≠ R409 ExtraLong@900 / ≠ R388@β0.1 / ≠ R390@r16 / ≠ R334–R412 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3099: R414 marsplan×Offline-DPO×Long×HiBeta×LongCtx α=32 r=16 β=0.5 lr=5e-6 @16384 max_steps=600 after R413 (R393×LongCtx compound; no HiRank/HiLR/ExtraLong/HiAlpha/LoBeta).
    ("mine-r414-marsplan-offline-dpo-long-hibeta-longctx-1", "R414", "marsplan×Offline-DPO×Long×HiBeta×LongCtx β=0.5 α=32 r=16 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R393 Offline-Long×HiBeta × LongCtx compound; ≠ R393@6144 / ≠ R399 HiBeta×HiLR@6144 / ≠ R401 HiBeta×HiRank@6144 / ≠ R402–R406 HiBeta×ExtraLong@900 / ≠ R411–R413 LoBeta×LongCtx / ≠ R408@6144 / ≠ R334–R413 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3100: R415 marsplan×Offline-DPO×Long×HiBeta×HiLR×LongCtx α=32 r=16 β=0.5 lr=2e-5 @16384 max_steps=600 after R414 (R414×HiLR / R399×LongCtx compound; no HiRank/ExtraLong/HiAlpha/LoBeta).
    ("mine-r415-marsplan-offline-dpo-long-hibeta-hilr-longctx-1", "R415", "marsplan×Offline-DPO×Long×HiBeta×HiLR×LongCtx β=0.5 α=32 r=16 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R414 Offline-Long×HiBeta×LongCtx × HiLR / R399 Offline-Long×HiBeta×HiLR × LongCtx compound; ≠ R414@5e-6 / ≠ R399@6144 / ≠ R404 HiBeta×HiLR×ExtraLong@900 / ≠ R412 LoBeta×HiLR×LongCtx / ≠ R393@6144 / ≠ R401 HiBeta×HiRank / ≠ R402–R406 ExtraLong / ≠ R411–R413 LoBeta×LongCtx / ≠ R334–R414 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3101: R416 marsplan×Offline-DPO×Long×HiBeta×HiRank×LongCtx α=32 r=64 β=0.5 lr=5e-6 @16384 max_steps=600 after R415 (R414×HiRank / R401×LongCtx compound; no HiLR/ExtraLong/HiAlpha/LoBeta).
    ("mine-r416-marsplan-offline-dpo-long-hibeta-hirank-longctx-1", "R416", "marsplan×Offline-DPO×Long×HiBeta×HiRank×LongCtx β=0.5 α=32 r=64 lr=5e-6 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R414 Offline-Long×HiBeta×LongCtx × HiRank / R401 Offline-Long×HiBeta×HiRank × LongCtx compound; ≠ R415 HiBeta×HiLR×LongCtx@r16@2e-5 / ≠ R414@r16 / ≠ R401@6144 / ≠ R411 LoBeta×HiRank×LongCtx / ≠ R413 LoBeta×HiRank×HiLR×LongCtx / ≠ R403 HiBeta×HiRank×HiLR@6144 / ≠ R405–R406 ExtraLong / ≠ R393@6144 / ≠ R334–R415 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3102: R417 marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×LongCtx α=32 r=64 β=0.5 lr=2e-5 @16384 max_steps=600 after R416 (R416×HiLR / R403×LongCtx / R415×HiRank compound; no ExtraLong/HiAlpha/LoBeta).
    ("mine-r417-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-1", "R417", "marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×LongCtx β=0.5 α=32 r=64 lr=2e-5 @16384 max_steps=600 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R416 Offline-Long×HiBeta×HiRank×LongCtx × HiLR / R403 Offline-Long×HiBeta×HiRank×HiLR × LongCtx / R415 Offline-Long×HiBeta×HiLR×LongCtx × HiRank compound; ≠ R416@5e-6 / ≠ R415@r16 / ≠ R403@6144 / ≠ R413 LoBeta×HiRank×HiLR×LongCtx / ≠ R414@r16 / ≠ R401@6144 / ≠ R406 ExtraLong / ≠ R334–R416 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3103: R418 marsplan×Offline-DPO×Long×HiBeta×LongCtx×ExtraLong α=32 r=16 β=0.5 lr=5e-6 @16384 max_steps=900 after R417 (R414×ExtraLong / R402×LongCtx compound; no HiRank/HiLR/HiAlpha/LoBeta).
    ("mine-r418-marsplan-offline-dpo-long-hibeta-longctx-extralong-1", "R418", "marsplan×Offline-DPO×Long×HiBeta×LongCtx×ExtraLong β=0.5 α=32 r=16 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R414 Offline-Long×HiBeta×LongCtx × ExtraLong / R402 Offline-Long×HiBeta×ExtraLong × LongCtx compound; ≠ R414@600 / ≠ R402@6144@900 / ≠ R417 HiBeta×HiRank×HiLR×LongCtx@r64@2e-5@600 / ≠ R415–R416 LongCtx@600 / ≠ R405–R406 ExtraLong@6144 / ≠ R334–R417 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3104: R419 marsplan×Offline-DPO×Long×HiBeta×HiLR×LongCtx×ExtraLong α=32 r=16 β=0.5 lr=2e-5 @16384 max_steps=900 after R418 (R415×ExtraLong / R404×LongCtx / R418×HiLR compound; no HiRank/HiAlpha/LoBeta).
    ("mine-r419-marsplan-offline-dpo-long-hibeta-hilr-longctx-extralong-1", "R419", "marsplan×Offline-DPO×Long×HiBeta×HiLR×LongCtx×ExtraLong β=0.5 α=32 r=16 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R415 Offline-Long×HiBeta×HiLR×LongCtx × ExtraLong / R404 Offline-Long×HiBeta×HiLR×ExtraLong × LongCtx / R418 Offline-Long×HiBeta×LongCtx×ExtraLong × HiLR compound; ≠ R418@5e-6@900 / ≠ R415@600 / ≠ R404@6144@900 / ≠ R417@r64@600 / ≠ R416 HiBeta×HiRank×LongCtx / ≠ R405–R406 ExtraLong@6144 / ≠ R334–R418 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3105: R420 marsplan×Offline-DPO×Long×HiBeta×HiRank×LongCtx×ExtraLong α=32 r=64 β=0.5 lr=5e-6 @16384 max_steps=900 after R419 (R416×ExtraLong / R405×LongCtx / R418×HiRank compound; no HiLR/HiAlpha/LoBeta).
    ("mine-r420-marsplan-offline-dpo-long-hibeta-hirank-longctx-extralong-1", "R420", "marsplan×Offline-DPO×Long×HiBeta×HiRank×LongCtx×ExtraLong β=0.5 α=32 r=64 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R416 Offline-Long×HiBeta×HiRank×LongCtx × ExtraLong / R405 Offline-Long×HiBeta×HiRank×ExtraLong × LongCtx / R418 Offline-Long×HiBeta×LongCtx×ExtraLong × HiRank compound; ≠ R419@r16@2e-5@900 / ≠ R418@r16 / ≠ R416@600 / ≠ R417@r64@2e-5@600 / ≠ R405@6144@900 / ≠ R406 ExtraLong×HiLR@6144 / ≠ R334–R419 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3106: R421 marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×LongCtx×ExtraLong α=32 r=64 β=0.5 lr=2e-5 @16384 max_steps=900 after R420 (R417×ExtraLong / R406×LongCtx / R420×HiLR compound; no HiAlpha/LoBeta).
    ("mine-r421-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-extralong-1", "R421", "marsplan×Offline-DPO×Long×HiBeta×HiRank×HiLR×LongCtx×ExtraLong β=0.5 α=32 r=64 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R417 Offline-Long×HiBeta×HiRank×HiLR×LongCtx × ExtraLong / R406 Offline-Long×HiBeta×HiRank×HiLR×ExtraLong × LongCtx / R420 Offline-Long×HiBeta×HiRank×LongCtx×ExtraLong × HiLR compound; ≠ R420@5e-6@900 / ≠ R419@r16@2e-5@900 / ≠ R417@600 / ≠ R406@6144@900 / ≠ R418@r16 / ≠ R416@600 / ≠ R334–R420 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3107: R422 marsplan×Offline-DPO×Long×LoBeta×LongCtx×ExtraLong α=32 r=16 β=0.02 lr=5e-6 @16384 max_steps=900 after R421 (R418×LoBeta / R400×LongCtx / R414×LoBeta×ExtraLong compound; no HiRank/HiLR/HiAlpha/HiBeta).
    ("mine-r422-marsplan-offline-dpo-long-lobeta-longctx-extralong-1", "R422", "marsplan×Offline-DPO×Long×LoBeta×LongCtx×ExtraLong β=0.02 α=32 r=16 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R418 Offline-Long×HiBeta×LongCtx×ExtraLong × LoBeta / R400 Offline-Long×LoBeta×ExtraLong × LongCtx / R414 Offline-Long×HiBeta×LongCtx × LoBeta×ExtraLong compound; ≠ R418@β0.5@900 / ≠ R414@600 / ≠ R400@6144@900 / ≠ R411–R413 LoBeta×LongCtx / ≠ R410 LoBeta×HiLR×ExtraLong / ≠ R421 HiBeta×HiRank×HiLR compound / ≠ R334–R421 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3108: R423 marsplan×Offline-DPO×Long×LoBeta×HiRank×LongCtx×ExtraLong α=32 r=64 β=0.02 lr=5e-6 @16384 max_steps=900 after R422 (R411×ExtraLong / R422×HiRank / R407×LongCtx compound; no HiLR/HiAlpha/HiBeta).
    ("mine-r423-marsplan-offline-dpo-long-lobeta-hirank-longctx-extralong-1", "R423", "marsplan×Offline-DPO×Long×LoBeta×HiRank×LongCtx×ExtraLong β=0.02 α=32 r=64 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R411 Offline-Long×LoBeta×HiRank×LongCtx × ExtraLong / R422 Offline-Long×LoBeta×LongCtx×ExtraLong × HiRank / R407 Offline-Long×LoBeta×HiRank×ExtraLong × LongCtx compound; ≠ R422@r16@900 / ≠ R411@600 / ≠ R407@6144@900 / ≠ R413 LoBeta×HiRank×HiLR×LongCtx / ≠ R420 HiBeta×HiRank×LongCtx×ExtraLong / ≠ R421 HiBeta compound / ≠ R334–R422 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3109: R424 marsplan×Offline-DPO×Long×LoBeta×HiLR×LongCtx×ExtraLong α=32 r=16 β=0.02 lr=2e-5 @16384 max_steps=900 after R423 (R412×ExtraLong / R422×HiLR / R410×LongCtx compound; no HiRank/HiAlpha/HiBeta).
    ("mine-r424-marsplan-offline-dpo-long-lobeta-hilr-longctx-extralong-1", "R424", "marsplan×Offline-DPO×Long×LoBeta×HiLR×LongCtx×ExtraLong β=0.02 α=32 r=16 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R412 Offline-Long×LoBeta×HiLR×LongCtx × ExtraLong / R422 Offline-Long×LoBeta×LongCtx×ExtraLong × HiLR / R410 Offline-Long×LoBeta×HiLR×ExtraLong × LongCtx compound; ≠ R422@5e-6@900 / ≠ R412@600 / ≠ R410@6144@900 / ≠ R423 LoBeta×HiRank×LongCtx×ExtraLong / ≠ R413 LoBeta×HiRank×HiLR×LongCtx@600 / ≠ R421 HiBeta compound / ≠ R334–R423 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3110: R425 marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×LongCtx×ExtraLong α=32 r=64 β=0.02 lr=2e-5 @16384 max_steps=900 after R424 (R413×ExtraLong / R423×HiLR / R424×HiRank / R409×LongCtx compound; no HiAlpha/HiBeta).
    ("mine-r425-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-extralong-1", "R425", "marsplan×Offline-DPO×Long×LoBeta×HiRank×HiLR×LongCtx×ExtraLong β=0.02 α=32 r=64 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R413 Offline-Long×LoBeta×HiRank×HiLR×LongCtx × ExtraLong / R423 Offline-Long×LoBeta×HiRank×LongCtx×ExtraLong × HiLR / R424 Offline-Long×LoBeta×HiLR×LongCtx×ExtraLong × HiRank / R409 Offline-Long×LoBeta×HiRank×HiLR×ExtraLong × LongCtx compound; ≠ R424@r16@2e-5@900 / ≠ R423@5e-6@900 / ≠ R413@600 / ≠ R409@6144@900 / ≠ R421 HiBeta compound / ≠ R334–R424 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3111: R426 marsplan×Offline-DPO×HiAlpha×LongCtx×ExtraLong α=128 r=16 β=0.1 lr=5e-6 @16384 max_steps=900 after R425 (R371×ExtraLong; no HiRank/HiLR/LoBeta/HiBeta).
    ("mine-r426-marsplan-offline-dpo-hialpha-longctx-extralong-1", "R426", "marsplan×Offline-DPO×HiAlpha×LongCtx×ExtraLong β=0.1 α=128 r=16 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R371 Offline-HiAlpha×LongCtx × ExtraLong compound; ≠ R371@600 / ≠ R369@6144 / ≠ R381 HiRank×HiLR / ≠ R425 LoBeta×HiRank×HiLR×LongCtx×ExtraLong / ≠ R373–R380 HiAlpha LongCtx ladder / ≠ R334–R425 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3112: R427 marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×ExtraLong α=128 r=64 β=0.1 lr=5e-6 @16384 max_steps=900 after R426 (R373×ExtraLong / R426×HiRank).
    ("mine-r427-marsplan-offline-dpo-hialpha-longctx-hirank-extralong-1", "R427", "marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×ExtraLong β=0.1 α=128 r=64 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R373 Offline-HiAlpha×LongCtx×HiRank × ExtraLong compound / R426×HiRank; ≠ R426@r16 / ≠ R373@600 / ≠ R371@r16@600 / ≠ R381 HiRank×HiLR / ≠ R425 LoBeta compound / ≠ R334–R426 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3113: R428 marsplan×Offline-DPO×HiAlpha×LongCtx×HiLR×ExtraLong α=128 r=16 β=0.1 lr=2e-5 @16384 max_steps=900 after R427 (R380×ExtraLong / R426×HiLR).
    ("mine-r428-marsplan-offline-dpo-hialpha-longctx-hilr-extralong-1", "R428", "marsplan×Offline-DPO×HiAlpha×LongCtx×HiLR×ExtraLong β=0.1 α=128 r=16 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R380 Offline-HiAlpha×LongCtx×HiLR × ExtraLong compound / R426×HiLR; ≠ R426@5e-6 / ≠ R380@600 / ≠ R427 HiRank×ExtraLong / ≠ R381 HiRank×HiLR / ≠ R425 LoBeta compound / ≠ R334–R427 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3114: R429 marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×HiLR×ExtraLong α=128 r=64 β=0.1 lr=2e-5 @16384 max_steps=900 after R428 (R381×ExtraLong / R427×HiLR / R428×HiRank).
    ("mine-r429-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-extralong-1", "R429", "marsplan×Offline-DPO×HiAlpha×LongCtx×HiRank×HiLR×ExtraLong β=0.1 α=128 r=64 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R381 Offline-HiAlpha×LongCtx×HiRank×HiLR × ExtraLong compound / R427×HiLR / R428×HiRank; ≠ R428@r16 / ≠ R427@5e-6 / ≠ R381@600 / ≠ R426@r16@5e-6 / ≠ R380@r16 / ≠ R425 LoBeta compound / ≠ R334–R428 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3115: R430 marsplan×Offline-DPO×HiAlpha×LoBeta×LongCtx×ExtraLong α=128 r=16 β=0.02 lr=5e-6 @16384 max_steps=900 after R429 (R426×LoBeta; first HiAlpha×LoBeta ExtraLong).
    ("mine-r430-marsplan-offline-dpo-hialpha-lobeta-longctx-extralong-1", "R430", "marsplan×Offline-DPO×HiAlpha×LoBeta×LongCtx×ExtraLong β=0.02 α=128 r=16 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R426 Offline-HiAlpha×LongCtx×ExtraLong × LoBeta β0.1→0.02 / R422 LoBeta family on HiAlpha; ≠ R426@β0.1 / ≠ R429 HiRank×HiLR@β0.1 / ≠ R422–R425 Long×LoBeta@α32 / ≠ R427–R428 / ≠ R334–R429 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3116: R431 marsplan×Offline-DPO×HiAlpha×HiBeta×LongCtx×ExtraLong α=128 r=16 β=0.5 lr=5e-6 @16384 max_steps=900 after R430 (R426×HiBeta; first HiAlpha×HiBeta ExtraLong).
    ("mine-r431-marsplan-offline-dpo-hialpha-hibeta-longctx-extralong-1", "R431", "marsplan×Offline-DPO×HiAlpha×HiBeta×LongCtx×ExtraLong β=0.5 α=128 r=16 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R426 Offline-HiAlpha×LongCtx×ExtraLong × HiBeta β0.1→0.5 / R402 HiBeta family on HiAlpha×LongCtx; ≠ R426@β0.1 / ≠ R430 LoBeta@β0.02 / ≠ R429 HiRank×HiLR@β0.1 / ≠ R402–R406 Long×HiBeta@α32 / ≠ R334–R430 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),

    # p3117: R432 marsplan×Offline-DPO×HiAlpha×LoBeta×HiRank×LongCtx×ExtraLong α=128 r=64 β=0.02 lr=5e-6 @16384 max_steps=900 after R431 (R430×HiRank / R427×LoBeta).
    ("mine-r432-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-extralong-1", "R432", "marsplan×Offline-DPO×HiAlpha×LoBeta×HiRank×LongCtx×ExtraLong β=0.02 α=128 r=64 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R430 Offline-HiAlpha×LoBeta×LongCtx×ExtraLong × HiRank r16→64 / R427×LoBeta β0.1→0.02; ≠ R430@r16 / ≠ R427@β0.1 / ≠ R431 HiBeta@β0.5 / ≠ R429 HiRank×HiLR@β0.1 / ≠ R423 Long×LoBeta×HiRank@α32 / ≠ R334–R431 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3118: R433 marsplan×Offline-DPO×HiAlpha×HiBeta×HiRank×LongCtx×ExtraLong α=128 r=64 β=0.5 lr=5e-6 @16384 max_steps=900 after R432 (R431×HiRank / R432×HiBeta).
    ("mine-r433-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-extralong-1", "R433", "marsplan×Offline-DPO×HiAlpha×HiBeta×HiRank×LongCtx×ExtraLong β=0.5 α=128 r=64 lr=5e-6 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R431 Offline-HiAlpha×HiBeta×LongCtx×ExtraLong × HiRank r16→64 / R432×HiBeta β0.02→0.5; ≠ R431@r16 / ≠ R432 LoBeta@β0.02 / ≠ R430 LoBeta@r16 / ≠ R429 HiRank×HiLR@β0.1 / ≠ R420 Long×HiBeta×HiRank@α32 / ≠ R334–R432 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3119: R434 marsplan×Offline-DPO×HiAlpha×LoBeta×LongCtx×HiLR×ExtraLong α=128 r=16 β=0.02 lr=2e-5 @16384 max_steps=900 after R433 (R430×HiLR / R428×LoBeta).
    ("mine-r434-marsplan-offline-dpo-hialpha-lobeta-longctx-hilr-extralong-1", "R434", "marsplan×Offline-DPO×HiAlpha×LoBeta×LongCtx×HiLR×ExtraLong β=0.02 α=128 r=16 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R430 Offline-HiAlpha×LoBeta×LongCtx×ExtraLong × HiLR 5e-6→2e-5 / R428×LoBeta β0.1→0.02; ≠ R430@5e-6 / ≠ R428@β0.1 / ≠ R432 HiRank@r64 / ≠ R433 HiBeta×HiRank / ≠ R429 HiRank×HiLR@β0.1 / ≠ R410 Long×LoBeta×HiLR@α32 / ≠ R334–R433 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3120: R435 marsplan×Offline-DPO×HiAlpha×HiBeta×HiLR×LongCtx×ExtraLong α=128 r=16 β=0.5 lr=2e-5 @16384 max_steps=900 after R434 (R431×HiLR / R428×HiBeta).
    ("mine-r435-marsplan-offline-dpo-hialpha-hibeta-longctx-hilr-extralong-1", "R435", "marsplan×Offline-DPO×HiAlpha×HiBeta×HiLR×LongCtx×ExtraLong β=0.5 α=128 r=16 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R431 Offline-HiAlpha×HiBeta×LongCtx×ExtraLong × HiLR 5e-6→2e-5 / R428×HiBeta β0.1→0.5; ≠ R431@5e-6 / ≠ R428@β0.1 / ≠ R434 LoBeta@β0.02 / ≠ R433 HiBeta×HiRank / ≠ R429 HiRank×HiLR@β0.1 / ≠ R404 Long×HiBeta×HiLR@α32 / ≠ R334–R434 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3121: R436 marsplan×Offline-DPO×HiAlpha×LoBeta×HiRank×LongCtx×HiLR×ExtraLong α=128 r=64 β=0.02 lr=2e-5 @16384 max_steps=900 after R435 (R432×HiLR / R434×HiRank).
    ("mine-r436-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-hilr-extralong-1", "R436", "marsplan×Offline-DPO×HiAlpha×LoBeta×HiRank×LongCtx×HiLR×ExtraLong β=0.02 α=128 r=64 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R432 Offline-HiAlpha×LoBeta×HiRank×LongCtx×ExtraLong × HiLR 5e-6→2e-5 / R434×HiRank r16→64; ≠ R432@5e-6 / ≠ R434@r16 / ≠ R433 HiBeta×HiRank@β0.5 / ≠ R435 HiBeta×HiLR@β0.5 / ≠ R429 HiRank×HiLR@β0.1 / ≠ R409 Long×LoBeta×HiRank×HiLR@α32 / ≠ R334–R435 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3122: R437 marsplan×Offline-DPO×HiAlpha×HiBeta×HiRank×LongCtx×HiLR×ExtraLong α=128 r=64 β=0.5 lr=2e-5 @16384 max_steps=900 after R436 (R433×HiLR / R435×HiRank).
    ("mine-r437-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-hilr-extralong-1", "R437", "marsplan×Offline-DPO×HiAlpha×HiBeta×HiRank×LongCtx×HiLR×ExtraLong β=0.5 α=128 r=64 lr=2e-5 @16384 max_steps=900 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R433 Offline-HiAlpha×HiBeta×HiRank×LongCtx×ExtraLong × HiLR 5e-6→2e-5 / R435×HiRank r16→64; ≠ R433@5e-6 / ≠ R435@r16 / ≠ R436 LoBeta×HiRank×HiLR@β0.02 / ≠ R434 LoBeta×HiLR@β0.02 / ≠ R432 LoBeta×HiRank@5e-6 / ≠ R429 HiRank×HiLR@β0.1 / ≠ R409 Long×LoBeta×HiRank×HiLR@α32 / ≠ R334–R436 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227 / ≠ R13 Tok Offline; n80 king=marsplan-queen)"),
    # p3123: R438 marsplan×Online-DPO×ExtraLong α=32 r=16 G=4 β=0.1 lr=5e-6 @6144 max_steps=900 after R437 (R351×ExtraLong isolate / R334 SIGNAL length 3×).
    ("mine-r438-marsplan-online-dpo-extralong-1", "R438", "marsplan×Online-DPO×ExtraLong max_steps=900 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R351 Online-Long × ExtraLong steps isolate; ≠ R351@600 / ≠ R334@300 / ≠ R336 G=8 / ≠ R337 HiLR / ≠ R395 Offline-ExtraLong / ≠ R352–R366 Long×LongCtx / ≠ R367–R437 Offline compounds / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3124: R439 marsplan×Online-DPO×ExtraLong×HiLR α=32 r=16 G=4 β=0.1 lr=2e-5 @6144 max_steps=900 after R438 (R438×R337 compound).
    ("mine-r439-marsplan-online-dpo-extralong-hilr-1", "R439", "marsplan×Online-DPO×ExtraLong×HiLR max_steps=900 β=0.1 α=32 r=16 G=4 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R438 ExtraLong × R337 HiLR compound; ≠ R438@5e-6 / ≠ R337@300 / ≠ R353 Long×HiLR@600 / ≠ R351@5e-6@600 / ≠ R334@300 / ≠ R336 G=8 / ≠ R395 Offline-ExtraLong / ≠ R352–R438 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3125: R440 marsplan×Online-DPO×ExtraLong×BigG α=32 r=16 G=8 β=0.1 lr=5e-6 @6144 max_steps=900 after R439 (R438×R336 compound; R336 REFUTE@151).
    ("mine-r440-marsplan-online-dpo-extralong-bigg-1", "R440", "marsplan×Online-DPO×ExtraLong×BigG max_steps=900 β=0.1 α=32 r=16 G=8 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R438 ExtraLong × R336 BigG compound; ≠ R438 G=4@900 / ≠ R336 G=8@300 REFUTE / ≠ R439 ExtraLong×HiLR / ≠ R337 HiLR / ≠ R351@600 / ≠ R334@300 / ≠ R395 Offline-ExtraLong / ≠ R352–R439 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3128: R443 marsplan×Online-DPO×ExtraLong×HiRank×HiLR α=32 r=64 G=4 β=0.1 lr=2e-5 @6144 max_steps=900 after R442 (R442×R439 / R438×R339×R337 compound).
    ("mine-r443-marsplan-online-dpo-extralong-hirank-hilr-1", "R443", "marsplan×Online-DPO×ExtraLong×HiRank×HiLR max_steps=900 β=0.1 α=32 r=64 G=4 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R442 ExtraLong×HiRank × R439 ExtraLong×HiLR; ≠ R442@5e-6 / ≠ R439 r=16 / ≠ R441 ExtraLong×BigG×HiLR / ≠ R440 ExtraLong×BigG / ≠ R438 ExtraLong / ≠ R341 HiRank×HiLR@300 / ≠ R339@300 / ≠ R336–R442 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3129: R444 marsplan×Online-DPO×ExtraLong×HiRank×BigG α=32 r=64 G=8 β=0.1 lr=5e-6 @6144 max_steps=900 after R443 (R442×R440 / R438×R339×R336 compound).
    ("mine-r444-marsplan-online-dpo-extralong-hirank-bigg-1", "R444", "marsplan×Online-DPO×ExtraLong×HiRank×BigG max_steps=900 β=0.1 α=32 r=64 G=8 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R442 ExtraLong×HiRank × R440 ExtraLong×BigG; ≠ R443 ExtraLong×HiRank×HiLR / ≠ R442 G=4 / ≠ R441 ExtraLong×BigG×HiLR / ≠ R440 ExtraLong×BigG r=16 / ≠ R438 ExtraLong / ≠ R336 G=8@300 REFUTE / ≠ R339@300 / ≠ R336–R443 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3130: R445 marsplan×Online-DPO×ExtraLong×HiRank×BigG×HiLR α=32 r=64 G=8 β=0.1 lr=2e-5 @6144 max_steps=900 after R444 (R444×R443 / R442×R440×R439 compound).
    ("mine-r445-marsplan-online-dpo-extralong-hirank-bigg-hilr-1", "R445", "marsplan×Online-DPO×ExtraLong×HiRank×BigG×HiLR max_steps=900 β=0.1 α=32 r=64 G=8 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R444 ExtraLong×HiRank×BigG × R443 ExtraLong×HiRank×HiLR; ≠ R444@5e-6 / ≠ R443 G=4@2e-5 / ≠ R442 G=4@5e-6 / ≠ R441 ExtraLong×BigG×HiLR r=16 / ≠ R440 ExtraLong×BigG / ≠ R439 ExtraLong×HiLR / ≠ R438 ExtraLong / ≠ R336 G=8@300 REFUTE / ≠ R339@300 / ≠ R336–R444 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3131: R446 marsplan×Online-DPO×ExtraLong×HiTemp α=32 r=16 G=4 β=0.1 lr=5e-6 temp=1.5 @6144 max_steps=900 after R445 (R438×temp↑ isolate; lean-warm crown 6,7).
    ("mine-r446-marsplan-online-dpo-extralong-hitemp-1", "R446", "marsplan×Online-DPO×ExtraLong×HiTemp max_steps=900 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=1.5 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R438 ExtraLong × HiTemp 1.2→1.5 isolate; ≠ R438@temp1.2 / ≠ R439–R445@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3132: R447 marsplan×Online-DPO×ExtraLong×UltraTemp α=32 r=16 G=4 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R446 (R446×temp↑ isolate; lean-warm crown 4,5).
    ("mine-r447-marsplan-online-dpo-extralong-ultratemp-1", "R447", "marsplan×Online-DPO×ExtraLong×UltraTemp max_steps=900 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R446 ExtraLong×HiTemp × UltraTemp 1.5→2.0 isolate; ≠ R446@temp1.5 / ≠ R438–R445@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),

    # p3133: R448 marsplan×Online-DPO×ExtraLong×UltraTemp×HiLR α=32 r=16 G=4 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R447 (R447×R439; lean-warm R262 4,5).
    ("mine-r448-marsplan-online-dpo-extralong-ultratemp-hilr-1", "R448", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiLR max_steps=900 β=0.1 α=32 r=16 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R447 ExtraLong×UltraTemp × R439 ExtraLong×HiLR; ≠ R447@5e-6 / ≠ R439@temp1.2 / ≠ R446@temp1.5 / ≠ R438–R445@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3134: R449 marsplan×Online-DPO×ExtraLong×UltraTemp×BigG α=32 r=16 G=8 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R448 (R447×R440; lean-warm R260 6,7).
    ("mine-r449-marsplan-online-dpo-extralong-ultratemp-bigg-1", "R449", "marsplan×Online-DPO×ExtraLong×UltraTemp×BigG max_steps=900 β=0.1 α=32 r=16 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R447 ExtraLong×UltraTemp × R440 ExtraLong×BigG; ≠ R447@G4 / ≠ R448 UltraTemp×HiLR / ≠ R440@temp1.2 / ≠ R441 BigG×HiLR / ≠ R446–R445 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3135: R450 marsplan×Online-DPO×ExtraLong×UltraTemp×BigG×HiLR α=32 r=16 G=8 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R449 (R449×R448 / R441×UltraTemp; queued rent — 0×8 + lean busy).
    ("mine-r450-marsplan-online-dpo-extralong-ultratemp-bigg-hilr-1", "R450", "marsplan×Online-DPO×ExtraLong×UltraTemp×BigG×HiLR max_steps=900 β=0.1 α=32 r=16 G=8 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R449 ExtraLong×UltraTemp×BigG × R448 ExtraLong×UltraTemp×HiLR; ≠ R449@5e-6 / ≠ R448@G4 / ≠ R447 UltraTemp / ≠ R441 BigG×HiLR@temp1.2 / ≠ R440–R446 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),    # p3136: R451 marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank α=32 r=64 G=4 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R450 (R447×R442 UltraTemp×HiRank; queued rent — 0×8 + lean busy).
    ("mine-r451-marsplan-online-dpo-extralong-ultratemp-hirank-1", "R451", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank max_steps=900 β=0.1 α=32 r=64 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R447 ExtraLong×UltraTemp × R442 ExtraLong×HiRank; ≠ R447@r16 / ≠ R442@temp1.2 / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R446–R445 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3137: R452 marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank×HiLR α=32 r=64 G=4 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R451 (R451×R448 / R443×UltraTemp; queued rent — 0×8 + lean busy).
    ("mine-r452-marsplan-online-dpo-extralong-ultratemp-hirank-hilr-1", "R452", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank×HiLR max_steps=900 β=0.1 α=32 r=64 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R451 ExtraLong×UltraTemp×HiRank × R448 ExtraLong×UltraTemp×HiLR; ≠ R451@5e-6 / ≠ R443@temp1.2 / ≠ R448@r16 / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R447 UltraTemp / ≠ R446–R445 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3138: R453 marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank×BigG α=32 r=64 G=8 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R452 (R451×R449 / R444×UltraTemp; queued rent — 0×8 + lean busy).
    ("mine-r453-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-1", "R453", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank×BigG max_steps=900 β=0.1 α=32 r=64 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R451 ExtraLong×UltraTemp×HiRank × R449 ExtraLong×UltraTemp×BigG; ≠ R451@G4 / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R444@temp1.2 / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449@r16 / ≠ R447 UltraTemp / ≠ R445 HiRank×BigG×HiLR@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3139: R454 marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank×BigG×HiLR α=32 r=64 G=8 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R453 (R453×HiLR / R452×BigG / R450×HiRank / R445×UltraTemp; queued rent — 0×8 + lean busy).
    ("mine-r454-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-hilr-1", "R454", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiRank×BigG×HiLR max_steps=900 β=0.1 α=32 r=64 G=8 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R453 ExtraLong×UltraTemp×HiRank×BigG × HiLR; ≠ R453@5e-6 / ≠ R452 UltraTemp×HiRank×HiLR@G4 / ≠ R450 UltraTemp×BigG×HiLR@r16 / ≠ R445@temp1.2 / ≠ R451 UltraTemp×HiRank / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3140: R455 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha α=128 r=16 G=4 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R454 (R447×HiAlpha; first HiAlpha on Online UltraTemp ExtraLong; queued rent — 0×8 + lean busy).
    ("mine-r455-marsplan-online-dpo-extralong-ultratemp-hialpha-1", "R455", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha max_steps=900 β=0.1 α=128 r=16 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R447 ExtraLong×UltraTemp × HiAlpha α32→128; ≠ R447@α32 / ≠ R454 UltraTemp×HiRank×BigG×HiLR / ≠ R453 UltraTemp×HiRank×BigG / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3141: R456 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiLR α=128 r=16 G=4 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R455 (R455×R448 / R455×HiLR; queued rent — 0×8 + lean busy).
    ("mine-r456-marsplan-online-dpo-extralong-ultratemp-hialpha-hilr-1", "R456", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiLR max_steps=900 β=0.1 α=128 r=16 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R455 ExtraLong×UltraTemp×HiAlpha × R448 ExtraLong×UltraTemp×HiLR; ≠ R455@5e-6 / ≠ R448@α32 / ≠ R454 UltraTemp×HiRank×BigG×HiLR / ≠ R453 UltraTemp×HiRank×BigG / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3142: R457 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×BigG α=128 r=16 G=8 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R456 (R455×R449 / R455×BigG; queued rent — 0×8 + lean busy).
    ("mine-r457-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-1", "R457", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×BigG max_steps=900 β=0.1 α=128 r=16 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R455 ExtraLong×UltraTemp×HiAlpha × R449 ExtraLong×UltraTemp×BigG; ≠ R456 UltraTemp×HiAlpha×HiLR / ≠ R455@G4 / ≠ R449@α32 / ≠ R454 UltraTemp×HiRank×BigG×HiLR / ≠ R453 UltraTemp×HiRank×BigG / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3143: R458 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×BigG×HiLR α=128 r=16 G=8 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R457 (R457×HiLR / R456×BigG; queued rent — 0×8 + lean busy).
    ("mine-r458-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-hilr-1", "R458", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×BigG×HiLR max_steps=900 β=0.1 α=128 r=16 G=8 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R457 ExtraLong×UltraTemp×HiAlpha×BigG × HiLR; ≠ R457@5e-6 / ≠ R456 UltraTemp×HiAlpha×HiLR@G4 / ≠ R455 UltraTemp×HiAlpha / ≠ R454 UltraTemp×HiRank×BigG×HiLR / ≠ R453 UltraTemp×HiRank×BigG / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),

    # p3144: R459 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank α=128 r=64 G=4 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R458 (R455×R451; queued rent — 0×8 + lean busy).
    ("mine-r459-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-1", "R459", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank max_steps=900 β=0.1 α=128 r=64 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R455 ExtraLong×UltraTemp×HiAlpha × R451 ExtraLong×UltraTemp×HiRank; ≠ R458 UltraTemp×HiAlpha×BigG×HiLR / ≠ R457 UltraTemp×HiAlpha×BigG / ≠ R456 UltraTemp×HiAlpha×HiLR / ≠ R455@r16 / ≠ R454 UltraTemp×HiRank×BigG×HiLR@α32 / ≠ R453 UltraTemp×HiRank×BigG / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451@α32 / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),

    # p3145: R460 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank×HiLR α=128 r=64 G=4 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R459 (R459×HiLR; queued rent — 0×8 + lean busy).
    ("mine-r460-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-hilr-1", "R460", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank×HiLR max_steps=900 β=0.1 α=128 r=64 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R459 ExtraLong×UltraTemp×HiAlpha×HiRank × HiLR; ≠ R459@5e-6 / ≠ R458 UltraTemp×HiAlpha×BigG×HiLR / ≠ R457 UltraTemp×HiAlpha×BigG / ≠ R456 UltraTemp×HiAlpha×HiLR / ≠ R455 UltraTemp×HiAlpha / ≠ R454 UltraTemp×HiRank×BigG×HiLR@α32 / ≠ R453 UltraTemp×HiRank×BigG / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3146: R461 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank×BigG α=128 r=64 G=8 β=0.1 lr=5e-6 temp=2.0 @6144 max_steps=900 after R460 (R459×BigG / R453×HiAlpha; queued rent — 0×8 + lean busy).
    ("mine-r461-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-1", "R461", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank×BigG max_steps=900 β=0.1 α=128 r=64 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R459 ExtraLong×UltraTemp×HiAlpha×HiRank × BigG; ≠ R460 UltraTemp×HiAlpha×HiRank×HiLR / ≠ R459@G4 / ≠ R458 UltraTemp×HiAlpha×BigG×HiLR@r16 / ≠ R457 UltraTemp×HiAlpha×BigG / ≠ R456 UltraTemp×HiAlpha×HiLR / ≠ R455 UltraTemp×HiAlpha / ≠ R454 UltraTemp×HiRank×BigG×HiLR@α32 / ≠ R453 UltraTemp×HiRank×BigG@α32 / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3147: R462 marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank×BigG×HiLR α=128 r=64 G=8 β=0.1 lr=2e-5 temp=2.0 @6144 max_steps=900 after R461 (R461×HiLR / R460×BigG; queued rent — 0×8 + lean busy).
    ("mine-r462-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-hilr-1", "R462", "marsplan×Online-DPO×ExtraLong×UltraTemp×HiAlpha×HiRank×BigG×HiLR max_steps=900 β=0.1 α=128 r=64 G=8 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R461 ExtraLong×UltraTemp×HiAlpha×HiRank×BigG × HiLR; ≠ R461@5e-6 / ≠ R460 UltraTemp×HiAlpha×HiRank×HiLR@G4 / ≠ R459 UltraTemp×HiAlpha×HiRank / ≠ R458 UltraTemp×HiAlpha×BigG×HiLR@r16 / ≠ R457 UltraTemp×HiAlpha×BigG / ≠ R456 UltraTemp×HiAlpha×HiLR / ≠ R455 UltraTemp×HiAlpha / ≠ R454 UltraTemp×HiRank×BigG×HiLR@α32 / ≠ R453 UltraTemp×HiRank×BigG@α32 / ≠ R452 UltraTemp×HiRank×HiLR / ≠ R451 UltraTemp×HiRank / ≠ R450 UltraTemp×BigG×HiLR / ≠ R449 UltraTemp×BigG / ≠ R448 UltraTemp×HiLR / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3148: R463 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx α=32 r=16 G=4 β=0.1 lr=5e-6 temp=2.0 @16384/1024 max_steps=900 after R462 (R447×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r463-marsplan-online-dpo-extralong-ultratemp-longctx-1", "R463", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx max_steps=900 β=0.1 α=32 r=16 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R447 ExtraLong×UltraTemp × LongCtx; ≠ R447@6144 / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R343 LongCtx@300 / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3149: R464 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiLR α=32 r=16 G=4 β=0.1 lr=2e-5 temp=2.0 @16384/1024 max_steps=900 after R463 (R463×HiLR / R448×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r464-marsplan-online-dpo-extralong-ultratemp-longctx-hilr-1", "R464", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiLR max_steps=900 β=0.1 α=32 r=16 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R463 ExtraLong×UltraTemp×LongCtx × HiLR; ≠ R463@5e-6 / ≠ R448 UltraTemp×HiLR@6144 / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R449 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3150: R465 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×BigG α=32 r=16 G=8 β=0.1 lr=5e-6 temp=2.0 @16384/1024 max_steps=900 after R464 (R463×BigG / R449×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r465-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-1", "R465", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×BigG max_steps=900 β=0.1 α=32 r=16 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R463 ExtraLong×UltraTemp×LongCtx × BigG; ≠ R463@G4 / ≠ R464 UltraTemp×LongCtx×HiLR / ≠ R449 UltraTemp×BigG@6144 / ≠ R450 UltraTemp×BigG×HiLR / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R440 ExtraLong×BigG@temp1.2 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3151: R466 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×BigG×HiLR α=32 r=16 G=8 β=0.1 lr=2e-5 temp=2.0 @16384/1024 max_steps=900 after R465 (R465×HiLR / R464×BigG; queued rent — 0×8 + lean busy).
    ("mine-r466-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-hilr-1", "R466", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×BigG×HiLR max_steps=900 β=0.1 α=32 r=16 G=8 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R465 ExtraLong×UltraTemp×LongCtx×BigG × HiLR; ≠ R465@5e-6 / ≠ R464 UltraTemp×LongCtx×HiLR@G4 / ≠ R463 UltraTemp×LongCtx / ≠ R450 UltraTemp×BigG×HiLR@6144 / ≠ R449 UltraTemp×BigG@6144 / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R440 ExtraLong×BigG@temp1.2 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3152: R467 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank α=32 r=64 G=4 β=0.1 lr=5e-6 temp=2.0 @16384/1024 max_steps=900 after R466 (R463×HiRank / R451×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r467-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-1", "R467", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank max_steps=900 β=0.1 α=32 r=64 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R463 ExtraLong×UltraTemp×LongCtx × HiRank; ≠ R463@r16 / ≠ R451 UltraTemp×HiRank@6144 / ≠ R466 UltraTemp×LongCtx×BigG×HiLR / ≠ R465 UltraTemp×LongCtx×BigG / ≠ R464 UltraTemp×LongCtx×HiLR / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R442 ExtraLong×HiRank@temp1.2 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3153: R468 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank×HiLR α=32 r=64 G=4 β=0.1 lr=2e-5 temp=2.0 @16384/1024 max_steps=900 after R467 (R467×HiLR / R464×HiRank / R452×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r468-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-hilr-1", "R468", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank×HiLR max_steps=900 β=0.1 α=32 r=64 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R467 ExtraLong×UltraTemp×LongCtx×HiRank × HiLR; ≠ R467@5e-6 / ≠ R464 UltraTemp×LongCtx×HiLR@r16 / ≠ R452 UltraTemp×HiRank×HiLR@6144 / ≠ R466 UltraTemp×LongCtx×BigG×HiLR / ≠ R465 UltraTemp×LongCtx×BigG / ≠ R463 UltraTemp×LongCtx / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R443 ExtraLong×HiRank×HiLR@temp1.2 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3154: R469 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank×BigG α=32 r=64 G=8 β=0.1 lr=5e-6 temp=2.0 @16384/1024 max_steps=900 after R468 (R467×BigG / R465×HiRank / R453×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r469-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-1", "R469", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank×BigG max_steps=900 β=0.1 α=32 r=64 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R467 ExtraLong×UltraTemp×LongCtx×HiRank × BigG; ≠ R467@G4 / ≠ R468 UltraTemp×LongCtx×HiRank×HiLR / ≠ R465 UltraTemp×LongCtx×BigG@r16 / ≠ R453 UltraTemp×HiRank×BigG@6144 / ≠ R466 UltraTemp×LongCtx×BigG×HiLR / ≠ R464 UltraTemp×LongCtx×HiLR / ≠ R463 UltraTemp×LongCtx / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R444 ExtraLong×HiRank×BigG@temp1.2 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3155: R470 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank×BigG×HiLR α=32 r=64 G=8 β=0.1 lr=2e-5 temp=2.0 @16384/1024 max_steps=900 after R469 (R469×HiLR / R468×BigG / R466×HiRank / R454×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r470-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-hilr-1", "R470", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiRank×BigG×HiLR max_steps=900 β=0.1 α=32 r=64 G=8 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R469 ExtraLong×UltraTemp×LongCtx×HiRank×BigG × HiLR; ≠ R469@5e-6 / ≠ R468 UltraTemp×LongCtx×HiRank×HiLR@G4 / ≠ R466 UltraTemp×LongCtx×BigG×HiLR@r16 / ≠ R454 UltraTemp×HiRank×BigG×HiLR@6144 / ≠ R467 UltraTemp×LongCtx×HiRank / ≠ R465 UltraTemp×LongCtx×BigG / ≠ R464 UltraTemp×LongCtx×HiLR / ≠ R463 UltraTemp×LongCtx / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R445 ExtraLong×HiRank×BigG×HiLR@temp1.2 / ≠ R359 ExtraLong×LongCtx@temp1.2 / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3156: R471 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha α=128 r=16 G=4 β=0.1 lr=5e-6 temp=2.0 @16384/1024 max_steps=900 after R470 (R463×HiAlpha / R455×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r471-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-1", "R471", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha max_steps=900 β=0.1 α=128 r=16 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R463 ExtraLong×UltraTemp×LongCtx × HiAlpha / R455 UltraTemp×HiAlpha × LongCtx; ≠ R455@6144 / ≠ R470 UltraTemp×LongCtx×HiRank×BigG×HiLR@α32 / ≠ R469 UltraTemp×LongCtx×HiRank×BigG / ≠ R468 UltraTemp×LongCtx×HiRank×HiLR / ≠ R467 UltraTemp×LongCtx×HiRank / ≠ R466 UltraTemp×LongCtx×BigG×HiLR / ≠ R465 UltraTemp×LongCtx×BigG / ≠ R464 UltraTemp×LongCtx×HiLR / ≠ R463@α32 / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR@6144 / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3157: R472 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha×HiLR α=128 r=16 G=4 β=0.1 lr=2e-5 temp=2.0 @16384/1024 max_steps=900 after R471 (R471×HiLR / R456×LongCtx; queued rent — 0×8 + lean busy).
    ("mine-r472-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hilr-1", "R472", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha×HiLR max_steps=900 β=0.1 α=128 r=16 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R471 ExtraLong×UltraTemp×LongCtx×HiAlpha × HiLR / R456 UltraTemp×HiAlpha×HiLR × LongCtx; ≠ R471@5e-6 / ≠ R456@6144 / ≠ R464 UltraTemp×LongCtx×HiLR@α32 / ≠ R470 UltraTemp×LongCtx×HiRank×BigG×HiLR@α32 / ≠ R469 UltraTemp×LongCtx×HiRank×BigG / ≠ R468 UltraTemp×LongCtx×HiRank×HiLR / ≠ R467 UltraTemp×LongCtx×HiRank / ≠ R466 UltraTemp×LongCtx×BigG×HiLR / ≠ R465 UltraTemp×LongCtx×BigG / ≠ R463 UltraTemp×LongCtx / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR@6144 / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3160: R473 marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha×HiRank α=128 r=64 G=4 β=0.1 lr=5e-6 temp=2.0 @16384/1024 max_steps=900 — lean on crown 6,7 (R447 keeps 4,5); also queued for rent if stock.
    ("mine-r473-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hirank-1", "R473", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha×HiRank max_steps=900 β=0.1 α=128 r=64 G=4 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R471 ExtraLong×UltraTemp×LongCtx×HiAlpha × HiRank / R459 UltraTemp×HiAlpha×HiRank × LongCtx; ≠ R472 HiLR@r16 / ≠ R471@r16 / ≠ R467 UltraTemp×LongCtx×HiRank@α32 / ≠ R459@6144 / ≠ R470 UltraTemp×LongCtx×HiRank×BigG×HiLR@α32 / ≠ R468–R466 / ≠ R463 UltraTemp×LongCtx / ≠ R462 UltraTemp×HiAlpha×HiRank×BigG×HiLR@6144 / ≠ R461–R448 UltraTemp@6144 / ≠ R447 UltraTemp / ≠ R446 HiTemp / ≠ R336–R437 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3161: R474 lean on lunar 6,7 — ExtraLong×UltraTemp×LongCtx×HiAlpha×HiRank×HiLR; also queued for rent if stock.
    ("mine-r474-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hirank-hilr-1", "R474", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha×HiRank×HiLR max_steps=900 β=0.1 α=128 r=64 G=4 lr=2e-5 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R473 × HiLR; ≠ R473@5e-6 / ≠ R472@r16 / ≠ R468@α32 / ≠ R460@6144 / ≠ R439–R471; n80 king=marsplan-queen)"),
    # p3163: R475 lean on crown 4,5 after R447 SIGNAL_POS_BELOW + chall purge — ExtraLong×UltraTemp×LongCtx×HiAlpha×HiRank×BigG; also queued for rent if stock.
    ("mine-r475-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hirank-bigg-1", "R475", "marsplan×Online-DPO×ExtraLong×UltraTemp×LongCtx×HiAlpha×HiRank×BigG max_steps=900 β=0.1 α=128 r=64 G=8 lr=5e-6 temp=2.0 FORCE_PREFIX min_gap=0 @16384/1024 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R473 × BigG; ≠ R473@G4 / ≠ R474 HiLR@G4 / ≠ R461@6144 / ≠ R469@α32 / ≠ R448–R472; n80 king=marsplan-queen)"),
    # p3127: R442 marsplan×Online-DPO×ExtraLong×HiRank α=32 r=64 G=4 β=0.1 lr=5e-6 @6144 max_steps=900 after R441 (R438×R339 ExtraLong×HiRank isolate).
    ("mine-r442-marsplan-online-dpo-extralong-hirank-1", "R442", "marsplan×Online-DPO×ExtraLong×HiRank max_steps=900 β=0.1 α=32 r=64 G=4 lr=5e-6 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R438 ExtraLong × R339 HiRank; ≠ R438 r=16 / ≠ R441 ExtraLong×BigG×HiLR / ≠ R440 ExtraLong×BigG / ≠ R439 ExtraLong×HiLR / ≠ R339@300 / ≠ R355 Long×HiRank / ≠ R336–R441 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # p3126: R441 marsplan×Online-DPO×ExtraLong×BigG×HiLR α=32 r=16 G=8 β=0.1 lr=2e-5 @6144 max_steps=900 after R440 (R440×R439 / R438×R336×R337 triple).
    ("mine-r441-marsplan-online-dpo-extralong-bigg-hilr-1", "R441", "marsplan×Online-DPO×ExtraLong×BigG×HiLR max_steps=900 β=0.1 α=32 r=16 G=8 lr=2e-5 temp=1.2 FORCE_PREFIX min_gap=0 @6144 from marsplan0624/affine-5gedzafcvg-queen@556d02a2 (R440 ExtraLong×BigG × R439 ExtraLong×HiLR triple; ≠ R440@5e-6 / ≠ R439 G=4@2e-5 / ≠ R438 G=4@5e-6 / ≠ R336 G=8@300 REFUTE / ≠ R337 HiLR@300 / ≠ R351@600 / ≠ R334@300 / ≠ R395 Offline-ExtraLong / ≠ R352–R440 / ≠ R335 BoN / ≠ R230–R232 / ≠ R225–R227; n80 king=marsplan-queen)"),
    # ("mine-r228-marsplan-odpo-1", "R228", "marsplan×Offline-DPO β=0.1 α=32 r=16 lr=5e-6 @6144 max_steps=200 from live reign-16 (R13 method on marsplan; ≠ R204–R227 GRPO/SFT/REINFORCE/FullFT ladder / ≠ R13 Tok-init Offline-DPO / ≠ R11 online DPO; n80 king=marsplan)"),
    # p2940: rent HEAD stays R252; next after demote R228 → R226 FullFT (still queued).
    # p2946: R229 lean-warm on mine-r165 lunar GPUs4–5 after R228 REFUTE+purge (isolated /root/r229; R212 keeps 6–7) — do not re-rent.
    # p2840: QUEUE#21 marsplan×Online-DPO — method axis (R11 recipe on live-king init; ≠ R228 Offline / ≠ R204–R227 GRPO/SFT/REINFORCE/FullFT / ≠ R11 Tok).
    # ("mine-r229-marsplan-online-dpo-1", "R229", "marsplan×Online-DPO β=0.1 α=32 r=16 G=2 lr=5e-6 @6144 max_steps=150 from live reign-16 (R11 method on marsplan; ≠ R228 Offline-DPO / ≠ R204–R227 GRPO/SFT/REINFORCE/FullFT / ≠ R11 Tok-init Online-DPO; n80 king=marsplan)"),
    # p2946: rent HEAD stays R252; next after demote R229 → R226 FullFT (still queued).
    # p2991: R230 lean-warm on mine-r165 lunar GPUs6–7 after R212 REFUTE+purge (isolated /root/r230; R229 keeps 4–5) — do not re-rent.
    # p2842: QUEUE#22 marsplan×BoN-CE — method axis (R12 recipe on live-king init; ≠ R204–R229 GRPO/SFT/REINFORCE/FullFT/DPO / ≠ R12 Tok).
    # ("mine-r230-marsplan-bon-1", "R230", "marsplan×BoN-CE α=32 r=16 G=4 lr=5e-6 @6144 max_steps=150 from live reign-16 (R12 method on marsplan; ≠ R204–R229 GRPO/SFT/REINFORCE/FullFT/DPO ladder / ≠ R12 Tok-init BoN; n80 king=marsplan)"),
    # p2999: R231 lean-warm on mine-r165 lunar GPUs4–5 (isolated /root/r231; R230 keeps 6–7) — do not re-rent.
    # p2848: QUEUE#23 marsplan×KL-GRPO — method axis (R32 recipe on live-king init; ≠ R204–R230 GRPO/SFT/REINFORCE/FullFT/DPO/BoN / ≠ R32 Tok).
    # ("mine-r231-marsplan-kl-grpo-1", "R231", "marsplan×KL-GRPO kl=0.02 α=32 r=16 G=4 lr=5e-6 @6144 max_steps=200 from live reign-16 (R32 method on marsplan; ≠ R204–R230 GRPO/SFT/REINFORCE/FullFT/DPO/BoN ladder / ≠ R32 Tok-init KL-GRPO; n80 king=marsplan)"),
    # p2998: R232 lean-warm on mine-crown-1 GPUs6–7 — do not re-rent.
    # p2849: QUEUE#24 marsplan×Teacher-ZC — method axis (R9 recipe on live-king init; ≠ R221 winner_za SFT / ≠ R222 DataFilt / ≠ R223–R231 / ≠ R9 Tok).
    # ("mine-r232-marsplan-teacher-zc-1", "R232", "marsplan×Teacher-ZC thought-only LoRA on expanded teacher z_C lr=1e-5 r=32 α=64 EP=3 @16384 from live reign-16 (R9 method on marsplan; ≠ R221 winner_za Reason-SFT / ≠ R222 DataFilt / ≠ R223–R231 GRPO/SFT/RL/DPO/BoN/KL / ≠ R9 Tok-init Teacher-ZC; n80 king=marsplan)"),
    # p2850: QUEUE#25 Genesis-nonking×HiAlpha-GRPO — structural non-king base (R204 knobs on Genesis; n80 king=marsplan; ≠ R204–R232 marsplan-init / ≠ R5 FullFT / ≠ R231 KL).
    ("mine-r233-genesis-nonking-grpo-1", "R233", "Genesis-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Genesis@abe89194 (R204 knobs on non-king base; ≠ R204–R232 marsplan-init ladder / ≠ R5 Genesis FullFT / ≠ R231 KL; n80 king=marsplan)"),
    # p2851: QUEUE#26 Tok-af10-nonking×HiAlpha-GRPO — structural non-king base #2 (R204 knobs on Tok af10 reign-4; n80 king=marsplan; ≠ R204–R233 marsplan/Genesis / ≠ Tok ladder R3–R133 / ≠ R5 FullFT).
    ("mine-r234-tok-nonking-grpo-1", "R234", "Tok-af10-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Tok331102/affine-5EqYW8McUc-af10@eb8bf9a (R204 knobs on reign-4 non-live base; ≠ R204–R233 marsplan/Genesis / ≠ Tok ladder R3–R133 / ≠ R5 Genesis FullFT / ≠ R233 Genesis-nonking; n80 king=marsplan)"),
    # p2852: QUEUE#27 Talent-nonking×HiAlpha-GRPO — structural non-king base #3 (R204 knobs on TalentPigs reign-3; n80 king=marsplan; ≠ R204–R234 marsplan/Genesis/Tok / ≠ Talent ladder R19–R133 / ≠ R5 FullFT).
    ("mine-r235-talent-nonking-grpo-1", "R235", "Talent-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from TalentPigs/affine-5ekxlcg3fx-abc@dbfbb3e2 (R204 knobs on reign-3 non-live base; ≠ R204–R234 marsplan/Genesis/Tok / ≠ Talent ladder R19–R133 / ≠ R5 Genesis FullFT / ≠ R233 Genesis / ≠ R234 Tok; n80 king=marsplan)"),
    # p2853: QUEUE#28 Kevin-nonking×HiAlpha-GRPO — structural non-king base #4 (R204 knobs on kevin954 reign-2; n80 king=marsplan; ≠ R204–R235 marsplan/Genesis/Tok/Talent / ≠ R5 FullFT / ≠ R233–R235).
    ("mine-r236-kevin-nonking-grpo-1", "R236", "Kevin-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from kevin954/Affine-5dfqbbh8ev-sft@6a5815fa (R204 knobs on reign-2 non-live base; ≠ R204–R235 marsplan/Genesis/Tok/Talent / ≠ R5 Genesis FullFT / ≠ R233 Genesis / ≠ R234 Tok / ≠ R235 Talent; n80 king=marsplan)"),
    # p2854: QUEUE#29 Pandora-nonking×HiAlpha-GRPO — structural non-king base #5 (R204 knobs on pandora-box reign-1; n80 king=marsplan; ≠ R204–R236 marsplan/Genesis/Tok/Talent/Kevin / ≠ R21 α=32 / ≠ R5 FullFT / ≠ R233–R236).
    ("mine-r237-pandora-nonking-grpo-1", "R237", "Pandora-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from pandora-box/Affine-5eqdtdzqle-ckpt300-m4@5218b138 (R204 knobs on reign-1 non-live base; ≠ R204–R236 marsplan/Genesis/Tok/Talent/Kevin / ≠ R21 Pandora α=32 / ≠ R5 Genesis FullFT / ≠ R233 Genesis / ≠ R234 Tok / ≠ R235 Talent / ≠ R236 Kevin; n80 king=marsplan)"),
    # p2855: QUEUE#30 Ckp333-nonking×HiAlpha-GRPO — structural non-king base #6 (R204 knobs on tolegend ckp333 reign-15; n80 king=marsplan; ≠ R204–R237 marsplan/Genesis/Tok/Talent/Kevin/Pandora / ≠ R39/R89 buried / ≠ R5 FullFT / ≠ R233–R237).
    ("mine-r238-ckp333-nonking-grpo-1", "R238", "Ckp333-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from tolegend/Affine-5fqbxvz29b-ckp333@24c137e8 (R204 knobs on reign-15 prior-crown base; ≠ R204–R237 marsplan/Genesis/Tok/Talent/Kevin/Pandora / ≠ R39 α=32 / ≠ R89 ckp333×HiAlpha buried / ≠ R47 ckp333×BigG / ≠ R5 Genesis FullFT / ≠ R233–R237; n80 king=marsplan)"),
    # p2856: QUEUE#31 Golden-nonking×HiAlpha-GRPO — structural non-king base #7 (R204 knobs on golden-crown seed-earner; n80 king=marsplan; ≠ R204–R238 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333 / ≠ R22/R88 buried / ≠ R5 FullFT / ≠ R233–R238).
    ("mine-r239-golden-nonking-grpo-1", "R239", "Golden-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from golden-crown/Affine-5EpvnXGu8jUAVc67oPGgJ3brR4JZqjBUSaTKhZuBoNAAzSJF@ee37f4f0 (R204 knobs on seed-earner non-live base; ≠ R204–R238 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333 / ≠ R22 α=32 / ≠ R88 golden×HiAlpha buried / ≠ R46 golden×BigG / ≠ R5 Genesis FullFT / ≠ R233–R238; n80 king=marsplan)"),
    # p2858: demote R240 after reign18 flip-back — isomsom-nonking×HiAlpha (was live-king HEAD under reign17; n80 now marsplan).
    ("mine-r240-isomsom-hialpha-1", "R240", "isomsom-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from isomsom/Affine-5esaxlk9vr-v11@5bfe4b6c (R204 knobs on reign-17 prior-crown; ≠ R204–R239 marsplan/nonking ladder / ≠ R165 awesome; n80 king=marsplan)"),
    # p2859: QUEUE#32 Diane-nonking×HiAlpha-GRPO — structural non-king base #9 (R204 knobs on diane613 seed-earner; n80 king=marsplan; ≠ R204–R240 / ≠ R23/R87 buried / ≠ Diane ladder / ≠ R5 FullFT / ≠ R233–R240).
    ("mine-r241-diane-nonking-grpo-1", "R241", "Diane-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from diane613/Affine-5CQLBK7Mmw1vsk7eQcBok9Qn44JNU5YVrfNmZpJHPxLV271B@ad0f3f11 (R204 knobs on seed-earner non-live base; ≠ R204–R240 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom / ≠ R23 α=32 / ≠ R87 diane×HiAlpha buried / ≠ R45 diane×BigG / ≠ Diane ladder R23–R106 / ≠ R5 Genesis FullFT / ≠ R233–R240; n80 king=marsplan)"),
    # p2860: QUEUE#33 Bittob-nonking×HiAlpha-GRPO — structural non-king base #10 (R204 knobs on Bittob11040 seed-earner; n80 king=marsplan; ≠ R204–R241 / ≠ R233–R241 / ≠ R5 FullFT / ≠ Bittoby1040 R2bl different repo).
    ("mine-r242-bittob-nonking-grpo-1", "R242", "Bittob-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Bittob11040/Affine_5DSW4cTwQt2U8rck6mFN1nNqoj37j1waqwszQDuz2zh9zC7z@0c04fe92 (R204 knobs on seed-earner non-live base; ≠ R204–R241 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane / ≠ R5 Genesis FullFT / ≠ R233–R241 / ≠ Bittoby1040 merge R2bl; n80 king=marsplan)"),
    # p2861: QUEUE#34 Everest-nonking×HiAlpha-GRPO — structural non-king base #11 (R204 knobs on everest12 seed-earner; n80 king=marsplan; ≠ R204–R242 / ≠ R233–R242 / ≠ R5 FullFT / ≠ R242 Bittob / ≠ H117/H130).
    ("mine-r243-everest-nonking-grpo-1", "R243", "Everest-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from everest12/affine-5EkhZHopy9CAoUhKmVTDsyGQi7Voo9gURYPnNDiMZX1pQZxp@a5ac5311 (R204 knobs on seed-earner non-live base; ≠ R204–R242 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob / ≠ R5 Genesis FullFT / ≠ R233–R242 / ≠ R242 Bittob-nonking / ≠ raw-everest H117 / ≠ Everest-FT H130; n80 king=marsplan)"),
    # p2862: QUEUE#35 Guass-nonking×HiAlpha-GRPO — structural non-king base #12 (R204 knobs on guass reign-14; n80 king=marsplan; ≠ R204–R243 / ≠ R233–R243 / ≠ R158–R164 guass-king / ≠ R5 FullFT / ≠ R243 Everest).
    ("mine-r244-guass-nonking-grpo-1", "R244", "Guass-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from ttttxxxxsada/Affine-5guassq3tu@e86758f5 (R204 knobs on reign-14 prior-crown non-live base; ≠ R204–R243 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest / ≠ R5 Genesis FullFT / ≠ R233–R243 / ≠ R158–R164 guass-king ladder / ≠ R243 Everest-nonking; n80 king=marsplan)"),
    # p2863: QUEUE#36 Afk1-nonking×HiAlpha-GRPO — structural non-king base #13 (R204 knobs on af-k1 reign-chain; n80 king=marsplan; ≠ R204–R244 / ≠ R233–R244 / ≠ R5 FullFT / ≠ R244 Guass).
    ("mine-r245-afk1-nonking-grpo-1", "R245", "Afk1-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from af-k1/Affine-5ECeJJpEMjW4pxM9eGyJ5ua3Sebfyr8kcVwLAdaiJLUC8pkW@ff6eb4bc (R204 knobs on reign-chain seed-earner non-live base; ≠ R204–R244 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass / ≠ R5 Genesis FullFT / ≠ R233–R244 / ≠ R244 Guass-nonking; n80 king=marsplan)"),
    # p2865: QUEUE#37 Awesome-nonking×HiAlpha-GRPO — structural non-king base #14 (R204 knobs on awesome reign-13; n80 king=marsplan; ≠ R204–R245 / ≠ R233–R245 / ≠ R165–R180 awesome-king / ≠ R5 FullFT / ≠ R245 Afk1).
    ("mine-r246-awesome-nonking-grpo-1", "R246", "Awesome-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from 0pentensor/Affine-5dflhtkufw-awesome-v11@450bdfc3 (R204 knobs on reign-13 prior-crown non-live base; ≠ R204–R245 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1 / ≠ R5 Genesis FullFT / ≠ R233–R245 / ≠ R165–R180 awesome-king ladder / ≠ R245 Afk1-nonking; n80 king=marsplan)"),
    # p2867: QUEUE#38 Legend-nonking×HiAlpha-GRPO — structural non-king base #15 (R204 knobs on legend reign-9; n80 king=marsplan; ≠ R204–R246 / ≠ R233–R246 / ≠ R159 legend-king / ≠ R5 FullFT / ≠ R246 Awesome).
    ("mine-r247-legend-nonking-grpo-1", "R247", "Legend-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from diceofgod/affine-5fjgc5jhxq-legend@d259cb38 (R204 knobs on reign-9 prior-crown non-live base; ≠ R204–R246 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome / ≠ R5 Genesis FullFT / ≠ R233–R246 / ≠ R159 legend-king ladder / ≠ R246 Awesome-nonking; n80 king=marsplan)"),
    # p2868: QUEUE#39 Thermopylae-nonking×HiAlpha-GRPO — structural non-king base #16 (R204 knobs on thermopylae reign-10; n80 king=marsplan; ≠ R204–R247 / ≠ R233–R247 / ≠ R160 thermopylae-king / ≠ R5 FullFT / ≠ R247 Legend).
    ("mine-r248-thermopylae-nonking-grpo-1", "R248", "Thermopylae-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from thermopylae-777/Affine-5eptsnvsre-v1@b5f748bf (R204 knobs on reign-10 prior-crown non-live base; ≠ R204–R247 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome/Legend / ≠ R5 Genesis FullFT / ≠ R233–R247 / ≠ R160 thermopylae-king ladder / ≠ R247 Legend-nonking; n80 king=marsplan)"),
    # p2869: QUEUE#40 Fjq-nonking×HiAlpha-GRPO — structural non-king base #17 (R204 knobs on dent1s2 fjq reign-7; n80 king=marsplan; ≠ R204–R248 / ≠ R233–R248 / ≠ R156–R157 fjq-king / ≠ R5 FullFT / ≠ R248 Thermopylae).
    ("mine-r249-fjq-nonking-grpo-1", "R249", "Fjq-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from dent1s2/Affine-5FjqRq3dGA-v1@cfd789c9 (R204 knobs on reign-7 prior-crown non-live base; ≠ R204–R248 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae / ≠ R5 Genesis FullFT / ≠ R233–R248 / ≠ R156–R157 fjq-king ladder / ≠ R248 Thermopylae-nonking; n80 king=marsplan)"),
    # p2870: QUEUE#41 Aftot-nonking×HiAlpha-GRPO — structural non-king base #18 (R204 knobs on aftot chal-00624 board challenger; n80 king=marsplan; ≠ R204–R249 / ≠ R233–R249 / ≠ R5 FullFT / ≠ R249 Fjq).
    ("mine-r250-aftot-nonking-grpo-1", "R250", "Aftot-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from aftot/Affine-5DqRzvm4BU-chpk200@1dc05c2d (R204 knobs on live chal-00624 board-challenger non-live base; ≠ R204–R249 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq / ≠ R5 Genesis FullFT / ≠ R233–R249 / ≠ R249 Fjq-nonking; n80 king=marsplan)"),
    # p2871: QUEUE#42 Leary-t3-nonking×HiAlpha-GRPO — structural non-king base #19 (R204 knobs on leary-criste chal-00626 queue#0; n80 king=marsplan; ≠ R204–R250 / ≠ R233–R250 / ≠ R5 FullFT / ≠ R250 Aftot / ≠ R184/R202 leary probes).
    ("mine-r251-leary-t3-nonking-grpo-1", "R251", "Leary-t3-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from leary-criste/affine-5g4yy75zuz-t3@1ee64fe6 (R204 knobs on chal-00626 board-queue non-live base; ≠ R204–R250 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot / ≠ R5 Genesis FullFT / ≠ R233–R250 / ≠ R250 Aftot-nonking / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2872/p2916: R252 promoted to QUEUE HEAD at vera reign21 (see top) — do not duplicate here.
    # ("mine-r252-vera-t4-nonking-grpo-1", "R252", "…"),
    # p2875: QUEUE#44 Crazyape-v3-nonking×HiAlpha-GRPO — structural non-king base #21 (R204 knobs on crazyape777 chal-00629 queue#2; n80 king=marsplan; ≠ R204–R252 / ≠ R233–R252 / ≠ R5 FullFT / ≠ R252 Vera-t4 / ≠ R197 crazyape-v9).
    ("mine-r253-crazyape-v3-nonking-grpo-1", "R253", "Crazyape-v3-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from crazyape777/Affine-5cawhezuhj-v3@411db7a5 (R204 knobs on chal-00629 board-queue non-live base; ≠ R204–R252 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4 / ≠ R5 Genesis FullFT / ≠ R233–R252 / ≠ R252 Vera-t4-nonking / ≠ R197 crazyape-v9 probe; n80 king=marsplan)"),
    # p2876: QUEUE#45 Pandora-st777-nonking×HiAlpha-GRPO — structural non-king base #22 (R204 knobs on pandora-box chal-00630 queue#3 st777; n80 king=marsplan; ≠ R204–R253 / ≠ R233–R253 / ≠ R5 FullFT / ≠ R253 Crazyape / ≠ R237 Pandora-ckpt300).
    ("mine-r254-pandora-st777-nonking-grpo-1", "R254", "Pandora-st777-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from pandora-box/Affine-5eqdtdzqle-st777@146c1999 (R204 knobs on chal-00630 board-queue non-live base; ≠ R204–R253 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/Diane/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3 / ≠ R5 Genesis FullFT / ≠ R233–R253 / ≠ R253 Crazyape-v3-nonking / ≠ R237 Pandora-ckpt300; n80 king=marsplan)"),
    # p2877: QUEUE#46 Diane-star-nonking×HiAlpha-GRPO — structural non-king base #23 (R204 knobs on diane613 chal-00631 queue#4 star; n80 king=marsplan; ≠ R204–R254 / ≠ R233–R254 / ≠ R5 FullFT / ≠ R254 Pandora-st777 / ≠ R241 Diane-seed / ≠ R183 Diane-star board).
    ("mine-r255-diane-star-nonking-grpo-1", "R255", "Diane-star-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from diane613/affine-5gedzafcvg-star@486ef178 (R204 knobs on chal-00631 board-queue non-live base; ≠ R204–R254 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777 / ≠ R5 Genesis FullFT / ≠ R233–R254 / ≠ R254 Pandora-st777-nonking / ≠ R241 Diane-seed / ≠ R183 Diane-star board probe; n80 king=marsplan)"),
    # p2878: QUEUE#47 Leary-t1-nonking×HiAlpha-GRPO — structural non-king base #24 (R204 knobs on leary-criste chal-00632 queue#5 t1; n80 king=marsplan; ≠ R204–R255 / ≠ R233–R255 / ≠ R5 FullFT / ≠ R255 Diane-star / ≠ R251 Leary-t3 / ≠ R184/R202).
    ("mine-r256-leary-t1-nonking-grpo-1", "R256", "Leary-t1-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from leary-criste/affine-5g4yy75zuz-t1@79e93491 (R204 knobs on chal-00632 board-queue non-live base; ≠ R204–R255 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar / ≠ R5 Genesis FullFT / ≠ R233–R255 / ≠ R255 Diane-star-nonking / ≠ R251 Leary-t3-nonking / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2879: QUEUE#48 Ammazon-sbs-v4-nonking×HiAlpha-GRPO — structural non-king base #25 (R204 knobs on ammazon chal-00633 queue#6 sbs-v4; n80 king=marsplan; ≠ R204–R256 / ≠ R233–R256 / ≠ R5 FullFT / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r257-ammazon-sbs-v4-nonking-grpo-1", "R257", "Ammazon-sbs-v4-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from ammazon/Affine-5dvqtektxx-sbs-v4@dac2b8c7 (R204 knobs on chal-00633 board-queue non-live base; ≠ R204–R256 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1 / ≠ R5 Genesis FullFT / ≠ R233–R256 / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2880: QUEUE#49 Sansaliu-v7-nonking×HiAlpha-GRPO — structural non-king base #26 (R204 knobs on Sansaliu chal-00634 queue#7 v7; n80 king=marsplan; ≠ R204–R257 / ≠ R233–R257 / ≠ R5 FullFT / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r258-sansaliu-v7-nonking-grpo-1", "R258", "Sansaliu-v7-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Sansaliu/Affine-5dm33ngrnj-v7@13a8a1f8 (R204 knobs on chal-00634 board-queue non-live base; ≠ R204–R257 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4 / ≠ R5 Genesis FullFT / ≠ R233–R257 / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2881: QUEUE#50 Michael-h2 — promoted to rent HEAD p2986 (live scoring chal-00636); do not re-rent duplicate.
    # ("mine-r259-michael-h2-nonking-grpo-1", "R259", "…"),
    # p2882: QUEUE#51 Elonmasky-ckp777 — promoted p2987 to rent HEAD (live chal-00637); duplicate entry removed.
    # ("mine-r260-elonmasky-ckp777-nonking-grpo-1", "R260", "…"),
    # p2883: QUEUE#52 Tok-happywolf18-nonking×HiAlpha-GRPO — structural non-king base #29 (R204 knobs on Tok331102 chal-00638 queue#11 happywolf18; n80 king=marsplan; ≠ R204–R260 / ≠ R233–R260 / ≠ R5 FullFT / ≠ R234 Tok-af10 / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r261-tok-happywolf18-nonking-grpo-1", "R261", "Tok-happywolf18-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Tok331102/affine-5EqYW8McUc-happywolf18@17148993 (R204 knobs on chal-00638 board-queue non-live base; ≠ R204–R260 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777 / ≠ R5 Genesis FullFT / ≠ R233–R260 / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2884: was QUEUE#53 Kevin-v5 — promoted to rent HEAD p3009 (live scoring chal-00639); do not re-rent duplicate.
    # ("mine-r262-kevin-v5-nonking-grpo-1", "R262", "Kevin-v5-nonking×HiAlpha-GRPO … see QUEUE HEAD"),
    # p2885: QUEUE#54 Tok-habibis19-nonking×HiAlpha-GRPO — structural non-king base #31 (R204 knobs on Tok331102 chal-00640 queue#13 habibis19; n80 king=marsplan; ≠ R204–R262 / ≠ R233–R262 / ≠ R5 FullFT / ≠ R192 Tok-habibis19-chal594 / ≠ R234 Tok-af10 / ≠ R261 Tok-happywolf18 / ≠ R262 Kevin-v5).
    ("mine-r263-tok-habibis19-nonking-grpo-1", "R263", "Tok-habibis19-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Tok331102/affine-5EqYW8McUc-habibis19@17340d3b (R204 knobs on chal-00640 board-queue non-live base; ≠ R204–R262 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5 / ≠ R5 Genesis FullFT / ≠ R233–R262 / ≠ R192 Tok-habibis19-chal594 / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2886: QUEUE#55 Talucampe-nonking×HiAlpha-GRPO — structural non-king base #32 (R204 knobs on Talucampe037 chal-00641 queue#14; n80 king=marsplan; ≠ R204–R263 / ≠ R233–R263 / ≠ R5 FullFT / ≠ R263 Tok-habibis19 / ≠ R262 Kevin-v5 / ≠ R261 Tok-happywolf18 / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r264-talucampe-nonking-grpo-1", "R264", "Talucampe-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Talucampe037/Affine-5f6xxabdmp@6afdc2a1 (R204 knobs on chal-00641 board-queue non-live base; ≠ R204–R263 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19 / ≠ R5 Genesis FullFT / ≠ R233–R263 / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),    # p2823: R216 lean-warm on mine-r160 GPUs 6–7 after R40 REFUTE+purge (isolated /root/r216; R209 keeps 4–5) — do not re-rent.

    # p2887: QUEUE#56 Athena-alloy-nonking×HiAlpha-GRPO — structural non-king base #33 (R204 knobs on athena2634 chal-00642 queue#15; n80 king=marsplan; ≠ R204–R264 / ≠ R233–R264 / ≠ R5 FullFT / ≠ R264 Talucampe / ≠ R263 Tok-habibis19 / ≠ R262 Kevin-v5 / ≠ R261 Tok-happywolf18 / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r265-athena-alloy-nonking-grpo-1", "R265", "Athena-alloy-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from athena2634/Affine-5h3msswruf-alloy@74a6ac4d (R204 knobs on chal-00642 board-queue non-live base; ≠ R204–R264 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe / ≠ R5 Genesis FullFT / ≠ R233–R264 / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2888: QUEUE#57 Llorite-tpc11-nonking×HiAlpha-GRPO — structural non-king base #34 (R204 knobs on llorite chal-00643 queue#16; n80 king=marsplan; ≠ R204–R265 / ≠ R233–R265 / ≠ R5 FullFT / ≠ R265 Athena-alloy / ≠ R264 Talucampe / ≠ R263 Tok-habibis19 / ≠ R262 Kevin-v5 / ≠ R261 Tok-happywolf18 / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r266-llorite-tpc11-nonking-grpo-1", "R266", "Llorite-tpc11-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from llorite/affine-5cjfxpsxn8-tpc11@285fea24 (R204 knobs on chal-00643 board-queue non-live base; ≠ R204–R265 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy / ≠ R5 Genesis FullFT / ≠ R233–R265 / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2889: QUEUE#58 Tok-af14-nonking×HiAlpha-GRPO — structural non-king base #35 (R204 knobs on Tok331102 chal-00644 queue#17; n80 king=marsplan; ≠ R204–R266 / ≠ R233–R266 / ≠ R5 FullFT / ≠ R266 Llorite-tpc11 / ≠ R265 Athena-alloy / ≠ R264 Talucampe / ≠ R263 Tok-habibis19 / ≠ R262 Kevin-v5 / ≠ R261 Tok-happywolf18 / ≠ R234 Tok-af10 / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r267-tok-af14-nonking-grpo-1", "R267", "Tok-af14-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Tok331102/affine-5EqYW8McUc-af14@888948cd (R204 knobs on chal-00644 board-queue non-live base; ≠ R204–R266 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11 / ≠ R5 Genesis FullFT / ≠ R233–R266 / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2890: QUEUE#59 Diane-sweet-nonking×HiAlpha-GRPO — structural non-king base #36 (R204 knobs on diane613 chal-00645 queue#18; n80 king=marsplan; ≠ R204–R267 / ≠ R233–R267 / ≠ R5 FullFT / ≠ R267 Tok-af14 / ≠ R266 Llorite-tpc11 / ≠ R265 Athena-alloy / ≠ R264 Talucampe / ≠ R263 Tok-habibis19 / ≠ R262 Kevin-v5 / ≠ R261 Tok-happywolf18 / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3).
    ("mine-r268-diane-sweet-nonking-grpo-1", "R268", "Diane-sweet-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from diane613/affine-5gedzafcvg-sweet@d77b3431 (R204 knobs on chal-00645 board-queue non-live base; ≠ R204–R267 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14 / ≠ R5 Genesis FullFT / ≠ R233–R267 / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2891: QUEUE#60 Magicworld-pizza-nonking×HiAlpha-GRPO — structural non-king base #37 (R204 knobs on magicworld7 chal-00647 queue#19; n80 king=marsplan; aurora storm 00646 HF404 skipped; ≠ R204–R268 / ≠ R233–R268 / ≠ R5 FullFT / ≠ R268 Diane-sweet / ≠ R267 Tok-af14 / ≠ R187 magicworld-venus).
    ("mine-r269-magicworld-pizza-nonking-grpo-1", "R269", "Magicworld-pizza-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from magicworld7/affine-5dtu4gucst-pizza@236eae33538b (R204 knobs on chal-00647 board-queue non-live base; ≠ R204–R268 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet / ≠ R5 Genesis FullFT / ≠ R233–R268 / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R234 Tok-af10 / ≠ R187 magicworld-venus / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2892: QUEUE#61 Thompsville-cgpb11-nonking×HiAlpha-GRPO — structural non-king base #38 (R204 knobs on thompsville chal-00648 queue#20; n80 king=marsplan; ≠ R204–R269 / ≠ R233–R269 / ≠ R5 FullFT / ≠ R269 Magicworld-pizza / ≠ R268 Diane-sweet).
    ("mine-r270-thompsville-cgpb11-nonking-grpo-1", "R270", "Thompsville-cgpb11-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from thompsville/affine-5dvegrgnsg-cgpb11@0724a3457148 (R204 knobs on chal-00648 board-queue non-live base; ≠ R204–R269 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza / ≠ R5 Genesis FullFT / ≠ R233–R269 / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),
    # p2893: QUEUE#62 Adsbasd-king-nonking×HiAlpha-GRPO — structural non-king base #39 (R204 knobs on adsbasd31badsf chal-00650 queue#22; TalentPigs-dog 00649 HF404 skipped; n80 king=marsplan; ≠ R204–R270 / ≠ R233–R270 / ≠ R5 FullFT / ≠ R270 Thompsville-cgpb11 / ≠ R269 Magicworld-pizza).
    ("mine-r271-adsbasd-king-nonking-grpo-1", "R271", "Adsbasd-king-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from adsbasd31badsf/affine-5ec3jw68ha-king@c8738c3f8c4b (R204 knobs on chal-00650 board-queue non-live base; ≠ R204–R270 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11 / ≠ R5 Genesis FullFT / ≠ R233–R270 / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2894: QUEUE#63 Saysth-r7-nonking×HiAlpha-GRPO — structural non-king base #40 (R204 knobs on saysth chal-00651 queue#23; n80 king=marsplan; ≠ R204–R271 / ≠ R233–R271 / ≠ R5 FullFT / ≠ R271 Adsbasd-king / ≠ R270 Thompsville-cgpb11).
    ("mine-r272-saysth-r7-nonking-grpo-1", "R272", "Saysth-r7-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from saysth/Affine-5ffaqdbhdh-r7@4559d4b6dc45 (R204 knobs on chal-00651 board-queue non-live base; ≠ R204–R271 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11/AdsbasdKing / ≠ R5 Genesis FullFT / ≠ R233–R271 / ≠ R271 Adsbasd-king-nonking / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2895: QUEUE#64 Wearetop-pa61s4q9-nonking×HiAlpha-GRPO — structural non-king base #41 (R204 knobs on wearetop chal-00652 queue#24; n80 king=marsplan; ≠ R204–R272 / ≠ R233–R272 / ≠ R5 FullFT / ≠ R272 Saysth-r7 / ≠ R271 Adsbasd-king).
    ("mine-r273-wearetop-pa61s4q9-nonking-grpo-1", "R273", "Wearetop-pa61s4q9-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from wearetop/Affine-5dvha3y7cd-pa61s4q9@5941a69a7495 (R204 knobs on chal-00652 board-queue non-live base; ≠ R204–R272 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11/AdsbasdKing/SaysthR7 / ≠ R5 Genesis FullFT / ≠ R233–R272 / ≠ R272 Saysth-r7-nonking / ≠ R271 Adsbasd-king-nonking / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2896: QUEUE#65 Wearetop-again1-nonking×HiAlpha-GRPO — structural non-king base #42 (R204 knobs on wearetop chal-00653 queue#25; n80 king=marsplan; ≠ R204–R273 / ≠ R233–R273 / ≠ R5 FullFT / ≠ R273 Wearetop-pa61s4q9 / ≠ R272 Saysth-r7).
    ("mine-r274-wearetop-again1-nonking-grpo-1", "R274", "Wearetop-again1-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from wearetop/affine-5gcl5uxakb-again1@c91884840f7f (R204 knobs on chal-00653 board-queue non-live base; ≠ R204–R273 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11/AdsbasdKing/SaysthR7/WearetopPa61s4q9 / ≠ R5 Genesis FullFT / ≠ R233–R273 / ≠ R273 Wearetop-pa61s4q9-nonking / ≠ R272 Saysth-r7-nonking / ≠ R271 Adsbasd-king-nonking / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2897: QUEUE#66 IntoLayer-v2-nonking×HiAlpha-GRPO — structural non-king base #43 (R204 knobs on IntoLayer chal-00654 queue#26; n80 king=marsplan; ≠ R204–R274 / ≠ R233–R274 / ≠ R5 FullFT / ≠ R274 Wearetop-again1 / ≠ R273 Wearetop-pa61s4q9).
    ("mine-r275-intolayer-v2-nonking-grpo-1", "R275", "IntoLayer-v2-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from IntoLayer/Affine-5g94ihdxwu-v2@bb5178f46c04 (R204 knobs on chal-00654 board-queue non-live base; ≠ R204–R274 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11/AdsbasdKing/SaysthR7/WearetopPa61s4q9/WearetopAgain1 / ≠ R5 Genesis FullFT / ≠ R233–R274 / ≠ R274 Wearetop-again1-nonking / ≠ R273 Wearetop-pa61s4q9-nonking / ≠ R272 Saysth-r7-nonking / ≠ R271 Adsbasd-king-nonking / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2898: QUEUE#67 Tok-happybaby15-nonking×HiAlpha-GRPO — structural non-king base #44 (R204 knobs on Tok331102 chal-00655 queue#28; n80 king=marsplan; ≠ R204–R275 / ≠ R233–R275 / ≠ R5 FullFT / ≠ R275 IntoLayer-v2 / ≠ R274 Wearetop-again1).
    ("mine-r276-tok-happybaby15-nonking-grpo-1", "R276", "Tok-happybaby15-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Tok331102/affine-5EqYW8McUc-happybaby15@f4306f87c1446f8b039d58895860dee8444e215e (R204 knobs on chal-00655 board-queue non-live base; ≠ R204–R275 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11/AdsbasdKing/SaysthR7/WearetopPa61s4q9/WearetopAgain1/IntoLayerV2 / ≠ R5 Genesis FullFT / ≠ R233–R275 / ≠ R275 IntoLayer-v2-nonking / ≠ R274 Wearetop-again1-nonking / ≠ R273 Wearetop-pa61s4q9-nonking / ≠ R272 Saysth-r7-nonking / ≠ R271 Adsbasd-king-nonking / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2904: QUEUE#68 Llorite-tpc12-nonking×HiAlpha-GRPO — structural non-king base #45 (R204 knobs on llorite chal-00656 queue#27; n80 king=marsplan-queen; ≠ R204–R276 / ≠ R233–R276 / ≠ R5 FullFT / ≠ R276 Tok-happybaby15 / ≠ R266 Llorite-tpc11).
    ("mine-r278-llorite-tpc12-nonking-grpo-1", "R278", "Llorite-tpc12-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from llorite/affine-5cjfxpsxn8-tpc12@8716fefb70e3ec553bbe1f6e0fad4a3c6bbf9d84 (R204 knobs on chal-00656 board-queue non-live base; ≠ R204–R276 marsplan/Genesis/Tok/Talent/Kevin/Pandora/Ckp333/Golden/isomsom/DianeSeed/Bittob/Everest/Guass/Afk1/Awesome/Legend/Thermopylae/Fjq/Aftot/LearyT3/VeraT4/CrazyapeV3/PandoraSt777/DianeStar/LearyT1/AmmazonSbsV4/SansaliuV7/MichaelH2/ElonmaskyCkp777/TokHappywolf18/KevinV5/TokHabibis19/Talucampe/AthenaAlloy/LloriteTpc11/TokAf14/DianeSweet/MagicworldPizza/ThompsvilleCgpb11/AdsbasdKing/SaysthR7/WearetopPa61s4q9/WearetopAgain1/IntoLayerV2/TokHappyBaby15 / ≠ R5 Genesis FullFT / ≠ R233–R276 / ≠ R276 Tok-happybaby15-nonking / ≠ R275 IntoLayer-v2-nonking / ≠ R274 Wearetop-again1-nonking / ≠ R273 Wearetop-pa61s4q9-nonking / ≠ R272 Saysth-r7-nonking / ≠ R271 Adsbasd-king-nonking / ≠ R270 Thompsville-cgpb11-nonking / ≠ R269 Magicworld-pizza-nonking / ≠ R268 Diane-sweet-nonking / ≠ R267 Tok-af14-nonking / ≠ R266 Llorite-tpc11-nonking / ≠ R265 Athena-alloy-nonking / ≠ R264 Talucampe-nonking / ≠ R263 Tok-habibis19-nonking / ≠ R262 Kevin-v5-nonking / ≠ R261 Tok-happywolf18-nonking / ≠ R260 Elonmasky-ckp777-nonking / ≠ R259 Michael-h2-nonking / ≠ R258 Sansaliu-v7-nonking / ≠ R257 Ammazon-sbs-v4-nonking / ≠ R256 Leary-t1-nonking / ≠ R251 Leary-t3-nonking / ≠ R255 Diane-star / ≠ R241 Diane-seed / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R184/R202 leary board probes; n80 king=marsplan)"),

    # p2905: QUEUE#69 Magicworld-earth-nonking×HiAlpha-GRPO — structural non-king base #46 (R204 knobs on magicworld7 chal-00657 queue#28; n80 king=marsplan-queen; ≠ R204–R278 / ≠ R269 pizza / ≠ R187 venus / ≠ R5 FullFT / ≠ R278 Llorite-tpc12).
    ("mine-r279-magicworld-earth-nonking-grpo-1", "R279", "Magicworld-earth-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from magicworld7/affine-5dtu4gucst-earth@5f53118abc78054e73d645b1a3d3881d9ab55941 (R204 knobs on chal-00657 board-queue non-live base; ≠ R204–R278 marsplan/nonking ladder / ≠ R278 Llorite-tpc12 / ≠ R276 Tok-happybaby15 / ≠ R275 IntoLayer-v2 / ≠ R274 Wearetop-again1 / ≠ R273 Wearetop-pa61s4q9 / ≠ R272 Saysth-r7 / ≠ R271 Adsbasd-king / ≠ R270 Thompsville-cgpb11 / ≠ R269 Magicworld-pizza / ≠ R268 Diane-sweet / ≠ R267 Tok-af14 / ≠ R266 Llorite-tpc11 / ≠ R265 Athena-alloy / ≠ R264 Talucampe / ≠ R263 Tok-habibis19 / ≠ R262 Kevin-v5 / ≠ R261 Tok-happywolf18 / ≠ R260 Elonmasky / ≠ R259 Michael-h2 / ≠ R258 Sansaliu / ≠ R257 Ammazon / ≠ R256 Leary-t1 / ≠ R251 Leary-t3 / ≠ R235 Talent-abc / ≠ R234 Tok-af10 / ≠ R187 magicworld-venus / ≠ R5 Genesis FullFT / ≠ R233–R278; n80 king=marsplan-queen)"),

    # p2925: QUEUE#70 Nerojimmy-ckp999-nonking×HiAlpha-GRPO — structural non-king base #47 (R204 knobs on nerojimmy chal-00660 queue#31; n80 king=marsplan-queen; ≠ R204–R279 / ≠ R238 Ckp333 / ≠ R5 FullFT; skipped 00658=R246 same-rev + 00659 our r172).
    ("mine-r280-nerojimmy-ckp999-nonking-grpo-1", "R280", "Nerojimmy-ckp999-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from nerojimmy/Affine-5fqbxvz29b-ckp999@1718a86bff8ccab579490820cd6466a025348b0d (R204 knobs on chal-00660 board-queue non-live base; ≠ R204–R279 marsplan/nonking ladder / ≠ R279 Magicworld-earth / ≠ R278 Llorite-tpc12 / ≠ R238 Ckp333 / ≠ R5 Genesis FullFT / ≠ R233–R279; n80 king=marsplan-queen)"),

    # p2926: QUEUE#71 Dora7-dance-nonking×HiAlpha-GRPO — structural non-king base #48 (R204 knobs on dora7 chal-00667 queue#32; n80 king=marsplan-queen; ≠ R204–R280 / ≠ R280 Nerojimmy-ckp999 / ≠ R5 FullFT).
    ("mine-r281-dora7-dance-nonking-grpo-1", "R281", "Dora7-dance-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from dora7/affine-5fhnbtexaw-dance@9a9e719c593000b951cba6789ba34b4725f1d9b5 (R204 knobs on chal-00667 board-queue non-live base; ≠ R204–R280 marsplan/nonking ladder / ≠ R280 Nerojimmy-ckp999 / ≠ R279 Magicworld-earth / ≠ R278 Llorite-tpc12 / ≠ R5 Genesis FullFT / ≠ R233–R280; n80 king=marsplan-queen)"),

    # p2927: QUEUE#72 Talucampe-sft-nonking×HiAlpha-GRPO — structural non-king base #49 (R204 knobs on Talucampe037 chal-00669 queue#33; n80 king=marsplan-queen; ≠ R204–R281 / ≠ R264 Talucampe-base@6afdc2a1 / ≠ R281 Dora7-dance / ≠ R5 FullFT).
    ("mine-r282-talucampe-sft-nonking-grpo-1", "R282", "Talucampe-sft-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Talucampe037/Affine-5f6xxabdmp-sft@74aeeffe16ca22aa8978e00e773762cce5df81f9 (R204 knobs on chal-00669 board-queue non-live base; ≠ R204–R281 marsplan/nonking ladder / ≠ R281 Dora7-dance / ≠ R280 Nerojimmy-ckp999 / ≠ R264 Talucampe-base / ≠ R5 Genesis FullFT / ≠ R233–R281; n80 king=marsplan-queen)"),

    # p2928: QUEUE#73 Tok-dirty20-nonking×HiAlpha-GRPO — structural non-king base #50 (R204 knobs on Tok331102 chal-00672 queue#34; n80 king=marsplan-queen; ≠ R204–R282 / ≠ prior Tok ladder af10/happywolf18/habibis19/af14/happybaby15 / ≠ R282 Talucampe-sft / ≠ R5 FullFT).
    ("mine-r283-tok-dirty20-nonking-grpo-1", "R283", "Tok-dirty20-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Tok331102/affine-5EqYW8McUc-dirty20@6540e5e2259089ddd881ce348218ca5797d20b5b (R204 knobs on chal-00672 board-queue non-live base; ≠ R204–R282 marsplan/nonking ladder / ≠ R282 Talucampe-sft / ≠ R281 Dora7-dance / ≠ R280 Nerojimmy-ckp999 / ≠ R276 Tok-happybaby15 / ≠ R267 Tok-af14 / ≠ R261 Tok-happywolf18 / ≠ R263 Tok-habibis19 / ≠ R234 Tok-af10 / ≠ R5 Genesis FullFT / ≠ R233–R282; n80 king=marsplan-queen)"),

    # p2929: QUEUE#74 Dent1s2-GNQK-nonking×HiAlpha-GRPO — structural non-king base #51 (R204 knobs on dent1s2 Affine-5GNQKpkkkC-v1@6ceabc8c intake chal-00569; n80 king=marsplan-queen; board exhausted at R283; ≠ R204–R283 / ≠ R249 Fjq / ≠ R196 screen-only / ≠ R5 FullFT).
    ("mine-r284-dent1s2-gnqk-nonking-grpo-1", "R284", "Dent1s2-GNQK-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from dent1s2/Affine-5GNQKpkkkC-v1@6ceabc8c561e071e3a5f2b28baaa15a3594b16ae (R204 knobs on intake chal-00569 non-live base; ≠ R204–R283 marsplan/nonking ladder / ≠ R283 Tok-dirty20 / ≠ R249 Fjq Affine-5FjqRq3dGA-v1@cfd789c9 / ≠ R196 screen-only / ≠ R5 Genesis FullFT / ≠ R233–R283; n80 king=marsplan-queen)"),

    # p2930: QUEUE#75 Bittoby-v1-nonking×HiAlpha-GRPO — structural non-king base #52 (R204 knobs on Bittoby1040 Affine-5cxncav2du-v1@55c84acf intake chal-00547; n80 king=marsplan-queen; board exhausted; ≠ R204–R284 / ≠ R242 Bittob11040 / ≠ R5 FullFT).
    ("mine-r285-bittoby-v1-nonking-grpo-1", "R285", "Bittoby-v1-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Bittoby1040/Affine-5cxncav2du-v1@55c84acfd0f5a0bc1c3505b73e626d2722b99848 (R204 knobs on intake chal-00547 non-live base; ≠ R204–R284 marsplan/nonking ladder / ≠ R284 Dent1s2-GNQK / ≠ R283 Tok-dirty20 / ≠ R242 Bittob11040@0c04fe92 / ≠ R5 Genesis FullFT / ≠ R233–R284; n80 king=marsplan-queen)"),

    # p2931: QUEUE#76 Elonmasky-jb13317k-nonking×HiAlpha-GRPO — structural non-king base #53 (R204 knobs on elonmasky Affine-5cj9mpkjrr-jb13317k@c131aa62 intake chal-00606; n80 king=marsplan-queen; board exhausted; ≠ R204–R285 / ≠ R260 Elonmasky-ckp777 / ≠ R195 screen-only / ≠ R5 FullFT).
    ("mine-r286-elonmasky-jb13317k-nonking-grpo-1", "R286", "Elonmasky-jb13317k-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from elonmasky/Affine-5cj9mpkjrr-jb13317k@c131aa6220c894e2c52056fd8db9ddcb5911e67a (R204 knobs on intake chal-00606 non-live base; ≠ R204–R285 marsplan/nonking ladder / ≠ R285 Bittoby-v1 / ≠ R260 Elonmasky-ckp777@9b4df167 / ≠ R195 screen-only / ≠ R5 Genesis FullFT / ≠ R233–R285; n80 king=marsplan-queen)"),

    # p2932: QUEUE#77 Windsword-testv1-nonking×HiAlpha-GRPO — structural non-king base #54 (R204 knobs on windsword8989 affine-5H8ewyxwJL-testv1@c3e8f63a intake chal-00620; n80 king=marsplan-queen; board exhausted; ≠ R204–R286 / ≠ R200 screen-only / ≠ R240 isomsom / ≠ R5 FullFT).
    ("mine-r287-windsword-testv1-nonking-grpo-1", "R287", "Windsword-testv1-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from windsword8989/affine-5H8ewyxwJL-testv1@c3e8f63a3c8f4a828200dcd1340150d5858858b9 (R204 knobs on intake chal-00620 non-live base; ≠ R204–R286 marsplan/nonking ladder / ≠ R286 Elonmasky-jb13317k / ≠ R200 screen-only / ≠ R240 isomsom-v11 / ≠ R5 Genesis FullFT / ≠ R233–R286; n80 king=marsplan-queen)"),

    # p2933: QUEUE#78 Shatoria-hope13-nonking×HiAlpha-GRPO — structural non-king base #55 (R204 knobs on Shatoria Affine-5ghntktyzq-hope13@0d40d787 intake chal-00612; n80 king=marsplan-queen; board exhausted; ≠ R204–R287 / ≠ R198 screen-only / ≠ R287 Windsword / ≠ R5 FullFT).
    ("mine-r288-shatoria-hope13-nonking-grpo-1", "R288", "Shatoria-hope13-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Shatoria/Affine-5ghntktyzq-hope13@0d40d787788c2aa6b8b68327912a6bcc00cf3a61 (R204 knobs on intake chal-00612 non-live base; ≠ R204–R287 marsplan/nonking ladder / ≠ R287 Windsword-testv1 / ≠ R198 screen-only / ≠ R5 Genesis FullFT / ≠ R233–R287; n80 king=marsplan-queen)"),

    # p2934: QUEUE#79 Crazyape-v9-nonking×HiAlpha-GRPO — structural non-king base #56 (R204 knobs on crazyape777 Affine-5dfvxyvetg-v9@b8a7cca0 intake chal-00610; n80 king=marsplan-queen; board exhausted; ≠ R204–R288 / ≠ R197 screen-only / ≠ R253 Crazyape-v3 / ≠ R288 Shatoria / ≠ R5 FullFT).
    ("mine-r289-crazyape-v9-nonking-grpo-1", "R289", "Crazyape-v9-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from crazyape777/Affine-5dfvxyvetg-v9@b8a7cca0cbd26bea7816c5f7ad119edef0607a61 (R204 knobs on intake chal-00610 non-live base; ≠ R204–R288 marsplan/nonking ladder / ≠ R288 Shatoria-hope13 / ≠ R253 Crazyape-v3 / ≠ R197 screen-only / ≠ R5 Genesis FullFT / ≠ R233–R288; n80 king=marsplan-queen)"),
    # p2935: QUEUE#80 Ichiro-chal672-nonking×HiAlpha-GRPO — structural non-king base #57 (R204 knobs on Ichiro1007 Affine-chal-00672@b3135f9a fresh HF; n80 king=marsplan-queen; intake exhausted; ≠ R204–R289 / ≠ R283 Tok-dirty20@6540e5e2 / ≠ R289 Crazyape-v9 / ≠ R5 FullFT).
    ("mine-r290-ichiro-chal672-nonking-grpo-1", "R290", "Ichiro-chal672-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00672@b3135f9a8eba4af1be1843dc2357e13b74976a2c (R204 knobs on fresh HF non-live base; ≠ R204–R289 marsplan/nonking ladder / ≠ R289 Crazyape-v9 / ≠ R283 Tok-dirty20@6540e5e2 / ≠ R288 Shatoria-hope13 / ≠ R5 Genesis FullFT / ≠ R233–R289; n80 king=marsplan-queen)"),

    # p2936: QUEUE#81 Ichiro-chal669-nonking×HiAlpha-GRPO — structural non-king base #58 (R204 knobs on Ichiro1007 Affine-chal-00669@fdd3604c fresh HF; n80 king=marsplan-queen; intake exhausted; ≠ R204–R290 / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R282 Talucampe-sft@74aeeffe / ≠ R5 FullFT).
    ("mine-r291-ichiro-chal669-nonking-grpo-1", "R291", "Ichiro-chal669-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00669@fdd3604c0855de23dbf097b7c92330ae6e4a8e69 (R204 knobs on fresh HF non-live base; ≠ R204–R290 marsplan/nonking ladder / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R282 Talucampe-sft@74aeeffe / ≠ R289 Crazyape-v9 / ≠ R5 Genesis FullFT / ≠ R233–R290; n80 king=marsplan-queen)"),
    ("mine-r292-ichiro-chal630-nonking-grpo-1", "R292", "Ichiro-chal630-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00630@b1dc77edc7a8ca413f82a77fee7788453a062671 (R204 knobs on fresh HF non-live base; named for live scoring chal-00630; sha ≠ pandora-st777@146c19998d08=R254; ≠ R204–R291 marsplan/nonking ladder / ≠ R291 Ichiro-chal669@fdd3604c / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R254 PandoraSt777 / ≠ R5 Genesis FullFT / ≠ R233–R291; n80 king=marsplan-queen)"),
    ("mine-r293-ichiro-chal629-nonking-grpo-1", "R293", "Ichiro-chal629-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00629@c9d0463f2b8ae3f87e1b4f695eaee22d8b23836a (R204 knobs on fresh HF Crazyape-v3 mirror; sha ≠ crazyape-v3@411db7a5=R253; ≠ R204–R292 marsplan/nonking ladder / ≠ R292 Ichiro-chal630@b1dc77ed / ≠ R291 Ichiro-chal669@fdd3604c / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R253 CrazyapeV3 / ≠ R289 Crazyape-v9 / ≠ R5 Genesis FullFT / ≠ R233–R292; n80 king=marsplan-queen)"),

    ("mine-r294-ichiro-chal631-nonking-grpo-1", "R294", "Ichiro-chal631-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00631@2037218bc61afd3f92e4f3069a8b0bcb0b8ebb84 (R204 knobs on fresh HF Diane-star live-scoring mirror; sha ≠ diane-star@486ef178=R255; ≠ R204–R293 marsplan/nonking ladder / ≠ R293 Ichiro-chal629@c9d0463f / ≠ R292 Ichiro-chal630@b1dc77ed / ≠ R291 Ichiro-chal669@fdd3604c / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R255 DianeStar / ≠ R289 Crazyape-v9 / ≠ R5 Genesis FullFT / ≠ R233–R293; n80 king=marsplan-queen)"),

    # p2942: QUEUE#85 Ichiro-chal658-nonking×HiAlpha-GRPO — structural non-king base #62 (R204 knobs on Ichiro1007 Affine-chal-00658@9a2eefce fresh HF Awesome-v11 mirror; sha ≠ awesome-v11@450bdfc3=R246; n80 king=marsplan-queen; board/intake exhausted; ≠ R204–R294 / ≠ R294 Ichiro-chal631 / ≠ R246 AwesomeV11 / ≠ R5 FullFT).
    ("mine-r295-ichiro-chal658-nonking-grpo-1", "R295", "Ichiro-chal658-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00658@9a2eefcebbb2dffddbd715f80409a0e709bc8d5d (R204 knobs on fresh HF Awesome-v11 mirror; sha ≠ awesome-v11@450bdfc3=R246; ≠ R204–R294 marsplan/nonking ladder / ≠ R294 Ichiro-chal631@2037218b / ≠ R293 Ichiro-chal629@c9d0463f / ≠ R292 Ichiro-chal630@b1dc77ed / ≠ R291 Ichiro-chal669@fdd3604c / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R246 AwesomeV11 / ≠ R289 Crazyape-v9 / ≠ R5 Genesis FullFT / ≠ R233–R294; n80 king=marsplan-queen)"),

    # p2943: QUEUE#86 Ichiro-chal667-nonking×HiAlpha-GRPO — structural non-king base #63 (R204 knobs on Ichiro1007 Affine-chal-00667@ea40b183 fresh HF Dora7-dance mirror; sha ≠ dora7@9a9e719c=R281; n80 king=marsplan-queen; board/intake exhausted; ≠ R204–R295 / ≠ R281 Dora7 / ≠ R5 FullFT).
    ("mine-r296-ichiro-chal667-nonking-grpo-1", "R296", "Ichiro-chal667-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00667@ea40b1839d29adc6d4a3e0c6b6e2ed153ad1491d (R204 knobs on fresh HF Dora7-dance mirror; sha ≠ dora7@9a9e719c=R281; ≠ R204–R295 marsplan/nonking ladder / ≠ R295 Ichiro-chal658@9a2eefce / ≠ R294 Ichiro-chal631@2037218b / ≠ R293 Ichiro-chal629@c9d0463f / ≠ R292 Ichiro-chal630@b1dc77ed / ≠ R291 Ichiro-chal669@fdd3604c / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R281 Dora7-dance / ≠ R289 Crazyape-v9 / ≠ R5 Genesis FullFT / ≠ R233–R295; n80 king=marsplan-queen)"),

    # p2944: QUEUE#87 Ichiro-chal595-nonking×HiAlpha-GRPO — structural non-king base #64 (R204 knobs on Ichiro1007 Affine-chal-00595@2a0980eb fresh HF Talucampe-base mirror; sha ≠ Talucampe037@6afdc2a1=R264; n80 king=marsplan-queen; board/intake exhausted; ≠ R204–R296 / ≠ R264 Talucampe / ≠ R5 FullFT).
    ("mine-r297-ichiro-chal595-nonking-grpo-1", "R297", "Ichiro-chal595-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00595@2a0980eb5eea1bb2323d801825afb90b1dd1a684 (R204 knobs on fresh HF Talucampe-base mirror; sha ≠ Talucampe037@6afdc2a1=R264; ≠ R204–R296 marsplan/nonking ladder / ≠ R296 Ichiro-chal667@ea40b183 / ≠ R295 Ichiro-chal658@9a2eefce / ≠ R294 Ichiro-chal631@2037218b / ≠ R293 Ichiro-chal629@c9d0463f / ≠ R292 Ichiro-chal630@b1dc77ed / ≠ R291 Ichiro-chal669@fdd3604c / ≠ R290 Ichiro-chal672@b3135f9a / ≠ R264 Talucampe-base / ≠ R282 Talucampe-sft / ≠ R5 Genesis FullFT / ≠ R233–R296; n80 king=marsplan-queen)"),

    # p2949: QUEUE#88 wire-fix — R298 scripts+prebuilt+HF existed since p2945 but were missing from rent QUEUE + bootstrap case (would stamp needs_axis_uploader). Ichiro-chal660-nonking×HiAlpha-GRPO on Ichiro1007@4883b8c6 Nerojimmy mirror ≠ R280@1718a86b.
    ("mine-r298-ichiro-chal660-nonking-grpo-1", "R298", "Ichiro-chal660-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00660@4883b8c61674c162d6e0cb52146f358e54388bce (R204 knobs on fresh HF Nerojimmy-ckp999 mirror; sha ≠ nerojimmy@1718a86b=R280; ≠ R204–R297 marsplan/nonking ladder / ≠ R297 Ichiro-chal595@2a0980eb / ≠ R280 Nerojimmy-ckp999 / ≠ R5 Genesis FullFT / ≠ R233–R297; n80 king=marsplan-queen)"),

    # p2950: QUEUE#89 Ichiro-chal634-nonking×HiAlpha-GRPO — structural non-king base #66 (R204 knobs on Ichiro1007 Affine-chal-00634@aed34c41 fresh HF Sansaliu-v7 mirror; sha ≠ Sansaliu@13a8a1f8=R258; n80 king=marsplan-queen; board/intake exhausted; ≠ R204–R298 / ≠ R258 Sansaliu / ≠ R5 FullFT).
    ("mine-r299-ichiro-chal634-nonking-grpo-1", "R299", "Ichiro-chal634-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00634@aed34c41fc61147190d26a7e5a67fb62c5502482 (R204 knobs on fresh HF Sansaliu-v7 mirror; sha ≠ Sansaliu@13a8a1f8=R258; ≠ R204–R298 marsplan/nonking ladder / ≠ R298 Ichiro-chal660@4883b8c6 / ≠ R297 Ichiro-chal595@2a0980eb / ≠ R258 Sansaliu-v7 / ≠ R5 Genesis FullFT / ≠ R233–R298; n80 king=marsplan-queen)"),

    # p2951: QUEUE#90 Ichiro-chal633-nonking×HiAlpha-GRPO — structural non-king base #67 (R204 knobs on Ichiro1007 Affine-chal-00633@a00f9d95 fresh HF Ammazon-sbs-v4 mirror of live scoring chal-00633; sha ≠ ammazon@dac2b8c7=R257; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R299 / ≠ R257 Ammazon / ≠ R299 Ichiro-chal634 / ≠ R5 FullFT).
    ("mine-r300-ichiro-chal633-nonking-grpo-1", "R300", "Ichiro-chal633-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00633@a00f9d9533ca93902fc2f80f80b6af338fc910c0 (R204 knobs on fresh HF Ammazon-sbs-v4 mirror of live chal-00633; sha ≠ ammazon@dac2b8c7=R257; ≠ R204–R299 marsplan/nonking ladder / ≠ R299 Ichiro-chal634@aed34c41 / ≠ R257 Ammazon-sbs-v4 / ≠ R5 Genesis FullFT / ≠ R233–R299; n80 king=marsplan-queen)"),

    # p2952: QUEUE#91 Ichiro-chal636-nonking×HiAlpha-GRPO — structural non-king base #68 (R204 knobs on Ichiro1007 Affine-chal-00636@f752ccc7 fresh HF Michael-h2 mirror of chal-00636; sha ≠ michael@6a68af69=R259; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R300 / ≠ R259 Michael-h2 / ≠ R300 Ichiro-chal633 / ≠ R5 FullFT).
    ("mine-r301-ichiro-chal636-nonking-grpo-1", "R301", "Ichiro-chal636-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00636@f752ccc7cc9edd1203a4e7333cf70d8d433db27d (R204 knobs on fresh HF Michael-h2 mirror of chal-00636; sha ≠ michael-chan-000@6a68af69=R259; ≠ R204–R300 marsplan/nonking ladder / ≠ R300 Ichiro-chal633@a00f9d95 / ≠ R259 Michael-h2 / ≠ R5 Genesis FullFT / ≠ R233–R300; n80 king=marsplan-queen)"),

    # p2953: QUEUE#92 Ichiro-chal637-nonking×HiAlpha-GRPO — structural non-king base #69 (R204 knobs on Ichiro1007 Affine-chal-00637@2999b8af fresh HF Elonmasky-ckp777 mirror of chal-00637; sha ≠ elonmasky@9b4df167=R260; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R301 / ≠ R260 Elonmasky-ckp777 / ≠ R301 Ichiro-chal636 / ≠ R5 FullFT).
    ("mine-r302-ichiro-chal637-nonking-grpo-1", "R302", "Ichiro-chal637-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00637@2999b8afe867fe52d032ed3523e34588d0e28846 (R204 knobs on fresh HF Elonmasky-ckp777 mirror of chal-00637; sha ≠ elonmasky@9b4df167=R260; ≠ R204–R301 marsplan/nonking ladder / ≠ R301 Ichiro-chal636@f752ccc7 / ≠ R260 Elonmasky-ckp777 / ≠ R5 Genesis FullFT / ≠ R233–R301; n80 king=marsplan-queen)"),

    # p2954: QUEUE#93 Ichiro-chal638-nonking×HiAlpha-GRPO — structural non-king base #70 (R204 knobs on Ichiro1007 Affine-chal-00638@4e9fcec2 fresh HF Tok-happywolf18 mirror of chal-00638; sha ≠ Tok331102@17148993=R261; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R302 / ≠ R302 Ichiro-chal637 / ≠ R261 Tok-happywolf18 / ≠ R5 FullFT).
    ("mine-r303-ichiro-chal638-nonking-grpo-1", "R303", "Ichiro-chal638-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00638@4e9fcec22a426f7169eddfdba16d30d9603c3d54 (R204 knobs on fresh HF Tok-happywolf18 mirror of chal-00638; sha ≠ Tok331102@17148993=R261; ≠ R204–R302 marsplan/nonking ladder / ≠ R302 Ichiro-chal637@2999b8af / ≠ R261 Tok-happywolf18 / ≠ R5 Genesis FullFT / ≠ R233–R302; n80 king=marsplan-queen)"),
    # p2955: QUEUE#94 Ichiro-chal639-nonking×HiAlpha-GRPO — structural non-king base #71 (R204 knobs on Ichiro1007 Affine-chal-00639@fa755131 fresh HF Kevin-v5 mirror of chal-00639; sha ≠ kevin954@b6575907=R262; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R303 / ≠ R303 Ichiro-chal638 / ≠ R262 Kevin-v5 / ≠ R5 FullFT).
    ("mine-r304-ichiro-chal639-nonking-grpo-1", "R304", "Ichiro-chal639-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00639@fa755131b21868a7d17ecee8c479701e9764d793 (R204 knobs on fresh HF Kevin-v5 mirror of chal-00639; sha ≠ kevin954@b6575907=R262; ≠ R204–R303 marsplan/nonking ladder / ≠ R303 Ichiro-chal638@4e9fcec2 / ≠ R262 Kevin-v5 / ≠ R5 Genesis FullFT / ≠ R233–R303; n80 king=marsplan-queen)"),

    # p2956: QUEUE#95 Ichiro-chal640-nonking×HiAlpha-GRPO — structural non-king base #72 (R204 knobs on Ichiro1007 Affine-chal-00640@92bf6e21 fresh HF Tok-habibis19 mirror of chal-00640; sha ≠ Tok331102@17340d3b=R263; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R304 / ≠ R304 Ichiro-chal639 / ≠ R263 Tok-habibis19 / ≠ R5 FullFT).
    ("mine-r305-ichiro-chal640-nonking-grpo-1", "R305", "Ichiro-chal640-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00640@92bf6e2171529890c65b51a845212cd86a1470a9 (R204 knobs on fresh HF Tok-habibis19 mirror of chal-00640; sha ≠ Tok331102@17340d3b=R263; ≠ R204–R304 marsplan/nonking ladder / ≠ R304 Ichiro-chal639@fa755131 / ≠ R263 Tok-habibis19 / ≠ R5 Genesis FullFT / ≠ R233–R304; n80 king=marsplan-queen)"),

    # p2957: QUEUE#96 Ichiro-chal641-nonking×HiAlpha-GRPO — structural non-king base #73 (R204 knobs on Ichiro1007 Affine-chal-00641@105cc660 fresh HF Talucampe-base mirror of chal-00641; sha ≠ Talucampe037@6afdc2a1=R264; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R305 / ≠ R305 Ichiro-chal640 / ≠ R264 Talucampe-base / ≠ R297 Ichiro-chal595 / ≠ R5 FullFT).
    ("mine-r306-ichiro-chal641-nonking-grpo-1", "R306", "Ichiro-chal641-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00641@105cc660ca0f044da3ad4005da2f3ef525e301b6 (R204 knobs on fresh HF Talucampe-base mirror of chal-00641; sha ≠ Talucampe037@6afdc2a1=R264; ≠ R204–R305 marsplan/nonking ladder / ≠ R305 Ichiro-chal640@92bf6e21 / ≠ R264 Talucampe-base / ≠ R297 Ichiro-chal595 / ≠ R5 Genesis FullFT / ≠ R233–R305; n80 king=marsplan-queen)"),
    # p2958: QUEUE#97 Ichiro-chal642-nonking×HiAlpha-GRPO — structural non-king base #74 (R204 knobs on Ichiro1007 Affine-chal-00642@5b49e2b5 fresh HF Athena-alloy mirror of chal-00642; sha ≠ athena2634@74a6ac4d=R265; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R306 / ≠ R306 Ichiro-chal641 / ≠ R265 Athena-alloy / ≠ R5 FullFT).
    ("mine-r307-ichiro-chal642-nonking-grpo-1", "R307", "Ichiro-chal642-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00642@5b49e2b52b6c213b6200b47fc0d47e5e2e756aff (R204 knobs on fresh HF Athena-alloy mirror of chal-00642; sha ≠ athena2634@74a6ac4d=R265; ≠ R204–R306 marsplan/nonking ladder / ≠ R306 Ichiro-chal641@105cc660 / ≠ R265 Athena-alloy / ≠ R5 Genesis FullFT / ≠ R233–R306; n80 king=marsplan-queen)"),


    # p2959: QUEUE#98 Ichiro-chal643-nonking×HiAlpha-GRPO — structural non-king base #75 (R204 knobs on Ichiro1007 Affine-chal-00643@199f4e1e fresh HF Llorite-tpc11 mirror of chal-00643; sha ≠ llorite@285fea24=R266; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R307 / ≠ R307 Ichiro-chal642 / ≠ R266 Llorite-tpc11 / ≠ R5 FullFT).
    ("mine-r308-ichiro-chal643-nonking-grpo-1", "R308", "Ichiro-chal643-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00643@199f4e1e26218f9670f82bc3eb2798ccf5219258 (R204 knobs on fresh HF Llorite-tpc11 mirror of chal-00643; sha ≠ llorite@285fea24=R266; ≠ R204–R307 marsplan/nonking ladder / ≠ R307 Ichiro-chal642@5b49e2b5 / ≠ R266 Llorite-tpc11 / ≠ R5 Genesis FullFT / ≠ R233–R307; n80 king=marsplan-queen)"),


    # p2960: QUEUE#99 Ichiro-chal644-nonking×HiAlpha-GRPO — structural non-king base #76 (R204 knobs on Ichiro1007 Affine-chal-00644@0224adbb fresh HF Tok-af14 mirror of chal-00644; sha ≠ Tok331102@888948cd=R267; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R308 / ≠ R308 Ichiro-chal643 / ≠ R267 Tok-af14 / ≠ R5 FullFT).
    ("mine-r309-ichiro-chal644-nonking-grpo-1", "R309", "Ichiro-chal644-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00644@0224adbb78892930558c4baf0e3f507b1d8eb776 (R204 knobs on fresh HF Tok-af14 mirror of chal-00644; sha ≠ Tok331102@888948cd=R267; ≠ R204–R308 marsplan/nonking ladder / ≠ R308 Ichiro-chal643@199f4e1e / ≠ R267 Tok-af14 / ≠ R5 Genesis FullFT / ≠ R233–R308; n80 king=marsplan-queen)"),

    # p2961: QUEUE#100 Ichiro-chal645-nonking×HiAlpha-GRPO — structural non-king base #77 (R204 knobs on Ichiro1007 Affine-chal-00645@9be5db28 fresh HF Diane-sweet mirror of chal-00645; sha ≠ diane613@d77b3431=R268; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R309 / ≠ R309 Ichiro-chal644 / ≠ R268 Diane-sweet / ≠ R5 FullFT).
    ("mine-r310-ichiro-chal645-nonking-grpo-1", "R310", "Ichiro-chal645-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00645@9be5db289fa0bf5873bd6c30c812607732bcd210 (R204 knobs on fresh HF Diane-sweet mirror of chal-00645; sha ≠ diane613@d77b3431=R268; ≠ R204–R309 marsplan/nonking ladder / ≠ R309 Ichiro-chal644@0224adbb / ≠ R268 Diane-sweet / ≠ R5 Genesis FullFT / ≠ R233–R309; n80 king=marsplan-queen)"),



    # p2962: QUEUE#101 Ichiro-chal647-nonking×HiAlpha-GRPO — structural non-king base #78 (R204 knobs on Ichiro1007 Affine-chal-00647@ea1925ef fresh HF Magicworld-pizza mirror of chal-00647; sha ≠ magicworld7@236eae33538b=R269; aurora storm 00646 HF404 skipped; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R310 / ≠ R310 Ichiro-chal645 / ≠ R269 Magicworld-pizza / ≠ R5 FullFT).
    ("mine-r311-ichiro-chal647-nonking-grpo-1", "R311", "Ichiro-chal647-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00647@ea1925ef8ae4919a14d5c576710f2145605fd12c (R204 knobs on fresh HF Magicworld-pizza mirror of chal-00647; sha ≠ magicworld7@236eae33538b=R269; ≠ R204–R310 marsplan/nonking ladder / ≠ R310 Ichiro-chal645@9be5db28 / ≠ R269 Magicworld-pizza / ≠ R5 Genesis FullFT / ≠ R233–R310; n80 king=marsplan-queen)"),

    # p2963: QUEUE#102 Ichiro-chal648-nonking×HiAlpha-GRPO — structural non-king base #79 (R204 knobs on Ichiro1007 Affine-chal-00648@6b2f6ecd fresh HF Thompsville-cgpb11 mirror of chal-00648; sha ≠ thompsville@0724a3457148=R270; TalentPigs-dog 00649 HF404 skipped; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R311 / ≠ R311 Ichiro-chal647 / ≠ R270 Thompsville-cgpb11 / ≠ R5 FullFT).
    ("mine-r312-ichiro-chal648-nonking-grpo-1", "R312", "Ichiro-chal648-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00648@6b2f6ecd7fa67609df8a045fabce35af35089c0c (R204 knobs on fresh HF Thompsville-cgpb11 mirror of chal-00648; sha ≠ thompsville@0724a3457148=R270; ≠ R204–R311 marsplan/nonking ladder / ≠ R311 Ichiro-chal647@ea1925ef / ≠ R270 Thompsville-cgpb11 / ≠ R5 Genesis FullFT / ≠ R233–R311; n80 king=marsplan-queen)"),

    # p2964: QUEUE#103 Ichiro-chal650-nonking×HiAlpha-GRPO — structural non-king base #80 (R204 knobs on Ichiro1007 Affine-chal-00650@96b27c2a fresh HF Adsbasd-king mirror of chal-00650; sha ≠ adsbasd31badsf@c8738c3f8c4b=R271; TalentPigs-dog 00649 HF404 skipped; after R312 chal648; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R312 / ≠ R312 Ichiro-chal648 / ≠ R271 Adsbasd-king / ≠ R5 FullFT).
    ("mine-r313-ichiro-chal650-nonking-grpo-1", "R313", "Ichiro-chal650-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00650@96b27c2a1bc8f49089072d366da7b5d36a00db38 (R204 knobs on fresh HF Adsbasd-king mirror of chal-00650; sha ≠ adsbasd31badsf@c8738c3f8c4b=R271; ≠ R204–R312 marsplan/nonking ladder / ≠ R312 Ichiro-chal648@6b2f6ecd / ≠ R271 Adsbasd-king / ≠ R5 Genesis FullFT / ≠ R233–R312; n80 king=marsplan-queen)"),

    # p2965: QUEUE#104 Ichiro-chal651-nonking×HiAlpha-GRPO — structural non-king base #81 (R204 knobs on Ichiro1007 Affine-chal-00651@a8a3b4b2 fresh HF Saysth-r7 mirror of chal-00651; sha ≠ saysth@4559d4b6dc45=R272; after R313 chal650; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R313 / ≠ R313 Ichiro-chal650 / ≠ R272 Saysth-r7 / ≠ R5 FullFT).
    ("mine-r314-ichiro-chal651-nonking-grpo-1", "R314", "Ichiro-chal651-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00651@a8a3b4b29105f45d18414f72d3970a830938437d (R204 knobs on fresh HF Saysth-r7 mirror of chal-00651; sha ≠ saysth@4559d4b6dc45=R272; ≠ R204–R313 marsplan/nonking ladder / ≠ R313 Ichiro-chal650@96b27c2a / ≠ R272 Saysth-r7 / ≠ R5 Genesis FullFT / ≠ R233–R313; n80 king=marsplan-queen)"),

    # p2966: QUEUE#105 Ichiro-chal652-nonking×HiAlpha-GRPO — structural non-king base #82 (R204 knobs on Ichiro1007 Affine-chal-00652@9e13c180 fresh HF Wearetop-pa61s4q9 mirror of chal-00652; sha ≠ wearetop@5941a69a7495=R273; after R314 chal651; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R314 / ≠ R314 Ichiro-chal651 / ≠ R273 Wearetop-pa61s4q9 / ≠ R5 FullFT).
    ("mine-r315-ichiro-chal652-nonking-grpo-1", "R315", "Ichiro-chal652-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00652@9e13c180da39509c4ed79bff6e74091923c519b6 (R204 knobs on fresh HF Wearetop-pa61s4q9 mirror of chal-00652; sha ≠ wearetop@5941a69a7495=R273; ≠ R204–R314 marsplan/nonking ladder / ≠ R314 Ichiro-chal651@a8a3b4b2 / ≠ R273 Wearetop-pa61s4q9 / ≠ R5 Genesis FullFT / ≠ R233–R314; n80 king=marsplan-queen)"),
    # p2967: QUEUE#106 Ichiro-chal653-nonking×HiAlpha-GRPO — structural non-king base #83 (R204 knobs on Ichiro1007 Affine-chal-00653@1132a16d fresh HF Wearetop-again1 mirror of chal-00653; sha ≠ wearetop@c91884840f7f=R274; after R315 chal652; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R315 / ≠ R315 Ichiro-chal652 / ≠ R274 Wearetop-again1 / ≠ R5 FullFT).
    ("mine-r316-ichiro-chal653-nonking-grpo-1", "R316", "Ichiro-chal653-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00653@1132a16dccc39541be87ca3deeda490a87d7735e (R204 knobs on fresh HF Wearetop-again1 mirror of chal-00653; sha ≠ wearetop@c91884840f7f=R274; ≠ R204–R315 marsplan/nonking ladder / ≠ R315 Ichiro-chal652@9e13c180 / ≠ R274 Wearetop-again1 / ≠ R5 Genesis FullFT / ≠ R233–R315; n80 king=marsplan-queen)"),

    # p2968: QUEUE#107 Ichiro-chal654-nonking×HiAlpha-GRPO — structural non-king base #84 (R204 knobs on Ichiro1007 Affine-chal-00654@36637a6d fresh HF IntoLayer-v2 mirror of chal-00654; sha ≠ IntoLayer@bb5178f46c04=R275; after R316 chal653; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R316 / ≠ R316 Ichiro-chal653 / ≠ R275 IntoLayer-v2 / ≠ R5 FullFT).
    ("mine-r317-ichiro-chal654-nonking-grpo-1", "R317", "Ichiro-chal654-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00654@36637a6da38b8a83f9ab17398d1cfe9f76c0b00c (R204 knobs on fresh HF IntoLayer-v2 mirror of chal-00654; sha ≠ IntoLayer@bb5178f46c04=R275; ≠ R204–R316 marsplan/nonking ladder / ≠ R316 Ichiro-chal653@1132a16d / ≠ R275 IntoLayer-v2 / ≠ R5 Genesis FullFT / ≠ R233–R316; n80 king=marsplan-queen)"),

    # p2969: QUEUE#108 Ichiro-chal655-nonking×HiAlpha-GRPO — structural non-king base #85 (R204 knobs on Ichiro1007 Affine-chal-00655@3a734155 fresh HF Tok-happybaby15 mirror of chal-00655; sha ≠ Tok331102@f4306f87c144=R276; after R317 chal654; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R317 / ≠ R317 Ichiro-chal654 / ≠ R276 Tok-happybaby15 / ≠ R5 FullFT).
    ("mine-r318-ichiro-chal655-nonking-grpo-1", "R318", "Ichiro-chal655-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00655@3a734155024b756980cb5a75229398f9cd8026e7 (R204 knobs on fresh HF Tok-happybaby15 mirror of chal-00655; sha ≠ Tok331102@f4306f87c144=R276; ≠ R204–R317 marsplan/nonking ladder / ≠ R317 Ichiro-chal654@36637a6d / ≠ R276 Tok-happybaby15 / ≠ R5 Genesis FullFT / ≠ R233–R317; n80 king=marsplan-queen)"),

    # p2970: QUEUE#109 Ichiro-chal656-nonking×HiAlpha-GRPO — structural non-king base #86 (R204 knobs on Ichiro1007 Affine-chal-00656@7b71fafa fresh HF Llorite-tpc12 mirror of chal-00656; sha ≠ llorite@8716fefb70e3=R278; after R318 chal655; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R318 / ≠ R318 Ichiro-chal655 / ≠ R278 Llorite-tpc12 / ≠ R5 FullFT).
    ("mine-r319-ichiro-chal656-nonking-grpo-1", "R319", "Ichiro-chal656-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00656@7b71fafa99571ab7b060fa6f1e938d57a619d84f (R204 knobs on fresh HF Llorite-tpc12 mirror of chal-00656; sha ≠ llorite@8716fefb70e3=R278; ≠ R204–R318 marsplan/nonking ladder / ≠ R318 Ichiro-chal655@3a734155 / ≠ R278 Llorite-tpc12 / ≠ R5 Genesis FullFT / ≠ R233–R318; n80 king=marsplan-queen)"),

    # p2971: QUEUE#110 Ichiro-chal657-nonking×HiAlpha-GRPO — structural non-king base #87 (R204 knobs on Ichiro1007 Affine-chal-00657@3d363d17 fresh HF Magicworld-earth mirror of chal-00657; sha ≠ magicworld7@5f53118a=R279; after R319 chal656; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R319 / ≠ R319 Ichiro-chal656 / ≠ R279 Magicworld-earth / ≠ R5 FullFT).
    ("mine-r320-ichiro-chal657-nonking-grpo-1", "R320", "Ichiro-chal657-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00657@3d363d17eb7e55c48994663e33031e8d8e324325 (R204 knobs on fresh HF Magicworld-earth mirror of chal-00657; sha ≠ magicworld7@5f53118a=R279; ≠ R204–R319 marsplan/nonking ladder / ≠ R319 Ichiro-chal656@7b71fafa / ≠ R279 Magicworld-earth / ≠ R5 Genesis FullFT / ≠ R233–R319; n80 king=marsplan-queen)"),

    # p2972: QUEUE#111 Ichiro-chal627-nonking×HiAlpha-GRPO — structural non-king base #88 (R204 knobs on Ichiro1007 Affine-chal-00627@7cadd0bc fresh HF Vera-t4 mirror of chal-00627; sha ≠ vera6@f44f6a37=R252; after R320 chal657; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R320 / ≠ R320 Ichiro-chal657 / ≠ R252 Vera-t4 / ≠ R5 FullFT).
    ("mine-r321-ichiro-chal627-nonking-grpo-1", "R321", "Ichiro-chal627-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00627@7cadd0bc80aea3453bc3dcbc0bb524187b205d6d (R204 knobs on fresh HF Vera-t4 mirror of chal-00627; sha ≠ vera6@f44f6a37=R252; ≠ R204–R320 marsplan/nonking ladder / ≠ R320 Ichiro-chal657@3d363d17 / ≠ R252 Vera-t4 / ≠ R5 Genesis FullFT / ≠ R233–R320; n80 king=marsplan-queen)"),

    # p2973: QUEUE#112 Ichiro-chal626-nonking×HiAlpha-GRPO — structural non-king base #89 (R204 knobs on Ichiro1007 Affine-chal-00626@bbb433c1 fresh HF Leary-t3 mirror of chal-00626; sha ≠ leary-criste@1ee64fe6=R251; after R321 chal627; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R321 / ≠ R321 Ichiro-chal627 / ≠ R251 Leary-t3 / ≠ R5 FullFT).
    ("mine-r322-ichiro-chal626-nonking-grpo-1", "R322", "Ichiro-chal626-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00626@bbb433c192bf66826be3185bcd99044881e7bae5 (R204 knobs on fresh HF Leary-t3 mirror of chal-00626; sha ≠ leary-criste@1ee64fe6=R251; ≠ R204–R321 marsplan/nonking ladder / ≠ R321 Ichiro-chal627@7cadd0bc / ≠ R251 Leary-t3 / ≠ R5 Genesis FullFT / ≠ R233–R321; n80 king=marsplan-queen)"),

    # p2974: QUEUE#113 Ichiro-chal620-nonking×HiAlpha-GRPO — structural non-king base #90 (R204 knobs on Ichiro1007 Affine-chal-00620@776012ac fresh HF Windsword-testv1 mirror of chal-00620; sha ≠ windsword8989@c3e8f63a=R287; after R322 chal626; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R322 / ≠ R322 Ichiro-chal626 / ≠ R287 Windsword-testv1 / ≠ R5 FullFT).
    ("mine-r323-ichiro-chal620-nonking-grpo-1", "R323", "Ichiro-chal620-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00620@776012ac3fe2d5b4e34adbce3f84a58ed3be6eb9 (R204 knobs on fresh HF Windsword-testv1 mirror of chal-00620; sha ≠ windsword8989@c3e8f63a=R287; ≠ R204–R322 marsplan/nonking ladder / ≠ R322 Ichiro-chal626@bbb433c1 / ≠ R287 Windsword-testv1 / ≠ R5 Genesis FullFT / ≠ R233–R322; n80 king=marsplan-queen)"),

    # p2975: QUEUE#114 Ichiro-chal612-nonking×HiAlpha-GRPO — structural non-king base #91 (R204 knobs on Ichiro1007 Affine-chal-00612@62a4ba01 fresh HF Shatoria-hope13 mirror of chal-00612; sha ≠ Shatoria@0d40d787=R288; after R323 chal620; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R323 / ≠ R323 Ichiro-chal620 / ≠ R288 Shatoria-hope13 / ≠ R5 FullFT).
    ("mine-r324-ichiro-chal612-nonking-grpo-1", "R324", "Ichiro-chal612-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00612@62a4ba016c2338263c65bb29e3055e94efa34c5e (R204 knobs on fresh HF Shatoria-hope13 mirror of chal-00612; sha ≠ Shatoria@0d40d787=R288; ≠ R204–R323 marsplan/nonking ladder / ≠ R323 Ichiro-chal620@776012ac / ≠ R288 Shatoria-hope13 / ≠ R5 Genesis FullFT / ≠ R233–R323; n80 king=marsplan-queen)"),
    ("mine-r325-ichiro-chal618-nonking-grpo-1", "R325", "Ichiro-chal618-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00618@ffd8051d41e2af6fcd9bf458841f214d71313104 (R204 knobs on fresh HF Isomsom-v11 mirror of chal-00618; sha ≠ isomsom@5bfe4b6c=R240; ≠ R204–R324 marsplan/nonking ladder / ≠ R324 Ichiro-chal612@62a4ba01 / ≠ R240 isomsom-v11 / ≠ R5 Genesis FullFT / ≠ R233–R324; n80 king=marsplan-queen)"),

    # p2977: QUEUE#116 Ichiro-chal599-nonking×HiAlpha-GRPO — structural non-king base #93 (R204 knobs on Ichiro1007 Affine-chal-00599@c26afeb4 fresh HF; after R325 chal618; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R325 / ≠ R325 Ichiro-chal618@ffd8051d / ≠ R240 isomsom / ≠ R5 FullFT).
    ("mine-r326-ichiro-chal599-nonking-grpo-1", "R326", "Ichiro-chal599-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00599@c26afeb4e3f860fc39ec1080c2e7e4aa48344b4c (R204 knobs on fresh HF Ichiro mirror of chal-00599; ≠ R204–R325 marsplan/nonking ladder / ≠ R325 Ichiro-chal618@ffd8051d / ≠ R240 isomsom-v11 / ≠ R5 Genesis FullFT / ≠ R233–R325; n80 king=marsplan-queen)"),

    # p2978: QUEUE#117 Ichiro-chal598-nonking×HiAlpha-GRPO — structural non-king base #94 (R204 knobs on Ichiro1007 Affine-chal-00598@a5a41ad6 fresh HF Crazyape-v3 mirror of chal-00598; sha ≠ crazyape777@411db7a5=R253; after R326 chal599; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R326 / ≠ R326 Ichiro-chal599@c26afeb4 / ≠ R253 Crazyape-v3 / ≠ R5 FullFT).
    ("mine-r327-ichiro-chal598-nonking-grpo-1", "R327", "Ichiro-chal598-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00598@a5a41ad62fb0b4c475abcf99d5b8c333152f3001 (R204 knobs on fresh HF Ichiro mirror of chal-00598 / crazyape777-v3; sha ≠ crazyape777@411db7a5=R253; ≠ R204–R326 marsplan/nonking ladder / ≠ R326 Ichiro-chal599@c26afeb4 / ≠ R253 Crazyape-v3 / ≠ R5 Genesis FullFT / ≠ R233–R326; n80 king=marsplan-queen)"),

    # p2979: QUEUE#118 Ichiro-chal565-nonking×HiAlpha-GRPO — structural non-king base #95 (R204 knobs on Ichiro1007 Affine-chal-00565@9fcbc8c74e52 fresh HF Thermopylae mirror of chal-00565; orig thermopylae-777 HF404; sha ≠ thermopylae@b5f748bf=R248 / ≠ R160; after R327 chal598; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R327 / ≠ R327 Ichiro-chal598@a5a41ad6 / ≠ R248 Thermopylae / ≠ R160 / ≠ R5 FullFT).
    ("mine-r328-ichiro-chal565-nonking-grpo-1", "R328", "Ichiro-chal565-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00565@9fcbc8c74e525c373954ea72cb67057118dcf040 (R204 knobs on fresh HF Ichiro mirror of chal-00565 / thermopylae-777; orig HF404; sha ≠ thermopylae@b5f748bf=R248; ≠ R204–R327 marsplan/nonking ladder / ≠ R327 Ichiro-chal598@a5a41ad6 / ≠ R248 Thermopylae / ≠ R160 thermopylae-king / ≠ R5 Genesis FullFT / ≠ R233–R327; n80 king=marsplan-queen)"),

    # p2980: QUEUE#119 Ichiro-chal544-nonking×HiAlpha-GRPO — structural non-king base #96 (R204 knobs on Ichiro1007 Affine-chal-00544@b8b6556d fresh HF Fjq mirror of chal-00544; orig dent1s2 HF404; sha ≠ dent1s2@cfd789c9=R249; after R328 chal565; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R328 / ≠ R328 Ichiro-chal565@9fcbc8c74e52 / ≠ R249 Fjq / ≠ R156–R157 / ≠ R5 FullFT).
    ("mine-r329-ichiro-chal544-nonking-grpo-1", "R329", "Ichiro-chal544-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00544@b8b6556d8ec7cc5960a03cbdd3bbe0f1ceaed542 (R204 knobs on fresh HF Ichiro mirror of chal-00544 / dent1s2-fjq; orig HF404; sha ≠ dent1s2@cfd789c9=R249; ≠ R204–R328 marsplan/nonking ladder / ≠ R328 Ichiro-chal565@9fcbc8c74e52 / ≠ R249 Fjq / ≠ R156–R157 fjq-king / ≠ R5 Genesis FullFT / ≠ R233–R328; n80 king=marsplan-queen)"),


    # p2981: QUEUE#120 Ichiro-chal551-nonking×HiAlpha-GRPO — structural non-king base #97 (R204 knobs on Ichiro1007 Affine-chal-00551@324a109a7beb fresh HF Tok-happybaby15 mirror of intake chal-00551; sha ≠ Tok331102@f4306f87c144=R276 / ≠ R318@3a734155; after R329 chal544; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R329 / ≠ R329 Ichiro-chal544@b8b6556d / ≠ R276 Tok-happybaby15 / ≠ R318 Ichiro-chal655 / ≠ R5 FullFT).
    ("mine-r330-ichiro-chal551-nonking-grpo-1", "R330", "Ichiro-chal551-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00551@324a109a7beb7281dbdbea74d280a8ff4f43d279 (R204 knobs on fresh HF Ichiro mirror of intake chal-00551 / Tok-happybaby15; sha ≠ Tok331102@f4306f87c144=R276; ≠ R204–R329 marsplan/nonking ladder / ≠ R329 Ichiro-chal544@b8b6556d / ≠ R318 Ichiro-chal655@3a734155 / ≠ R276 Tok-happybaby15 / ≠ R5 Genesis FullFT / ≠ R233–R329; n80 king=marsplan-queen)"),


    # p2982: QUEUE#121 Ichiro-chal541-nonking×HiAlpha-GRPO — structural non-king base #98 (R204 knobs on Ichiro1007 Affine-chal-00541@c9fe17e8e0a8 fresh HF Adsbasd-king mirror of intake chal-00541; sha ≠ adsbasd31badsf@c8738c3f=R271 / ≠ R313@96b27c2a; after R330 chal551; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R330 / ≠ R330 Ichiro-chal551@324a109a7beb / ≠ R271 Adsbasd-king / ≠ R313 Ichiro-chal650 / ≠ R5 FullFT).
    ("mine-r331-ichiro-chal541-nonking-grpo-1", "R331", "Ichiro-chal541-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00541@c9fe17e8e0a8035048c0090149d6b59ccf366876 (R204 knobs on fresh HF Ichiro mirror of intake chal-00541 / adsbasd-king; sha ≠ adsbasd31badsf@c8738c3f8c4b=R271; ≠ R204–R330 marsplan/nonking ladder / ≠ R330 Ichiro-chal551@324a109a7beb / ≠ R313 Ichiro-chal650@96b27c2a / ≠ R271 Adsbasd-king / ≠ R5 Genesis FullFT / ≠ R233–R330; n80 king=marsplan-queen)"),

    # p2983: QUEUE#122 Ichiro-chal547-nonking×HiAlpha-GRPO — structural non-king base #99 (R204 knobs on Ichiro1007 Affine-chal-00547@6d518562849d1471 fresh HF Bittoby-v1 mirror of intake chal-00547; sha ≠ Bittoby1040@55c84acf=R285; after R331 chal541; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R331 / ≠ R331 Ichiro-chal541@c9fe17e8e0a8 / ≠ R285 Bittoby-v1 / ≠ R5 FullFT).
    ("mine-r332-ichiro-chal547-nonking-grpo-1", "R332", "Ichiro-chal547-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00547@6d518562849d1471bd2237ccc6bb41873f33a439 (R204 knobs on fresh HF Ichiro mirror of intake chal-00547 / Bittoby-v1; sha ≠ Bittoby1040@55c84acfd0f5=R285; ≠ R204–R331 marsplan/nonking ladder / ≠ R331 Ichiro-chal541@c9fe17e8e0a8 / ≠ R285 Bittoby-v1 / ≠ R5 Genesis FullFT / ≠ R233–R331; n80 king=marsplan-queen)"),

    # p2984: QUEUE#123 Ichiro-chal610-nonking×HiAlpha-GRPO — structural non-king base #100 (R204 knobs on Ichiro1007 Affine-chal-00610@2815558fdd23702a7b30499719d245339313ea5e fresh HF Crazyape-v9 mirror of intake chal-00610; sha ≠ crazyape777@b8a7cca0=R289; after R332 chal547; n80 king=marsplan-queen; 0×8 stock; ≠ R204–R332 / ≠ R332 Ichiro-chal547@6d518562849d1471 / ≠ R289 Crazyape-v9 / ≠ R253 Crazyape-v3 / ≠ R5 FullFT).
    ("mine-r333-ichiro-chal610-nonking-grpo-1", "R333", "Ichiro-chal610-nonking×HiAlpha-GRPO α=128 r=16 G=4 lr=5e-6 @6144 max_steps=200 from Ichiro1007/Affine-chal-00610@2815558fdd23702a7b30499719d245339313ea5e (R204 knobs on fresh HF Ichiro mirror of intake chal-00610 / crazyape777 Affine-5dfvxyvetg-v9; sha ≠ crazyape777@b8a7cca0cbd26bea=R289; ≠ R204–R332 marsplan/nonking ladder / ≠ R332 Ichiro-chal547@6d518562849d1471 / ≠ R289 Crazyape-v9 / ≠ R253 Crazyape-v3 / ≠ R5 Genesis FullFT / ≠ R233–R332; n80 king=marsplan-queen)"),
    # p3006: R335 moved up after R225 (was QUEUE#124 here).

    # ("mine-r216-marsplan-hialpha-hilr-longctx-1", "R216", "marsplan×HiAlpha×HiLR×LongCtx α=128 r=16 G=4 lr=2e-5 16384/1024 from live reign-16 (R176 recipe on marsplan; ≠ R206 HiLR@6144 / R208 LongCtx@5e-6 / R209 HiRank / R176 awesome parent; n80 king=marsplan)"),
    # p2813: R209 lean-warm on mine-r160 GPUs 4–5 after R39 SIGNAL_POS_BELOW+purge (isolated /root/r209; R40 keeps 6–7) — do not re-rent.
    # ("mine-r209-marsplan-hialpha-hirank-1", "R209", "marsplan×HiAlpha×HiRank α=128 r=64 G=4 lr=5e-6 @6144 from live reign-16 (≠ R204 r=16 / R207 HiRank×HiLR / R40 ckp333×LongCtx; n80 king=marsplan)"),
    ("mine-r47-ckp333-bigg-1", "R47", "ckp333×BigG G=16 demoted after reign16 (R39 ckp333 × R27 BigG; ≠ R41 Talent×BigG / R45 Diane×BigG / R46 Golden×BigG / R44 HiLR / R40 LongCtx / R39 G=4 / R27 Tok)"),
    ("mine-r163-guass-hialpha-longctx-1", "R163", "guass×HiAlpha×LongCtx α=128 r=16 G=4 lr=5e-6 16384/1024 demoted after reign15 (≠ R158@6144 / R161 BigG@6144 / R162 HiLR; n80 was guass)"),
    ("mine-r164-guass-hialpha-bigg-hilr-1", "R164", "guass×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 demoted after reign15 (≠ R161@5e-6 / R162 G=4 / R158 G=4@5e-6; n80 was guass)"),
    # Non-HiLR parent×SIGNAL axes (R19/R24) before demoted HiLR-primary R34.
    # ("mine-r35-talent-longctx-1", "R35", "Talent×LongCtx max_len=16384 (R19 SIGNAL_POS × R24 LongCtx)"),  # p2756 lean on crown
    # p2759: R37 lean-warm on mine-r160 GPUs 6–7 after R176 FALSE_PROBE+purge; R177 keeps 4–5 — do not re-rent.
    # ("mine-r37-golden-longctx-1", "R37", "Golden×LongCtx max_len=16384 (R22 golden × R24 LongCtx SIGNAL; ≠ R22@6144)"),
    ("mine-r38-diane-longctx-1", "R38", "Diane×LongCtx max_len=16384 (R23 diane × R24 LongCtx SIGNAL; ≠ R23@6144)"),
    ("mine-r41-talent-bigg-1", "R41", "Talent×BigG G=16 (R19 SIGNAL_POS × R27 G isolate; ≠ R35 LongCtx / R36 HiLR / R27 Tok)"),
    ("mine-r45-diane-bigg-1", "R45", "Diane×BigG G=16 (R23 diane × R27 BigG; ≠ R41 Talent×BigG / R43 Diane×HiLR / R38 LongCtx / R27 Tok)"),
    ("mine-r46-golden-bigg-1", "R46", "Golden×BigG G=16 (R22 golden × R27 BigG; ≠ R41 Talent×BigG / R45 Diane×BigG / R42 Golden×HiLR / R37 LongCtx / R27 Tok)"),
    ("mine-r57-longctx-bigg-1", "R57", "Tok LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R24×R27 compound; ≠ R34 LongCtx×HiLR / R48 BigG×HiLR / R24 / R27)"),
    ("mine-r58-talent-longctx-bigg-1", "R58", "Talent×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R19×R24×R27; ≠ R57 Tok / R53 Talent×LongCtx×HiLR / R41 Talent×BigG@6144 / R35 LongCtx / R49 BigG×HiLR)"),
    ("mine-r59-diane-longctx-bigg-1", "R59", "Diane×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R23×R24×R27; ≠ R58 Talent / R57 Tok / R54 Diane×LongCtx×HiLR / R45 Diane×BigG@6144 / R38 LongCtx / R50 Diane×BigG×HiLR)"),
    ("mine-r60-golden-longctx-bigg-1", "R60", "Golden×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R22×R24×R27; ≠ R59 Diane / R58 Talent / R57 Tok / R55 Golden×LongCtx×HiLR / R46 Golden×BigG@6144 / R37 LongCtx / R51 BigG×HiLR)"),
    ("mine-r61-ckp333-longctx-bigg-1", "R61", "ckp333×LongCtx×BigG 16384/1024 G=16 lr=5e-6 (R39/R40×R24×R27; ≠ R60 Golden / R59 Diane / R58 Talent / R57 Tok / R56 ckp333×LongCtx×HiLR / R47 ckp333×BigG@6144 / R40 LongCtx@G=4 / R52 BigG×HiLR)"),
    # Demoted: R28 HiLR REFUTED (p2294) — keep in queue but after SIGNAL compounds.
    ("mine-r34-longctx-hilr-1", "R34", "Tok LongCtx×HiLR max_len=16384 lr=2e-5 (R24×R28 compound)"),
    ("mine-r36-talent-hilr-1", "R36", "Talent×HiLR lr=2e-5 (R19 SIGNAL_POS × R28 HiLR; ≠ R35 LongCtx)"),
    ("mine-r42-golden-hilr-1", "R42", "Golden×HiLR lr=2e-5 (R22 golden × R28 HiLR; ≠ R37 LongCtx / R28 Tok / R36 Talent×HiLR)"),
    ("mine-r43-diane-hilr-1", "R43", "Diane×HiLR lr=2e-5 (R23 diane × R28 HiLR; ≠ R38 LongCtx / R42 Golden×HiLR / R36 Talent×HiLR)"),
    ("mine-r44-ckp333-hilr-1", "R44", "ckp333×HiLR lr=2e-5 (R39 ckp333 × R28 HiLR; ≠ R39@5e-6 / R40 LongCtx / R42 Golden×HiLR)"),
    ("mine-r48-bigg-hilr-1", "R48", "Tok BigG×HiLR G=16 lr=2e-5 (R27×R28 compound; ≠ R27 G-only / R28 lr-only / R34 LongCtx×HiLR / R3b)"),
    ("mine-r49-talent-bigg-hilr-1", "R49", "Talent×BigG×HiLR G=16 lr=2e-5 (R41×R36/R28 compound; ≠ R41 G-only / R36 HiLR@G=4 / R48 Tok BigG×HiLR / R35 LongCtx)"),
    ("mine-r50-diane-bigg-hilr-1", "R50", "Diane×BigG×HiLR G=16 lr=2e-5 (R45×R43/R28 compound; ≠ R45 G-only / R43 HiLR@G=4 / R49 Talent BigG×HiLR / R48 Tok / R38 LongCtx)"),
    ("mine-r51-golden-bigg-hilr-1", "R51", "Golden×BigG×HiLR G=16 lr=2e-5 (R46×R42/R28 compound; ≠ R46 G-only / R42 HiLR@G=4 / R50 Diane BigG×HiLR / R49 Talent / R48 Tok / R37 LongCtx)"),
    ("mine-r52-ckp333-bigg-hilr-1", "R52", "ckp333×BigG×HiLR G=16 lr=2e-5 (R47×R44/R28 compound; ≠ R47 G-only / R44 HiLR@G=4 / R51 Golden BigG×HiLR / R50 Diane / R49 Talent / R48 Tok / R40 LongCtx)"),
    ("mine-r53-talent-longctx-hilr-1", "R53", "Talent×LongCtx×HiLR 16384/1024 lr=2e-5 (R35×R36/R28 compound; ≠ R35 LongCtx@5e-6 / R36 HiLR@6144 / R34 Tok LongCtx×HiLR / R49 Talent×BigG×HiLR)"),
    ("mine-r54-diane-longctx-hilr-1", "R54", "Diane×LongCtx×HiLR 16384/1024 lr=2e-5 (R38×R43/R28 compound; ≠ R38 LongCtx@5e-6 / R43 HiLR@6144 / R34 Tok LongCtx×HiLR / R50 Diane×BigG×HiLR / R53 Talent×LongCtx×HiLR)"),
    ("mine-r55-golden-longctx-hilr-1", "R55", "Golden×LongCtx×HiLR 16384/1024 lr=2e-5 (R37×R42/R28 compound; ≠ R37 LongCtx@5e-6 / R42 HiLR@6144 / R34 Tok LongCtx×HiLR / R51 Golden×BigG×HiLR / R53 Talent×LongCtx×HiLR / R54 Diane×LongCtx×HiLR)"),
    ("mine-r56-ckp333-longctx-hilr-1", "R56", "ckp333×LongCtx×HiLR 16384/1024 lr=2e-5 (R40×R44/R28 compound; ≠ R40 LongCtx@5e-6 / R44 HiLR@6144 / R34 Tok LongCtx×HiLR / R52 ckp333×BigG×HiLR / R53–R55 LongCtx×HiLR parents)"),
    ("mine-r62-longctx-bigg-hilr-1", "R62", "Tok LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R24×R27×R28; ≠ R57@5e-6 / R34 LongCtx×HiLR@G=4 / R48 BigG×HiLR@6144 / R24 / R27 / R28)"),
    ("mine-r63-talent-longctx-bigg-hilr-1", "R63", "Talent×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R19×R24×R27×R28; ≠ R62 Tok triple / R58@5e-6 / R53 LongCtx×HiLR@G=4 / R49 BigG×HiLR@6144 / R41 BigG / R35 LongCtx)"),
    ("mine-r64-diane-longctx-bigg-hilr-1", "R64", "Diane×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R23×R24×R27×R28; ≠ R63 Talent triple / R62 Tok / R59@5e-6 / R54 LongCtx×HiLR@G=4 / R50 BigG×HiLR@6144 / R45 BigG / R38 LongCtx)"),
    ("mine-r65-golden-longctx-bigg-hilr-1", "R65", "Golden×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R22×R24×R27×R28; ≠ R64 Diane triple / R63 Talent / R62 Tok / R60@5e-6 / R55 LongCtx×HiLR@G=4 / R51 BigG×HiLR@6144 / R46 BigG / R37 LongCtx)"),
    ("mine-r66-ckp333-longctx-bigg-hilr-1", "R66", "ckp333×LongCtx×BigG×HiLR 16384/1024 G=16 lr=2e-5 (R39/R40×R24×R27×R28; ≠ R65 Golden triple / R64 Diane / R63 Talent / R62 Tok / R61@5e-6 / R56 LongCtx×HiLR@G=4 / R52 BigG×HiLR@6144 / R47 BigG / R40 LongCtx)"),
    ("mine-r80-talent-hialpha-1", "R80", "Talent×HiAlpha α=128 r=16 lr=5e-6 @6144 (R19×R30; ≠ R19@α32 / R30 Tok / R36 Talent×HiLR / R35 LongCtx / R74 Tok×HiAlpha×HiLR)"),
    ("mine-r81-talent-hialpha-longctx-1", "R81", "Talent×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R80×R24; ≠ R80@6144 / R35@α32 / R73 Tok×HiAlpha×LongCtx / R53 Talent×LongCtx×HiLR / R19@α32)"),
    ("mine-r82-talent-hialpha-hilr-1", "R82", "Talent×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R80×R36/R28; ≠ R80@5e-6 / R36@α32 / R74 Tok×HiAlpha×HiLR / R81 LongCtx / R30 Tok)"),
    ("mine-r83-talent-hialpha-bigg-1", "R83", "Talent×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R80×R41/R27; ≠ R80@G4 / R41@α32 / R75 Tok×HiAlpha×BigG / R82 HiLR / R81 LongCtx / R30 Tok)"),
    ("mine-r84-talent-hialpha-longctx-hilr-1", "R84", "Talent×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R81×R82/R28; ≠ R81@5e-6 / R82@6144 / R76 Tok×HiAlpha×LongCtx×HiLR / R53@α32 / R80@6144)"),
    ("mine-r85-talent-hialpha-longctx-bigg-1", "R85", "Talent×HiAlpha×LongCtx×BigG α=128 r=16 16384/1024 G=16 lr=5e-6 (R81×R83/R27; ≠ R81@G4 / R83@6144 / R78 Tok×HiAlpha×LongCtx×BigG / R58@α32 / R84 HiLR)"),
    ("mine-r86-talent-hialpha-longctx-bigg-hilr-1", "R86", "Talent×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 16384/1024 G=16 lr=2e-5 (R85×R84/R28; ≠ R85@5e-6 / R84@G4 / R79 Tok quintuple / R63@α32 / R83@6144)"),
    ("mine-r87-diane-hialpha-1", "R87", "Diane×HiAlpha α=128 r=16 lr=5e-6 @6144 (R23×R30; ≠ R23@α32 / R80 Talent×HiAlpha / R30 Tok / R43 Diane×HiLR / R38 LongCtx)"),
    ("mine-r88-golden-hialpha-1", "R88", "Golden×HiAlpha α=128 r=16 lr=5e-6 @6144 (R22×R30; ≠ R22@α32 / R87 Diane×HiAlpha / R80 Talent×HiAlpha / R30 Tok / R42 Golden×HiLR / R37 LongCtx)"),
    ("mine-r89-ckp333-hialpha-1", "R89", "ckp333×HiAlpha α=128 r=16 lr=5e-6 @6144 (R39×R30; ≠ R39@α32 / R88 Golden×HiAlpha / R87 Diane×HiAlpha / R80 Talent×HiAlpha / R30 Tok / R44 ckp333×HiLR / R40 LongCtx)"),
    ("mine-r90-diane-hialpha-longctx-1", "R90", "Diane×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R87×R24; ≠ R87@6144 / R81 Talent×HiAlpha×LongCtx / R73 Tok×HiAlpha×LongCtx / R38 Diane×LongCtx@α32 / R88 Golden×HiAlpha / R89 ckp333×HiAlpha / R30 Tok)"),
    ("mine-r91-golden-hialpha-longctx-1", "R91", "Golden×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R88×R24; ≠ R88@6144 / R90 Diane×HiAlpha×LongCtx / R81 Talent×HiAlpha×LongCtx / R73 Tok×HiAlpha×LongCtx / R37 Golden×LongCtx@α32 / R89 ckp333×HiAlpha / R30 Tok)"),
    ("mine-r92-ckp333-hialpha-longctx-1", "R92", "ckp333×HiAlpha×LongCtx α=128 r=16 16384/1024 lr=5e-6 (R89×R24; ≠ R89@6144 / R91 Golden×HiAlpha×LongCtx / R90 Diane×HiAlpha×LongCtx / R81 Talent×HiAlpha×LongCtx / R73 Tok×HiAlpha×LongCtx / R40 ckp333×LongCtx@α32 / R30 Tok)"),
    ("mine-r93-diane-hialpha-hilr-1", "R93", "Diane×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R87×R43/R28; ≠ R87@5e-6 / R43@α32 / R90 LongCtx / R82 Talent×HiAlpha×HiLR / R74 Tok×HiAlpha×HiLR / R30 Tok)"),
    ("mine-r94-golden-hialpha-hilr-1", "R94", "Golden×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R88×R42/R28; ≠ R88@5e-6 / R42@α32 / R91 LongCtx / R93 Diane×HiAlpha×HiLR / R82 Talent×HiAlpha×HiLR / R74 Tok×HiAlpha×HiLR / R30 Tok)"),
    ("mine-r95-ckp333-hialpha-hilr-1", "R95", "ckp333×HiAlpha×HiLR α=128 r=16 lr=2e-5 @6144 (R89×R44/R28; ≠ R89@5e-6 / R44@α32 / R92 LongCtx / R94 Golden×HiAlpha×HiLR / R93 Diane×HiAlpha×HiLR / R82 Talent×HiAlpha×HiLR / R74 Tok×HiAlpha×HiLR / R30 Tok)"),
    ("mine-r96-diane-hialpha-bigg-1", "R96", "Diane×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R87×R45/R27; ≠ R87@G4 / R45@α32 / R93 HiLR / R90 LongCtx / R83 Talent×HiAlpha×BigG / R75 Tok×HiAlpha×BigG / R50 Diane×BigG×HiLR / R30 Tok)"),
    ("mine-r97-golden-hialpha-bigg-1", "R97", "Golden×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R88×R46/R27; ≠ R88@G4 / R46@α32 / R94 HiLR / R91 LongCtx / R96 Diane×HiAlpha×BigG / R75 Tok×HiAlpha×BigG / R51 Golden×BigG×HiLR / R30 Tok)"),
    ("mine-r98-ckp333-hialpha-bigg-1", "R98", "ckp333×HiAlpha×BigG α=128 r=16 G=16 lr=5e-6 @6144 (R89×R47/R27; ≠ R89@G4 / R47@α32 / R95 HiLR / R92 LongCtx / R97 Golden×HiAlpha×BigG / R96 Diane×HiAlpha×BigG / R75 Tok×HiAlpha×BigG / R52 ckp333×BigG×HiLR / R30 Tok)"),
    ("mine-r99-talent-hialpha-bigg-hilr-1", "R99", "Talent×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R83×R82/R28; ≠ R83@5e-6 / R82@G4 / R49@α32 / R86 LongCtx / R77 Tok×HiAlpha×BigG×HiLR / R98 ckp333×HiAlpha×BigG / R30 Tok)"),
    ("mine-r100-diane-hialpha-bigg-hilr-1", "R100", "Diane×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R96×R93/R28; ≠ R96@5e-6 / R93@G4 / R50@α32 / R90 LongCtx / R99 Talent×HiAlpha×BigG×HiLR / R77 Tok×HiAlpha×BigG×HiLR / R30 Tok)"),
    ("mine-r101-golden-hialpha-bigg-hilr-1", "R101", "Golden×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R97×R94/R28; ≠ R97@5e-6 / R94@G4 / R51@α32 / R91 LongCtx / R100 Diane×HiAlpha×BigG×HiLR / R99 Talent×HiAlpha×BigG×HiLR / R77 Tok×HiAlpha×BigG×HiLR / R30 Tok)"),
    ("mine-r102-ckp333-hialpha-bigg-hilr-1", "R102", "ckp333×HiAlpha×BigG×HiLR α=128 r=16 G=16 lr=2e-5 @6144 (R98×R95/R28; ≠ R98@5e-6 / R95@G4 / R52@α32 / R92 LongCtx / R101 Golden×HiAlpha×BigG×HiLR / R100 Diane×HiAlpha×BigG×HiLR / R99 Talent×HiAlpha×BigG×HiLR / R77 Tok×HiAlpha×BigG×HiLR / R30 Tok)"),
    ("mine-r103-diane-hialpha-longctx-hilr-1", "R103", "Diane×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R90×R93/R28; ≠ R90@5e-6 / R93@6144 / R100 BigG×HiLR@6144 / R84 Talent×HiAlpha×LongCtx×HiLR / R76 Tok×HiAlpha×LongCtx×HiLR / R54@α32 / R30 Tok)"),
    ("mine-r104-golden-hialpha-longctx-hilr-1", "R104", "Golden×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R91×R94/R28; ≠ R91@5e-6 / R94@6144 / R101 BigG×HiLR@6144 / R103 Diane×HiAlpha×LongCtx×HiLR / R84 Talent×HiAlpha×LongCtx×HiLR / R76 Tok×HiAlpha×LongCtx×HiLR / R55@α32 / R30 Tok)"),
    ("mine-r105-ckp333-hialpha-longctx-hilr-1", "R105", "ckp333×HiAlpha×LongCtx×HiLR α=128 r=16 16384/1024 lr=2e-5 (R92×R95/R28; ≠ R92@5e-6 / R95@6144 / R102 BigG×HiLR@6144 / R104 Golden×HiAlpha×LongCtx×HiLR / R103 Diane×HiAlpha×LongCtx×HiLR / R84 Talent×HiAlpha×LongCtx×HiLR / R76 Tok×HiAlpha×LongCtx×HiLR / R56@α32 / R30 Tok)"),
    ("mine-r106-diane-hialpha-longctx-bigg-hilr-1", "R106", "Diane×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=16 16384/1024 lr=2e-5 (R103×R100/R27; ≠ R103@G4 / R100@6144 / R86 Talent×HiAlpha×LongCtx×BigG×HiLR / R79 Tok×HiAlpha×LongCtx×BigG×HiLR / R64@α32 / R105 ckp333×HiAlpha×LongCtx×HiLR / R104 Golden×HiAlpha×LongCtx×HiLR)"),
    ("mine-r107-golden-hialpha-longctx-bigg-hilr-1", "R107", "Golden×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=16 16384/1024 lr=2e-5 (R104×R101/R27; ≠ R104@G4 / R101@6144 / R106 Diane×HiAlpha×LongCtx×BigG×HiLR / R65@α32 / R86 Talent×HiAlpha×LongCtx×BigG×HiLR / R79 Tok×HiAlpha×LongCtx×BigG×HiLR / R105 ckp333×HiAlpha×LongCtx×HiLR)"),
    ("mine-r108-ckp333-hialpha-longctx-bigg-hilr-1", "R108", "ckp333×HiAlpha×LongCtx×BigG×HiLR α=128 r=16 G=16 16384/1024 lr=2e-5 (R105×R102/R27; ≠ R105@G4 / R102@6144 / R107 Golden×HiAlpha×LongCtx×BigG×HiLR / R106 Diane×HiAlpha×LongCtx×BigG×HiLR / R66@α32 / R86 Talent×HiAlpha×LongCtx×BigG×HiLR / R79 Tok×HiAlpha×LongCtx×BigG×HiLR)"),
    ("mine-r109-nodrop-longctx-1", "R109", "Tok NoDrop×LongCtx drop=0.0 16384/1024 lr=5e-6 r=16 α=32 G=4 (R31×R24; ≠ R31@6144 / R24 drop=0.05 / R67 HiRank×LongCtx / R108 HiAlpha quintuple)"),
    ("mine-r110-nodrop-hilr-1", "R110", "Tok NoDrop×HiLR drop=0.0 lr=2e-5 @6144/512 r=16 α=32 G=4 (R31×R28; ≠ R31@5e-6 / R28 drop=0.05 / R109 LongCtx / R68 HiRank×HiLR)"),
    ("mine-r111-nodrop-bigg-1", "R111", "Tok NoDrop×BigG drop=0.0 G=16 lr=5e-6 @6144/512 r=16 α=32 (R31×R27; ≠ R31@G4 / R27 drop=0.05 / R109 LongCtx / R110 HiLR / R69 HiRank×BigG)"),
    ("mine-r112-nodrop-hirank-1", "R112", "Tok NoDrop×HiRank drop=0.0 r=64 α=128 lr=5e-6 @6144/512 G=4 (R31×R29; ≠ R31@r16 / R29 drop=0.05 / R109 LongCtx / R110 HiLR / R111 BigG / R67–R69 HiRank compounds)"),
    ("mine-r113-nodrop-hialpha-1", "R113", "Tok NoDrop×HiAlpha drop=0.0 α=128 r=16 lr=5e-6 @6144/512 G=4 (R31×R30; ≠ R31@α32 / R30 drop=0.05 / R112 NoDrop×HiRank r=64 / R109 LongCtx / R110 HiLR / R111 BigG / R73–R79 HiAlpha compounds)"),
    ("mine-r114-nodrop-kl-1", "R114", "Tok NoDrop×KL drop=0.0 kl_coef=0.02 r=16 α=32 lr=5e-6 @6144/512 G=4 (R31×R32; ≠ R31 kl=0 / R32 drop=0.05 / R109–R113 NoDrop compounds all kl=0)"),
    ("mine-r115-nodrop-longctx-hilr-1", "R115", "Tok NoDrop×LongCtx×HiLR drop=0.0 16384/1024 lr=2e-5 r=16 α=32 G=4 (R109×R110; ≠ R109@5e-6 / R110@6144 / R34 LongCtx×HiLR drop=0.05 / R70 HiRank×LongCtx×HiLR / R114 NoDrop×KL)"),
    ("mine-r116-nodrop-longctx-bigg-1", "R116", "Tok NoDrop×LongCtx×BigG drop=0.0 16384/1024 G=16 lr=5e-6 r=16 α=32 (R109×R111; ≠ R109@G=4 / R111@6144 / R115 NoDrop×LongCtx×HiLR @G=4/2e-5 / R57 LongCtx×BigG drop=0.05 / R114 NoDrop×KL)"),
    ("mine-r117-nodrop-longctx-hirank-1", "R117", "Tok NoDrop×LongCtx×HiRank drop=0.0 r=64 α=128 16384/1024 G=4 lr=5e-6 (R109×R112; ≠ R109@r16 / R112@6144 / R67 HiRank×LongCtx drop=0.05 / R115 NoDrop×LongCtx×HiLR / R116 NoDrop×LongCtx×BigG)"),
    ("mine-r118-nodrop-longctx-hialpha-1", "R118", "Tok NoDrop×LongCtx×HiAlpha drop=0.0 α=128 r=16 16384/1024 G=4 lr=5e-6 (R109×R113; ≠ R109@α32 / R113@6144 / R73 HiAlpha×LongCtx drop=0.05 / R115 NoDrop×LongCtx×HiLR / R116 NoDrop×LongCtx×BigG / R117 NoDrop×LongCtx×HiRank)"),
    ("mine-r119-nodrop-longctx-kl-1", "R119", "Tok NoDrop×LongCtx×KL drop=0.0 kl=0.02 r=16 α=32 16384/1024 G=4 lr=5e-6 (R109×R114; ≠ R109@kl=0 / R114@6144 / R32 drop=0.05 / R115–R118 NoDrop doubles all kl=0)"),
    ("mine-r120-nodrop-longctx-bigg-hilr-1", "R120", "Tok NoDrop×LongCtx×BigG×HiLR drop=0.0 G=16 lr=2e-5 r=16 α=32 16384/1024 (R115×R116; ≠ R115@G=4 / R116@5e-6 / R62 drop=0.05 / R119 KL / R109–R114)"),
    ("mine-r121-nodrop-longctx-hirank-hilr-1", "R121", "Tok NoDrop×LongCtx×HiRank×HiLR drop=0.0 r=64 α=128 16384/1024 lr=2e-5 G=4 (R117×R115; ≠ R117@5e-6 / R115@r16 / R120@r16G16 / R70 drop=0.05)"),
    ("mine-r122-nodrop-longctx-hialpha-hilr-1", "R122", "Tok NoDrop×LongCtx×HiAlpha×HiLR drop=0.0 α=128 r=16 16384/1024 lr=2e-5 G=4 (R118×R115; ≠ R118@5e-6 / R115@α32 / R121@r64 / R120@r16G16 / R76 drop=0.05)"),
    ("mine-r123-nodrop-longctx-kl-hilr-1", "R123", "Tok NoDrop×LongCtx×KL×HiLR drop=0.0 kl=0.02 r=16 α=32 16384/1024 lr=2e-5 G=4 (R119×R115; ≠ R119@5e-6 / R115@kl=0 / R114@6144 / R122@α128 / R121@r64 / R120@G16 / R32 drop=0.05)"),
    ("mine-r124-nodrop-longctx-kl-bigg-1", "R124", "Tok NoDrop×LongCtx×KL×BigG drop=0.0 kl=0.02 r=16 α=32 16384/1024 G=16 lr=5e-6 (R119×R116; ≠ R119@G=4 / R116@kl=0 / R120@kl=0+HiLR / R123@G=4+HiLR / R32 drop=0.05)"),
    ("mine-r125-nodrop-longctx-kl-bigg-hilr-1", "R125", "Tok NoDrop×LongCtx×KL×BigG×HiLR drop=0.0 kl=0.02 r=16 α=32 16384/1024 G=16 lr=2e-5 (R124×R123; ≠ R124@5e-6 / R123@G=4 / R120@kl=0 / R119@G=4 / R32 drop=0.05)"),
    ("mine-r126-nodrop-longctx-hirank-bigg-1", "R126", "Tok NoDrop×LongCtx×HiRank×BigG drop=0.0 r=64 α=128 16384/1024 G=16 lr=5e-6 (R117×R116; ≠ R117@G=4 / R116@r16 / R121@HiLR / R71 drop=0.05 / R125 KL)"),
    ("mine-r127-nodrop-longctx-hialpha-bigg-1", "R127", "Tok NoDrop×LongCtx×HiAlpha×BigG drop=0.0 α=128 r=16 16384/1024 G=16 lr=5e-6 (R118×R116; ≠ R118@G=4 / R116@α32 / R122@HiLR / R75 drop=0.05 / R126 HiRank×BigG)"),
    ("mine-r128-nodrop-longctx-hirank-bigg-hilr-1", "R128", "Tok NoDrop×LongCtx×HiRank×BigG×HiLR drop=0.0 r=64 α=128 16384/1024 G=16 lr=2e-5 (R126×R121; ≠ R126@5e-6 / R121@G=4 / R71 drop=0.05 / R125 KL / R127 HiAlpha×BigG)"),
    ("mine-r129-nodrop-longctx-hialpha-bigg-hilr-1", "R129", "Tok NoDrop×LongCtx×HiAlpha×BigG×HiLR drop=0.0 α=128 r=16 16384/1024 G=16 lr=2e-5 (R127×R122; ≠ R127@5e-6 / R122@G=4 / R128 HiRank×BigG×HiLR / R75 drop=0.05 / R125 KL)"),
    ("mine-r130-nodrop-longctx-kl-hirank-bigg-hilr-1", "R130", "Tok NoDrop×LongCtx×KL×HiRank×BigG×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=2e-5 (R125×R128; ≠ R125@α32 / R128@kl=0 / R129 HiAlpha / R124@5e-6 / R126 no-KL)"),
    ("mine-r131-nodrop-longctx-kl-hialpha-bigg-hilr-1", "R131", "Tok NoDrop×LongCtx×KL×HiAlpha×BigG×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=2e-5 (R125×R129; ≠ R125@α32 / R129@kl=0 / R130 HiRank / R122@G=4 / R127@5e-6)"),
    ("mine-r132-nodrop-longctx-kl-hirank-bigg-1", "R132", "Tok NoDrop×LongCtx×KL×HiRank×BigG drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=5e-6 (R126×R124; ≠ R130@2e-5 HiLR / R126@kl=0 / R131 HiAlpha / R124@r16 / R128@kl=0+HiLR)"),
    ("mine-r133-nodrop-longctx-kl-hialpha-bigg-1", "R133", "Tok NoDrop×LongCtx×KL×HiAlpha×BigG drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=5e-6 (R127×R124; ≠ R131@2e-5 HiLR / R127@kl=0 / R132 HiRank / R124@α32 / R129@kl=0+HiLR)"),
    ("mine-r134-nodrop-longctx-kl-hirank-1", "R134", "Tok NoDrop×LongCtx×KL×HiRank drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=5e-6 (R117×R119; ≠ R132@G16 BigG / R130@G16+HiLR / R117@kl=0 / R121@kl=0+HiLR / R119@r16 / R133 HiAlpha×BigG)"),
    ("mine-r135-nodrop-longctx-kl-hialpha-1", "R135", "Tok NoDrop×LongCtx×KL×HiAlpha drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=5e-6 (R118×R119; ≠ R133@G16 BigG / R131@G16+HiLR / R118@kl=0 / R122@kl=0+HiLR / R134 HiRank@G4 / R119@α32)"),
    ("mine-r136-nodrop-longctx-kl-hirank-hilr-1", "R136", "Tok NoDrop×LongCtx×KL×HiRank×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=2e-5 (R134×HiLR; ≠ R134@5e-6 / R130@G16 BigG+HiLR / R121@kl=0 / R123@α32 / R132@G16@5e-6 / R135 HiAlpha@G4)"),
    ("mine-r137-nodrop-longctx-kl-hialpha-hilr-1", "R137", "Tok NoDrop×LongCtx×KL×HiAlpha×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=2e-5 (R135×HiLR; ≠ R135@5e-6 / R131@G16 BigG+HiLR / R122@kl=0 / R123@α32 / R133@G16@5e-6 / R136 HiRank@G4)"),
    ("mine-r138-nodrop-longctx-kl-hialpha-hitemp-1", "R138", "Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=5e-6 temp=1.2 (R135×HiTemp; ≠ R135@temp0.8 / R137@HiLR@temp0.8 / R25@drop0.05 short / R131@G16 BigG+HiLR / R122@kl=0 / R136 HiRank×HiLR)"),
    ("mine-r139-nodrop-longctx-kl-hirank-hitemp-1", "R139", "Tok NoDrop×LongCtx×KL×HiRank×HiTemp drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=5e-6 temp=1.2 (R134×HiTemp; ≠ R134@temp0.8 / R136@HiLR@temp0.8 / R138 HiAlpha×HiTemp / R25@drop0.05 short / R130@G16 BigG+HiLR / R121@kl=0)"),
    ("mine-r140-nodrop-longctx-kl-hialpha-hitemp-hilr-1", "R140", "Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=4 lr=2e-5 temp=1.2 (R138×HiLR; ≠ R138@lr5e-6@temp1.2 / R137@HiLR@temp0.8 / R139 HiRank×HiTemp / R25@drop0.05 short / R131@G16 BigG+HiLR / R122@kl=0)"),
    ("mine-r141-nodrop-longctx-kl-hirank-hitemp-hilr-1", "R141", "Tok NoDrop×LongCtx×KL×HiRank×HiTemp×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=4 lr=2e-5 temp=1.2 (R139×HiLR; ≠ R139@lr5e-6@temp1.2 / R136@HiLR@temp0.8 / R140 HiAlpha×HiTemp×HiLR / R138 HiAlpha×HiTemp / R25@drop0.05 short / R130@G16 BigG+HiLR)"),
    ("mine-r142-nodrop-longctx-kl-hialpha-hitemp-bigg-1", "R142", "Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp×BigG drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=5e-6 temp=1.2 (R138×BigG; ≠ R138@G4@temp1.2 / R140@HiLR@G4 / R133@BigG@temp0.8 / R131@BigG+HiLR / R141 HiRank×HiTemp×HiLR / R25@drop0.05 short)"),
    ("mine-r143-nodrop-longctx-kl-hirank-hitemp-bigg-1", "R143", "Tok NoDrop×LongCtx×KL×HiRank×HiTemp×BigG drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=5e-6 temp=1.2 (R139×BigG; ≠ R139@G4@temp1.2 / R141@HiLR@G4 / R142 HiAlpha×HiTemp×BigG / R132@BigG@temp0.8 / R130@BigG+HiLR / R25@drop0.05 short)"),
    ("mine-r144-nodrop-longctx-kl-hialpha-hitemp-bigg-hilr-1", "R144", "Tok NoDrop×LongCtx×KL×HiAlpha×HiTemp×BigG×HiLR drop=0.0 kl=0.02 r=16 α=128 16384/1024 G=16 lr=2e-5 temp=1.2 (R142×HiLR; ≠ R142@5e-6@G16@temp1.2 / R140@HiLR@G4 / R143 HiRank×HiTemp×BigG / R141 HiRank×HiTemp×HiLR / R131@BigG+HiLR@temp0.8 / R25@drop0.05 short)"),
    ("mine-r145-nodrop-longctx-kl-hirank-hitemp-bigg-hilr-1", "R145", "Tok NoDrop×LongCtx×KL×HiRank×HiTemp×BigG×HiLR drop=0.0 kl=0.02 r=64 α=128 16384/1024 G=16 lr=2e-5 temp=1.2 (R143×HiLR; ≠ R143@5e-6@G16@temp1.2 / R141@HiLR@G4 / R144 HiAlpha×HiTemp×BigG×HiLR / R130@BigG+HiLR@temp0.8 / R132@BigG@temp0.8 / R25@drop0.05 short)"),
    ("mine-r146-nodrop-longctx-kl-megarank-hitemp-bigg-hilr-1", "R146", "Tok NoDrop×LongCtx×KL×MegaRank×HiTemp×BigG×HiLR drop=0.0 kl=0.02 r=128 α=256 16384/1024 G=16 lr=2e-5 temp=1.2 (R145×MegaRank; ≠ R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R29@r64@6144 / R25@drop0.05 short)"),
    ("mine-r147-nodrop-longctx-kl-megarank-ultratemp-bigg-hilr-1", "R147", "Tok NoDrop×LongCtx×KL×MegaRank×UltraTemp×BigG×HiLR drop=0.0 kl=0.02 r=128 α=256 16384/1024 G=16 lr=2e-5 temp=1.5 (R146×UltraTemp; ≠ R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"),
    ("mine-r148-nodrop-longctx-kl-megarank-supertemp-bigg-hilr-1", "R148", "Tok NoDrop×LongCtx×KL×MegaRank×SuperTemp×BigG×HiLR drop=0.0 kl=0.02 r=128 α=256 16384/1024 G=16 lr=2e-5 temp=2.0 (R147×SuperTemp; ≠ R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"),
    ("mine-r149-nodrop-longctx-kl-ultramegarank-supertemp-bigg-hilr-1", "R149", "Tok NoDrop×LongCtx×KL×UltraMegaRank×SuperTemp×BigG×HiLR drop=0.0 kl=0.02 r=256 α=512 16384/1024 G=16 lr=2e-5 temp=2.0 (R148×UltraMegaRank; ≠ R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"),
    ("mine-r150-nodrop-longctx-kl-ultramegarank-extremetemp-bigg-hilr-1", "R150", "Tok NoDrop×LongCtx×KL×UltraMegaRank×ExtremeTemp×BigG×HiLR drop=0.0 kl=0.02 r=256 α=512 16384/1024 G=16 lr=2e-5 temp=2.5 (R149×ExtremeTemp; ≠ R149@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R143@r64@5e-6 / R130@r64@temp0.8 / R25@drop0.05 short)"),
    ("mine-r151-nodrop-longctx-kl-hypermegarank-extremetemp-bigg-hilr-1", "R151", "Tok NoDrop×LongCtx×KL×HyperMegaRank×ExtremeTemp×BigG×HiLR drop=0.0 kl=0.02 r=512 α=1024 16384/1024 G=16 lr=2e-5 temp=2.5 (R150×HyperMegaRank; ≠ R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"),
    ("mine-r152-nodrop-longctx-kl-hypermegarank-infernotemp-bigg-hilr-1", "R152", "Tok NoDrop×LongCtx×KL×HyperMegaRank×InfernoTemp×BigG×HiLR drop=0.0 kl=0.02 r=512 α=1024 16384/1024 G=16 lr=2e-5 temp=3.0 (R151×InfernoTemp; ≠ R151@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"),
    ("mine-r153-nodrop-longctx-kl-gigarank-infernotemp-bigg-hilr-1", "R153", "Tok NoDrop×LongCtx×KL×GigaRank×InfernoTemp×BigG×HiLR drop=0.0 kl=0.02 r=1024 α=2048 16384/1024 G=16 lr=2e-5 temp=3.0 (R152×GigaRank; ≠ R152@r512@temp3.0 / R151@r512@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"),
    ("mine-r154-nodrop-longctx-kl-gigarank-plasmatemp-bigg-hilr-1", "R154", "Tok NoDrop×LongCtx×KL×GigaRank×PlasmaTemp×BigG×HiLR drop=0.0 kl=0.02 r=1024 α=2048 16384/1024 G=16 lr=2e-5 temp=3.5 (R153×PlasmaTemp; ≠ R153@r1024@temp3.0 / R152@r512@temp3.0 / R151@r512@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"),
    ("mine-r155-nodrop-longctx-kl-terarank-plasmatemp-bigg-hilr-1", "R155", "Tok NoDrop×LongCtx×KL×TeraRank×PlasmaTemp×BigG×HiLR drop=0.0 kl=0.02 r=2048 α=4096 16384/1024 G=16 lr=2e-5 temp=3.5 (R154×TeraRank; ≠ R154@r1024@temp3.5 / R153@r1024@temp3.0 / R152@r512@temp3.0 / R151@r512@temp2.5 / R150@r256@temp2.5 / R149@r256@temp2.0 / R148@r128 / R147@temp1.5 / R146@temp1.2 / R145@r64 / R144 HiAlpha×HiTemp×BigG×HiLR@r16 / R25@drop0.05 short)"),
    # R156 also queued at HEAD (after warm-skips) — see QUEUE top after R73 comment.
    # p2235: R33 warm-armed on mine-crown-1 after R26 SIGNAL_POS_BELOW — do not re-rent.
    # ("mine-r33-guass-grpo-1", "R33", "guass-init Reason-GRPO (≠ R3 Tok / R19–R23; LoRA from live king)"),
    # R11 REFUTED p2175 n80 m=-0.0055 z=-0.82 — do not re-rent.
    # R12–R17 / R20–R21 warm or REFUTED — do not re-rent.
    # R20 REFUTED p2211 vs guass — do not re-rent.
    # R21 REFUTED p2235 vs guass — do not re-rent.
    # R26 SIGNAL_POS_BELOW p2235 vs guass — do not re-rent.
]

BASE = os.environ.get("LIUM_BASE_URL", "https://lium.io/api")


def log(msg: str) -> None:
    line = f"[fleet-rent] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)


def api_key() -> str:
    env = os.environ.get("LIUM_API_KEY")
    if env:
        return env
    cfg = configparser.ConfigParser()
    cfg.read(str(Path.home() / ".lium/config.ini"))
    key = cfg.get("api", "api_key", fallback=cfg.get("default", "api_key", fallback=""))
    if not key:
        raise SystemExit("no Lium API key")
    return key


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "X-API-KEY": api_key(),
            "X-Source": "fleet-api",
            "X-Lium-Client-Version": "0.0.32",
        }
    )
    return s


def _get_json(sess: requests.Session, path: str, params: dict | None = None, retries: int = 6):
    """GET JSON with 429/5xx backoff. Returns parsed body or None on hard fail."""
    url = f"{BASE}{path}"
    backoff = 0.5
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log(f"GET {path} neterr attempt={attempt+1}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue
        if r.status_code == 429 or 500 <= r.status_code < 600:
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else backoff
            except ValueError:
                wait = backoff
            wait = max(wait, backoff)
            log(f"GET {path} http={r.status_code} backoff={wait:.2f}s attempt={attempt+1}")
            time.sleep(wait)
            backoff = min(backoff * 2, 8.0)
            continue
        if not r.ok:
            log(f"GET {path} http={r.status_code} body={r.text[:160]}")
            return None
        try:
            return r.json()
        except Exception as e:
            log(f"GET {path} jsonerr: {e}")
            return None
    return None


def _is_b300(machine: str) -> bool:
    return "B300" in (machine or "").upper()


def _is_b200(machine: str) -> bool:
    u = (machine or "").upper()
    return "B200" in u and "B300" not in u


def _gpu_label(n: dict) -> str:
    """machine_name plus specs.gpu.details[0].name (p2582: some rows label only in specs)."""
    mn = n.get("machine_name") or ""
    details = (((n.get("specs") or {}).get("gpu") or {}).get("details") or [])
    gname = ""
    if isinstance(details, list) and details and isinstance(details[0], dict):
        gname = details[0].get("name") or ""
    return f"{mn} {gname}".strip()


def list_b300_b200_nodes(
    sess: requests.Session,
) -> tuple[str, list[dict], int] | tuple[None, None, int]:
    """One unfiltered 8× poll; prefer B300, else B200.

    Returns (gpu_label, nodes, n_blacklisted_b300_b200). None,None,_ = API fail.
    """
    data = _get_json(
        sess,
        "/executors",
        params={
            "size": 1000,
            "gpu_count_gte": 8,
            "gpu_count_lte": 8,
        },
    )
    if not isinstance(data, list):
        return None, None, 0
    b300: list[dict] = []
    b200: list[dict] = []
    n_bl = 0
    # p2702: reload each poll — module-level EXECUTOR_BLACKLIST freezes at
    # import, so a mid-run bad-host append (e.g. R178 0e1d41d2…) would be
    # re-rented until the process restarts.
    bl = _load_executor_blacklist()
    for n in data:
        if not isinstance(n, dict):
            continue
        if n.get("has_no_pending_rental") is False:
            continue
        eid = n.get("id")
        if not eid:
            continue
        label = _gpu_label(n)
        is_target = _is_b300(label) or _is_b200(label)
        if eid in bl:
            if is_target:
                n_bl += 1
            continue
        if _is_b300(label):
            b300.append(n)
        elif _is_b200(label):
            b200.append(n)
    if b300:
        return "B300", b300, n_bl
    if b200:
        return "B200", b200, n_bl
    return "", [], n_bl


def mine_pods_or_none(sess: requests.Session) -> list[dict] | None:
    """Return mine-* pods, or None if /pods failed (do NOT treat as empty)."""
    data = _get_json(sess, "/pods")
    if not isinstance(data, list):
        return None
    mines = []
    for p in data:
        if not isinstance(p, dict):
            continue
        name = p.get("pod_name") or p.get("name") or ""
        if isinstance(name, str) and name.startswith("mine-"):
            p = dict(p)
            p["name"] = name
            mines.append(p)
    return mines


def mine_names_cli_fallback() -> set[str] | None:
    """Last-resort live mine names via `lium ps` when /pods is 429'd."""
    try:
        raw = subprocess.check_output(
            ["lium", "ps", "--format", "json"], text=True, timeout=60
        )
        pods = json.loads(raw)
        if isinstance(pods, dict):
            pods = pods.get("pods") or pods.get("data") or []
        if not isinstance(pods, list):
            return None
        out = set()
        for p in pods:
            if not isinstance(p, dict):
                continue
            name = p.get("name") or p.get("pod_name") or ""
            if isinstance(name, str) and name.startswith("mine-"):
                out.add(name)
        return out
    except Exception as e:
        log(f"lium ps fallback fail: {e}")
        return None


def resolve_live(sess: requests.Session) -> set[str] | None:
    pods = mine_pods_or_none(sess)
    if pods is not None:
        return {p["name"] for p in pods}
    log("/pods failed — trying lium ps fallback before any rent")
    return mine_names_cli_fallback()


def balance_ok() -> bool:
    try:
        raw = subprocess.check_output(["lium", "balance"], text=True, timeout=30)
    except Exception:
        return True
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw.replace(",", ""))
    if not m:
        return True
    return float(m.group(1)) >= 10000.0


def next_slots(live: set[str], want: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for name, axis, note in QUEUE:
        if name in live:
            continue
        out.append((name, axis, note))
        if len(out) >= want:
            break
    return out


def _ttl_hours(ttl: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([hmdHMD]?)\s*", ttl or "")
    if not m:
        return 24
    n = int(m.group(1))
    unit = (m.group(2) or "h").lower()
    if unit == "m":
        return max(1, (n + 59) // 60)
    if unit == "d":
        return n * 24
    return n


def _ssh_pubkey() -> str:
    try:
        return SSH_PUBKEY_PATH.read_text().strip()
    except Exception:
        return ""


def _schedule_ttl(sess: requests.Session, pod_id: str, hours: int) -> None:
    when = datetime.now(timezone.utc) + timedelta(hours=hours)
    iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = sess.post(
            f"{BASE}/pods/{pod_id}/schedule-removal",
            json={"removal_scheduled_at": iso},
            timeout=30,
        )
        if r.ok:
            log(f"TTL ok pod={pod_id} removal={iso}")
        else:
            log(f"TTL fail pod={pod_id} http={r.status_code} body={r.text[:160]}")
    except Exception as e:
        log(f"TTL err pod={pod_id}: {e}")


def try_rent_node_api(node_id: str, name: str) -> bool:
    """POST /executors/{id}/rent — beats CLI fork+auth when stock flickers."""
    if not name.startswith("mine-"):
        log(f"REFUSE non-mine name={name}")
        return False
    body = {
        "pod_name": name,
        "template_id": TEMPLATE_ID,
        "gpu_count": 8,
        "initial_port_count": 12,
        "enable_volume_encryption": True,
        "enable_jupyter": False,
    }
    pk = _ssh_pubkey()
    if pk:
        body["user_public_key"] = pk
    hours = _ttl_hours(TTL)
    body["termination_hours"] = hours
    log(f"attempting API rent executor={node_id} name={name} ttl={hours}h")
    # Per-call session: ThreadPoolExecutor must not share one Session.
    sess = make_session()
    try:
        r = sess.post(
            f"{BASE}/executors/{node_id}/rent",
            json=body,
            timeout=60,
        )
    except Exception as e:
        log(f"API rent neterr name={name} err={e}")
        return False
    if not r.ok:
        log(f"API rent fail name={name} http={r.status_code} body={r.text[:200]}")
        return False
    pod_id = None
    try:
        data = r.json()
        if isinstance(data, dict):
            pod_id = data.get("id") or (data.get("pod") or {}).get("id")
    except Exception:
        data = None
    log(f"API rent ok name={name} pod={pod_id or '?'} http={r.status_code}")
    if pod_id:
        _schedule_ttl(sess, str(pod_id), hours)
    return True


def try_rent_node_cli(node_id: str, name: str) -> bool:
    cmd = [
        "lium",
        "up",
        node_id,
        "--name",
        name,
        "--ttl",
        TTL,
        "--no-ssh",
        "-y",
    ]
    log(f"attempting CLI fallback {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, timeout=180)
        return r.returncode == 0
    except Exception as e:
        log(f"CLI rent fail name={name} err={e}")
        return False


def try_rent_node(node_id: str, name: str) -> bool:
    if try_rent_node_api(node_id, name):
        return True
    return try_rent_node_cli(node_id, name)


def write_stamp(
    name: str,
    axis: str,
    gpu: str,
    note: str,
    sess: requests.Session,
    executor_id: str | None = None,
) -> None:
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    path = STAMP_DIR / f"rented_{name.replace('/', '_')}.json"
    pods = mine_pods_or_none(sess)
    # Prefer rent-time node id; else pull executor_id from /pods (p2584 BAD_HOST).
    eid = executor_id
    if not eid and isinstance(pods, list):
        for p in pods:
            if not isinstance(p, dict):
                continue
            n = p.get("name") or p.get("pod_name") or ""
            if n == name:
                eid = p.get("executor_id")
                if not eid and isinstance(p.get("executor"), dict):
                    eid = p["executor"].get("id")
                break
    ps = json.dumps(pods)[:4000] if pods is not None else "pods_unavailable"
    path.write_text(
        json.dumps(
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pass": PASS,
                "name": name,
                "axis": axis,
                "gpu": gpu,
                "note": note,
                "ttl": TTL,
                "mode": "api-unfiltered-8x-POST-rent",
                "executor_id": eid,
                "ps_json": ps,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"STAMP_OK {path} executor_id={eid or '?'}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except Exception:
        return ""


def main() -> int:
    EXP.joinpath("logs").mkdir(parents=True, exist_ok=True)
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    # Single-instance: a second launch (even `… --help` — no argparse) was
    # double-polling /executors → 429 storms that blind stock sightings (p2150).
    # SKIP_PID_LOCK=1: pass-level burst while long waiter is SIGSTOP'd (p2183).
    # Still refuses a second *running* snatcher unless the operator pauses it.
    skip_lock = os.environ.get("SKIP_PID_LOCK", "").strip() in ("1", "true", "yes")
    if PIDF.exists() and not skip_lock:
        try:
            old = int(PIDF.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            cmd = _cmdline(old)
            if "wait_fleet_b300_api.py" in cmd:
                print(
                    f"[fleet-rent] ABORT already running pid={old} cmd={cmd[:120]}",
                    flush=True,
                )
                return 1
    if not skip_lock:
        PIDF.write_text(str(os.getpid()) + "\n")
    else:
        print(
            f"[fleet-rent] SKIP_PID_LOCK=1 pass-burst (pidfile left for long waiter)",
            flush=True,
        )

    # Launcher already redirects stdout → LOG; only Tee when it does not.
    # Pass-bursts (SKIP_PID_LOCK) must not append into the long waiter's log.
    stdout_path = ""
    try:
        stdout_path = os.readlink(f"/proc/{os.getpid()}/fd/1")
    except Exception:
        stdout_path = ""
    already_logging = os.path.abspath(stdout_path) == os.path.abspath(str(LOG))
    if not already_logging and not skip_lock:
        log_f = open(LOG, "a", buffering=1)

        class Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                log_f.write(s)

            def flush(self):
                sys.__stdout__.flush()
                log_f.flush()

        sys.stdout = Tee()  # type: ignore
        sys.stderr = sys.stdout  # type: ignore
    else:
        sys.stderr = sys.stdout

    sess = make_session()
    log(
        f"start target={TARGET} cap={CAP} empty_sleep={EMPTY_SLEEP}s "
        f"parallel={PARALLEL_N} max_iters={MAX_ITERS} pass={PASS} "
        f"mode=api-unfiltered-8x-POST-rent template={TEMPLATE_ID[:8]}"
    )

    live = resolve_live(sess)
    if live is None:
        log("ABORT cannot resolve live mine-* at start (/pods + lium ps failed)")
        return 5
    log(f"live_mines={' '.join(sorted(live))}|count={len(live)}")

    for i in range(1, MAX_ITERS + 1):
        if i == 1 or i % 40 == 0:
            if not balance_ok():
                log("ABORT balance below $10k floor")
                return 4
            live = resolve_live(sess)
            if live is None:
                log(f"iter={i} live-resolve FAIL — skip rent, sleep {PODS_FAIL_SLEEP}s")
                time.sleep(PODS_FAIL_SLEEP)
                continue
            n = len(live)
            if n >= TARGET:
                log(f"TARGET reached mine_count={n} >= {TARGET} — exit")
                return 0
            if n >= CAP:
                log(f"ABORT at cap mine_count={n} >= {CAP}")
                return 3

        gpu_label, nodes, n_bl = list_b300_b200_nodes(sess)
        if gpu_label is None:
            if i % 40 == 1:
                log(f"iter={i} /executors FAIL — backoff sleep")
            time.sleep(max(EMPTY_SLEEP, PODS_FAIL_SLEEP))
            continue

        if not nodes:
            if i % 40 == 1:
                live = resolve_live(sess)
                if live is None:
                    log(f"iter={i} api-empty but live-resolve FAIL")
                    time.sleep(PODS_FAIL_SLEEP)
                    continue
                slots = next_slots(live, 6)
                try:
                    bal = subprocess.check_output(
                        ["lium", "balance"], text=True, timeout=30
                    ).strip()
                except Exception:
                    bal = "?"
                # p2586: CLI often shows the blacklisted 8×B200 ghost — log skip count.
                bl_note = f" bl_skip={n_bl}" if n_bl else ""
                log(
                    f"iter={i} api-empty B300/B200×8{bl_note}; mine={len(live)}/{TARGET} "
                    f"(cap {CAP}) next={' '.join(s[0] for s in slots)}… bal={bal}"
                )
            time.sleep(EMPTY_SLEEP)
            continue

        # STOCK — must know live count before claiming (never assume 0 on API fail).
        live = resolve_live(sess)
        if live is None:
            log(
                f"STOCK {gpu_label}×8 n={len(nodes)} but live-resolve FAIL — "
                f"NO RENT (cap protect); sleep {PODS_FAIL_SLEEP}s"
            )
            time.sleep(PODS_FAIL_SLEEP)
            continue

        n = len(live)
        if n >= TARGET:
            log(f"TARGET reached mine_count={n} >= {TARGET} — exit")
            return 0
        remain = CAP - n
        if remain < 1:
            log(f"ABORT at cap mine_count={n} >= {CAP}")
            return 3
        want = min(PARALLEL_N, remain, len(nodes))
        slots = next_slots(live, want)
        if not slots:
            log(f"QUEUE exhausted with mine_count={n} < target={TARGET} — exit")
            return 0

        n_claim = min(len(nodes), len(slots))
        log(
            f"STOCK {gpu_label}×8 n={len(nodes)} — claiming {n_claim} axes: "
            f"{' '.join(s[0] for s in slots[:n_claim])}"
        )

        rented: dict[str, tuple[str, str, str, str]] = {}
        with ThreadPoolExecutor(max_workers=n_claim) as pool:
            futs = {}
            for j in range(n_claim):
                name, axis, note = slots[j]
                node_id = nodes[j]["id"]
                futs[pool.submit(try_rent_node, node_id, name)] = (
                    name,
                    axis,
                    note,
                    node_id,
                )
            for fut in as_completed(futs):
                name, axis, note, node_id = futs[fut]
                ok = False
                try:
                    ok = fut.result()
                except Exception as e:
                    log(f"rent future err name={name} err={e}")
                if ok:
                    rented[name] = (axis, note, gpu_label, node_id)

        if not rented:
            log(
                f"iter={i} STOCK sighting but 0 rents ({gpu_label} n={len(nodes)}) "
                "— keep polling"
            )
            time.sleep(EMPTY_SLEEP)
            continue

        time.sleep(8)
        live = resolve_live(sess) or set()
        for name, (axis, note, gpu, node_id) in rented.items():
            if name in live:
                log(f"RENTED ok gpu={gpu} name={name} axis={axis} executor={node_id}")
                write_stamp(name, axis, gpu, note, sess, executor_id=node_id)
            else:
                log(f"up rc=0 but {name} not in ps — keep polling")

    live_final = resolve_live(sess) or set()
    log(f"TIMEOUT after {MAX_ITERS} iters — mine_count={len(live_final)}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
