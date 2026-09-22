# A-term v2 live read

Duels: 68 (chal-00470 … chal-00549). Reference pool = every teacher reference action of these duels, normalised per dialect (evalsrv/amatch.py).

## Reference pool and generic lists

| dialect | refs | distinct normalised actions | generic @ f=0.0005 (n actions / share of refs) | generic @ f=0.001 (n actions / share of refs) | generic @ f=0.002 (n actions / share of refs) | generic @ f=0.005 (n actions / share of refs) |
|---|---|---|---|---|---|---|
| bash | 120876 | 90338 | 36 / 9.0% | 13 / 7.3% | 3 / 6.1% | 2 / 5.8% |
| boxed | 6220 | 1537 | 368 / 74.1% | 222 / 62.2% | 106 / 45.2% | 18 / 16.8% |
| terminus_json | 4006 | 1870 | 78 / 52.6% | 23 / 48.2% | 14 / 46.8% | 4 / 43.7% |
| tool_call | 87361 | 71896 | 9 / 1.4% | 4 / 1.0% | 1 / 0.5% | 1 / 0.5% |

### Command-head pool (v2b filter, f_head = 0.01)

| dialect | heads (refs) | distinct heads | generic heads | share of refs |
|---|---|---|---|---|
| bash | 120876 | 1843 | 13 | 73.8% |
  - bash generic heads: `cat` (17.64%), `ls` (15.54%), `echo` (9.03%), `grep` (5.67%), `python3` (5.54%), `nl` (5.24%), `sed` (3.59%), `python` (3.32%), `which` (3.18%), `find` (1.77%), `git log` (1.18%), `mkdir` (1.09%), `git diff` (1.06%)
| tool_call | 87361 | 2091 | 17 | 70.0% |
  - tool_call generic heads: `tool:Bash` (23.92%), `grep` (6.76%), `tool:Read` (4.71%), `tool:Edit` (4.10%), `cat` (3.91%), `ls` (3.85%), `tool:read` (3.50%), `sed` (3.32%), `tool:edit` (3.27%), `git show` (2.35%), `python` (2.15%), `tool:wiki_click_link` (1.78%), `git log` (1.48%), `find` (1.35%), `timeout` (1.25%), `git diff` (1.19%), `python3` (1.11%)

Staged list: f = 0.001 → `generic_actions_f0.001.json`. Top entries:

- **bash** (13): `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTP` (4.31%), `ls -la /app/task_file/ 2>/dev/null || ec` (1.54%), `ls -la /app/task_file/ 2>/dev/null && fi` (0.22%), `git diff` (0.15%), `cd /testbed && git diff` (0.14%), `ls -la /app/task_file/ 2>/dev/null && fi` (0.13%), `cd /app/task_file && python solution.py` (0.13%), `cat /app/task_file/output/results.txt` (0.13%), `ls -la /app/task_file/ && ls -la /app/ta` (0.12%), `ls -la /app/task_file/output/ && python3` (0.12%), `ls -la /app/task_file/ && find /app/task` (0.11%), `ls -la /app/task_file/ 2>/dev/null && ec` (0.11%)
- **tool_call** (4): `[["wiki_go_back", {}]]` (0.50%), `[["bash", {"command": "cd /testbed && gi` (0.20%), `[["TaskUpdate", {"status": "in_progress"` (0.16%), `[["Bash", {"command": "git diff", "descr` (0.14%)

## Re-scoring under v1 / v2 (w = 0.01 and 0.02)

| duel | wvk | n | stored | z S0 | z v1 .01 | z v2 .01 | z v2b .01 | z v2b .02 | margin S0 → v2b .01 | decision S0 / v1 .01 / v2 .01 / v2b .01 |
|---|---|---|---|---|---|---|---|---|---|---|
| chal-00470 | 16 | 1251 | lose | 2.17 | 2.15 | 2.16 | 2.18 | 2.19 | +0.0015 → +0.0015 | lose / lose / lose / lose |
| chal-00472 | 16 | 1254 | lose | -1.97 | -2.11 | -2.13 | -2.09 | -2.20 | -0.0015 → -0.0016 | lose / lose / lose / lose |
| chal-00473 | 16 | 1255 | lose | -1.50 | -1.57 | -1.54 | -1.57 | -1.64 | -0.0010 → -0.0011 | lose / lose / lose / lose |
| chal-00474 | 16 | 1242 | lose | 0.53 | 0.57 | 0.56 | 0.49 | 0.45 | +0.0005 → +0.0004 | lose / lose / lose / lose |
| chal-00475 | 16 | 1245 | lose | -0.27 | -0.31 | -0.34 | -0.25 | -0.24 | -0.0002 → -0.0002 | lose / lose / lose / lose |
| chal-00476 | 16 | 1235 | lose | -0.26 | -0.22 | -0.23 | -0.21 | -0.17 | -0.0002 → -0.0002 | lose / lose / lose / lose |
| chal-00477 | 16 | 1245 | lose | -0.08 | -0.12 | -0.10 | -0.08 | -0.08 | -0.0001 → -0.0001 | lose / lose / lose / lose |
| chal-00478 | 16 | 1238 | lose | 0.19 | 0.16 | 0.15 | 0.19 | 0.20 | +0.0001 → +0.0001 | lose / lose / lose / lose |
| chal-00480 | 16 | 1224 | lose | -1.81 | -1.82 | -1.80 | -1.80 | -1.78 | -0.0014 → -0.0014 | lose / lose / lose / lose |
| chal-00481 | 16 | 1253 | lose | -1.13 | -1.24 | -1.24 | -1.16 | -1.19 | -0.0010 → -0.0010 | lose / lose / lose / lose |
| chal-00482 | 16 | 1238 | lose | 1.90 | 1.96 | 1.93 | 1.91 | 1.92 | +0.0014 → +0.0014 | lose / lose / lose / lose |
| chal-00483 | 16 | 1230 | lose | 0.72 | 0.73 | 0.73 | 0.65 | 0.57 | +0.0006 → +0.0005 | lose / lose / lose / lose |
| chal-00484 | 16 | 1255 | lose | 1.36 | 1.33 | 1.35 | 1.38 | 1.40 | +0.0012 → +0.0012 | lose / lose / lose / lose |
| chal-00485 | 16 | 1233 | lose | 0.33 | 0.22 | 0.24 | 0.33 | 0.33 | +0.0003 → +0.0003 | lose / lose / lose / lose |
| chal-00488 | 16 | 1249 | lose | -5.37 | -5.40 | -5.37 | -5.39 | -5.41 | -0.0053 → -0.0053 | lose / lose / lose / lose |
| chal-00489 | 16 | 1233 | lose | -0.70 | -0.56 | -0.57 | -0.64 | -0.59 | -0.0005 → -0.0005 | lose / lose / lose / lose |
| chal-00490 | 16 | 1245 | lose | -0.28 | -0.33 | -0.34 | -0.27 | -0.26 | -0.0002 → -0.0002 | lose / lose / lose / lose |
| chal-00491 | 16 | 1244 | lose | -2.21 | -2.47 | -2.46 | -2.27 | -2.33 | -0.0021 → -0.0022 | lose / lose / lose / lose |
| chal-00492 | 16 | 1238 | lose | 1.38 | 1.44 | 1.44 | 1.44 | 1.50 | +0.0011 → +0.0012 | lose / lose / lose / lose |
| chal-00493 | 17 | 1286 | lose | -1.42 | -1.40 | -1.40 | -1.38 | -1.33 | -0.0010 → -0.0010 | lose / lose / lose / lose |
| chal-00494 | 17 | 1289 | lose | 1.20 | 1.12 | 1.14 | 1.21 | 1.23 | +0.0009 → +0.0009 | lose / lose / lose / lose |
| chal-00495 | 17 | 1296 | lose | 1.16 | 1.09 | 1.08 | 1.14 | 1.11 | +0.0010 → +0.0010 | lose / lose / lose / lose |
| chal-00496 | 17 | 1291 | lose | 2.10 | 2.10 | 2.08 | 2.10 | 2.10 | +0.0016 → +0.0016 | lose / lose / lose / lose |
| chal-00497 | 17 | 1293 | lose | 2.37 | 2.39 | 2.40 | 2.44 | 2.51 | +0.0018 → +0.0019 | lose / lose / lose / lose |
| chal-00498 | 17 | 1290 | lose | -0.34 | -0.30 | -0.32 | -0.33 | -0.31 | -0.0003 → -0.0003 | lose / lose / lose / lose |
| chal-00499 | 17 | 1285 | lose | 1.93 | 1.94 | 1.96 | 1.99 | 2.05 | +0.0014 → +0.0014 | lose / lose / lose / lose |
| chal-00500 | 17 | 1288 | lose | 0.69 | 0.64 | 0.66 | 0.66 | 0.63 | +0.0005 → +0.0005 | lose / lose / lose / lose |
| chal-00501 | 17 | 1290 | lose | -0.55 | -0.47 | -0.50 | -0.56 | -0.57 | -0.0004 → -0.0004 | lose / lose / lose / lose |
| chal-00502 | 17 | 1287 | win | 2.69 | 2.70 | 2.72 | 2.75 | 2.81 | +0.0021 → +0.0021 | win / win / win / win |
| chal-00503 | 17 | 1295 | lose | -0.27 | -0.22 | -0.22 | -0.24 | -0.20 | -0.0002 → -0.0002 | lose / lose / lose / lose |
| chal-00504 | 17 | 1296 | lose | 0.34 | 0.36 | 0.36 | 0.39 | 0.44 | +0.0003 → +0.0003 | lose / lose / lose / lose |
| chal-00505 | 17 | 1291 | lose | 1.50 | 1.49 | 1.50 | 1.51 | 1.51 | +0.0012 → +0.0012 | lose / lose / lose / lose |
| chal-00506 | 17 | 1290 | lose | 1.21 | 1.28 | 1.25 | 1.18 | 1.15 | +0.0010 → +0.0010 | lose / lose / lose / lose |
| chal-00507 | 17 | 1283 | lose | 2.54 | 2.53 | 2.54 | 2.56 | 2.58 | +0.0019 → +0.0019 | lose / lose / lose / lose |
| chal-00508 | 17 | 1283 | lose | -0.85 | -0.80 | -0.83 | -0.84 | -0.82 | -0.0007 → -0.0007 | lose / lose / lose / lose |
| chal-00509 | 17 | 1291 | lose | -5.02 | -5.07 | -5.07 | -5.07 | -5.11 | -0.0054 → -0.0054 | lose / lose / lose / lose |
| chal-00510 | 17 | 1287 | lose | -1.05 | -0.98 | -0.99 | -1.02 | -0.98 | -0.0009 → -0.0008 | lose / lose / lose / lose |
| chal-00511 | 17 | 1289 | lose | -0.80 | -0.75 | -0.77 | -0.73 | -0.67 | -0.0007 → -0.0006 | lose / lose / lose / lose |
| chal-00512 | 17 | 1292 | lose | 0.48 | 0.29 | 0.29 | 0.44 | 0.39 | +0.0004 → +0.0003 | lose / lose / lose / lose |
| chal-00515 | 17 | 1286 | lose | -0.50 | -0.57 | -0.55 | -0.53 | -0.55 | -0.0004 → -0.0004 | lose / lose / lose / lose |
| chal-00516 | 17 | 1288 | lose | -1.39 | -1.37 | -1.38 | -1.40 | -1.41 | -0.0013 → -0.0013 | lose / lose / lose / lose |
| chal-00517 | 17 | 1287 | win | 2.51 | 2.38 | 2.38 | 2.47 | 2.42 | +0.0023 → +0.0022 | win / win / win / win |
| chal-00518 | 17 | 1285 | lose | -2.53 | -2.44 | -2.46 | -2.50 | -2.46 | -0.0022 → -0.0022 | lose / lose / lose / lose |
| chal-00519 | 17 | 1288 | lose | -1.73 | -1.57 | -1.60 | -1.71 | -1.69 | -0.0014 → -0.0014 | lose / lose / lose / lose |
| chal-00520 | 17 | 1284 | lose | 0.70 | 0.82 | 0.81 | 0.76 | 0.81 | +0.0006 → +0.0006 | lose / lose / lose / lose |
| chal-00521 | 17 | 1288 | lose | 2.07 | 2.18 | 2.12 | 2.09 | 2.10 | +0.0017 → +0.0017 | lose / lose / lose / lose |
| chal-00522 | 17 | 1289 | lose | 0.57 | 0.82 | 0.79 | 0.67 | 0.78 | +0.0004 → +0.0005 | lose / lose / lose / lose |
| chal-00524 | 17 | 1293 | lose | -16.31 | -16.50 | -16.52 | -16.35 | -16.38 | -0.0225 → -0.0226 | lose / lose / lose / lose |
| chal-00525 | 17 | 1288 | lose | -1.04 | -0.81 | -0.83 | -0.93 | -0.82 | -0.0009 → -0.0008 | lose / lose / lose / lose |
| chal-00526 | 17 | 1295 | lose | 0.04 | 0.10 | 0.09 | 0.01 | -0.02 | +0.0000 → +0.0000 | lose / lose / lose / lose |
| chal-00527 | 17 | 1290 | lose | 0.53 | 0.66 | 0.65 | 0.65 | 0.78 | +0.0003 → +0.0003 | lose / lose / lose / lose |
| chal-00528 | 17 | 1291 | lose | -1.74 | -1.67 | -1.68 | -1.75 | -1.75 | -0.0013 → -0.0014 | lose / lose / lose / lose |
| chal-00531 | 17 | 1285 | lose | -2.05 | -1.88 | -1.90 | -2.02 | -1.98 | -0.0017 → -0.0016 | lose / lose / lose / lose |
| chal-00533 | 17 | 1286 | lose | 0.36 | 0.56 | 0.48 | 0.39 | 0.41 | +0.0003 → +0.0003 | lose / lose / lose / lose |
| chal-00534 | 17 | 1282 | lose | 0.32 | 0.24 | 0.23 | 0.27 | 0.21 | +0.0002 → +0.0002 | lose / lose / lose / lose |
| chal-00535 | 17 | 1287 | lose | -1.32 | -1.18 | -1.20 | -1.30 | -1.28 | -0.0010 → -0.0010 | lose / lose / lose / lose |
| chal-00536 | 18+ | 1292 | lose | 0.67 | 0.71 | 0.66 | 0.67 | 0.67 | +0.0005 → +0.0005 | lose / lose / lose / lose |
| chal-00538 | 18+ | 1294 | lose | -1.24 | -1.19 | -1.21 | -1.25 | -1.26 | -0.0008 → -0.0008 | lose / lose / lose / lose |
| chal-00539 | 18+ | 1289 | lose | 2.16 | 2.24 | 2.18 | 2.12 | 2.08 | +0.0015 → +0.0015 | lose / lose / lose / lose |
| chal-00540 | 18+ | 1290 | lose | 0.24 | 0.27 | 0.29 | 0.26 | 0.28 | +0.0002 → +0.0002 | lose / lose / lose / lose |
| chal-00541 | 18+ | 1288 | lose | -4.14 | -4.20 | -4.21 | -4.22 | -4.28 | -0.0031 → -0.0031 | lose / lose / lose / lose |
| chal-00543 | 18+ | 1290 | lose | 0.49 | 0.35 | 0.32 | 0.36 | 0.22 | +0.0003 → +0.0002 | lose / lose / lose / lose |
| chal-00544 | 18+ | 1291 | lose | 1.40 | 1.31 | 1.32 | 1.36 | 1.31 | +0.0009 → +0.0009 | lose / lose / lose / lose |
| chal-00545 | 18+ | 1294 | lose | -0.17 | 0.03 | -0.03 | -0.08 | 0.01 | -0.0001 → -0.0001 | lose / lose / lose / lose |
| chal-00546 | 18+ | 1294 | lose | -2.06 | -2.11 | -2.13 | -2.11 | -2.16 | -0.0013 → -0.0013 | lose / lose / lose / lose |
| chal-00547 | 18+ | 1294 | lose | -0.41 | -0.24 | -0.24 | -0.32 | -0.23 | -0.0003 → -0.0002 | lose / lose / lose / lose |
| chal-00548 | 18+ | 1295 | lose | 0.60 | 0.81 | 0.76 | 0.73 | 0.86 | +0.0005 → +0.0006 | lose / lose / lose / lose |
| chal-00549 | 18+ | 1295 | lose | 0.87 | 1.02 | 1.05 | 0.95 | 1.03 | +0.0005 → +0.0005 | lose / lose / lose / lose |

### Flips and Δz vs S0

- v1 w=0.01: decision flips 0 / 68; mean Δz +0.020; max |Δz| 0.259
- v2 w=0.01: decision flips 0 / 68; mean Δz +0.010; max |Δz| 0.249
- v2b w=0.01: decision flips 0 / 68; mean Δz +0.008; max |Δz| 0.138
- v1 w=0.02: decision flips 0 / 68; mean Δz +0.040; max |Δz| 0.506
- v2 w=0.02: decision flips 0 / 68; mean Δz +0.020; max |Δz| 0.485
- v2b w=0.02: decision flips 0 / 68; mean Δz +0.016; max |Δz| 0.277

### Separation on wvk-18 flat duels (|z S0| < 1): 7

| duel | z S0 | z v1 .01 | z v2 .01 | z v2b .01 | Δ|z| v1 | Δ|z| v2 | Δ|z| v2b |
|---|---|---|---|---|---|---|---|
| chal-00536 | 0.67 | 0.71 | 0.66 | 0.67 | +0.04 | -0.01 | +0.00 |
| chal-00540 | 0.24 | 0.27 | 0.29 | 0.26 | +0.03 | +0.05 | +0.02 |
| chal-00543 | 0.49 | 0.35 | 0.32 | 0.36 | -0.15 | -0.17 | -0.14 |
| chal-00545 | -0.17 | 0.03 | -0.03 | -0.08 | -0.14 | -0.14 | -0.09 |
| chal-00547 | -0.41 | -0.24 | -0.24 | -0.32 | -0.17 | -0.17 | -0.09 |
| chal-00548 | 0.60 | 0.81 | 0.76 | 0.73 | +0.21 | +0.16 | +0.13 |
| chal-00549 | 0.87 | 1.02 | 1.05 | 0.95 | +0.15 | +0.18 | +0.08 |

separates (Δ|z| > 0.25): v1 0 / 7, v2 0 / 7, v2b 0 / 7

## Generic credit share (w = 0.01; generic = E5 regex on bash / bash-tool actions)

| dialect | valid turns (both sides) | v1: positive credit to generic / all positive | v2 (exact list + no terminus) | v2b (+ head list) |
|---|---|---|---|---|
| bash | 80351 | 6.332 / 34.580 = 18.3% | 3.918 / 28.387 = 13.8% | 0.123 / 8.437 = 1.5% |
| boxed | 3676 | 0.000 / 1.697 = 0.0% | 0.000 / 1.697 = 0.0% | 0.000 / 1.697 = 0.0% |
| terminus_json | 2665 | 0.000 / 0.927 = 0.0% | 0.000 / 0.000 = 0.0% | 0.000 / 0.000 = 0.0% |
| text | 20159 | 0.000 / 0.000 = 0.0% | 0.000 / 0.000 = 0.0% | 0.000 / 0.000 = 0.0% |
| tool_call | 59044 | 0.863 / 19.372 = 4.5% | 0.863 / 18.758 = 4.6% | 0.020 / 5.930 = 0.3% |
