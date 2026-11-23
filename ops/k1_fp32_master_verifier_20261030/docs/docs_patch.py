#!/usr/bin/env python3
"""RA/20261030-fp32-verifier: llms.txt + k1/spec/step-recipe.json (+ /k1/replay/host/fresh_adam_v833f.md). usage: docs_patch.py TRAINER_FROM VERIFIER_FROM"""
import json, sys, time, shutil, os
from pathlib import Path
W = Path('/home/const/subnet120/affine/website'); TF, VF = sys.argv[1], sys.argv[2]
ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()); MARK = '<!-- fp32-verifier-20261030 -->'
L = W / 'llms.txt'; s = L.read_text()
if MARK not in s:
    shutil.copy2(L, str(L) + '.bak-pre-fp32-verifier-' + ts)
    step = '''**fp32 head master (from epoch %s).** The trainer now keeps a persistent fp32 master copy of the head. The step is the same one AdamW step from zero moments (lr 5e-7, per-tensor relative cap 2e-5, no moments carried), but it is taken on the head's fp32 master M_H instead of fp32(bf16 head): `d_t = lr * (g_t/(|g_t|+1e-8) + 0.01*M_H,t*[2-D, not bias])`, `r_t = min(1, 2e-5*(1-1e-6)*||M_H,t||/||d_t||)`, `M_C = M_H - fp32(s*r_t)*d_t`, and the bf16 candidate is `bf16_round_nearest_even(M_C)`. Small updates that round away in bf16 are kept in the master and accumulate across GREEN turns instead of being lost. RED candidates are discarded with their master; a GREEN makes M_C the new head master. The first v8.3.3f turn (epoch %s) starts from the exact fp32 upcast of the bf16 head. Each turn commits `training.optimizer.sgd.fp32_master` in turn.json: `input_master_digest` / `output_master_digest` (sha256 of the canonical JSON `{tensor name: sha256(little-endian fp32 bytes)}`), the bf16 sha256s and the master source (`bootstrap-upcast`, `memory` or `disk`). On every GREEN the new head master is published: safetensors shards plus `meta.json` in R2 under `checkpoints/<C>/fp32-master/` (same bucket and access as `checkpoints/<C>/model/`), with a manifest at `https://affine.io/k1/fp32-master/<C>.json` and an index at `https://affine.io/k1/fp32-master/index.json`. A replay of turn N needs the head master from the previous GREEN's manifest (or the bf16 upcast for the bootstrap turn); `bf16_rne(master)` must equal the published bf16 head. Recipe: `https://affine.io/k1/spec/step-recipe.json` -> `optimizer.fp32_master`.

''' % (TF, TF)
    anchor = '**The loss (what the trainer actually runs).**'
    assert anchor in s; s = s.replace(anchor, step + anchor, 1)
    ver = '''**Verifier fix (gate grading, from epoch %s).** The gate's `affine_math` grade now also accepts correct answers the strict projection or `verify.py` scored 0 (Allan's fix, `catoneone/sn120-k1-fixes` 786cdb9, `code/lenient_grader.py`, rules P1/P2/E1-E7: a bare box after a closed `$$`, trailing text or several boxes (the last complete box is graded), `\\text{}`/thousands separators, multiple-choice letters mapped to their option, pmatrix vs tuple, single-variable inequalities vs intervals, `kI` vs an explicit matrix, `i` as the imaginary unit). It is applied only when the live grade is not positive and the reply ended with EOS; capped replies stay unresolved, and a live positive is never changed. On the 4,073 finished K=1 rollouts of epochs 196-260 it recovers 481 correct answers and changes no wrong answer. It applies to the head and candidate runs alike (it is part of the probe worker). The open held-out round was closed when it went live, and a fresh round with a new baseline started at epoch %s on the same head. **Miner-side grading (the native grades in your batch and the trainer reward) is unchanged for now**: they must match the validator's native grader, so that changes only with a coordinated client release.

''' % (VF, VF)
    anchor2 = '**Credit.** Each of the last'
    assert anchor2 in s; s = s.replace(anchor2, ver + anchor2, 1)
    s = s.replace('## K=1 per-UID training (LIVE from epoch 196) — added October 28, 2026', '## K=1 per-UID training (LIVE from epoch 196) — added October 28, 2026; fp32 head master and gate verifier fix October 30, 2026 ' + MARK, 1)
    t = Path(str(L) + '.tmp'); t.write_text(s); os.chmod(t, 0o644); os.replace(t, L)
R = W / 'k1' / 'spec' / 'step-recipe.json'; d = json.loads(R.read_text())
if 'fp32_master' not in d['optimizer']:
    shutil.copy2(R, str(R) + '.bak-pre-fp32-verifier-' + ts)
    d['optimizer']['fp32_master'] = dict(
        applies_from_epoch=int(TF), trainer='v8.3.3f (engine sgd_fp32_master: true)',
        rule='M_H = the head fp32 master; d_t and r_t are computed from M_H (not fp32(bf16 W)); M_C = M_H - fp32(s * r_t) * d_t; candidate bf16 = bf16_round_nearest_even(M_C); GREEN -> M_C becomes the head master; RED -> discarded',
        bootstrap='the first v8.3.3f turn (epoch %s) uses M_H = fp32(bf16 head) exactly' % TF,
        moments='not persisted (fresh zero state every turn, as before)',
        digest='sha256 of canonical JSON {tensor name: sha256(little-endian fp32 bytes)} (sorted keys, separators , :); bf16 digest: same over bf16 bytes',
        committed_in='turn.json training.optimizer.sgd.fp32_master (input/output master digest, input/output bf16 sha256, source bootstrap-upcast|memory|disk)',
        published='on every GREEN: R2 checkpoints/<C>/fp32-master/fp32-master-XXXXX-of-XXXXX.safetensors + meta.json; manifest https://affine.io/k1/fp32-master/<C>.json; index https://affine.io/k1/fp32-master/index.json',
        check='bf16_round_nearest_even(published master) == published bf16 head of <C>; master digest == turn.json output_master_digest')
    d['optimizer']['update_v833f'] = 'W_new = bf16_round_nearest_even(M_C), M_C = M_H - fp32(s * r_t) * d_t (from epoch %s)' % TF
    d['grading_note'] = dict(applies_from_epoch=int(VF), gate='gate probe grades use the live native grade plus Allan lenient rules (catoneone/sn120-k1-fixes 786cdb9) for non-positive EOS replies',
                             training='training rewards use the native grades in the signed dispatch rows (unchanged; miner and validator grader must match)')
    t = Path(str(R) + '.tmp'); t.write_text(json.dumps(d, indent=1) + '\n'); os.chmod(t, 0o644); os.replace(t, R)
print('ok', L.stat().st_size, R.stat().st_size)
