#!/usr/bin/env python3
"""usage: patch_texts_prior.py apply|revert "<switch time CR>" LLMS_TXT STATE_DIR
apply: saves LLMS_TXT byte-exact to STATE_DIR/llms.pre-prior.txt, then replaces sections 5-7 (from '## 5. The task pool' up to
'## 8. What winning means') with llms_sections_5_7.prior-v1.md (__SWITCH__ -> the switch time) and updates the header, the
section 9 'Borderline difficulty' bullet and the section 10 pool bullet. revert: restores the saved copy (byte-exact)."""
import os, re, shutil, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
mode, T, p, st = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]); bk = st / 'llms.pre-prior.txt'
def write(path, text):
    t = Path(str(path) + '.tmp.pt'); t.write_text(text); os.chmod(t, os.stat(path).st_mode & 0o777); os.replace(t, path)
if mode == 'revert':
    if not bk.exists(): sys.exit('no saved copy %s' % bk)
    shutil.copyfile(bk, str(p) + '.tmp.pt'); os.chmod(str(p) + '.tmp.pt', os.stat(p).st_mode & 0o777); os.replace(str(p) + '.tmp.pt', p)
    print('revert', p); sys.exit(0)
s = open(p).read()
if '<!-- prior-v1-publish -->' in s: print('already applied', p); sys.exit(0)
a = s.index('## 5. The task pool'); b = s.index('## 8. What winning means')
new = (HERE / 'llms_sections_5_7.prior-v1.md').read_text().replace('__SWITCH__', T)
out = s[:a] + new + s[b:]
out = re.sub(r'^Last updated: [^\n]*?\. ', 'Last updated: %s (UTC-6). ' % T, out, count=1, flags=re.M)
out = out.replace('gate heldout ivw-v2', 'gate heldout prior-v1', 1)
out = out.replace('keeps the candidate only if it beats the head on a secret held-out sample of the task pool (GREEN).',
                  'keeps the candidate only if it beats the head on a sample of the public gate pool that is drawn after the upload\ndeadline from a committed secret and a future block hash (GREEN).', 1)
old9 = s[s.index('- Borderline difficulty:'):s.index('- Fewer capped replies:')] if '- Borderline difficulty:' in s else None
if old9:
    out = out.replace(old9, '- Borderline difficulty: the draw favours tasks the head sometimes solves (pi ~ sqrt(p(1-p)) from the prior), and every\n'
                            '  drawn task counts with weight 1/pi, so the estimate covers the whole pool (affine_math + dapo). Correct samples on\n'
                            '  problems the head usually fails or caps out on are where the signal is.\n', 1)
out = out.replace('- the 7,496-task pool;', '- the 7,496-task affine_math file and the 19,305-task gate pool (miner/data/gate-pool-v1.jsonl + manifest);', 1)
st.mkdir(parents=True, exist_ok=True)
if not bk.exists(): shutil.copyfile(p, bk)
write(p, out); print('apply', p)
