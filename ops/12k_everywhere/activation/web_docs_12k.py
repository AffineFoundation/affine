#!/usr/bin/env python3
"""RA/20261102-12k-everywhere public docs on affine.io (marker 12k-everywhere-20261102): llms.txt says 12,288 output tokens everywhere.
  build FROM_EPOCH OUTDIR   patched copy in OUTDIR (nothing live touched)
  apply FROM_EPOCH          build from the CURRENT live llms.txt, back it up to RA/20261102-12k-everywhere/web/backup, install
  revert                    restore the backup (only if llms.txt is still what apply installed; otherwise left as is + reported)"""
import json, os, re, shutil, sys, hashlib, time
from pathlib import Path
W = Path('/home/const/subnet120/affine/website'); D = Path('/home/const/subnet120-rewrite/state/root-audits/20261102-12k-everywhere'); WB = D / 'web'
def rep(s, o, n):
    c = s.count(o)
    if c != 1: raise SystemExit('llms: anchor count %d: %r' % (c, o[:90]))
    return s.replace(o, n)
def llms(s, fe):
    fe = str(fe)
    s = rep(s, '- EOS rule: an output of up to 8191 tokens gets `<|im_end|>` appended (it ended). Exactly 8192 tokens = cut at the cap\n'
               '  (no EOS; capped). More than 8192 is invalid. Prompt + output can be at most 12,288 tokens.\n'
               '  (This is the batch-file submission limit and is unchanged. The gate evaluation cap is 12,288 output tokens, section 5.)\n',
            '- EOS rule: an output of up to 12,287 tokens gets `<|im_end|>` appended (it ended). Exactly 12,288 tokens = cut at the cap\n'
            '  (no EOS; capped). More than 12,288 is invalid. Prompt + output can be at most 16,384 tokens.  <!-- 12k-everywhere-20261102 -->\n'
            '  12,288 output tokens apply everywhere from epoch %s: generation (sample with max_tokens 12,288), batch files, training\n'
            '  and the gate (section 5). Before epoch %s batch files were capped at 8,192 output / 12,288 context.\n'
            '  Transition: files generated at 8K still count. An output of exactly 8,192 tokens (no EOS) is still read as capped.\n' % (fe, fe))
    s = rep(s, '  The trainer adds an overlong penalty to the reward: -0.5 x clip((len - 6000) / (8192 - 6000), 0, 1).\n',
            '  The trainer adds an overlong penalty to the reward: -0.5 x clip((len - 10096) / (12288 - 10096), 0, 1) (len = output\n'
            '  tokens incl. the EOS; from epoch %s, same 2,192-token ramp moved to the new cap; before: (len - 6000) / (8192 - 6000)).\n' % fe)
    s = rep(s, 'Trainer v8.3.3l, gate', 'Trainer v8.3.3m, gate')
    s = rep(s, '## 4. Training: the exact v8.3.3l step (one stateless step per turn)', '## 4. Training: the exact v8.3.3m step (one stateless step per turn)')
    s = rep(s, 'miner/validator/trainer-v833l (+ engine/) and the live settings are in miner/validator/trainer-v833l/trainer-config.public.json.\n',
            'miner/validator/trainer-v833m (+ engine/) and the live settings are in miner/validator/trainer-v833m/trainer-config.public.json.\n'
            'v8.3.3m (from epoch %s) = v8.3.3l + the 12K cap: rollouts up to 16,384 tokens of prompt + output are trained in full\n'
            '(no truncation) and the free-file overlong window is 10096..12288. Everything else is the v8.3.3l step.\n' % fe)
    s = rep(s, '- the trainer v8.3.3l host + engine;', '- the trainer v8.3.3m host + engine;')
    s = rep(s, 'miner/validator/trainer-v833l + engine/. kit/train_step.py', 'miner/validator/trainer-v833m + engine/. kit/train_step.py')
    s, n = re.subn(r'Last updated: [0-9-]+ [0-9:]+ CR \(UTC-6\)\.', 'Last updated: %s CR (UTC-6).' % time.strftime('%Y-%m-%d %H:%M', time.gmtime(time.time() - 6 * 3600)), s, count=1)
    if n != 1: raise SystemExit('llms: Last updated anchor')
    return s
def sha(b): return hashlib.sha256(b).hexdigest()
def build(fe, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'llms.txt').write_text(llms((W / 'llms.txt').read_text(), fe)); return out / 'llms.txt'
def put(src, dst):
    tmp = dst.with_name(dst.name + '.12k-new'); shutil.copyfile(src, tmp); os.chmod(tmp, 0o644); os.replace(tmp, dst)
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build': print('built', build(sys.argv[2], sys.argv[3]))
    elif cmd == 'apply':
        if (WB / 'APPLIED.json').exists(): raise SystemExit('already applied')
        (WB / 'backup').mkdir(parents=True, exist_ok=True); shutil.copy2(W / 'llms.txt', WB / 'backup/llms.txt')
        n = build(sys.argv[2], WB / 'next'); put(n, W / 'llms.txt')
        (WB / 'APPLIED.json').write_text(json.dumps(dict(from_epoch=int(sys.argv[2]), llms_sha256=sha(n.read_bytes()), at=time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime(time.time() - 6 * 3600)))) + '\n')
        print('applied', sys.argv[2])
    elif cmd == 'revert':
        a = json.loads((WB / 'APPLIED.json').read_text())
        if sha((W / 'llms.txt').read_bytes()) != a['llms_sha256']: raise SystemExit('llms.txt changed since apply - left as is (backup: %s)' % (WB / 'backup/llms.txt'))
        put(WB / 'backup/llms.txt', W / 'llms.txt'); os.replace(WB / 'APPLIED.json', WB / ('REVERTED-%d.json' % time.time())); print('reverted')
    else: raise SystemExit(__doc__)
