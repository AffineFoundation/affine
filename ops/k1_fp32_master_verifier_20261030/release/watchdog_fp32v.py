#!/usr/bin/env python3
"""fp32v-watchdog (RA/20261030-fp32-verifier): watches the first 2 epochs after each change goes live and auto-reverts.
Trainer v8.3.3f revert if: TRAINING-RESULT missing 16 min after TRAINING-JOB,
  or training (job->result) > 13 min (baseline ~9 min), or the epoch is long/stalled with training as the slow part.
Verifier revert if: a watched epoch's gate decision is a held-out fail-closed/error, or the epoch is long/stalled with the gate as the slow part.
Epoch limit: 36 min fin-to-fin (baseline ~29.6 min, +20%); 45 min for the round-opening epoch (fresh held-out baseline, same rule as ho-watchdog).
env FROM=<first epoch with any change>."""
import json, os, subprocess, time
from pathlib import Path
D = Path('/home/const/subnet120-rewrite/state/root-audits/20261030-fp32-verifier'); AV = D / 'activation'
STATE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1'); E = STATE / 'epochs'
GE = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs')
LOG = AV / 'watchdog.log'; FROM = int(os.environ['FROM'])
LIMIT, LIMIT_OPEN, TRAIN_MAX, TRAIN_STALL = 36 * 60, 45 * 60, 13 * 60, 16 * 60
def cr(): return time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))
def log(msg, prog=False):
    line = '%s %s' % (cr(), msg); open(LOG, 'a').write(line + '\n'); print(line, flush=True)
    if prog: open(D / 'PROGRESS.md', 'a').write('- %s: fp32v-watchdog: %s\n' % (cr()[:5] + ' CR', msg))
def rj(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def mt(p): return os.path.getmtime(p) if os.path.exists(p) else None
def revert(kind, why):
    flag = AV / ('REVERTED-%s.json' % kind)
    if flag.exists(): return
    flag.write_text(json.dumps(dict(kind=kind, why=why, at=cr())))
    log('AUTO-REVERT %s: %s' % (kind, why), prog=True)
    subprocess.run(['systemd-run', '--user', '--unit', 'fp32v-revert-%s' % kind, '--collect', '/usr/bin/env', 'MODE=revert-%s' % kind,
                    '/usr/bin/bash', str(AV / 'activate_fp32v.sh')], check=False)
def main():
    done = {}
    log('armed: FROM=e%d, watching 2 epochs per change; limits %d/%d min, training %d min' % (FROM, LIMIT // 60, LIMIT_OPEN // 60, TRAIN_MAX // 60), prog=True)
    t_start = time.time()
    while True:
        tf = rj(AV / 'TRAINER-f.json'); vf = rj(AV / 'VERIFIER-ON.json')
        watch = {}
        if vf: watch.update({e: watch.get(e, set()) | {'verifier'} for e in (vf['from_epoch'], vf['from_epoch'] + 1)})
        if tf: watch.update({e: watch.get(e, set()) | {'trainer'} for e in (tf['from_epoch'], tf['from_epoch'] + 1)})
        if not tf and time.time() - t_start > 3 * 3600:
            log('trainer not switched within 3 h; watchdog exits', prog=True); return
        for e, kinds in sorted(watch.items()):
            if done.get(e): continue
            if (AV / 'REVERTED-trainer.json').exists(): kinds = kinds - {'trainer'}
            if (AV / 'REVERTED-verifier.json').exists(): kinds = kinds - {'verifier'}
            if not kinds: done[e] = 'reverted'; continue
            prevfin = mt(E / str(e - 1) / 'FINALIZED.json')
            if not prevfin: continue
            ed = E / str(e)
            if (ed / 'ABANDONED-SPECULATIVE.json').exists():
                done[e] = 'abandoned'; log('e%d abandoned (speculative) - not counted' % e)
                # the watch window moves one epoch on
                for k, f in (('trainer', 'TRAINER-f.json'), ('verifier', 'VERIFIER-ON.json')):
                    d = rj(AV / f)
                    if d and k in kinds and d['from_epoch'] == e: d['from_epoch'] = e + 1; (AV / f).write_text(json.dumps(d))
                continue
            opening = vf is not None and e == vf['from_epoch']
            limit = LIMIT_OPEN if opening else LIMIT
            tj, tr, fin = mt(ed / 'TRAINING-JOB.json'), mt(ed / 'TRAINING-RESULT.json'), mt(ed / 'FINALIZED.json')
            dec = rj(GE / str(e) / 'DECISION.json')
            train_s = (tr - tj) if (tj and tr) else ((time.time() - tj) if tj else 0)
            if 'trainer' in kinds and tj:
                if not tr and time.time() - tj > TRAIN_STALL: revert('trainer', 'e%d no TRAINING-RESULT %d min after TRAINING-JOB' % (e, TRAIN_STALL // 60))
                elif tr and tr - tj > TRAIN_MAX: revert('trainer', 'e%d training took %.1f min (> %d)' % (e, (tr - tj) / 60, TRAIN_MAX // 60))
            if 'verifier' in kinds and dec:
                rs = ' '.join(map(str, dec.get('reasons') or []))
                if 'fail' in rs.lower() and 'closed' in rs.lower() or dec.get('decision') == 'error':
                    revert('verifier', 'e%d gate decision %s: %s' % (e, dec.get('decision'), rs[:200]))
            el = (fin or time.time()) - prevfin
            if el > limit + (0 if fin else 120):
                slow_train = train_s > TRAIN_MAX
                if slow_train and 'trainer' in kinds: revert('trainer', 'e%d epoch %.1f min > %d (training %.1f min)' % (e, el / 60, limit // 60, train_s / 60))
                elif 'verifier' in kinds: revert('verifier', 'e%d epoch %.1f min > %d (gate side; training %.1f min)' % (e, el / 60, limit // 60, train_s / 60))
                elif 'trainer' in kinds: revert('trainer', 'e%d epoch %.1f min > %d' % (e, el / 60, limit // 60))
                done[e] = 'long'
            elif fin:
                done[e] = 'ok'
                log('e%d OK: epoch %.1f min (limit %d), training %.1f min, gate %s z %s [%s]' % (
                    e, el / 60, limit // 60, train_s / 60, (dec or {}).get('decision'), (dec or {}).get('z', ((dec or {}).get('evaluated') or {}).get('z')), ','.join(sorted(kinds))), prog=True)
        if watch and all(done.get(e) for e in watch) and (tf or (AV / 'REVERTED-trainer.json').exists() or os.environ.get('NO_TRAINER')):
            log('all watched epochs done: %s - watchdog exits' % json.dumps({str(k): v for k, v in sorted(done.items())}), prog=True); return
        time.sleep(20)
if __name__ == '__main__': main()
