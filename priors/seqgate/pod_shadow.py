"""affine-seqgate-shadow pod: rent (ONE pod, 8xH100/8xH200 whole node, <= $30/h), provision as a probe clone, release.
Uses ops/pods/replace_pod.py functions (rent ledger, golden copy, provisioning steps, ssh) with an in-memory role spec;
roles.json is NOT modified. Never touches any other pod: release removes only the exact id this tool rented, and only if Lium
still lists it under the name affine-seqgate-shadow.
  python3 pod_shadow.py plan | rent | provision | health | release [--why TEXT] | status"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/subnet120-rewrite/ops/pods'))
import replace_pod as RP

NAME = 'affine-seqgate-shadow'; ROLE = 'seqgate-shadow'
D = Path(os.path.expanduser('~/subnet120-rewrite/state/root-audits/20261102-seqgate-shadow/state')); D.mkdir(parents=True, exist_ok=True)
PJ = D / 'pod.json'
P = RP.ROLES['probe']
RP.ROLES[ROLE] = dict(what='SN120 sequential-gate SHADOW pod (Jacob 2026-11-02 11:14 CR); never live gate', name=NAME, names=[NAME],
                      gpu_specs=[dict(gpu='H100', count=8, whole_node=True), dict(gpu='H200', count=8, whole_node=True)],
                      max_price_h=30, min_disk_gb=800, tiers=['secure'], max_rents_24h=1, golden=dict(from_role='probe'),
                      provision=P['provision'], health=P['health'])


def cands():
    out = []
    for spec in RP.ROLES[ROLE]['gpu_specs']:
        RP.ROLES[ROLE]['gpu_specs'] = [spec]; out += RP.candidates(ROLE)
    RP.ROLES[ROLE]['gpu_specs'] = [dict(gpu='H100', count=8, whole_node=True), dict(gpu='H200', count=8, whole_node=True)]
    return sorted(out, key=lambda r: (r['price_h'], r['gpu'] != 'H200', -(r['down'] or 0)))


def mine():
    p = RP.rjson(PJ)
    if not p: return None, None
    ps = RP.lium_ps(); live = next((x for x in ps if x['id'] == p['id']), None)
    return p, live


def pod_obj(p): return RP.Pod(p['id'], p['ip'], p['port'], p.get('huid'), NAME)


def main():
    cmd = sys.argv[1]
    if cmd == 'plan':
        for c in cands()[:6]: print(c)
        print('rents 24h global', RP.rents_24h(), 'role', RP.rents_24h(ROLE)); return 0
    if cmd == 'status':
        p, live = mine(); print(json.dumps(dict(pod=p, lium=live and dict(status=live.get('status'), name=live.get('name'), price=live.get('price_per_hour'))), default=str)); return 0
    if cmd == 'rent':
        p, live = mine()
        if live: print('already have', p['huid'], p['id']); return 0
        if any(x.get('name') == NAME for x in RP.lium_ps()): print('refuse: a pod named %s already exists' % NAME); return 2
        if RP.rents_24h(ROLE) >= 1 or RP.rents_24h() >= RP.LIM['max_rents_24h_global']: print('refuse: rent cap'); return 2
        for c in cands()[:3]:
            if c['price_h'] > 30: continue
            print('renting', c); RP.append(RP.ST / 'rents.jsonl', dict(at=RP.now(), at_cr=RP.cr(), role=ROLE, event='rent', node=c['huid'], price_h=c['price_h'], gpu='%dx%s' % (c['count'], c['gpu'])))
            rc, o, e = RP.lium('up', c['huid'], '-c', str(c['count']), '--name', NAME, '--verify-gpus', '--strict-gpus', '--no-volume-encryption', '-y', '--json', '--timeout', '900', timeout=1000)
            if rc == 0:
                try:
                    new = json.loads(o)['pod']
                    rec = dict(id=new['id'], huid=new.get('huid'), ip=new['ip'], port=int((new.get('ports') or {}).get('22')), price_h=c['price_h'], gpu='%dx%s' % (c['count'], c['gpu']),
                               node=c['huid'], rented_at=time.time(), rented_at_cr=RP.cr())
                    RP.wjson(PJ, rec, 0o600); RP.append(RP.ST / 'rents.jsonl', dict(at=RP.now(), at_cr=RP.cr(), role=ROLE, event='pod', id=rec['id'], huid=rec['huid']))
                    print('rented', rec); return 0
                except Exception as x: print('parse failed', x, o[-300:])
            print('rent failed', rc, (e or o)[-300:])
        return 1
    if cmd == 'provision':
        p, live = mine()
        if not live: print('no pod'); return 2
        pod = pod_obj(p)
        for _ in range(40):
            if RP.ssh_ok(pod.ip, pod.port, pod.id): break
            time.sleep(15)
        ctx = {}
        steps = RP.ROLES[ROLE]['provision']
        if '--skip-golden' in sys.argv:   # golden copy already restored by parallel subset streams: only (re)make the links
            links = RP.gspec(ROLE).get('links', {})
            pod.run('; '.join('mkdir -p $(dirname %s); [ -L %s ] || [ ! -e %s ] && ln -sfn %s %s' % (k, k, k, v, k) for k, v in links.items()) + '; echo linked')
            steps = [x for x in steps if x['type'] != 'golden_restore']
        for i, s in enumerate(steps):
            t0 = time.time(); r = RP.step(pod, ROLE, s, ctx, False); print('step', i + 1, s['type'], 'ok %ds' % (time.time() - t0), str(r).strip()[-160:], flush=True)
        print(pod.run(RP.ROLES[ROLE]['health'], timeout=600)); p['provisioned_at_cr'] = RP.cr(); RP.wjson(PJ, p, 0o600); return 0
    if cmd == 'health':
        p, live = mine(); print(pod_obj(p).run(RP.ROLES[ROLE]['health'], timeout=600)); return 0
    if cmd == 'release':
        why = sys.argv[sys.argv.index('--why') + 1] if '--why' in sys.argv else 'release'
        p, live = mine()
        if not p: print('nothing rented'); return 0
        if live and live.get('name') == NAME and not RP.excluded(live):
            rc, o, e = RP.lium('rm', p['id'], '-y', timeout=300); print('lium rm', p['id'], rc, (o or e)[-200:])
            ok = rc == 0
        else: ok = True; print('not listed (already gone)')
        if ok:
            hrs = (time.time() - p['rented_at']) / 3600
            p.update(released_at=time.time(), released_at_cr=RP.cr(), hours=round(hrs, 3), spend_usd=round(hrs * p['price_h'], 2), why=why)
            RP.append(RP.ST / 'rents.jsonl', dict(at=RP.now(), at_cr=RP.cr(), role=ROLE, event='released', id=p['id'], hours=p['hours'], spend_usd=p['spend_usd']))
            os.replace(PJ, D / ('pod.released-%d.json' % int(time.time()))); RP.wjson(D / ('pod.released-%d.json' % int(time.time())), p, 0o600)
            print('released', p['spend_usd'], 'USD'); return 0
        return 1
    print(__doc__); return 2


if __name__ == '__main__': sys.exit(main())
