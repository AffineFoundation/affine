#!/usr/bin/env python3
"""caps-check: diff every consumer of the SN120 token caps against caps.json (read-only).

  python3 tools/caps/caps_check.py --scope repo            # CI / pre-push: files in this repo
  python3 tools/caps/caps_check.py --scope host,pods       # validator host: live files + live pods over ssh (read-only)
  python3 tools/caps/caps_check.py --scope public          # affine.io + GitHub copies
  options: --json OUT (machine-readable result; '-' = stdout, table to stderr), --quiet (only problems), --caps PATH
Exit 0 = every live/derived/docs value equals caps.json and no old cap value appears anywhere; 1 = a mismatch or an old cap
value (FAIL); 2 = could not check something required.
Old-cap scan (consumers.OLD_CAP_*): every scanned file of the chosen scopes must hold no old token-cap value / legacy field /
transition wording; reviewed non-cap hits (OLD_CAP_ALLOW) are listed as INFO.
Consumers and classes: tools/caps/consumers.py. Pods: the current pod of each roles.json role (ops/pods/replace_pod.py
current_ref); excluded pods (roles.json _excluded) are never touched; commands are read-only (cat / grep / curl).
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capslib as L
import consumers as CS

ACTIVE = Path(CS.REPO) / 'state/caps/ACTIVE.json'   # written by the caps-v1 activation (gates the 'after' checks)


def places(con, scopes, root):
    out = []
    if 'repo' in scopes and con.get('repo'):
        for p in ([con['repo']] if isinstance(con['repo'], str) else con['repo']):
            out.append(('repo', p, lambda p=p: (root / p).read_text()))
    if 'host' in scopes and con.get('host'):
        for h in ([con['host']] if isinstance(con['host'], str) else con['host']):
            out.append(('host', h, lambda h=h: Path(h).read_text()))
    if 'host' in scopes and con.get('host_glob'):
        out.append(('host', con['host_glob'], lambda c=con: newest(c)))
    if 'pods' in scopes and con.get('pod'): out.append(('pod', '%s: %s' % con['pod'], ('pod',) + tuple(con['pod'])))
    if 'public' in scopes and con.get('url'): out.append(('public', con['url'], lambda u=con['url']: fetch(u)))
    return out


class Skip(Exception):
    pass


def newest(con):
    """The newest file matching con['host_glob'] written after con['newer_than'] (a marker file); Skip if there is none yet."""
    import glob
    m = Path(con['newer_than']) if con.get('newer_than') else None
    if m is not None and not m.exists(): raise Skip('marker %s not there yet' % m)
    t0 = m.stat().st_mtime if m is not None else 0
    fs = sorted((f for f in glob.glob(con['host_glob']) if os.path.getmtime(f) > t0), key=os.path.getmtime)
    if not fs: raise Skip('no file newer than %s yet' % (m or 'epoch 0'))
    return Path(fs[-1]).read_text()


def fetch(url):
    req = urllib.request.Request(url + ('&' if '?' in url else '?') + 'capscheck=%d' % time.time(), headers={'User-Agent': 'sn120-caps-check'})
    return urllib.request.urlopen(req, timeout=30).read().decode()


_PODS = {}
def pod_run(role, cmd):
    if 'rp' not in _PODS:
        sys.path.insert(0, CS.REPO + '/ops/pods'); sys.argv = sys.argv[:1]
        import replace_pod as rp; _PODS['rp'] = rp
    rp = _PODS['rp']
    if role not in _PODS:
        ref = rp.current_ref(role) or {}
        _PODS[role] = rp.Pod(ref['id'], ref['ip'], ref['port']) if ref.get('id') and ref.get('ip') else None
    if _PODS[role] is None: raise RuntimeError('no current pod for role %s' % role)
    return _PODS[role].run(cmd, timeout=90, check=False)


def old_cap_scan(scopes, root):
    """One row per scanned file: FAIL on any old cap value / legacy field / transition wording not in OLD_CAP_ALLOW."""
    import glob
    rows = []
    for scope in ('repo', 'host', 'public'):
        if scope not in scopes: continue
        if scope == 'repo':
            files = sorted({f for g in CS.OLD_CAP_SCAN['repo'] for f in glob.glob(str(root / g)) if os.path.isfile(f)})
            todo = [(os.path.relpath(f, root), (lambda f=f: Path(f).read_text(errors='replace'))) for f in files]
        elif scope == 'host':
            hosts = [(max(glob.glob(h), key=os.path.getmtime, default=h) if '*' in h else h) for h in CS.OLD_CAP_SCAN['host']]   # '*' = newest match
            todo = [(h, (lambda h=h: Path(h).read_text(errors='replace'))) for h in hosts]
        else:
            todo = [(u, (lambda u=u: fetch(u))) for u in CS.OLD_CAP_SCAN['public']]
        for name, get in todo:
            row = dict(id='no_old_cap', cls='live', scope=scope, where=name)
            try: text = get()
            except Exception as e: rows.append(dict(row, status='ERROR', detail='%s: %s' % (type(e).__name__, str(e)[:200]))); continue
            key = name.split('?')[0]
            fails, infos = L.old_cap_hits(key, text, CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
            if fails:
                rows.append(dict(row, status='FAIL', detail='old cap value(s): %s' % '; '.join('%s: %s (%s)' % h for h in fails[:4])))
            elif infos:
                rows.append(dict(row, status='INFO', detail='reviewed non-cap: %s' % '; '.join('%s (%s)' % (h[0], h[2][:90]) for h in infos[:3])))
            else:
                rows.append(dict(row, status='OK', detail='no old cap value'))
    return rows


def run(scopes, root, C, quiet=False):
    rows = []
    active = ACTIVE.exists()
    for con in CS.CONSUMERS:
        for scope, where, get in places(con, scopes, root):
            row = dict(id=con['id'], cls=con['cls'], scope=scope, where=where)
            if con.get('after') and scope in ('host', 'public') and not active:
                rows.append(dict(row, status='SKIP', detail='before the caps-v1 activation')); continue
            try:
                text = pod_run(*get[1:]) if isinstance(get, tuple) else get()
            except Skip as e:
                rows.append(dict(row, status='SKIP', detail=str(e)[:200])); continue
            except Exception as e:
                rows.append(dict(row, status='ERROR', detail='%s: %s' % (type(e).__name__, str(e)[:200]))); continue
            if con.get('optional') and not text.strip():
                rows.append(dict(row, status='SKIP', detail='not serving')); continue
            res = L.check_text(con, text, C)
            bad = [r for r in res if not r['ok']]
            st = 'OK' if not bad else 'FAIL'
            rows.append(dict(row, status=st, results=res,
                             detail='; '.join('%s=%s found %s (%d)' % (r['key'], r['expected'], ','.join(r['found']) or 'nothing', r['n']) for r in (bad or res))))
    rows += old_cap_scan(scopes, root)
    if 'host' in scopes:
        for nl in CS.NO_LITERALS:
            row = dict(id=nl['id'], cls='live', scope='host', where=nl['host'])
            if nl.get('after') and not active: rows.append(dict(row, status='SKIP', detail='before the caps-v1 activation')); continue
            vals = sorted({C[k] for k in nl['keys']})
            try: text = Path(nl['host']).read_text()
            except Exception as e: rows.append(dict(row, status='ERROR', detail=str(e)[:200])); continue
            hits = [(i + 1, l.strip()[:120]) for i, l in enumerate(text.splitlines())
                    if re.search(r'(?<![\d.])(%s)(?![\d])' % '|'.join(map(str, vals)), l)]
            rows.append(dict(row, status='FAIL' if hits else 'OK', detail=('hard-coded cap literal(s): %s' % hits[:5]) if hits else
                             'no hard-coded %s (reads caps.py)' % vals))
        try:   # roles.json: every push_file must survive replace_pod's secret filter (the serve.env lesson of 2026-11-03)
            import fnmatch
            sys.path.insert(0, CS.REPO + '/ops/pods'); sys.argv = sys.argv[:1]
            import replace_pod as rp; _PODS['rp'] = rp
            bad = [(role, s['src']) for role, R in rp.ROLES.items() for s in R.get('provision', [])
                   if s.get('type') == 'push_file' and any(fnmatch.fnmatch(os.path.basename(rp.X(s['src'])), p) for p in rp.SECRET_PATTERNS)]
            rows.append(dict(id='roles_secret_filter', cls='live', scope='host', where=CS.REPO + '/ops/pods/roles.json', status='FAIL' if bad else 'OK',
                             detail=('push_file steps replace_pod would refuse: %s' % bad) if bad else 'no push_file step is refused by SECRET_PATTERNS'))
        except Exception as e:
            rows.append(dict(id='roles_secret_filter', cls='live', scope='host', where='ops/pods/roles.json', status='ERROR', detail=str(e)[:200]))
        pins = json.loads(Path(CS.FF + '/FF-PINS.json').read_text())['files']
        sha = hashlib.sha256(Path(CS.FF + '/ff_format.py').read_bytes()).hexdigest()
        rows.append(dict(id='ff_pins', cls='derived', scope='host', where=CS.FF + '/FF-PINS.json', status='OK' if pins.get('ff_format.py') == sha else 'FAIL',
                         detail='ff_format.py sha %s pinned %s' % (sha[:12], str(pins.get('ff_format.py'))[:12])))
    if 'public' in scopes:
        try:
            pub = json.loads(fetch('https://raw.githubusercontent.com/AffineFoundation/affine/main/caps.json'))
            same = pub == L.stored(C)
            rows.append(dict(id='caps_json_github', cls='derived', scope='public', where='github main caps.json', status='OK' if same else 'FAIL',
                             detail='max_output_tokens %s' % pub.get('max_output_tokens')))
        except Exception as e:
            rows.append(dict(id='caps_json_github', cls='derived', scope='public', where='github main caps.json', status='ERROR', detail=str(e)[:200]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scope', default='repo'); ap.add_argument('--root', default=str(L.ROOT)); ap.add_argument('--caps')
    ap.add_argument('--json'); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    C, _ = L.load_caps(a.caps)
    scopes = set(a.scope.replace('all', 'repo,host,pods,public').split(','))
    rows = run(scopes, Path(a.root), C)
    n = {s: sum(r['status'] == s for r in rows) for s in ('OK', 'FAIL', 'ERROR', 'INFO', 'SKIP')}
    out = sys.stderr if a.json == '-' else sys.stdout
    for r in rows:
        if a.quiet and r['status'] in ('OK', 'SKIP'): continue
        print('%-5s %-7s %-6s %-26s %s | %s' % (r['status'], r['cls'], r['scope'], r['id'], r['where'][-70:], r['detail'][:300]), file=out)
    head = 'caps-check %s: max_output %d / context %d / overlong %d..%d (caps.json, %s) -- %s' % (
        ','.join(sorted(scopes)), C['max_output_tokens'], C['max_context_tokens'], C['overlong_soft_start'], C['overlong_max_tokens'],
        L.load_caps()[1].URL, ' '.join('%s %d' % kv for kv in n.items()))
    print(head, file=out)
    if a.json:
        (sys.stdout.write if a.json == '-' else Path(a.json).write_text)(json.dumps(dict(version='sn120-caps-check-v1', at=time.time(), at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime(time.time() - 21600)),
                                                caps={k: C[k] for k in ('max_output_tokens', 'max_context_tokens', 'overlong_soft_start', 'overlong_max_tokens')},
                                                summary=n, rows=rows), indent=None if a.json == '-' else 1, default=str) + '\n')
    sys.exit(1 if n['FAIL'] else 2 if n['ERROR'] else 0)


if __name__ == '__main__':
    main()
