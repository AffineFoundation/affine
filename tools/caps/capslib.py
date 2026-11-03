"""Shared check / render logic for tools/caps (stdlib only)."""
import importlib.util, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_caps(path=None):
    spec = importlib.util.spec_from_file_location('sn120_caps', str(ROOT / 'caps.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.load(path), m


def fmt(v, f):
    return '{:,}'.format(v) if f == ',' else str(v)


def num(s):
    return int(str(s).replace(',', ''))


def expected(rule, C):
    if rule['kind'] == 'signed': return C['signed_manifest'][rule['key']]
    return C[rule['key']]


def _walk(obj, path):
    """-> [(value, parent, key)] for a path with '*' wildcards, 'last_trained' (last list item with a training block)."""
    out = [(obj, None, None)]
    for p in path:
        nxt = []
        for o, _, _ in out:
            if p == '*':
                items = o.items() if isinstance(o, dict) else enumerate(o) if isinstance(o, list) else []
                nxt += [(v, o, k) for k, v in items]
            elif p == 'last_trained' and isinstance(o, list):
                tr = [x for x in o if isinstance(x, dict) and isinstance(x.get('training'), dict) and x['training'].get('trained_tokens')]
                if tr: nxt.append((tr[-1], o, o.index(tr[-1])))
            elif isinstance(o, dict) and p in o: nxt.append((o[p], o, p))
            elif isinstance(o, list) and isinstance(p, int) and -len(o) <= p < len(o): nxt.append((o[p], o, p))
        out = nxt
    return out


def check_text(con, text, C):
    """-> list of dict(rule, expected, found, ok, n). found = the distinct values seen; n = matches."""
    res = []
    for r in con['rules']:
        exp = expected(r, C); desc = r.get('rx') or r.get('sub') or '.'.join(map(str, r.get('path', [])))
        if r['kind'] == 'json':
            try: vals = [v for v, _, _ in _walk(json.loads(text), r['path'])]
            except ValueError as e: res.append(dict(rule=desc, key=r['key'], expected=exp, found=['bad json: %s' % e], ok=False, n=0)); continue
            found = sorted({json.dumps(v) for v in vals}); ok = bool(vals) and all(v == exp for v in vals)
        else:
            rx = r.get('rx') or r.get('sub')
            vals = [m.group(1) for m in re.finditer(rx, text, r.get('flags', re.M))]
            found = sorted(set(vals)); ok = bool(vals) and all(num(v) == exp for v in vals)
        res.append(dict(rule=desc, key=r['key'], expected=exp, found=found, ok=ok, n=len(vals), signed=r['kind'] == 'signed'))
    return res


def _json_dump_like(text, obj):
    """Dump obj in the same layout as text (indent/sort_keys/ensure_ascii/trailing newline), or None if no layout reproduces text."""
    d = json.loads(text)
    for ind in (1, 2, None):
        for sk in (False, True):
            for ea in (True, False):
                for nl in ('\n', ''):
                    if json.dumps(d, indent=ind, sort_keys=sk, ensure_ascii=ea) + nl == text:
                        return json.dumps(obj, indent=ind, sort_keys=sk, ensure_ascii=ea) + nl
    return None


def render_text(con, text, C):
    """-> the text with every rule's value set to caps (byte-identical when everything already matches)."""
    for r in con['rules']:
        if r['kind'] == 'signed': continue
        exp = expected(r, C)
        if r['kind'] == 'json':
            obj = json.loads(text); hits = _walk(obj, r['path'])
            if not hits: raise ValueError('%s: json path %s not found' % (con['id'], r['path']))
            if all(v == exp for v, _, _ in hits): continue
            for _, parent, k in hits: parent[k] = exp
            new = _json_dump_like(text, obj)
            if new is None: raise ValueError('%s: cannot re-dump json in its own layout' % con['id'])
            text = new
        else:
            def sub(m):
                s, e = m.span(1); return m.group(0)[:s - m.start()] + fmt(exp, r['fmt']) + m.group(0)[e - m.start():]
            text, n = re.subn(r['rx'], sub, text, flags=r.get('flags', re.M))
            if not n: raise ValueError('%s: pattern not found: %s' % (con['id'], r['rx']))
    return text
