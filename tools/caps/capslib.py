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
    return rule.get('value', C[rule['key']])


def stored(C):
    """The caps.json content (the stored keys only) of a loaded caps dict."""
    m = load_caps()[1]
    return {k: C[k] for k in m.STORED if k in C}


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
        if r['kind'] == 'same':   # the file is caps.json (same JSON content)
            try: same = json.loads(text) == json.loads((ROOT / 'caps.json').read_text())
            except ValueError as e: same = False
            res.append(dict(rule='= caps.json', key=r['key'], expected='caps.json', found=['same' if same else 'differs'], ok=same, n=1, signed=False)); continue
        if r['kind'] == 'absent':   # the path must not exist (a stale field that must not come back)
            try: vals = [v for v, _, _ in _walk(json.loads(text), r['path'])]
            except ValueError as e: vals = ['bad json: %s' % e]
            res.append(dict(rule=desc, key=r['key'], expected='absent', found=sorted({json.dumps(v) for v in vals}), ok=not vals, n=len(vals), signed=False)); continue
        if r['kind'] == 'json':
            try: vals = [v for v, _, _ in _walk(json.loads(text), r['path'])]
            except ValueError as e: res.append(dict(rule=desc, key=r['key'], expected=exp, found=['bad json: %s' % e], ok=False, n=0)); continue
            found = sorted({json.dumps(v) for v in vals}); ok = bool(vals) and all(v == exp for v in vals)
        else:
            rx = r.get('rx') or r.get('sub')
            vals = [m.group(1) for m in re.finditer(rx, text, r.get('flags', re.M))]
            found = sorted(set(vals)); ok = bool(vals) and all(num(v) == exp for v in vals)
        res.append(dict(rule=desc, key=r['key'], expected=exp, found=found, ok=ok, n=len(vals), signed=False))
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
        if r['kind'] in ('absent', 'same'): continue
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


_CAPKEY = re.compile(r'token|cap|max_|context|overlong|window|model_len', re.I)


def _json_lines(obj, path=''):
    """-> ['path: value'] for every string value (and key) and every number under a token / cap key of a JSON document."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _json_lines(v, '%s.%s' % (path, k))
            out.append('%s.%s = key %s' % (path, k, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj): out += _json_lines(v, '%s[%d]' % (path, i))
    elif isinstance(obj, str): out.append('%s = %s' % (path, obj))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and _CAPKEY.search(path.rsplit('.', 1)[-1].split('[')[0]):
        out.append('%s = %s' % (path, obj))
    return out


def old_cap_hits(name, text, rx_list, allow):
    """-> (fails, infos): [(line no / json path, line, why)] for every old-cap pattern hit; allow = [(glob, line regex, why)]."""
    import fnmatch
    try:
        lines = _json_lines(json.loads(text)) if name.endswith('.json') else text.splitlines()
        js = name.endswith('.json')
    except ValueError:
        lines, js = text.splitlines(), False
    fails, infos = [], []
    for i, line in enumerate(lines):
        for rx, why in rx_list:
            if re.search(rx, line):
                where = line.split(' = ')[0] if js else 'line %d' % (i + 1)
                ok = [w for g, lr, w in allow if fnmatch.fnmatch(name, g) and re.search(lr, line)]
                (infos if ok else fails).append((where, line.strip()[:160], ok[0] if ok else why))
                break
    return fails, infos
