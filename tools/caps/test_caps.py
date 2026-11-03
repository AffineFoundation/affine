"""tools/caps tests (stdlib + pytest; also runs as `python3 tools/caps/test_caps.py`).

* every repo consumer equals caps.json (the CI / pre-push gate);
* rendering is the identity today (rendered == current bytes);
* a hypothetical cap change renders to files that pass the check at the new cap and renders back to the exact bytes;
* the loader derives the documented values.
"""
import copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capslib as L
import consumers as CS

ROOT = L.ROOT


def repo_cons():
    return [c for c in CS.CONSUMERS if c.get('repo') and (ROOT / c['repo']).exists()]


def other_caps(C, out=16384):
    raw = json.loads((ROOT / 'caps.json').read_text()); raw['max_output_tokens'] = out; raw['legacy_capped_output_tokens'] = C['max_output_tokens']
    return L.load_caps()[1].derive(raw)


def test_loader_derives():
    C, m = L.load_caps()
    assert C['max_context_tokens'] == C['max_output_tokens'] + C['prompt_headroom_tokens'] == C['max_model_len']
    assert C['overlong_window'] == [C['max_output_tokens'] - C['overlong_ramp_tokens'], C['max_output_tokens']]
    assert C['last_eos_output_tokens'] == C['max_output_tokens'] - 1
    assert m.at_epoch(0)['max_output_tokens'] <= C['max_output_tokens']


def test_repo_consumers_match_caps():
    C, _ = L.load_caps(); bad = []
    for c in repo_cons():
        for r in L.check_text(c, (ROOT / c['repo']).read_text(), C):
            if not r['ok']: bad.append((c['id'], c['repo'], r['key'], r['expected'], r['found'], r['n']))
    assert not bad, 'consumers that disagree with caps.json (run tools/caps/caps_apply.py --scope repo --write): %s' % bad


def test_render_is_identity_today():
    C, _ = L.load_caps()
    for c in repo_cons():
        if c.get('render') in (True, 'json', 'sign'):
            t = (ROOT / c['repo']).read_text(); assert L.render_text(c, t, C) == t, c['id']


def test_cap_change_roundtrip():
    C, _ = L.load_caps(); C2 = other_caps(C)
    for c in repo_cons():
        if c.get('render') not in (True, 'json', 'sign'): continue
        t = (ROOT / c['repo']).read_text(); t2 = L.render_text(c, t, C2)
        assert t2 != t, '%s did not move with the cap' % c['id']
        assert all(r['ok'] for r in L.check_text(c, t2, C2)), (c['id'], [r for r in L.check_text(c, t2, C2) if not r['ok']])
        assert L.render_text(c, t2, C) == t, '%s: render back is not byte-identical' % c['id']


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
