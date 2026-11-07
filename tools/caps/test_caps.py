"""tools/caps tests (stdlib + pytest; also runs as `python3 tools/caps/test_caps.py`).

* every repo consumer equals caps.json (the CI / pre-push gate);
* rendering is the identity today (rendered == current bytes);
* a hypothetical cap change renders to files that pass the check at the new cap and renders back to the exact bytes;
* the loader derives the documented values, and refuses history / legacy keys in caps.json;
* the old-cap scan finds no old cap value in the repo and does catch a planted one.
"""
import copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capslib as L
import consumers as CS

ROOT = L.ROOT


def repo_cons():
    """-> [(consumer, repo path)] (a consumer may list several repo copies)."""
    return [(c, p) for c in CS.CONSUMERS if c.get('repo') for p in ([c['repo']] if isinstance(c['repo'], str) else c['repo']) if (ROOT / p).exists()]


def other_caps(C, out=16384):
    raw = json.loads((ROOT / 'caps.json').read_text()); raw['max_output_tokens'] = out
    return L.load_caps()[1].derive(raw)


def test_loader_derives():
    C, m = L.load_caps()
    assert C['max_context_tokens'] == C['max_output_tokens'] + C['prompt_headroom_tokens'] == C['max_model_len']
    assert C['overlong_window'] == [C['max_output_tokens'] - C['overlong_ramp_tokens'], C['max_output_tokens']]
    assert C['last_eos_output_tokens'] == C['max_output_tokens'] - 1
    raw = json.loads((ROOT / 'caps.json').read_text())
    for bad in ('history', 'legacy_capped_output_tokens', 'signed_manifest'):
        try: m.derive(dict(raw, **{bad: 1})); raise AssertionError('caps.json accepted %s' % bad)
        except ValueError: pass


def test_no_old_cap_in_repo():
    import caps_check as CC
    rows = CC.old_cap_scan({'repo'}, ROOT)
    assert rows and not [r for r in rows if r['status'] in ('FAIL', 'ERROR')], [r for r in rows if r['status'] in ('FAIL', 'ERROR')]


def test_old_cap_scan_catches_a_planted_value():
    old = 8 * 1024
    for name, text in (('llms.txt', 'Exactly %d tokens = cut at the cap' % old), ('llms.txt', 'cap: {:,} output tokens'.format(old)),
                       ('mining.json', json.dumps({'caps': {'max_output_tokens': old}})), ('x.py', 'LEGACY_CAP = 1'),
                       ('mining.json', json.dumps({'training': {'caps': {'history': [], 'legacy_capped_output_tokens': 1}}})),
                       ('serve.sh', '--max-model-len %d' % (12 * 1024)), ('README.md', 'files generated at %dK still count' % (old // 1024))):
        fails, _ = L.old_cap_hits(name, text, CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
        assert fails, (name, text)
    fails, _ = L.old_cap_hits('mining.json', json.dumps({'k1': {'served': old, 'max_output_tokens': 12 * 1024}}), CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
    assert not fails, 'a count is not a cap'
    fails, _ = L.old_cap_hits('gate.json', json.dumps({'commitment': 'caea27399300%de1ee4' % old}), CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
    assert not fails, 'a hash is not a cap'


def test_signed_policy_exemption_is_narrow():
    old = 8 * 1024
    h = {'harness': {'max_output_tokens': old, 'max_context_tokens': 12 * 1024}}
    plan = {'manifest': {'environments': [dict(h)], 'mimo_release_policy': {'payload': {'scope': {'immutable_manifest': {'environments': [dict(h)]}}}}}}
    fails, infos = L.old_cap_hits('epochs/692/PLAN.json', json.dumps(plan), CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
    assert not fails and infos, 'the signed release-policy harness is the documented exemption (INFO)'
    for bad in ({'manifest': {'max_output_tokens': old}}, {'manifest': {'training': {'overlong_max_tokens': old}}},
                {'manifest': {'environments': [{'harness': {'max_tokens': old}}]}}, {'other': {'environments': [dict(h)]}}):
        fails, _ = L.old_cap_hits('epochs/692/PLAN.json', json.dumps(bad), CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
        assert fails, bad
    fails, _ = L.old_cap_hits('trainer-config.public.json', json.dumps({'trainer_config': {'settings': {'overlong_max_tokens': old}}}), CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
    assert fails, 'the signed trainer overlong window is 12K now: an old value FAILs'
    fails, _ = L.old_cap_hits('trainer-config.public.json', json.dumps({'trainer_config': {'settings': {'loss_scale_factor': old}}}), CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
    assert not fails, 'loss_scale_factor is GRPO math, not a cap'
    src = 'MAX_CONTEXT = %d\n    need(type(value[\'max_output_tokens\']) is int and 1 <= value[\'max_output_tokens\'] <= %d,\n' % (old, old)
    fails, infos = L.old_cap_hits('subnet/complete_group_harness.py', src, CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
    assert not fails and len(infos) == 2, 'the signed MiMo harness source bound is the documented exemption (INFO)'
    for name, bad in (('subnet/complete_group_harness.py', 'DEFAULT_MAX_TOKENS = %d\n' % old), ('subnet/mimo_parallel_mining.py', 'MAX_OUT = %d\n' % old),
                      ('subnet/harness.py', 'MAX_CONTEXT = %d\n' % old)):
        fails, _ = L.old_cap_hits(name, bad, CS.OLD_CAP_RX, CS.OLD_CAP_ALLOW)
        assert fails, (name, bad)


def test_repo_consumers_match_caps():
    C, _ = L.load_caps(); bad = []
    for c, p in repo_cons():
        for r in L.check_text(c, (ROOT / p).read_text(), C):
            if not r['ok']: bad.append((c['id'], p, r['key'], r['expected'], r['found'], r['n']))
    assert not bad, 'consumers that disagree with caps.json (run tools/caps/caps_apply.py --scope repo --write): %s' % bad


def test_render_is_identity_today():
    C, _ = L.load_caps()
    for c, p in repo_cons():
        if c.get('render') in (True, 'json', 'sign'):
            t = (ROOT / p).read_text(); assert L.render_text(c, t, C) == t, c['id']


def test_cap_change_roundtrip():
    C, _ = L.load_caps(); C2 = other_caps(C)
    for c, p in repo_cons():
        if c.get('render') not in (True, 'json', 'sign'): continue
        t = (ROOT / p).read_text(); t2 = L.render_text(c, t, C2)
        if c['id'] in CS._GATE_64K_CONSUMERS:
            assert t2 == t, '%s changed with training caps' % c['id']
        else:
            assert t2 != t, '%s did not move with the cap' % c['id']
        assert all(r['ok'] for r in L.check_text(c, t2, C2)), (c['id'], [r for r in L.check_text(c, t2, C2) if not r['ok']])
        assert L.render_text(c, t2, C) == t, '%s: render back is not byte-identical' % c['id']


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
