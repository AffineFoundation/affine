"""Isolated regression tests for public JSON timestamp redaction.

Publisher modules are exercised with temporary paths or extracted pure/write
functions. No host state, credentials, network, or production service is used.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = '2026-01-02 03:04:05 CR (UTC-6)'
LEGACY_UNIX = 1767344645
SANITIZERS = [
    'ops/zbar_flat/services/gate_ledger.py',
    'ops/green_only_reveal/services/gate_ledger.py',
    'ops/fp32_master_publish/fp32_master_publish.py',
    'ops/ff_train_params/release/ff_service.py',
    'ops/free_file/release/ff_service.py',
    'ops/ff_train_params/services/k1_transparency.py',
    'ops/green_publish/services/k1_transparency.py',
    'ops/green_publish_default/services/k1_transparency.py',
    'ops/green_only_reveal/services/k1_transparency.py',
    'ops/renorm_step/services/k1_transparency.py',
    'ops/gain_mult/services/k1_service.py',
    'ops/gain_mult_removal/services/k1_service.py',
    'ops/k1_pass_rates/services/k1_service.py',
    'ops/caps_v1/services/training_params.py',
    'ops/ff_train_params/services/training_params.py',
    'ops/gain_mult/services/training_params.py',
    'ops/gain_mult_removal/services/training_params.py',
    'ops/green_only_reveal/services/training_params.py',
    'ops/green_publish_default/services/training_params.py',
    'ops/k1_lr_chart/services/training_params.py',
    'ops/renorm_step/services/training_params.py',
    'ops/zbar_flat/services/training_params.py',
    'miner/validator/gate/gate_service.py',
    'ops/promotion_gate/gate_service.py',
    'ops/green_only_reveal/services/gate_service.py',
    'ops/performance_eval/perf_service.py',
]
TRAINING_PUBLISHERS = [p for p in SANITIZERS if p.endswith('/training_params.py')]
K1_SERVICES = [p for p in SANITIZERS if p.endswith('/k1_service.py')]
TRANSPARENCY_PUBLISHERS = [p for p in SANITIZERS if p.endswith('/k1_transparency.py')]
FF_SERVICES = [p for p in SANITIZERS if p.endswith('/ff_service.py')]
GATE_SERVICES = [p for p in SANITIZERS if p.endswith('/gate_service.py')]
GATE_LEDGERS = [p for p in SANITIZERS if p.endswith('/gate_ledger.py')]


def module(path):
    path = ROOT / path
    spec = importlib.util.spec_from_file_location('timestamp_test_' + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def functions(path, names, **values):
    source = (ROOT / path).read_text()
    mod = types.ModuleType('timestamp_test_functions')
    mod.__dict__.update(values)
    for name in names:
        start = source.index('\ndef ' + name + '(') + 1
        end = source.find('\ndef ', start + 1)
        if end < 0:
            end = len(source)
        exec(source[start:end], mod.__dict__)
    return mod


def public_functions(path, **values):
    import os
    import re
    return functions(path, ('public_json', 'wj'), os=os, re=re, Path=Path, json=json, **values)


def read(path):
    return json.loads(Path(path).read_text())


def training_functions(path):
    import os
    names = ['heldout_public', 'public_json', 'write']
    if 'def heldout_se(' in (ROOT / path).read_text():
        names.insert(0, 'heldout_se')
    return functions(
        path, names,
        Path=Path, json=json, os=os,
        rj=lambda p: read(p) if Path(p).exists() else None,
        HOME=Path('/unused'), PG_FLAG=Path('/unused/flag.json'),
    )


def assert_public(value):
    if isinstance(value, dict):
        assert not any(str(k).endswith('_cr') for k in value), value
        for child in value.values():
            assert_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_public(child)
    elif isinstance(value, str):
        assert LEGACY not in value


def sanitizer_case(mod):
    original = dict(
        generated_at=123,
        generated_at_cr=LEGACY,
        decided_at_cr=LEGACY,
        nested=[dict(drawn_at_cr=LEGACY), LEGACY, 'unchanged prose'],
    )
    cleaned = mod.public_json(original)
    assert cleaned == dict(
        generated_at=123,
        decided_at=LEGACY_UNIX,
        nested=[dict(drawn_at=LEGACY_UNIX), LEGACY_UNIX, 'unchanged prose'],
    )
    assert original['generated_at_cr'] == LEGACY


@pytest.mark.parametrize('path', SANITIZERS)
def test_public_json_recursive_legacy_conversion(path):
    if path.endswith('/ff_service.py') or path.endswith('/perf_service.py'):
        mod = public_functions(path)
    elif path.endswith('/training_params.py'):
        mod = training_functions(path)
    elif path.endswith('/gate_service.py'):
        mod = public_functions(path, SITE=Path('/unused'))
    else:
        mod = module(path)
    sanitizer_case(mod)


@pytest.mark.parametrize('path', TRAINING_PUBLISHERS)
def test_training_public_document_is_sanitized(path, tmp_path):
    mod = training_functions(path)
    mod.PG_FLAG = tmp_path / 'flag.json'
    mod.PG_FLAG.write_text(json.dumps(dict(heldout=dict(enabled=True))))
    mod.HOME = tmp_path
    current = tmp_path / 'subnet120/affine/website/gate/heldout/current.json'
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(
        round_id='r', opened_at=10, opened_at_cr=LEGACY,
        nested=dict(decided_at_cr=LEGACY),
    )))
    heldout = mod.heldout_public()
    assert heldout['current_round']['round_id'] == 'r'
    assert heldout['current_round']['opened_at'] == 10
    assert_public(heldout)
    mod.OUT = tmp_path / 'out.json'
    mod.SITE_OUT = tmp_path / 'training.json'
    mod.build = lambda: dict(nested=dict(updated_at_cr=LEGACY), generated_at=123)
    result = mod.write()
    assert_public(result)
    assert result['nested']['updated_at'] == LEGACY_UNIX
    assert_public(read(mod.OUT))
    assert_public(read(mod.SITE_OUT))


@pytest.mark.parametrize('path', K1_SERVICES + FF_SERVICES)
def test_mode_publication_preserves_private_records(path, tmp_path):
    mod = public_functions(path)
    doc = dict(generated_at_cr=LEGACY, rows=[dict(decided_at=456, decided_at_cr=LEGACY)])
    pub, private = tmp_path / 'public.json', tmp_path / 'private.json'
    mod.wj(pub, doc, 0o644)
    mod.wj(private, doc, 0o600)
    assert_public(read(pub))
    assert read(pub)['rows'][0]['decided_at'] == 456
    assert read(private) == doc
    assert (pub.stat().st_mode & 0o777) == 0o644
    assert (private.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize('path', GATE_SERVICES)
def test_gate_publication_is_scoped_to_site(path, tmp_path):
    mod = public_functions(path, SITE=tmp_path / 'site')
    doc = dict(opened_at=123, opened_at_cr=LEGACY)
    mod.wj(mod.SITE / 'index.json', doc, 0o644)
    mod.wj(tmp_path / 'state.json', doc, 0o600)
    assert_public(read(mod.SITE / 'index.json'))
    assert read(tmp_path / 'state.json') == doc


@pytest.mark.parametrize('path', TRANSPARENCY_PUBLISHERS)
def test_transparency_write_is_sanitized(path, tmp_path):
    mod = module(path)
    output = tmp_path / 'turn.json'
    mod.wj_public(output, dict(issued_at_cr=LEGACY, rows=[dict(decided_at=789, decided_at_cr=LEGACY)]))
    doc = read(output)
    assert_public(doc)
    assert doc == dict(issued_at=LEGACY_UNIX, rows=[dict(decided_at=789)])


@pytest.mark.parametrize('path', GATE_LEDGERS)
def test_gate_ledger_website_records_are_sanitized(path, tmp_path):
    mod = module(path)
    mod.EP = tmp_path / 'epochs'
    mod.EP.mkdir()
    mod.LEDGER = tmp_path / 'LEDGER.public.jsonl'
    mod.REJECTED = tmp_path / 'REJECTED.private.jsonl'
    mod.ST = tmp_path
    mod.SITE_OUT = tmp_path / 'site/gate.json'
    mod.SITE_OUT.parent.mkdir()
    sys.modules['prior_gate'] = types.SimpleNamespace(z_bar_public=lambda: dict(computed_at=123))
    historical = dict(
        epoch=1, candidate='c', decision='green', verdict='promoted',
        decided_at_cr=LEGACY,
        timestamps=dict(decided_at=123),
        thresholds=dict(z_min=1),
    )
    mod.LEDGER.write_text(json.dumps(historical) + '\n')
    mod.tick()
    gate = read(mod.SITE_OUT)
    zbar = read(mod.SITE_OUT.with_name('gate') / 'zbar.json')
    assert_public(gate)
    assert_public(zbar)
    assert gate['entries'][0]['decided_at'] == LEGACY_UNIX
    assert gate['entries'][0]['timestamps']['decided_at'] == 123
    assert isinstance(gate['updated_at'], (int, float))
    if 'turns' in zbar:
        assert zbar['turns'][0]['decided_at'] == 123


@pytest.mark.parametrize('path', ['miner/validator/gate/prior_gate.py', 'priors/gate/prior_gate.py'])
def test_prior_public_commitments_use_unix_times(path, tmp_path):
    baseline = types.ModuleType('baseline_v1')
    baseline.ROUND_SCHEMA = 'round-v1'
    baseline.DEFAULTS = dict(scoring_mode='separate', tau_run_separate=0, tau_run_same=0,
                             bar_z=1.0, z_inclusive=False, max_tokens=100)
    baseline.IRT_DEFAULTS = dict(a_shrink=0, a_lo=0, a_hi=0, theta_prior_sd=0)
    guard = types.ModuleType('probe_error_guard')
    heldout = types.ModuleType('heldout')
    old = {name: sys.modules.get(name) for name in ('baseline_v1', 'probe_error_guard', 'heldout')}
    sys.modules.update(baseline_v1=baseline, probe_error_guard=guard, heldout=heldout)
    try:
        mod = module(path)
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    mod._mt = lambda pv: 100
    mod._ctx = lambda pv: 200
    cfg = dict(baseline.DEFAULTS, probe_env='affine_math', pool_url='https://example.invalid/pool.jsonl', k_cand=4)
    private = dict(id='r', cfg=cfg, head='h', pool={}, secret_commitment='sc', table=dict(sha256='t'),
                   opened_at=10, opened_at_cr=LEGACY, opened_epoch=1, target_fixed_at=20,
                   target_fixed_at_cr=LEGACY, drawn_at=30, drawn_at_cr=LEGACY)
    commitment = mod.public_commit(private)
    assert commitment['opened_at'] == 10
    assert commitment['target_block_fixed_at'] == 20
    assert commitment['draw']['drawn_at'] == 30
    assert_public(commitment)
    params = mod.public_params(dict(prior_v1=cfg), dict(commitment, opened_at_cr=LEGACY))
    assert params['current_round']['opened_at'] == 10
    assert_public(params['current_round'])
    mod.FLAGP = tmp_path / 'missing.json'
    mod.FLAGP.write_text('{}')
    assert mod.z_bar_public(now=100)['computed_at'] == 100
    assert_public(mod.z_bar_public(now=100))


def test_heldout_commitment_excludes_display_time():
    mod = module('miner/validator/gate/heldout.py')
    record = dict(id='r', head='h', n=100, pool=1000, env='affine_math', commitment='c',
                  opened_at=123, opened_at_cr=LEGACY, opened_epoch=1,
                  blocks=4, k_block=4, k_cand=4, bar_z=1.0, z_inclusive=False)
    doc = mod.Rounds.public_commit(record)
    assert doc['opened_at'] == 123
    assert_public(doc)


def test_fp32_all_manifest_rewrites_are_public(tmp_path):
    mod = module('ops/fp32_master_publish/fp32_master_publish.py')
    mod.OUTD = tmp_path / 'fp32'
    mod.PUBLIC_HEADS = 1
    mod.time = types.SimpleNamespace(time=lambda: 200.0)
    checkpoint = 'a' * 64
    old = dict(published_at_cr=LEGACY, signed_at_cr=LEGACY,
               nested=dict(expires_at_cr=LEGACY),
               files={'head.safetensors': dict(r2_key='head', sha256='x')},
               meta_json=dict(r2_key='meta'))
    (mod.OUTD / (checkpoint + '.json')).parent.mkdir(parents=True)
    (mod.OUTD / (checkpoint + '.json')).write_text(json.dumps(old))
    (mod.OUTD / 'index.json').write_text(json.dumps(dict(heads=[dict(
        epoch=1, checkpoint=checkpoint, manifest='/k1/fp32-master/a.json', master_digest='d')])) )
    class Bucket:
        name = 'unused'
        client = types.SimpleNamespace(head_object=lambda **kw: {})
        presign = lambda self, *args: 'https://example.invalid/signed'
    mod.bucket = lambda: Bucket()
    mod.log = lambda **kw: None
    mod.refresh_links(force=True)
    manifest = read(mod.OUTD / (checkpoint + '.json'))
    current = read(mod.OUTD / 'current.json')
    assert_public(manifest)
    assert_public(current)
    assert manifest['published_at'] == LEGACY_UNIX
    assert manifest['signed_at'] == LEGACY_UNIX
    assert manifest['nested']['expires_at'] == LEGACY_UNIX
    assert manifest['download'] == dict(
        expires_at=200.0 + mod.URL_SECONDS, signed_at=200.0,
        files={'head.safetensors': 'https://example.invalid/signed'},
        meta_json='https://example.invalid/signed',
        note='presigned R2 GET links, valid 7 days, re-signed when fewer than 3 days are left; '
             'check every file against files.<name>.sha256')
    assert current['updated_at'] == 200.0
    # Expiring a link must sanitize the rest of a historical manifest too.
    mod.PUBLIC_HEADS = 0
    mod.rj = lambda path, default=None: read(path) if Path(path).exists() else default
    mod.refresh_links()
    expired = read(mod.OUTD / (checkpoint + '.json'))
    assert 'download' not in expired
    assert_public(expired)


def test_ff_readiness_separates_public_status_and_private_ready(tmp_path):
    mod = module('ops/free_file/release/ff_service.py')
    mod.STATUS = tmp_path / 'status.json'
    mod.FL = types.SimpleNamespace(
        flag=lambda: {}, FF_READY=tmp_path / 'FF-READY.json',
        tokenizer=lambda f, files: object(), caps=lambda: (1, 1),
        listing=lambda bucket: {'hotkey': ('etag', 1)},
    )
    mod.current = lambda: (1, dict(checkpoint=dict(files={'tokenizer.json': 't'})))
    mod.time = types.SimpleNamespace(time=lambda: 123.0)
    mod.wj(mod.FL.FF_READY, dict(checked_at=123, checked_at_cr=LEGACY), 0o600)
    mod.wj(mod.STATUS, dict(generated_at=123, generated_at_cr=LEGACY,
                            miners={'hotkey': dict(checked_at=123)}), 0o644)
    assert read(mod.FL.FF_READY)['checked_at_cr'] == LEGACY
    assert_public(read(mod.STATUS))
    assert read(mod.STATUS)['generated_at'] == 123


def test_gate_public_index_and_heldout_writes(tmp_path):
    mod = functions(
        'miner/validator/gate/gate_service.py',
        ('public_json', 'wj', '_ho_write_public', 'publish'),
        Path=Path, json=json, os=os, time=time, SITE=tmp_path / 'gate',
        EP=tmp_path / 'epochs', G=types.SimpleNamespace(VERSION='test'),
        heldout_publish=lambda: None, opts=lambda: dict(standing=False),
        rj=lambda path: read(path) if Path(path).exists() else None,
        _HO_FORBID_OPEN=('"seed"',),
    )
    epoch = mod.EP / '1'
    epoch.mkdir(parents=True)
    (epoch / 'COMMIT.json').write_text(json.dumps(dict(
        head='h', sample_size=1, commitment='c', drawn_at=10, drawn_at_cr=LEGACY)))
    (epoch / 'DECISION.json').write_text(json.dumps(dict(
        decision='green', decided_at=20, decided_at_cr=LEGACY, evaluated={})))
    mod.publish(1)
    doc = read(mod.SITE / 'index.json')
    assert_public(doc)
    assert doc['entries'][0]['drawn_at'] == 10
    assert doc['entries'][0]['decided_at'] == 20
    mod._ho_write_public(mod.SITE / 'heldout/index.json',
                         dict(opened_at=10, baseline=dict(done_at_cr=LEGACY)), True)
    heldout = read(mod.SITE / 'heldout/index.json')
    assert_public(heldout)
    assert heldout['baseline']['done_at'] == LEGACY_UNIX


def test_gain_multiplier_public_record(tmp_path):
    mod = module('ops/gain_mult/gain_mult.py')
    mod.CFG_PATH = tmp_path / 'config.json'
    mod.CFG_PATH.write_text(json.dumps(dict(
        enabled=True, activation_epoch=1, cap_regimes=[dict(from_t=0, cap=1)],
        public_rule='test', reference_frozen=[], min_shared=1, ref_window=1, se_mult=1,
        **{'lambda': 0, 'm_min': 1, 'm_max': 1},
    )))
    mod.PG_FLAG = tmp_path / 'missing.json'
    mod.GATE = tmp_path / 'epochs'
    mod.GATE.mkdir()
    mod.PRIV = tmp_path / 'private.json'
    mod.PUB = tmp_path / 'public.json'
    mod.ST = tmp_path
    mod.DRY = False
    mod.main()
    assert_public(read(mod.PUB))
    assert read(mod.PUB)['generated_at'] > 0
    assert read(mod.PRIV)['generated_at_cr']


def test_performance_public_king_list_and_private_results(tmp_path):
    mod = functions(
        'ops/performance_eval/perf_service.py',
        ('public_json', 'wj', 'publish'),
        json=json, os=os, re=__import__('re'), time=time,
        now=lambda: 123.0, cr=lambda t: LEGACY,
        OUTP=str(tmp_path / 'private.json'), OUTJ=str(tmp_path / 'performance.json'),
        RES=str(tmp_path), S=dict(fail={}, unavailable=[]), MAX_FAIL=3,
        rj=lambda *args: None, claim_owner=lambda king: False,
        HOLD=dict(n=1, sha256='h', source='test', rule='test', removed=0, hash_rule='test'),
        K=dict(full=1, holdout=1), SEED='test', POD='test', GPU='test', COST_H=0,
        PODS={}, XC=str(tmp_path), holdout_guard=lambda: {}, backfill_summary=lambda ks: {},
    )
    mod.publish([dict(king='k', epoch=1, head_from_epoch=1, era='gate', parent='p',
                      crowned_at=100, crowned_at_cr=LEGACY)])
    public = read(mod.OUTJ)
    assert_public(public)
    assert public['updated_at'] == 123.0
    assert public['kings'][0]['crowned_at'] == 100
    assert read(mod.OUTP)['updated_at_cr'] == LEGACY
