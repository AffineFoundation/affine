"""End-to-end simulation (RA/20261031-ff-train-params) on live state copies; writes only to a temp dir and the R2 TEST prefix.
Three miner files in queue order: [0] affine-batch-v2 with an out-of-range lr (must be skipped), [1] affine-batch-v1 (trains with the
defaults), [2] affine-batch-v2 with valid params. Path: snapshot -> format check -> select -> documents/signed dispatch/job (build_job)
-> lifecycle validate_job -> worker fetch by presigned URL -> trainer v8.3.3j ff_trainer.prepare + train_params -> train_update_v833j
(exact source, stub inner update) -> lr multiplier + kl applied. Also FF-TRAIN disabled -> v2 invalid, no 'train' in the dispatch.
Args: <release dir with ff_*.py> <trainer host dir v8.3.3j>"""
import ast, dataclasses, json, sys, tempfile, textwrap, time, urllib.request
from pathlib import Path
from types import SimpleNamespace
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
REL, HOSTJ = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path[:0] = [str(REL), str(HOSTJ)]
import ff_launcher
L = ff_launcher.load_launcher()
import ff_lifecycle as FL, ff_format as F, ff_service as S
assert Path(FL.__file__).resolve().parent == REL and Path(F.__file__).resolve().parent == REL
FL.L = L
from subnet.storage import Identity
from subnet.mimo_release_policy import sign
from subnet.backend_jobs import signed
from subnet.mimo_training_worker import validate_job
from subnet.mimo_grpo.config import TrainerConfig
import ff_trainer as FT
assert Path(FT.__file__).resolve().parent == HOSTJ

LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
AUTH = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
cur = json.loads((LIFE / 'CURRENT.json').read_bytes()); e = cur['epoch_index']
ed = next(LIFE / 'epochs' / str(x) for x in range(e, e - 8, -1) if (LIFE / 'epochs' / str(x) / 'coordinator/DISPATCH.json').exists()
          and (lambda P: P.get('payload', P))(json.loads((LIFE / 'epochs' / str(x) / 'PLAN.json').read_bytes()))['input']['checkpoint']['id'] == cur['checkpoint']['id'])
opening = json.loads((ed / 'coordinator/OPENING.json').read_bytes()); m = signed(opening, AUTH)
sealed = signed(json.loads((ed / 'coordinator/DISPATCH.json').read_bytes()), AUTH)
job0 = next(json.loads(p.read_bytes())['payload'] for p in sorted(LIFE.glob('epochs/*/TRAINING-JOB.json'), key=lambda p: -int(p.parent.name)))
CFG = TrainerConfig(**job0['training']['config']['settings'])
b = S.bucket()
FL.TURN_PREFIX = 'private/ff/v1/test/turns/'
TESTKEY = {}
FL.object_key = lambda hk: TESTKEY[hk]
FLAG = dict(on=True); FL.train_enabled = lambda: FLAG['on']
f = dict(enabled=True, from_epoch=0, legacy_cutoff_unix=0, tokenizer_dir=str(RA / '20261030-determinism/tok'))
k1f = L.k1_flag()
chain = json.loads(L.K1_CHAIN.read_bytes())
q = FL.queue(chain, L.k1_turns(before_epoch=e + 1))
(p0, u0, r0), (p1, u1, r1), (p2, u2, r2) = [x for x in q if x[0] >= 0][:3]
q = [x for x in q if x[0] >= 0]
FL.queue = lambda chain, turns: q                              # grants (negative positions) left out of the simulation
stamp = int(time.time())
GROUPS = [dict(prompt='Compute %d+%d.' % (i, i), rollouts=[dict(output='<think>\nok</think>\n\n%d' % (2 * i + k), reward=float(k == 0))
                                                          for k in range(1 + i % 4)]) for i in range(10)]


def put(hk, version, **extra):
    key = 'private/ff/v1/test/miners-tp-%d/%s/batch.json' % (stamp, hk); TESTKEY[hk] = key
    raw = json.dumps(dict(dict(version=version, hotkey=hk, groups=GROUPS), **extra)).encode()
    b.client.put_object(Bucket=b.name, Key=key, Body=raw); return raw


adapter = SimpleNamespace(bucket=b, trainer_config=job0['training']['config'], job_seconds=L.TRAIN_JOB_SECONDS,
                          source_files=job0['source_files'], runtime_versions=job0['runtime_versions'], trainer_worker=job0['trainer_worker'],
                          checkpoint_urls=lambda c: job0['checkpoint_read_urls'], publisher_id=job0['training']['publisher_id'],
                          role_config=job0['training']['role_config_file'])
ident = Identity()
from subnet.mimo_release_policy import policy_for_manifest
rp = policy_for_manifest(m, AUTH, allow_public=True)
fake = SimpleNamespace(identity=ident, policy=SimpleNamespace(namespace=lambda: rp.namespace(m)))
# the exact train_update_v833j source from the v8.3.3j launcher, with a stub inner update
src = (HOSTJ / 'mimo_host_trainer_v83.py').read_text(); tree = ast.parse(src)
inst = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install'][0]
fn = [n for n in ast.walk(inst) if isinstance(n, ast.FunctionDef) and n.name == 'train_update_v833j'][0]
calls = []
def inner(runtime, optimizer, groups, binding, *, config, **kw):
    calls.append(dict(kl=config.kl_coefficient, same_config=config is CFG, mult=W['TURN']['lr_multiplier'], opt_cfg_kl=optimizer.config.kl_coefficient))
    W['TURN']['last'] = dict(path='scaled' if W['TURN']['lr_multiplier'] != 1.0 else 'v833i-exact')
    return dict(status='updated', config=config.document())
W = dict(TURN=dict(lr_multiplier=1.0, last=None), TURN_PARAMS=dict(params=None), train_update_v833i=inner, _dc=dataclasses,
         sgd_lr=5e-07, sgd_mode='fresh-adam', sgd_fp32=True, sgd_dir='normgrad', sgd_trc=2e-05)
exec(textwrap.dedent(ast.get_source_segment(src, fn)), W); wrap = W['train_update_v833j']
results = {}


def run_turn(td, name, listing, expect_uid):
    FL.listing = lambda bucket: listing
    FL.TURN_PREFIX = 'private/ff/v1/test/turns/tp-%d-%s/' % (stamp, name)     # immutable per-epoch keys: one prefix per phase
    d = Path(td) / name; d.mkdir()
    turn = FL.select(fake, d, e, m, sealed, cur, adapter, f, k1f)
    assert turn['status'] == 'turn' and turn['route'] == 'ff' and turn['uid'] == expect_uid, {k: turn.get(k) for k in ('status', 'route', 'uid', 'reason', 'skipped_not_ready')}
    FL.build_job(fake, d, e, m, opening, sealed, cur, adapter, turn)
    job = signed(json.loads((d / 'TRAINING-JOB.json').read_bytes()), ident.id)
    validate_job(job, m, AUTH, allow_public=True)
    with tempfile.TemporaryDirectory() as wd:
        w = Path(wd)
        dsp = urllib.request.urlopen(job['training']['dispatch']['url'], timeout=60).read()
        for h, obj in job['training']['documents'].items():
            (w / (h + '.json')).write_bytes(urllib.request.urlopen(obj['url'], timeout=60).read())
        shell = SimpleNamespace(checkpoint_id=m['checkpoint']['id'], spec=SimpleNamespace(to_dict=lambda: m['mimo_runtime']),
                                vocab_size=m['group_transport_policy']['vocab_size'], eos_token_ids=tuple(m['mimo_runtime']['eos_token_ids']))
        p = FT.prepare(sign(ident, m), json.loads(dsp), authority=ident.id, documents_root=w, runtime=shell, optimizer_step=cur['optimizer_step'])
        W['TURN_PARAMS']['params'] = FT.train_params(p['dispatch'], p['binding'])
        opt = SimpleNamespace(config=CFG)
        out = wrap(None, opt, p['groups'], p['binding'], config=CFG)
        assert opt.config is CFG and out['config'] == CFG.document()
    return turn, signed(json.loads(dsp), ident.id), out


with tempfile.TemporaryDirectory() as td:
    raw0 = put(r0['hotkey'], F.FORMAT_V2, train=dict(lr=2e-06, kl=0.02))
    raw1 = put(r1['hotkey'], F.FORMAT)
    raw2 = put(r2['hotkey'], F.FORMAT_V2, train=dict(lr=1e-06, kl=0.05))
    # A: [0] out-of-range -> skipped; [1] v1 -> turn with the defaults
    turn, disp, out = run_turn(td, 'a', {r0['hotkey']: ('e0', len(raw0)), r1['hotkey']: ('e1', len(raw1)), r2['hotkey']: ('e2', len(raw2))}, u1)
    sk = turn['skipped_not_ready']
    assert sk[0]['uid'] == u0 and 'train.lr must be a number in [1.25e-07, 1e-06]' in sk[0]['reason'], sk
    assert turn['format'] == F.FORMAT and turn['train'] == dict(lr=5e-07, kl=0.02) and turn['train_given'] == []
    assert disp['train'] == dict(lr=5e-07, kl=0.02, given=[], defaults=dict(lr=5e-07, kl=0.02))
    assert calls[-1] == dict(kl=0.02, same_config=True, mult=1.0, opt_cfg_kl=0.02) and out['train_params']['defaults_used'] is True
    results['out_of_range_v2'] = dict(uid=u0, skipped=True, reason=sk[0]['reason'][:120])
    results['v1_file'] = dict(uid=u1, format=turn['format'], dispatch_train=disp['train'], applied=dict(lr_multiplier=calls[-1]['mult'], kl=calls[-1]['kl']),
                              same_config_object=True, path=out['train_params']['step']['path'])
    # B: [1] gone -> [2] v2 with params
    turn, disp, out = run_turn(td, 'b', {r0['hotkey']: ('e0', len(raw0)), r2['hotkey']: ('e2', len(raw2))}, u2)
    assert turn['format'] == F.FORMAT_V2 and turn['train'] == dict(lr=1e-06, kl=0.05) and turn['train_given'] == ['kl', 'lr']
    assert disp['format'] == F.FORMAT_V2 and disp['train'] == dict(lr=1e-06, kl=0.05, given=['kl', 'lr'], defaults=dict(lr=5e-07, kl=0.02))
    assert calls[-1] == dict(kl=0.05, same_config=False, mult=2.0, opt_cfg_kl=0.05), calls[-1]
    tp = out['train_params']
    assert tp['lr'] == 1e-06 and tp['kl'] == 0.05 and tp['lr_multiplier'] == 2.0 and tp['kl_applied'] == 0.05 and not tp['defaults_used']
    rec = json.loads((Path(td) / 'b' / 'K1-TURN.json').read_bytes()) if (Path(td) / 'b' / 'K1-TURN.json').exists() else None
    results['v2_valid_params'] = dict(uid=u2, format=turn['format'], dispatch_train=disp['train'], applied=dict(lr_multiplier=calls[-1]['mult'], kl=calls[-1]['kl']),
                                      signed_config_reported_unchanged=True)
    # C: FF-TRAIN disabled -> v2 invalid (skip), v1 trains with no 'train' in the dispatch (exact previous behaviour)
    FLAG['on'] = False
    turn, disp, out = run_turn(td, 'c', {r0['hotkey']: ('e0', len(raw0)), r1['hotkey']: ('e1', len(raw1)), r2['hotkey']: ('e2', len(raw2))}, u1)
    assert 'train' not in disp and 'train' not in turn and 'train_params' not in out and calls[-1]['same_config'] and calls[-1]['mult'] == 1.0
    assert all('version must be affine-batch-v1' in x['reason'] or 'top level keys' in x['reason'] for x in turn['skipped_not_ready'][:1])
    results['flag_off'] = dict(v2_files='invalid (skipped)', v1_file_dispatch_has_train=False)
print(json.dumps(dict(ok=True, epoch=e, queue=[p0, p1, p2], **results), indent=1))
