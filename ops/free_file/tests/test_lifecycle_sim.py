"""Lifecycle simulation on live state copies (writes only to a temp dir and the R2 TEST prefix): queue order -> snapshot copy
-> format check -> FF documents/dispatch/job (ff_lifecycle.select/build_job) -> host validate_job -> worker-side download of
dispatch + documents via the job's presigned URLs -> trainer ff_trainer.prepare. Also: skip on missing / invalid files."""
import json, sys, tempfile, time, urllib.request
from pathlib import Path
from types import SimpleNamespace
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA / '20261030-free-file'
sys.path[:0] = [str(D / 'release'), str(D / 'stage/host-v833h')]
import ff_launcher
L = ff_launcher.load_launcher()
import ff_lifecycle as FL, ff_format as F, ff_service as S
FL.L = L
from subnet.storage import Identity
from subnet.mimo_release_policy import sign
from subnet.backend_jobs import signed
from subnet.mimo_training_worker import validate_job
import ff_trainer as FT

LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
AUTH = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
cur = json.loads((LIFE / 'CURRENT.json').read_bytes()); e = cur['epoch_index']
ed = next(LIFE / 'epochs' / str(x) for x in range(e, e - 5, -1) if (LIFE / 'epochs' / str(x) / 'coordinator/DISPATCH.json').exists()
          and (lambda P: P.get('payload', P))(json.loads((LIFE / 'epochs' / str(x) / 'PLAN.json').read_bytes()))['input']['checkpoint']['id'] == cur['checkpoint']['id'])
opening = json.loads((ed / 'coordinator/OPENING.json').read_bytes()); m = signed(opening, AUTH)
sealed = signed(json.loads((ed / 'coordinator/DISPATCH.json').read_bytes()), AUTH)
job0 = next(json.loads(p.read_bytes())['payload'] for p in sorted(LIFE.glob('epochs/*/TRAINING-JOB.json'), key=lambda p: -int(p.parent.name)))
b = S.bucket()
FL.TURN_PREFIX = 'private/ff/v1/test/turns/'
TESTKEY = {}
FL.object_key = lambda hk: TESTKEY[hk]
f = dict(enabled=True, from_epoch=0, legacy_cutoff_unix=0, tokenizer_dir=str(RA / '20261030-determinism/tok'))
k1f = L.k1_flag()
chain = json.loads(L.K1_CHAIN.read_bytes())
q = FL.queue(chain, L.k1_turns(before_epoch=e + 1))
(p0, u0, r0), (p1, u1, r1), (p2, u2, r2) = q[:3]
stamp = int(time.time())


def put(hk, groups, valid_hotkey=True):
    key = 'private/ff/v1/test/miners-%d/%s/batch.json' % (stamp, hk); TESTKEY[hk] = key
    raw = json.dumps(dict(version=F.FORMAT, hotkey=hk if valid_hotkey else 'x', groups=groups)).encode()
    b.client.put_object(Bucket=b.name, Key=key, Body=raw); return raw


# queue[0]: invalid file, queue[1]: no file, queue[2]: valid file
put(r0['hotkey'], [dict(prompt='p', rollouts=[dict(output='a<|im_end|>', reward=1)])])
raw = put(r2['hotkey'], [dict(prompt='Compute %d+%d.' % (i, i), rollouts=[dict(output='<think>\nok</think>\n\n%d' % (2 * i + k), reward=float(k == 0))
                                                                         for k in range(1 + i % 4)]) for i in range(20)])
listing = {r0['hotkey']: ('e0', 10), r2['hotkey']: ('e2', len(raw))}
FL.listing = lambda bucket: listing
adapter = SimpleNamespace(bucket=b, trainer_config=job0['training']['config'], job_seconds=L.TRAIN_JOB_SECONDS,
                          source_files=job0['source_files'], runtime_versions=job0['runtime_versions'], trainer_worker=job0['trainer_worker'],
                          checkpoint_urls=lambda c: job0['checkpoint_read_urls'], publisher_id=job0['training']['publisher_id'],
                          role_config=job0['training']['role_config_file'])
ident = Identity()
from subnet.mimo_release_policy import policy_for_manifest
rp = policy_for_manifest(m, AUTH, allow_public=True)
fake = SimpleNamespace(identity=ident, policy=SimpleNamespace(namespace=lambda: rp.namespace(m)))
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    turn = FL.select(fake, d, e, m, sealed, cur, adapter, f, k1f)
    assert turn['status'] == 'turn' and turn['route'] == 'ff' and turn['uid'] == u2 and turn['queue_position'] == p2, {k: turn.get(k) for k in ('status', 'route', 'uid', 'reason')}
    sk = [(x['uid'], x['reason'][:40]) for x in turn['skipped_not_ready']]
    assert sk[0][0] == u0 and sk[0][1].startswith('invalid-file') and sk[1] == (u1, 'no-file'), sk
    assert (d / 'ff' / (r2['hotkey'] + '.json')).read_bytes() == raw
    assert turn['tasks'] == 20 and turn['rollouts'] == sum(1 + i % 4 for i in range(20))
    filtered = FL.build_job(fake, d, e, m, opening, sealed, cur, adapter, turn)
    env = json.loads((d / 'TRAINING-JOB.json').read_bytes()); job = signed(env, ident.id)
    config = validate_job(job, m, AUTH, allow_public=True)           # the lifecycle's own job check (inner) passes
    with tempfile.TemporaryDirectory() as wd:                       # worker side: fetch by presigned URL, prepare
        w = Path(wd)
        dsp = urllib.request.urlopen(job['training']['dispatch']['url'], timeout=60).read()
        (w / 'DISPATCH.json').write_bytes(dsp)
        for h, obj in job['training']['documents'].items():
            data = urllib.request.urlopen(obj['url'], timeout=60).read(); assert len(data) == obj['size']
            (w / (h + '.json')).write_bytes(data)
        shell = SimpleNamespace(checkpoint_id=m['checkpoint']['id'], spec=SimpleNamespace(to_dict=lambda: m['mimo_runtime']),
                                vocab_size=m['group_transport_policy']['vocab_size'], eos_token_ids=tuple(m['mimo_runtime']['eos_token_ids']))
        p = FT.prepare(sign(ident, m), json.loads(dsp), authority=ident.id, documents_root=w, runtime=shell, optimizer_step=cur['optimizer_step'])
        assert len(p['groups']) == 20 and sum(len(g.trajectories) for g in p['groups']) == turn['rollouts']
        assert p['dispatch']['hotkey'] == r2['hotkey']
    # nobody valid -> skip
    listing.pop(r2['hotkey'])
    t2 = FL.select(fake, Path(td) / 'x', e, m, sealed, cur, adapter, dict(f, legacy_cutoff_unix=0), k1f) if (Path(td) / 'x').mkdir() is None else None
    assert t2['status'] == 'skip' and t2['reason'] == 'no-unserved-hotkey-with-valid-batch'
    # before the cutoff, an old-format-ready member ahead of the file owner takes the turn through the K1 path
    listing[r2['hotkey']] = ('e2', len(raw))
    fake_sealed = dict(sealed, selected=[dict(miner=r1['miner_hex'], env_id='affine_math', index=i, batch_sha256='%064x' % i,
                                              document=dict(received_at=0, slot=i), native_grades={}) for i in range(8)], excluded=[])
    (Path(td) / 'y').mkdir()
    t3 = FL.select(fake, Path(td) / 'y', e, m, fake_sealed, cur, adapter, dict(f, legacy_cutoff_unix=int(time.time()) + 3600), k1f)
    assert t3['status'] == 'turn' and t3['route'] == 'legacy' and t3['uid'] == u1, t3
    (Path(td) / 'z').mkdir()
    t4 = FL.select(fake, Path(td) / 'z', e, m, fake_sealed, cur, adapter, dict(f, legacy_cutoff_unix=int(time.time()) - 1), k1f)
    assert t4['route'] == 'ff' and t4['uid'] == u2, 'after the cutoff old-format rows are ignored'
print(json.dumps(dict(ok=True, epoch=e, queue_first=[p0, p1, p2], turn=dict(uid=u2, tasks=turn['tasks'], rollouts=turn['rollouts']),
                      skipped=sk, job_documents=len(job['training']['documents']), validate_job='ok', worker_prepare='ok',
                      empty='skip', legacy_before_cutoff='K1 route', after_cutoff='ff route')))
