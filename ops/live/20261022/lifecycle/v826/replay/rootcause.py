"""Read-only root-cause measurement for e76 capture: bytes per miner and a cProfile of original collect. Aggregates only."""
import cProfile, pstats, io, json, sys, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.complete_group_native import resolve_task
from subnet.backend_jobs import signed
from subnet.complete_group_transport import paths
live = rohost.STATE/'epochs/76/coordinator'
m = signed(json.loads((live/'OPENING.json').read_bytes()), actor.id)
out = dict(opening_bytes=(live/'OPENING.json').stat().st_size)
# --- bytes: list commitments + snapshot prefixes (LIST only)
def listing(prefix):
    c = real.commitment_read_client(); tok_ = None; rows = []
    try:
        while True:
            kw = dict(Bucket=real.name, Prefix=prefix, MaxKeys=1000)
            if tok_: kw['ContinuationToken'] = tok_
            p = c.list_objects_v2(**kw); rows += [(e['Key'], e['Size']) for e in p.get('Contents', [])]
            if not p.get('IsTruncated'): return rows
            tok_ = p['NextContinuationToken']
    finally: c.close()
ep = m['epoch']
commits = listing('private/%s/commitments/' % ep)
snaps = listing('private/%s/complete-group-snapshots/' % ep)
caps = {p.stem: signed(json.loads(p.read_bytes()), actor.id) for p in (live/'captured').glob('*.json')}
final_prefix = {c['snapshot'] for c in caps.values()}
fin = [s for k, s in snaps if k.rsplit('/', 2)[0] in final_prefix or k.rsplit('/', 3)[0] in final_prefix]
read_per_miner = []
proof_bytes = 0; ngroups = 0
for miner, c in caps.items():
    # what one original full collect GETs: commitment + every compact document of the cumulative commitment
    cb = [s for k, s in commits if k.endswith('/' + miner + '.json')]
    db = sum(s for k, s in snaps if k.startswith(c['snapshot'] + '/training/'))
    read_per_miner.append((cb[0] if cb else 0) + db); ngroups += len(c['documents'])
out['bytes'] = dict(commitment_objects=len(commits), commitment_bytes_total=sum(s for _, s in commits),
    snapshot_objects_all_versions=len(snaps), snapshot_bytes_all_versions=sum(s for _, s in snaps),
    distinct_snapshot_versions=len({k.rsplit('/training/', 1)[0].rsplit('/commitment.json', 1)[0] for k, _ in snaps}),
    final_groups=ngroups, per_miner_get_bytes_median=sorted(read_per_miner)[len(read_per_miner)//2],
    per_miner_get_bytes_max=max(read_per_miner), full_pass_get_bytes=sum(read_per_miner))
# --- profile original collect for 3 miners
coord, _ = rohost.coordinator_copy(76, Path('/tmp/v826-rootcause-root'), bucket, actor, lambda a, e, i: resolve_task(a, e, i, tok), policy)
miners = sorted(caps)[100:103]
pr = cProfile.Profile(); t = time.time(); pr.enable()
for mm in miners: coord.collect(mm, now=int(time.time()))
pr.disable(); out['collect_wall_per_miner'] = round((time.time() - t) / 3, 2)
out['r2'] = {k: round(v, 2) for k, v in stats.items() if k != 'lock'}
s = io.StringIO(); ps = pstats.Stats(pr, stream=s)
want = ['_source_hash', 'create_session', 'reset', 'render', '_resolved', 'resolve_task', 'validate_document', 'validate_commitment',
        'validate_group', 'verify', 'loads', 'canonical', 'admit', 'save_manifest', '_immutable', '_submission_get', '_submission_response',
        'decode', 'encode', 'apply_chat_template', 'entries', 'binding', 'sha', 'read', 'poll', 'fsync', 'execute']
rows = {}
for (f, l, n), (cc, nc, tt, ct, callers) in ps.stats.items():
    if n in want or any(w in n for w in ('verify', 'tokeniz')):
        key = '%s:%s' % (Path(f).name, n)
        r = rows.setdefault(key, [0, 0.0, 0.0]); r[0] += nc; r[1] += tt; r[2] += ct
out['profile_cumtime_s_3miners'] = {k: dict(calls=v[0], self=round(v[1], 3), cum=round(v[2], 3)) for k, v in sorted(rows.items(), key=lambda x: -x[1][2])[:40]}
print(json.dumps(out, indent=1))
json.dump(out, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay/ROOTCAUSE-e76.json', 'w'), indent=1)
