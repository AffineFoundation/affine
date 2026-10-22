"""v8.2.8 end-to-end composition test: the SIGNED launcher runs as __main__ (runpy) with the real operator config, so the
full v8.2.7+v8.2.8 install happens and native-grading spawn children re-import the launcher exactly as in production.
ops.mimo_lifecycle.main is swapped for a test body BEFORE the launcher runs, so run() / services NEVER start.
The test body builds the pool the way run() does (config native_grading -> patched 32 workers), grades N e<E> documents
(READ-ONLY R2) and compares every native grade with the live DISPATCH. Aggregates only."""
import json, runpy, sys, time
from pathlib import Path
REL = Path('/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/release-v828-signed')
SRC = '/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
EPOCH, N = int(sys.argv[1]), int(sys.argv[2])
def body():
    sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
    import rohost, multiprocessing
    config, actor, policy, tok, real = rohost.build()
    stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
    from subnet.mimo_native_pool import NativeGradingPool
    from subnet.mimo_lifecycle import Lifecycle
    import ops.mimo_lifecycle as OPS
    from subnet.backend_jobs import signed
    from subnet.storage import sha, canonical
    from subnet.complete_group_capture import _get
    from subnet.complete_group_codec import limits, loads
    from subnet.complete_group_native_receipts import compact_native_grades
    out = dict(main_module_file=sys.modules['__main__'].__file__, training_wrapped=Lifecycle._training.__module__,
               audit_population_wrapped=Lifecycle._audit_population.__code__.co_filename.endswith('mimo_v828_launcher.py'),
               scheduler=OPS.archive_scoped_audit_scheduler.__code__.co_filename.endswith('mimo_v828_launcher.py'),
               tick_wrapped=Lifecycle.tick.__code__.co_filename.endswith('mimo_host_capture_launcher.py'))
    pool = NativeGradingPool(Path(config['tokenizer_directory']).absolute(), policy.scope['initial_checkpoint']['files'],
        policy.scope['immutable_manifest']['mimo_runtime']['chat_template_sha256'], policy.scope['task_assets'], **config.get('native_grading', {}))
    out['pool_workers'] = pool.workers
    live = rohost.STATE/'epochs'/str(EPOCH)/'coordinator'
    manifest = signed(json.loads((live/'OPENING.json').read_bytes()), actor.id)
    pop = signed(json.loads((live/'POPULATION.json').read_bytes()), actor.id)
    disp = signed(json.loads((live/'DISPATCH.json').read_bytes()), actor.id)
    graded = {(r['miner'], r['batch_sha256']): r['native_grades'] for r in disp['selected'] + disp['excluded'] if r.get('native_grades')}
    items = [(signed(c, actor.id)['miner'], it) for c in pop['captures'] for it in signed(c, actor.id)['documents']][:N]
    def inputs():
        for miner, it in items:
            data, _ = _get(bucket, it['key'], limits(manifest).metadata_bytes); assert sha(data) == it['sha256']
            yield miner, it, loads(data, limits(manifest).metadata_bytes)['batch']
    t = time.time(); same = 0; n = 0
    for (miner, it, batch), g in pool(inputs(), manifest):
        n += 1; same += compact_native_grades(g) == graded.get((miner, g['batch_sha256']))
    out.update(documents=n, identical_native_grades=same, seconds=round(time.time() - t, 1),
               spawn_children_ok=True, start_method_used='spawn')
    print(json.dumps(out, indent=1))
    json.dump(out, open(str(REL.parent/'replay'/('TEST-COMPOSE-SPAWN-e%d.json' % EPOCH)), 'w'), indent=1)
if __name__ == '__main__':
    sys.path.insert(0, SRC)
    import ops.mimo_lifecycle as OPS
    OPS.main = body
    sys.argv = [str(REL/'mimo_v828_launcher.py'), '--operator-config', str(REL/'v828-operator.ROOT-ADMITTED.private.json')]
    runpy.run_path(str(REL/'mimo_v828_launcher.py'), run_name='__main__')
