"""v8.2.8 behaviour test of the staged launcher overrides (READ-ONLY R2; temp epoch copy; no queue, no services).
Real original Lifecycle._audit_population on live e<N> data in the background thread; fake trainer result. Aggregates only."""
import importlib.util, json, shutil, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
def main():
    sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
    import rohost
    EPOCH = int(sys.argv[1]); STAGE = Path(sys.argv[2])
    config, actor, policy, tok, real = rohost.build()
    stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
    spec = importlib.util.spec_from_file_location('v828_stage', STAGE/'mimo_v828_launcher.py')
    L = importlib.util.module_from_spec(spec); sys.modules[spec.name] = L; spec.loader.exec_module(L)
    from subnet.mimo_lifecycle import Lifecycle
    import subnet.mimo_native_pool as NP
    R = {}
    state = dict(result=None)
    def fake_inner(self, directory, opening, dispatch, current, now): return state['result']
    fake_inner.__qualname__ = 'Lifecycle._training'
    Lifecycle._training = fake_inner
    stub = SimpleNamespace(archive_scoped_audit_scheduler=None)
    R['install'] = L.install_v828(stub, Lifecycle, 32)
    pool = NP.NativeGradingPool(Path(config['tokenizer_directory']).absolute(), policy.scope['initial_checkpoint']['files'],
        policy.scope['immutable_manifest']['mimo_runtime']['chat_template_sha256'], policy.scope['task_assets'], workers=8)
    R['pool_workers_from_config8'] = pool.workers
    try: NP.NativeGradingPool('x', {}, 'y', {}, workers=16); R['pool_rejects_unreviewed'] = False
    except ValueError: R['pool_rejects_unreviewed'] = True
    cls = stub.archive_scoped_audit_scheduler(object, {'a' * 64})
    sch = cls(bucket=bucket, queue=None, identity=actor, manifest_guard=None, metadata=None, directory='/nonexistent', budget_per_tick=3)
    R['scheduler'] = dict(name=cls.__name__, admit=sch.admit({'x': 1}), tick=sch.tick())
    E = rohost.STATE/'epochs'/str(EPOCH)
    root = Path('/tmp/v828-offpath-e%d' % EPOCH)
    if root.exists(): shutil.rmtree(root)
    (root/'coordinator').mkdir(parents=True, mode=0o700)
    for n in ('OPENING.json', 'POPULATION.json', 'DISPATCH.json'): shutil.copy2(E/'coordinator'/n, root/'coordinator'/n)
    plan = json.loads((E/'PLAN.json').read_bytes()); current = plan['input']
    opening = json.loads((root/'coordinator/OPENING.json').read_bytes()); dispatch = json.loads((root/'coordinator/DISPATCH.json').read_bytes())
    s = SimpleNamespace(_load=lambda p: json.loads(p.read_bytes()), bucket=bucket, identity=actor, policy=policy)
    coord = SimpleNamespace(root=root/'coordinator')
    t = time.time(); Lifecycle._audit_population(s, root, coord, opening, dispatch, plan, current); R['tick_audit_call_seconds'] = round(time.time() - t, 3)
    n1 = sum(t.name == 'v828-audit-population' for t in threading.enumerate())
    t = time.time(); Lifecycle._audit_population(s, root, coord, opening, dispatch, plan, current); R['second_tick_audit_call_seconds'] = round(time.time() - t, 3)
    R['threads_after_two_calls'] = sum(t.name == 'v828-audit-population' for t in threading.enumerate()); R['threads_after_one'] = n1
    t = time.time(); R['training_pending_returns'] = Lifecycle._training(s, root, opening, dispatch, current, 0); R['training_pending_seconds'] = round(time.time() - t, 3)
    R['file_exists_while_training_pending'] = (root/'AUDIT-POPULATION.json').exists()
    state['result'] = dict(status='updated')
    t = time.time(); R['finalize_returns'] = Lifecycle._training(s, root, opening, dispatch, current, 0); R['finalize_waited_seconds'] = round(time.time() - t, 1)
    R['audit_population_bytes_identical_to_live'] = (root/'AUDIT-POPULATION.json').read_bytes() == (E/'AUDIT-POPULATION.json').read_bytes()
    R['r2_after_build'] = {k: v for k, v in stats.items() if k in ('put_suppressed_existing', 'would_write_absent')}
    t = time.time(); Lifecycle._audit_population(s, root, coord, opening, dispatch, plan, current); R['after_done_call_seconds'] = round(time.time() - t, 3)
    R['threads_after_done'] = sum(t.name == 'v828-audit-population' and t.is_alive() for t in threading.enumerate())
    # failure path: population missing -> background error -> finalize refused -> next tick restarts the build
    bad = Path('/tmp/v828-offpath-bad-e%d' % EPOCH)
    if bad.exists(): shutil.rmtree(bad)
    (bad/'coordinator').mkdir(parents=True, mode=0o700)
    for n in ('OPENING.json', 'DISPATCH.json'): shutil.copy2(E/'coordinator'/n, bad/'coordinator'/n)
    bc = SimpleNamespace(root=bad/'coordinator')
    Lifecycle._audit_population(s, bad, bc, opening, dispatch, plan, current)
    try: Lifecycle._training(s, bad, opening, dispatch, current, 0); R['failure_finalize_refused'] = False
    except ValueError: R['failure_finalize_refused'] = True
    R['failure_error_type'] = s._v828_audit_jobs[str(bad)].error
    shutil.copy2(E/'coordinator/POPULATION.json', bad/'coordinator/POPULATION.json')
    Lifecycle._audit_population(s, bad, bc, opening, dispatch, plan, current)
    R['failure_restarted_thread'] = s._v828_audit_jobs[str(bad)].thread.is_alive()
    s._v828_audit_jobs[str(bad)].thread.join(); R['failure_recovered_done'] = s._v828_audit_jobs[str(bad)].done
    R['recovered_bytes_identical'] = (bad/'AUDIT-POPULATION.json').read_bytes() == (E/'AUDIT-POPULATION.json').read_bytes()
    R['r2_ops'] = {k: round(v, 1) for k, v in stats.items() if k != 'lock'}
    shutil.rmtree(root); shutil.rmtree(bad)
    print(json.dumps(R, indent=1, default=str))
    json.dump(R, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/replay/TEST-OFFPATH-e%d.json' % EPOCH, 'w'), indent=1, default=str)
if __name__ == '__main__': main()
