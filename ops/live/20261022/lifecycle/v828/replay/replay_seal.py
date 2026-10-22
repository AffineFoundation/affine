"""v8.2.8 offline seal replay (READ-ONLY R2; private temp coordinator root). Re-grades EVERY document of the live frozen
population with NativeGradingPool(workers=W) and the live SELECTION seed, then proves byte identity:
  (1) local replay DISPATCH bytes == live DISPATCH.json bytes; (2) the original immutable PUT of mimo/dispatch.json hits the
  existing R2 object (412, never written) and the original readback compare passes. Aggregates only (no miner data)."""
import json, shutil, sys, time
from pathlib import Path
def main():
    sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
    import rohost
    EPOCH, W = int(sys.argv[1]), int(sys.argv[2])
    config, actor, policy, tok, real = rohost.build()
    stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
    from subnet.mimo_coordinator import Coordinator
    from subnet.mimo_native_pool import NativeGradingPool
    from subnet.complete_group_native import resolve_task
    from subnet.backend_jobs import signed
    from subnet.storage import sha
    live = rohost.STATE/'epochs'/str(EPOCH)/'coordinator'
    root = Path('/tmp/v828-seal-e%d-w%d' % (EPOCH, W))
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True, mode=0o700)
    for n in ('OPENING.json', 'OPENING-REQUEST.json', 'POPULATION.json', 'SELECTION.json'): shutil.copy2(live/n, root/n)
    pool = NativeGradingPool(Path(config['tokenizer_directory']).absolute(), policy.scope['initial_checkpoint']['files'],
        policy.scope['immutable_manifest']['mimo_runtime']['chat_template_sha256'], policy.scope['task_assets'], workers=W)
    coord = Coordinator(root, bucket, actor, resolve_task=lambda m, e, i: resolve_task(m, e, i, tok), grade_batch=None,
                        release_policy=policy, grade_batches=pool)
    t = time.time(); err = None
    try: result = coord.seal(max_tasks=256, now=int(time.time()))
    except Exception as e: err = '%s: %s' % (type(e).__name__, str(e)[:200]); result = None
    wall = round(time.time() - t, 1)
    lb = (live/'DISPATCH.json').read_bytes()
    rb = (root/'DISPATCH.json').read_bytes() if (root/'DISPATCH.json').exists() else b''
    out = dict(epoch=EPOCH, workers=W, seal_wall_seconds=wall, error=err, dispatch_bytes_identical=(lb == rb),
               live_dispatch_sha256=sha(lb), replay_dispatch_sha256=sha(rb) if rb else None,
               r2_put_suppressed_existing=stats.get('put_suppressed_existing', 0), r2_would_write_absent=stats.get('would_write_absent', 0))
    if result is not None:
        L, R = signed(json.loads(lb), actor.id), signed(result, actor.id)
        key = lambda r: (r['miner'], r['batch_sha256'])
        out.update(selected=len(R['selected']), same_selected_groups=[key(r) for r in L['selected']] == [key(r) for r in R['selected']],
                   same_selected_rewards=[[x['reward'] for x in r['native_grades']['results']] for r in L['selected']] ==
                                         [[x['reward'] for x in r['native_grades']['results']] for r in R['selected']],
                   same_excluded=L['excluded'] == R['excluded'], documents_graded=len(R['selected']) + len(R['excluded']))
    out['r2_ops'] = {k: round(v, 1) for k, v in stats.items() if k != 'lock'}
    print(json.dumps(out, indent=1))
    json.dump(out, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/replay/SEAL-REPLAY-e%d-w%d.json' % (EPOCH, W), 'w'), indent=1)
    shutil.rmtree(root)
if __name__ == '__main__': main()
