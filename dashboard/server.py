"""SQLite-backed, public allowlisted projection of subnet epoch records."""
import argparse
import json
import math
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = Path(__file__).parent / 'public'


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class Database:
    def __init__(self, path, source=ROOT/'state'):
        self.path, self.source = Path(path), Path(source)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS epochs(id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def refresh(self):
        epochs = {}
        folders=[(folder,folder.name) for folder in self.source.iterdir()
            if folder.name in ('live', 'e2e-final', 'registered-test', 'registered-test-compatible', 'multi-environment', 'service-conformance', 'gpu-continuous', 'gpu-wide', 'native-agent-common', 'native-sql-common', 'native-eog-common', 'native-math-common')]
        folders.append((self.source/'prospective-separated-hopper-math-v1/controller-state','separated-hopper-math'))
        folders.append((self.source/'prospective-separated-hopper-math-recovery-v1/controller-state','separated-hopper-math-recovery'))
        folders.append((self.source/'prospective-separated-hopper-math-v2/controller-state','separated-hopper-math-v2'))
        folders.append((self.source/'prospective-separated-hopper-math-v3/controller-state','separated-hopper-math-v3'))
        folders.append((self.source/'prospective-separated-hopper-math-v6/controller-state','separated-hopper-math-v6'))
        folders.append((self.source/'prospective-separated-hopper-math-v7/controller-state','separated-hopper-math-v7'))
        folders.append((self.source/'prospective-separated-hopper-math-v8/controller-state','separated-hopper-math-v8'))
        folders.append((self.source/'prospective-separated-hopper-math-v9/controller-state','separated-hopper-math-v9'))
        folders.append((self.source/'prospective-separated-hopper-math-v10/controller-state','separated-hopper-math-v10'))
        for folder,source_name in folders:
            if not folder.is_dir():
                continue
            health = read(folder/'health.json', {})
            for path in folder.glob('*-manifest.json'):
                doc = read(path, {})
                eid = doc.get('epoch')
                if not eid:
                    continue
                scorepath = folder/f'{eid}-scores.json'
                scores = read(scorepath, {})
                verified = read(folder/f'{eid}-verified.json', {})
                if not verified:
                    verified = {}
                    for reportpath in folder.glob(f'{eid}-*-report.json'):
                        identity = reportpath.name[len(eid)+1:-len('-report.json')]
                        verified[identity] = read(reportpath, {})
                reports = list(verified.values()) if isinstance(verified, dict) else []
                valid = [r for r in reports if isinstance(r, dict) and r.get('accepted')]
                batches = sum(len(r.get('outcomes', [])) for r in reports)
                accepted_batches = sum(len(r.get('accepted', [])) for r in reports)
                unchecked_batches = sum(
                    outcome.get('valid', 'missing') is None and not outcome.get('fully_audited', False)
                    for report in reports for outcome in report.get('outcomes', [])
                    if isinstance(outcome, dict)
                )
                registered = {}
                for registry_path in (self.source/'live/epoch-registrations.json',
                                      folder/'epoch-registrations.json', folder/'registrations.json',
                                      folder/f'{eid}-registrations.json'):
                    registry = read(registry_path, {})
                    if not isinstance(registry, dict):
                        continue
                    entries = registry.get('by_identity', registry)
                    if isinstance(entries, dict):
                        registered.update(entries)
                identity_uids = {r.get('public_key'): r.get('uid') for r in registered.values() if isinstance(r,dict)}
                trial = read(folder/'report.json', {})
                if trial.get('uid') is not None and len(verified) == 1:
                    identity_uids.setdefault(next(iter(verified)), trial['uid'])
                grid = [0]*256
                outside = 0
                for identity, audit in verified.items():
                    uid = identity_uids.get(identity)
                    count = len(audit.get('outcomes', []))
                    if type(uid) is int and 0 <= uid < 256:
                        grid[uid] += count
                    else:
                        outside += count
                points = scores.get('points', {})
                checkpoint = doc.get('checkpoint', {})
                if isinstance(checkpoint, dict):
                    checkpoint = checkpoint.get('id') or checkpoint.get('hash') or checkpoint.get('sha256') or ''
                phase = 'collecting' if time.time() < doc.get('deadline', 0) else ('verified' if reports else 'closed')
                if health.get('epoch') == eid and health.get('status') == 'paused_no_verified_training_data':
                    phase = 'awaiting data'
                row = dict(id=eid, finalized=scorepath.exists(), mode='test' if eid.startswith(('mock-', 'test-', 'nonpayable-')) else 'live',
                           payable=doc.get('payable', not eid.startswith(('mock-', 'test-', 'nonpayable-'))),
                           start=doc.get('start', 0), deadline=doc.get('deadline', 0), phase=phase,
                           checkpoint=str(checkpoint), indices=len(doc.get('indices', [])), k=doc.get('K', 0), l=doc.get('L', 0),
                           participants=len(doc.get('capabilities', {})), submissions=len(reports),
                           accepted=accepted_batches, rejected=batches-accepted_batches-unchecked_batches,
                           unchecked=unchecked_batches, points=sum(points.values()),
                           batches=batches, grid=grid, unassigned_batches=outside,
                           miners=[dict(identity=identity, points=value, weight=scores.get('weights', {}).get(identity, 0))
                                   for identity, value in points.items()], training=None,
                           audit_policy=doc.get('audit_policy', ''), source=source_name)
                epochs[eid] = row
                metrics = read(folder/f'{eid}-training-metrics.json', {})
                if metrics:
                    row['training'] = {key:metrics.get(key) for key in ('steps','losses','weights_changed','checkpoint','objective','input_pairs')}
                    if metrics.get('weights_changed'):
                        row['phase'] = 'trained'
                else:
                    abort = read(folder/f'{eid}-aborted-training-admission.json', {}) or read(folder/f'{eid}-aborted-evaluation.json', {})
                    abort = abort.get('payload', {}) if isinstance(abort, dict) else {}
                    if not isinstance(abort, dict):
                        abort = {}
                    # Public projection of operator state, not signature verification.
                    # Never export the failed worker logs or raw exception reason.
                    if (abort.get('status') in ('aborted_evaluation','aborted_training_admission') and abort.get('epoch') == eid
                            and abort.get('checkpoint') == abort.get('next_checkpoint') == row['checkpoint']
                            and abort.get('optimizer_ran') is False
                            and type(abort.get('steps')) is int and abort['steps'] == 0):
                        row['phase'] = ('training admission rejected' if abort['status']=='aborted_training_admission'
                                        else 'aborted evaluation')
            report = read(folder/'report.json', {})
            eid = report.get('epoch')
            if eid in epochs and report.get('training'):
                training = report['training']
                epochs[eid]['training'] = {key: training.get(key) for key in ('steps', 'losses', 'weights_changed', 'checkpoint', 'objective', 'input_pairs')}
                epochs[eid]['phase'] = 'trained' if training.get('weights_changed') else epochs[eid]['phase']
                epochs[eid]['uid'] = report.get('uid')
                epochs[eid]['hotkey'] = report.get('hotkey')
        evaluations = {}
        keys = ('run_id','epoch_id','checkpoint','model','env_id','environment_version','harness',
                'dataset_id','seed','count','successes','mean_reward','timestamp',
                'training_steps','status','fixed_task_ids','reward_standard_error',
                'requested_count','completed_count','taskset_hash','policy_kind','model_runtime_revision',
                'original_error_count','recovered_count','status_detail')
        evaluation_paths=list((self.source/'evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v1/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-recovery-v1/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v2/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v3/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v6/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v7/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v8/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v9/evaluations').glob('*.json'))
        evaluation_paths.extend((self.source/'prospective-separated-hopper-math-v10/evaluations').glob('*.json'))
        for path in evaluation_paths:
            raw = read(path,{})
            if not isinstance(raw,dict) or not all(isinstance(raw.get(k),str) for k in ('run_id','env_id','dataset_id','status')):
                continue
            row = {key:raw[key] for key in keys if key in raw}
            harness_config=raw.get('harness_config')
            budget=harness_config.get('max_output_tokens') if isinstance(harness_config,dict) else None
            if type(budget) is int and 1<=budget<=8192:
                row['output_token_budget']=budget
            # Public model identifiers are names, never local checkpoint paths.
            if 'model' in row and (not isinstance(row['model'], str) or
                    re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?', row['model']) is None):
                row.pop('model')
            if type(row.get('timestamp')) not in (int,float) or not math.isfinite(row['timestamp']):
                continue
            # Reject malformed optional metadata too: one nonfinite value must
            # not prevent all otherwise valid records from being published.
            try:
                json.dumps(row, allow_nan=False)
            except (ValueError, TypeError):
                continue
            if any(key in row and not isinstance(row[key], str)
                   for key in ('checkpoint','epoch_id','environment_version','harness','taskset_hash','model_runtime_revision')):
                continue
            if 'policy_kind' in row and row['policy_kind'] not in ('curated-control','autoregressive'):
                continue
            if any(key in row and (type(row[key]) is not int or row[key] < 0)
                   for key in ('original_error_count', 'recovered_count')):
                continue
            if 'status_detail' in row and row['status_detail'] not in (
                    'original-complete', 'original-partial', 'completed-with-explicit-recovery'):
                continue
            if row['status'] == 'complete':
                if type(row.get('count')) is not int or row['count'] <= 0:
                    continue
                if type(row.get('successes')) is not int or not 0 <= row['successes'] <= row['count']:
                    continue
                if type(row.get('mean_reward')) not in (int,float) or not math.isfinite(row['mean_reward']):
                    continue
            # Never export free-text errors, paths, token traces, or credentials.
            evaluations[row['run_id']] = row
        inventory = read(self.source/'live-chain/registration-inventory.json', {})
        registration_count = len(inventory) if isinstance(inventory, dict) else len(inventory or [])
        readiness = read(self.source/'live/readiness.json', {})
        live_health = read(self.source/'live/health.json', {})
        models = sorted({r['model'] for r in evaluations.values()
                         if r['status']=='complete' and r.get('model')})
        summary = dict(updated_at=time.time(), network='Finney', netuid=120,
                       model=models[0] if len(models)==1 else 'Multiple pinned models' if models else 'Pinned checkpoints',
                       models=models,
                       environments=sorted({r['env_id'] for r in evaluations.values() if r['status']=='complete'}),
                       registered_miners=max(registration_count, readiness.get('registered_miners', 0)),
                       controller=live_health.get('status', 'unknown'),
                       payout_status='Pilot · chain payouts not activated',
                       numerical_policy='Pinned runtime per evaluation · strict verification',
                       epochs=len(epochs), accepted=sum(e['accepted'] for e in epochs.values()),
                       rejected=sum(e['rejected'] for e in epochs.values()),
                       unchecked=sum(e['unchecked'] for e in epochs.values()),
                       training_steps=sum((e.get('training') or {}).get('steps', 0) or 0 for e in epochs.values()),
                       evaluated_environments=len({r['env_id'] for r in evaluations.values() if r['status']=='complete'}))
        with self.connect() as db:
            db.execute('DELETE FROM epochs')
            db.executemany('INSERT INTO epochs VALUES(?,?)', [(k, json.dumps(v, allow_nan=False)) for k,v in epochs.items()])
            db.execute('INSERT OR REPLACE INTO snapshots VALUES(1,?)', (json.dumps(summary),))
            db.executemany('INSERT OR REPLACE INTO evaluations VALUES(?,?)',[(k,json.dumps(v,allow_nan=False)) for k,v in evaluations.items()])

    def snapshot(self):
        with self.connect() as db:
            summary = db.execute('SELECT data FROM snapshots WHERE id=1').fetchone()
            epochs = [json.loads(r[0]) for r in db.execute('SELECT data FROM epochs')]
            evaluations = [json.loads(r[0]) for r in db.execute('SELECT data FROM evaluations')]
        return dict(summary=json.loads(summary[0]) if summary else {}, epochs=sorted(epochs, key=lambda x:x['start'], reverse=True),
                    evaluations=sorted(evaluations,key=lambda x:x['timestamp']))


def serve(db, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/api/network', '/network-data.json', '/health'):
                data = json.dumps(db.snapshot() if path != '/health' else {'status':'ok'}).encode()
                kind = 'application/json'
            else:
                files = {'/':'index.html', '/index.html':'index.html', '/llms.txt':'llms.txt', '/network.js':'network.js', '/network.css':'network.css', '/network-favicon.svg':'network-favicon.svg', '/network-haffer.ttf':'network-haffer.ttf', '/network-mono.ttf':'network-mono.ttf'}
                if path not in files:
                    self.send_error(404)
                    return
                file = PUBLIC/files[path]
                data = file.read_bytes()
                kind = {'.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.svg':'image/svg+xml', '.ttf':'font/ttf', '.txt':'text/plain'}[file.suffix]
            self.send_response(200)
            self.send_header('Content-Type', kind+'; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', 'https://affine.io')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def export_snapshot(snapshot, destination, public=PUBLIC):
    """Publish data and preserve the current guide against legacy regeneration."""
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot, allow_nan=False))
    temporary.replace(destination)
    # This export target is the production static website directory. Legacy
    # website pushes regenerate a historical guide, so restore our canonical
    # public guide atomically whenever it differs. No private URLs are used.
    guide = Path(public)/'llms.txt'
    target = destination.parent/'llms.txt'
    content = guide.read_bytes()
    if not target.exists() or target.read_bytes() != content:
        temporary_guide = target.with_suffix('.tmp')
        temporary_guide.write_bytes(content)
        temporary_guide.replace(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', default=str(ROOT/'state/dashboard/network.sqlite'))
    p.add_argument('--source', default=str(ROOT/'state'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8794)
    p.add_argument('--snapshot', action='store_true')
    p.add_argument('--export', help='Atomically publish the public SQLite snapshot as JSON')
    args=p.parse_args()
    db=Database(args.db,args.source)
    db.refresh()
    def export():
        if args.export:
            export_snapshot(db.snapshot(), args.export)
    export()
    if args.snapshot:
        print(json.dumps(db.snapshot()))
        return
    def poll():
        while True:
            time.sleep(15)
            try:
                db.refresh()
                export()
            except Exception:
                # Preserve previous committed snapshot on incomplete source writes.
                pass
    threading.Thread(target=poll,daemon=True).start()
    serve(db,args.host,args.port)


if __name__=='__main__':
    main()
