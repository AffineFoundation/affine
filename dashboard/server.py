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
        for folder in self.source.iterdir():
            if not folder.is_dir() or folder.name not in ('live', 'e2e-final', 'registered-test', 'registered-test-compatible', 'multi-environment', 'service-conformance', 'gpu-continuous', 'gpu-wide'):
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
                           audit_policy=doc.get('audit_policy', ''), source=folder.name)
                epochs[eid] = row
                metrics = read(folder/f'{eid}-training-metrics.json', {})
                if metrics:
                    row['training'] = {key:metrics.get(key) for key in ('steps','losses','weights_changed','checkpoint','objective','input_pairs')}
                    if metrics.get('weights_changed'):
                        row['phase'] = 'trained'
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
                'requested_count','completed_count','taskset_hash','policy_kind','model_runtime_revision')
        for path in (self.source/'evaluations').glob('*.json'):
            raw = read(path,{})
            if not isinstance(raw,dict) or not all(isinstance(raw.get(k),str) for k in ('run_id','env_id','dataset_id','status')):
                continue
            row = {key:raw[key] for key in keys if key in raw}
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
                files = {'/':'index.html', '/index.html':'index.html', '/network.js':'network.js', '/network.css':'network.css', '/network-favicon.svg':'network-favicon.svg', '/network-haffer.ttf':'network-haffer.ttf', '/network-mono.ttf':'network-mono.ttf'}
                if path not in files:
                    self.send_error(404)
                    return
                file = PUBLIC/files[path]
                data = file.read_bytes()
                kind = {'.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.svg':'image/svg+xml', '.ttf':'font/ttf'}[file.suffix]
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
            path=Path(args.export)
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(db.snapshot(),allow_nan=False))
            temporary.replace(path)
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
