import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet import mimo_audit_scheduler as mas
from subnet.storage import canonical
from ops.mimo_lifecycle import archive_scoped_audit_scheduler, archive_execution_resolver

V7, V82, V821, V822 = 'a7' * 32, 'b8' * 32, 'c9' * 32, 'd0' * 32
ID = 'f' * 64


class FakeQueue:
    authority = ID
    def __init__(self):
        self.db = sqlite3.connect(':memory:'); self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT)'); self.enqueued = []
    @contextmanager
    def transaction(self): yield self.db
    def enqueue(self, envelope):
        self.enqueued.append(envelope['payload']['job_id'])
        self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)', (envelope['payload']['job_id'], 'queued'))
    def set(self, job_id, status): self.db.execute('UPDATE jobs SET status=? WHERE id=?', (status, job_id))


def population(epoch, archive, n):
    import hashlib
    hx = lambda *a: hashlib.sha256(repr(a).encode()).hexdigest()
    records = [dict(epoch=epoch, round=1, checkpoint=hx('c', epoch), miner=hx('m', i), env_id='e', index=i, batch_sha256=hx('b', epoch, i),
                    proof_sha256=hx('p', epoch, i), commitment_sha256=hx('k', epoch, i), verifier_contract_sha256=hx('v'), committed_at=0)
               for i in range(n)]
    return dict(payload=dict(version='v', committed_at=0, records=records,
                             manifest_document=dict(payload=dict(epoch=epoch, source_bundle=dict(sha256=archive)))))


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)/'auditor'; root.mkdir(mode=0o700)
        self.saved = {k: getattr(mas, k) for k in ('signed', 'build_audit_job', '_immutable', 'authenticate_population')}
        self.auth_calls = []
        mas.signed = lambda d, a: d['payload']
        def auth(doc, authority, manifest_guard): self.auth_calls.append(doc['payload']['manifest_document']['payload']['epoch']); return doc['payload']
        mas.authenticate_population = auth
        mas.build_audit_job = lambda p, rows, **k: dict(job_id=k['job_id'], expires_at=k['expires_at'], manifest=p['manifest_document'], rows=rows)
        mas._immutable = lambda *a: None
        self.queue = FakeQueue()
        Scoped = archive_scoped_audit_scheduler(mas.AuditScheduler, {V821, V822})
        self.s = Scoped(root, bucket=SimpleNamespace(presign=lambda *a: 'u'), queue=self.queue, identity=SimpleNamespace(id=ID),
                        manifest_guard=lambda m: m, metadata=lambda m: dict(source_files={'subnet/a.py': '1'}, runtime_versions={}),
                        max_inflight=3, budget_per_tick=3, max_execution_retries=3)
        self.s._signed = lambda job: dict(payload=job)
        pops = {'e07': population('e07', V7, 40), 'e39': population('e39', V82, 20), 'e40': population('e40', V821, 12), 'e41': population('e41', V822, 6)}
        state = dict(version='mimo-audit-scheduler-v1', populations={}, draws={})
        for epoch, doc in pops.items():
            c = mas.digest(doc); (root/(c + '.population.json')).write_bytes(canonical(doc)); state['populations'][epoch] = c
        r7 = pops['e07']['payload']['records']
        state['draws'] = {
            mas.digest(r7[0]): dict(row=r7[0], seed='s', selected_at=1, generation=3, job_id='mimo-audit-old-3', expires_at=10, prior_jobs=['a', 'b', 'c']),
            mas.digest(r7[1]): dict(row=r7[1], seed='s', selected_at=1, generation=4, prior_jobs=['a', 'b', 'c', 'd']),
            mas.digest(r7[2]): dict(row=r7[2], seed='s', selected_at=1, generation=0, job_id='mimo-audit-old-ok', expires_at=10)}
        self.queue.db.execute("INSERT INTO jobs VALUES ('mimo-audit-old-3','failed'),('mimo-audit-old-ok','complete')")
        (root/'state.json').write_bytes(canonical(state)); self.root = root; self.old = copy.deepcopy(state['draws'])

    def tearDown(self):
        for k, v in self.saved.items(): setattr(mas, k, v)
        self.tmp.cleanup()

    def raw_state(self): return json.loads((self.root/'state.json').read_bytes())

    def test_only_auditable_archives_are_drawn_and_old_draws_never_renewed(self):
        for i in range(10):
            self.s.tick(now=1000 + i)
            for j in list(self.queue.enqueued): self.queue.set(j, 'failed' if i % 2 else 'complete')
        st = self.raw_state()
        new = {k: d for k, d in st['draws'].items() if k not in self.old}
        self.assertTrue(new)
        self.assertEqual({d['row']['epoch'] for d in new.values()} - {'e40', 'e41'}, set())
        for k, d in self.old.items(): self.assertEqual(st['draws'][k], d)           # byte-identical history, never renewed
        self.assertFalse([j for j in self.queue.enqueued if 'old' in j])
        self.assertTrue(all(any(d.get('job_id') == j or j in d.get('prior_jobs', []) for d in new.values()) for j in self.queue.enqueued))

    def test_failed_auditable_draw_is_renewed_up_to_cap_3(self):
        for i in range(12):
            self.s.tick(now=1000 + i)
            for j in list(self.queue.enqueued): self.queue.set(j, 'failed')
        gens = [d.get('generation', 0) for d in self.raw_state()['draws'].values() if d['row']['epoch'] in ('e40', 'e41')]
        self.assertTrue(gens and max(gens) == 3)

    def test_excluded_populations_not_authenticated_and_reported(self):
        self.s.tick(now=1000)
        self.assertEqual(set(self.auth_calls), {'e40', 'e41'})
        rep = json.loads((self.root/'UNAUDITABLE-BY-DESIGN.json').read_bytes())
        self.assertEqual(sorted(rep['excluded_epochs']), ['e07', 'e39'])
        self.assertEqual(rep['excluded_epochs']['e07']['records'], 40); self.assertEqual(rep['excluded_epochs']['e07']['draws_parked'], 3)
        self.assertEqual(rep['excluded_epochs']['e39']['archive'], V82); self.assertEqual(rep['excluded_records'], 60)
        self.assertEqual(rep['auditable_archives'], sorted([V821, V822]))

    def test_load_save_round_trip_keeps_parked_draws(self):
        st = self.s._load(); self.assertEqual(set(st['draws']), set()); self.assertEqual(set(st['_parked_unauditable']), set(self.old))
        self.s._save(st); self.assertEqual(self.raw_state()['draws'], self.old); self.assertNotIn('_parked_unauditable', self.raw_state())

    def test_requires_explicit_auditable_set(self):
        with self.assertRaises(ValueError): archive_scoped_audit_scheduler(mas.AuditScheduler, set())


class AuditableArchiveTests(unittest.TestCase):
    def test_auditable_archives_are_those_matching_live_closure(self):
        class P:
            def __init__(s, sha, a): s.policy_sha256 = sha; s.scope = {'immutable_manifest': {'source_bundle': {'sha256': a}}}; s.envelope = sha
        class C:
            def __init__(s, *ps): s.members = [(i, p) for i, p in enumerate(ps)]; s.active = ps[-1]
            policy_sha256 = property(lambda s: s.active.policy_sha256)
        inv = lambda n: dict(source_files={'subnet/x.py': n}, runtime_versions={'t': '1'})
        live = inv('821')
        r = archive_execution_resolver(C(P('1', V7), P('2', V82), P('3', V821), P('4', V822)),
                                       {V7: inv('7'), V82: inv('82'), V821: live, V822: live}, live)
        self.assertEqual(r.auditable_archives, frozenset({V821, V822}))


if __name__ == '__main__':
    unittest.main()
