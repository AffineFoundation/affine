import copy
import json
import hashlib
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from nacl.signing import SigningKey
from ops.running_api_validation import canonical, digest, sign
from ops import trusted_cpu_service_lifetime as lifetime
from ops.trusted_cpu_service_lifetime import Supervisor, api_ready, auditor_ready


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.events = []
        self.starts = 0
        self.stops = 0
        self.ready = True
        self.old = dict(systemd=dict(MainPID='10', InvocationID='old', ActiveState='active'),
                        process=dict(pid=10, ticks=1, argv=['old'], started_at=10))
        self.empty = dict(systemd=dict(MainPID='0', InvocationID='', ActiveState='inactive'), process=None)
        self.fresh = dict(systemd=dict(MainPID='20', InvocationID='new', ActiveState='active'),
                          process=dict(pid=20, ticks=2, argv=['new'], started_at=100))
        self.actual = copy.deepcopy(self.empty)
        service = dict(unit='fixture', initial_instance=self.old, expected_argv=['new'], max_failures=2,
                       circuit_seconds=60, backoff_seconds=1, max_backoff_seconds=4, readiness_seconds=10)
        self.policy = dict(authority='fixture', services={'api': service})
        self.guard_calls = 0
        def guard(): self.guard_calls += 1
        def sleep(seconds): self.now += seconds
        def start():
            self.starts += 1
            self.actual = copy.deepcopy(self.fresh)
        def stop():
            self.stops += 1
            self.actual = copy.deepcopy(self.empty)
        self.start, self.stop, self.sleep = start, stop, sleep
        self.supervisor = Supervisor(self.policy, 'api', observe=lambda s: copy.deepcopy(self.actual),
                                     start=start, stop=stop, ready=lambda since: self.ready,
                                     record=self.events.append, guard=guard, clock=lambda: self.now,
                                     wall=lambda: self.now, sleep=sleep)

    def test_adopts_exact_original_without_restart_or_expiry_alias(self):
        self.actual = copy.deepcopy(self.old)
        self.assertEqual(self.supervisor.step(), 'live')
        self.assertEqual((self.starts, self.stops), (0, 0))
        self.assertTrue(self.events[0]['adopted_original'])

    def test_start_after_exit_and_authentic_readiness(self):
        self.assertEqual(self.supervisor.step(), 'restarted')
        self.assertEqual((self.starts, self.stops), (1, 0))
        self.assertGreaterEqual(self.guard_calls, 2)
        self.assertEqual(self.events[-1]['event'], 'restart_ready')

    def test_alive_unbound_is_not_declared_ready_and_is_retired(self):
        self.ready = False
        self.assertEqual(self.supervisor.step(), 'failed')
        self.assertEqual((self.starts, self.stops), (1, 1))
        self.assertFalse(any(event['event'] == 'restart_ready' for event in self.events))

    def test_start_timeout_but_bound_genuine_instance_is_reconciled(self):
        def timeout():
            self.start()
            raise TimeoutError('start observation timeout')
        self.supervisor.start = timeout
        self.assertEqual(self.supervisor.step(), 'restarted')
        self.assertEqual(self.stops, 0)

        self.assertEqual(self.events[-1]['event'], 'restart_ready_after_start_timeout')

    def test_start_timeout_unbound_cleanup_is_not_orphaned(self):
        self.ready = False
        def timeout():
            self.start()
            raise TimeoutError('start observation timeout')
        self.supervisor.start = timeout
        self.assertEqual(self.supervisor.step(), 'failed')
        self.assertEqual(self.stops, 1)

    def test_unrelated_live_process_is_preserved(self):
        self.actual = copy.deepcopy(self.fresh)
        self.actual['process']['argv'] = ['unrelated']
        with self.assertRaisesRegex(ValueError, 'unrelated live'):
            self.supervisor.step()
        self.assertEqual((self.starts, self.stops), (0, 0))

    def test_instance_race_during_backoff_refuses_start(self):
        def sleep(seconds):
            self.sleep(seconds)
            self.actual = copy.deepcopy(self.fresh)
            self.actual['process']['argv'] = ['unrelated']
        self.supervisor.sleep = sleep
        with self.assertRaisesRegex(ValueError, 'changed during backoff'):
            self.supervisor.step()
        self.assertEqual(self.starts, 0)

    def test_changed_pins_after_backoff_refuse_start(self):
        self.supervisor.guard = lambda: (_ for _ in ()).throw(ValueError('pin changed'))
        with self.assertRaisesRegex(ValueError, 'pin changed'):
            self.supervisor.step()
        self.assertEqual(self.starts, 0)

    def test_bounded_failure_circuit_and_automatic_cooldown(self):
        self.ready = False
        self.assertEqual(self.supervisor.step(), 'failed')
        self.assertEqual(self.supervisor.step(), 'failed')
        attempts = self.starts
        self.assertEqual(self.supervisor.step(), 'circuit_open')
        self.assertEqual(self.supervisor.step(), 'circuit_open')
        self.assertEqual(self.starts, attempts)
        self.assertEqual(sum(event['event'] == 'circuit_open' for event in self.events), 1)
        self.now += 61
        self.ready = True
        self.assertEqual(self.supervisor.step(), 'restarted')

    def test_short_lived_successful_restarts_also_trip_circuit(self):
        self.assertEqual(self.supervisor.step(), 'restarted')
        self.actual = copy.deepcopy(self.empty)
        self.assertEqual(self.supervisor.step(), 'restarted')
        self.actual = copy.deepcopy(self.empty)
        self.assertEqual(self.supervisor.step(), 'circuit_open')
        self.assertEqual(self.starts, 2)

    def test_supervisor_recovery_never_labels_unknown_live_instance_ready(self):
        self.actual = copy.deepcopy(self.fresh)
        self.ready = False
        self.assertEqual(self.supervisor.step(), 'awaiting_readiness')
        self.assertEqual((self.starts, self.stops), (0, 0))
        self.now += 11
        self.assertEqual(self.supervisor.step(), 'failed')
        self.assertEqual(self.stops, 1)

    def test_readiness_identity_race_never_stops_reused_unrelated_pid(self):
        def ready(since):
            self.actual['process']['ticks'] = 77
            self.actual['process']['argv'] = ['unrelated']
            return True
        self.supervisor.ready = ready
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            self.supervisor.step()
        self.assertEqual(self.stops, 0)


class APIReadinessTests(unittest.TestCase):
    def setUp(self):
        self.code = 403
        self.body = b'{"error":"request rejected"}'
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                fixture.request_body = self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(fixture.code)
                self.end_headers()
                self.wfile.write(fixture.body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.service = dict(readiness_url='http://127.0.0.1:'+str(self.server.server_port)+'/request')
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_rejected_empty_unauthenticated_request_is_real_readiness(self):
        self.assertTrue(api_ready(self.service))
        self.assertEqual(self.request_body, b'{}')

    def test_success_response_cannot_fake_the_authentication_guard(self):
        self.code = 200
        self.assertFalse(api_ready(self.service))

    def test_wrong_or_malformed_rejection_not_readiness(self):
        for body in (b'{"error":"wrong"}', b'not JSON'):
            with self.subTest(body=body):
                self.body = body
                self.assertFalse(api_ready(self.service))

    def test_unbound_server_is_not_readiness(self):
        self.server.shutdown()
        self.server.server_close()
        self.assertFalse(api_ready(self.service))


class AuditorReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.job = dict(job_id='original', role='verify', created_at=100, source_files={'runtime.py': 'pin'},
                        manifest=sign(self.key, dict(source_bundle={'sha256': 'source'})))
        (self.root/'original-job.json').write_bytes(canonical(sign(self.key, self.job)))
        state = self.root/'audit-state.json'
        state.write_bytes(canonical(dict(jobs={'original': dict(job_sha256=digest(self.job))})))
        config = self.root/'config.json'
        config.write_bytes(canonical(dict(continuous_audit_service=dict(
            source_admission=sign(self.key, dict(approved_sources={'source': self.job['source_files']}))))))
        health = self.root/'health.json'
        health.write_bytes(canonical(dict(at=99)))
        self.database = self.root/'queue.sqlite3'
        with sqlite3.connect(self.database) as db:
            db.execute('CREATE TABLE jobs(id TEXT,digest TEXT,status TEXT)')
            db.execute('INSERT INTO jobs VALUES(?,?,?)', ('original', digest(self.job), 'leased'))
        self.policy = dict(database=str(self.database))
        self.service = dict(state_path=str(state), config_path=str(config), health_path=str(health))

    def test_genuine_signed_job_progress_independent_of_completed_tick(self):
        result = auditor_ready(self.policy, self.service, 100, self.authority)
        self.assertTrue(result['producer_progress'])
        self.assertFalse(result['completed_tick'])

    def test_wrong_queue_digest_refused(self):
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE jobs SET digest='wrong'")
        self.assertIsNone(auditor_ready(self.policy, self.service, 100, self.authority))

    def test_busy_database_is_pending_not_fabricated_empty_queue(self):
        db = sqlite3.connect(self.database)
        db.execute('BEGIN EXCLUSIVE')
        try:
            self.assertIsNone(auditor_ready(self.policy, self.service, 100, self.authority))
        finally:
            db.rollback()
            db.close()

    def test_stale_job_not_readiness(self):
        self.assertIsNone(auditor_ready(self.policy, self.service, 101, self.authority))

    def test_tampered_signed_job_refused(self):
        envelope = json.loads((self.root/'original-job.json').read_bytes())
        envelope['payload']['source_files'] = {'evil': 'wrong'}
        (self.root/'original-job.json').write_bytes(canonical(envelope))
        with self.assertRaises(Exception):
            auditor_ready(self.policy, self.service, 100, self.authority)

    def test_actual_completed_tick_is_distinct_from_new_job_progress(self):
        Path(self.service['health_path']).write_bytes(canonical(dict(at=101)))
        result = auditor_ready(self.policy, self.service, 100, self.authority)
        self.assertTrue(result['completed_tick'])
        self.assertFalse(result['producer_progress'])


class LifetimeGrantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.key = SigningKey.generate()
        cls.authority = cls.key.verify_key.encode().hex()
        cls.files = {}
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            cls.files[str(path)] = hashlib.sha256(value).hexdigest()
        cls.write = staticmethod(write)
        state = cls.root/'state'
        queue = state/'roles/verifier-queue.sqlite3'
        write(queue, b'fixture queue; never opened by validation')
        cls.files.pop(str(queue))
        st = queue.stat()
        inode = [st.st_dev, st.st_ino]
        credentials = cls.root/'R2.fixture.env'
        write(credentials, b'fixture only; no real credential\n')
        credentials.chmod(0o600)
        seed = cls.root/'authority.seed'
        seed.write_text(cls.key.encode().hex())
        seed.chmod(0o600)
        registry = cls.root/'registry.json'
        source = cls.root/'source/runtime.py'
        write(source, b'fixture immutable science')
        admission = sign(cls.key, dict(approved_sources={'source': {'runtime.py': cls.files[str(source)]}}))
        write(registry, canonical(sign(cls.key, dict(version='source-specific-sampling-api-admission-v1',
            sources={'source': dict(runtime_files={'runtime.py': cls.files[str(source)]}, runtime_versions={}, sampling_versions=[None])}))))
        entrypoint = str(Path(lifetime.__file__).resolve())
        cls.files[entrypoint] = hashlib.sha256(Path(entrypoint).read_bytes()).hexdigest()
        services = {}
        histories = {}
        for role, count in [('api', 173), ('auditor', 2102)]:
            config = dict(state=str(state), bucket=dict(credentials_file=str(credentials)),
                          remote=dict(verifier_queue=dict(port=19080)),
                          continuous_audit_service=dict(source_admission=admission))
            config_path = cls.root/(role+'-config.json')
            write(config_path, canonical(config))
            tree = cls.root/role
            pins = {}
            for index in range(count):
                name = 'subnet/module'+str(index)+'.py'
                write(tree/name, b'fixture operator')
                pins[name] = cls.files[str(tree/name)]
            fragment = cls.root/(role+'.service')
            write(fragment, b'fixture unit')
            history = cls.root/(role+'-historical.json')
            write(history, canonical(sign(cls.key, dict(created_at=1, expires_at=20, queue_inode=inode))))
            histories[role] = dict(path=str(history), sha256=cls.files[str(history)])
            initial = dict(systemd=dict(MainPID='10', ActiveState='active', InvocationID=role+'old'),
                           process=dict(pid=10, ticks=1, started_at=10,
                                        argv=['old', '--scope', str(history), '--config', str(config_path)]))
            services[role] = dict(role=role, unit=role+'.service', config_path=str(config_path),
                                 config_sha256=digest(config), operator_tree=str(tree), operator_files=pins,
                                 unit_fragment=str(fragment), dropins=[], environment_files=[],
                                 initial_instance=initial, max_failures=3, circuit_seconds=1800,
                                 backoff_seconds=5, max_backoff_seconds=300, readiness_seconds=120,
                                 expected_argv=[sys.executable, '-I', '-B', entrypoint, 'run-service',
                                                '--policy', str(cls.root/'policy.json'), '--authority', cls.authority, '--role', role])
        services['api'].update(registry_path=str(registry), source_trees={'source': str(source.parent)},
                               readiness_url='http://127.0.0.1:19080/request')
        cls.base = dict(version=lifetime.VERSION, execute_allowed=False, authority=cls.authority,
                        job_permissions_unchanged=True, scientific_contracts_unchanged=True, GPU_execution_allowed=False,
                        created_at=50, not_after=None, authorization_status_path=str(cls.root/'authorization.json'),
                        host_uid=os.getuid(), host_machine_id_sha256=hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest(),
                        files=cls.files, database=str(queue), queue_inode=inode, services=services,
                        historical_execution_scopes=histories, entrypoint=entrypoint, python=sys.executable,
                        policy_path=str(cls.root/'policy.json'), authority_seed_path=str(seed))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.policy = copy.deepcopy(self.base)

    def validate(self, require_execution=False):
        Path(self.policy['authorization_status_path']).write_bytes(canonical(sign(self.key, dict(
            version=lifetime.STATUS, active_policy_sha256=digest(self.policy)))))
        return lifetime.validate(sign(self.key, self.policy), self.authority, now=100, require_execution=require_execution)

    def test_default_off_full_173_2102_pins_and_expired_history_check_only(self):
        self.assertEqual(self.validate()['version'], lifetime.VERSION)
        self.assertNotEqual(self.policy['version'], 'independent-queue-source-aware-api173-v1')
        with self.assertRaisesRegex(ValueError, 'default-off'):
            self.validate(require_execution=True)

    def test_explicit_new_lifetime_grant_authorizes_recovery_without_rewriting_old_scopes(self):
        before = {role: Path(value['path']).read_bytes() for role, value in self.policy['historical_execution_scopes'].items()}
        self.policy['execute_allowed'] = True
        self.validate(require_execution=True)
        self.assertEqual(before, {role: Path(value['path']).read_bytes() for role, value in self.policy['historical_execution_scopes'].items()})

    def test_revoked_grant_refused(self):
        self.validate()
        Path(self.policy['authorization_status_path']).write_bytes(canonical(sign(self.key, dict(
            version=lifetime.STATUS, active_policy_sha256='revoked'))))
        with self.assertRaisesRegex(ValueError, 'revoked'):
            lifetime.validate(sign(self.key, self.policy), self.authority, now=100)

    def test_wrong_host_refused(self):
        self.policy['host_machine_id_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'same trusted host'):
            self.validate()

    def test_missing_full_operator_map_refused(self):
        self.policy['services']['auditor']['operator_files'].pop('subnet/module0.py')
        with self.assertRaisesRegex(ValueError, 'full exact'):
            self.validate()

    def test_changed_source_or_R2_file_refused(self):
        for path in [self.root/'source/runtime.py', self.root/'R2.fixture.env']:
            before = path.read_bytes()
            path.write_bytes(b'changed')
            try:
                with self.assertRaisesRegex(ValueError, 'file changed'):
                    self.validate()
            finally:
                path.write_bytes(before)

    def test_wrong_queue_inode_refused(self):
        self.policy['queue_inode'][1] += 1
        with self.assertRaisesRegex(ValueError, 'queue inode'):
            self.validate()

    def test_extra_service_permission_refused(self):
        self.policy['services']['trainer'] = self.policy['services']['auditor']
        with self.assertRaisesRegex(ValueError, 'exact two'):
            self.validate()

    def test_external_or_writing_readiness_route_refused(self):
        self.policy['services']['api']['readiness_url'] = 'https://wrong/request'
        with self.assertRaisesRegex(ValueError, 'local non-mutating'):
            self.validate()

    def test_original_process_start_must_be_authorized(self):
        self.policy['services']['api']['initial_instance']['process']['started_at'] = 21
        with self.assertRaisesRegex(ValueError, 'execution provenance'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
