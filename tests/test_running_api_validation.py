import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from nacl.signing import SigningKey
from ops.running_api_validation import (LOCATOR, POLICY, VALIDATION, authenticate, canonical,
                                        ensure_validation, sign, validate_dependency)


class RunningApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.seed = self.root/'authority.seed'
        self.seed.write_text(self.key.encode().hex())
        self.seed.chmod(0o600)
        self.files = {}
        def write(path, raw):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self.files[str(path)] = hashlib.sha256(raw).hexdigest()
            return self.files[str(path)]
        self.write = write
        parent_files = {}
        operator_files = {}
        for index in range(173):
            name = 'subnet/file'+str(index)+'.py'
            operator_files[name] = write(self.root/'api173'/name, b'unchanged')
            if index < 172:
                parent_files[name] = write(self.root/'api172'/name, b'unchanged')
        self.config = self.root/'config.json'
        write(self.config, canonical({'queue': 'original'}))
        parent = dict(created_at=1, expires_at=20, operator_tree=str(self.root/'api172'),
                      operator_files=parent_files)
        parent_path = self.root/'original-parent.json'
        parent_sha = write(parent_path, canonical(sign(self.key, parent)))
        source_root = self.root/'source'
        source_pin = write(source_root/'runtime.py', b'unchanged science')
        registry = self.root/'registry.json'
        write(registry, canonical(sign(self.key, dict(version='source-specific-sampling-api-admission-v1',
              sources={'source': dict(runtime_files={'runtime.py': source_pin}, runtime_versions={}, sampling_versions=[None])}))))
        self.queue = self.root/'queue.sqlite3'
        self.queue.write_bytes(b'original queue must never change')
        api = dict(version='independent-queue-source-aware-api173-v1', execute_allowed=True,
                   created_at=1, expires_at=20, parent_scope_path=str(parent_path),
                   parent_scope_sha256=parent_sha, config_sha256=hashlib.sha256(canonical({'queue': 'original'})).hexdigest(),
                   operator_tree=str(self.root/'api173'), operator_files=operator_files,
                   registry_path=str(registry), source_trees={'source': str(source_root)}, database=str(self.queue))
        self.execution = self.root/'original-execution.json'
        execution_sha = write(self.execution, canonical(sign(self.key, api)))
        process = dict(pid=17, ticks=19, started_at=10., argv=['launcher', '--scope', str(self.execution), '--config', str(self.config)],
                       systemd=dict(MainPID='17', InvocationID='original', ActiveState='active'))
        st = self.queue.stat()
        self.policy = dict(version=POLICY, enabled=True, validation_only=True, API_restart_allowed=False,
                           created_at=5, validity_seconds=600, renew_before_seconds=60, unit='original-api.service',
                           invocation='original', process=process, execution_scope_path=str(self.execution),
                           execution_scope_sha256=execution_sha, config_path=str(self.config), database=str(self.queue),
                           queue_inode=[st.st_dev, st.st_ino], files=dict(self.files),
                           validation_directory=str(self.root/'private-validation'))
        self.inspect = lambda unit, pid: copy.deepcopy(self.policy['process'])

    def run_validation(self, now=100):
        return ensure_validation(sign(self.key, self.policy), self.authority, self.seed,
                                 now=now, inspect=self.inspect)

    def test_expired_execution_window_preserved_and_validation_version_not_executable(self):
        original = self.execution.read_bytes()
        queue = self.queue.read_bytes()
        locator = self.run_validation()
        value = authenticate(json.loads(Path(locator['validation_path']).read_bytes()), self.authority)
        self.assertEqual(value['version'], VALIDATION)
        self.assertNotEqual(value['version'], 'independent-queue-source-aware-api173-v1')
        self.assertNotIn('execute_allowed', value)
        self.assertFalse(value['API_restart_allowed'])
        self.assertEqual(self.execution.read_bytes(), original)
        self.assertEqual(self.queue.read_bytes(), queue)

    def test_reuses_fresh_validation_and_renews_immutably(self):
        first = self.run_validation()
        first_raw = Path(first['validation_path']).read_bytes()
        self.assertEqual(first, self.run_validation(now=101))
        second = self.run_validation(now=641)
        self.assertNotEqual(first['validation_path'], second['validation_path'])
        self.assertEqual(Path(first['validation_path']).read_bytes(), first_raw)
        latest = authenticate(json.loads((Path(self.policy['validation_directory'])/'latest.json').read_bytes()), self.authority)
        self.assertEqual(latest['version'], LOCATOR)
        self.assertEqual(latest, second)

    def test_changed_process_never_renews(self):
        self.inspect = lambda unit, pid: dict(self.policy['process'], ticks=20)
        with self.assertRaisesRegex(ValueError, 'same live original'):
            self.run_validation()

    def test_disabled_policy_requires_explicit_authorization(self):
        self.policy['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'explicit validation-only'):
            self.run_validation()

    def test_source_or_operator_change_fails_closed_even_on_fresh_locator(self):
        self.run_validation()
        for path in [self.root/'api173/subnet/file1.py', self.root/'source/runtime.py']:
            original = path.read_bytes()
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'file changed'):
                self.run_validation(now=101)
            path.write_bytes(original)

    def test_missing_full_map_pin_rejected(self):
        self.policy['files'].pop(str(self.root/'api172/subnet/file1.py'))
        with self.assertRaisesRegex(ValueError, 'complete operator'):
            self.run_validation()

    def test_queue_inode_replacement_rejected(self):
        replacement = self.root/'new.sqlite3'
        replacement.write_bytes(self.queue.read_bytes())
        os.replace(replacement, self.queue)
        with self.assertRaisesRegex(ValueError, 'same original queue'):
            self.run_validation()

    def test_wrong_seed_does_not_create_locator(self):
        self.seed.write_text(SigningKey.generate().encode().hex())
        with self.assertRaisesRegex(ValueError, 'same local authority'):
            self.run_validation()
        self.assertFalse((Path(self.policy['validation_directory'])/'latest.json').exists())

    def test_start_outside_original_scope_rejected_even_if_new_policy_signed(self):
        self.policy['process']['started_at'] = 21
        with self.assertRaisesRegex(ValueError, 'started under original scope'):
            self.run_validation()

    def test_locator_tamper_rejected(self):
        self.run_validation()
        path = Path(self.policy['validation_directory'])/'latest.json'
        value = json.loads(path.read_bytes())
        value['payload']['validation_path'] = str(self.root/'unowned.json')
        path.write_text(json.dumps(value))
        with self.assertRaises(Exception):
            self.run_validation()


if __name__ == '__main__':
    unittest.main()
