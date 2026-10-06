import base64
import hashlib
import json
import shutil
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet import environments as e
from subnet import native_session_validation as v

class NativeValidationControls(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=self.root, prefix='native-source-control-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        legacy = self.directory/'legacy'
        shutil.copytree(e.LEGACY_ROOT/'rollouts/envs/affine_math_v1', legacy/'rollouts/envs/affine_math_v1')
        research = self.directory/'research'
        (research/'environments').mkdir(parents=True)
        import sys
        sys.path.insert(0, str(legacy/'rollouts/envs/affine_math_v1'))
        from affine_math_v1.taskset import MathData, MathConfig, SYSTEM
        rows = []
        for i in range(2):
            data = MathData(idx=i, name='control-'+str(i), system_prompt=SYSTEM, prompt='Compute '+str(i)+'+2.',
                            problem='Compute '+str(i)+'+2.', answer=str(i+2), subject='Algebra', level='1')
            rows.append(dict(task_class='MathTask', data=data.model_dump(mode='json'),
                             task_config=MathConfig().task.model_dump(mode='json')))
        self.snapshot = self.directory/'tasks.json'
        self.snapshot.write_bytes(v.canonical(rows))
        self.spec = e.build_spec('affine_math', {'task_snapshot':str(self.snapshot)}, legacy_root=legacy,
                                 research_root=research, num_samples=2, max_turns=1)
        files = [Path(v.__file__), Path(e.__file__), e.PACKAGE_ROOT/'native_math_grader.py', self.snapshot]
        files += [p for p in legacy.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__'not in p.parts]
        self.key = SigningKey.generate()
        now = time.time()
        self.scope = dict(version=v.VERSION, execute_allowed=True, created_at=now-1, expires_at=now+120,
                          job_id='CPU-control', source_root=str(self.root),
                          source_files={str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                          environment_sha256=hashlib.sha256(v.canonical(self.spec.to_dict())).hexdigest())
        self.legacy = legacy

    def cache(self, scope=None, job='CPU-control'):
        payload = scope or self.scope
        signed = dict(payload=payload, signer=self.key.verify_key.encode().hex(),
                      signature=base64.b64encode(self.key.sign(v.canonical(payload)).signature).decode())
        return v.JobSourceValidation(signed, authority=signed['signer'], job_id=job, spec=self.spec)

    def test_exact_source_hash_once_and_fresh_native_outcomes_no_state_leak(self):
        with patch.object(e, '_source_hash', wraps=e._source_hash) as hashing:
            cache = self.cache()
            for index, answer, classification in [(0,'2','positive'), (1,'2','negative'), (0,'9','negative')]:
                session = e.create_session(self.spec, source_validation=cache)
                try:
                    initial = session.reset(index, 17)
                    self.assertEqual(initial['task_name'], 'control-'+str(index))
                    self.assertEqual(session.turns, 0)
                    result = session.step({'text':r'\boxed{'+answer+'}'})
                    self.assertEqual(result['classification'], classification)
                finally:
                    session.close()
            self.assertEqual(hashing.call_count, 1)

    def test_default_path_revalidates_original_each_new_session(self):
        with patch.object(e, '_source_hash', wraps=e._source_hash) as hashing:
            for _ in range(2):
                e.create_session(self.spec).close()
            self.assertEqual(hashing.call_count, 2)

    def test_cross_job_environment_and_expiry_refuse(self):
        with self.assertRaises(ValueError):
            self.cache(job='other')
        cache = self.cache()
        with self.assertRaises(ValueError):
            e.create_session(replace(self.spec, max_output_tokens=9), source_validation=cache)
        cache.expires = time.time()-1
        with self.assertRaises(ValueError):
            e.create_session(self.spec, source_validation=cache)

    def test_actual_snapshot_same_size_restored_mtime_replacement_and_new_file(self):
        cache = self.cache()
        before = self.snapshot.stat()
        data = self.snapshot.read_bytes().replace(b'Compute 0+2.', b'Compute 9+2.')
        self.assertEqual(len(data), before.st_size)
        self.snapshot.write_bytes(data)
        import os
        os.utime(self.snapshot, ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaises(ValueError):
            cache.snapshot_row(self.spec, 0)
        self.snapshot.write_bytes(v.canonical(json.loads(data)))
        self.spec = replace(self.spec, source_hash=e._source_hash(self.spec))
        self.scope['source_files'][str(self.snapshot.relative_to(self.root))] = hashlib.sha256(self.snapshot.read_bytes()).hexdigest()
        self.scope['environment_sha256'] = hashlib.sha256(v.canonical(self.spec.to_dict())).hexdigest()
        cache = self.cache()
        (self.legacy/'rollouts/envs/added.py').write_text('new source')
        with self.assertRaises(ValueError):
            cache.validate(self.spec)

    def test_source_digest_inventory_and_snapshot_rows_are_not_trusted_mutably(self):
        scope = dict(self.scope, source_files=dict(self.scope['source_files']))
        scope['source_files'][str(Path(e.__file__).relative_to(self.root))] = '0'*64
        with self.assertRaises(ValueError):
            self.cache(scope)
        cache = self.cache()
        row = cache.snapshot_row(self.spec, 0)
        row['data']['answer'] = 'forged'
        self.assertEqual(cache.snapshot_row(self.spec,0)['data']['answer'],'2')
        with self.assertRaises(ValueError):
            e.create_session(self.spec, source_validation=object())

if __name__ == '__main__':
    unittest.main()
