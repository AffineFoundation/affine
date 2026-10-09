import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops" / "heldout_monitor"))

import base64
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
import failed_cache_retirement as cleanup
from serial_worker import canonical, digest, lease, save


class FailedCacheRetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        proc_scan = patch.object(cleanup, 'unused_model'); proc_scan.start(); self.addCleanup(proc_scan.stop)
        self.root = Path(self.temp.name) / 'monitor'; self.root.mkdir()
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        self.job = 'step-2-fixture'; self.directory = self.root / 'jobs' / self.job
        self.directory.mkdir(parents=True); (self.directory / 'evaluation.lease').touch()
        self.files = {'a.safetensors': digest(b'alpha'), 'config.json': digest(b'{}')}
        self.cp = digest(canonical(self.files)); self.destination = self.root.parent / 'checkpoints' / self.cp
        self.destination.parent.mkdir()
        descriptor = self.sign(dict(id=self.cp, files=self.files))
        self.read = self.sign(dict(kind='immutable-checkpoint-read-hydration-v1', role='heldout128-evaluate',
            retained_UUID='fixture', checkpoint=self.cp, destination=str(self.destination),
            checkpoint_descriptor=descriptor, checkpoint_descriptor_sha256=digest(canonical(descriptor)),
            objects={n:dict(sha256=h,bytes=5 if n=='a.safetensors' else 2) for n,h in self.files.items()}))
        self.read_hash = digest(canonical(self.read))
        self.assignment = self.sign(dict(version='serial-heldout128-assignment-v1',root=str(self.root),job_id=self.job,
            checkpoint=self.cp,checkpoint_path=str(self.destination),retained_UUID='fixture',
            program_files={'read-plan.json':self.read_hash}))
        self.archive = self.sign(dict(version='serial-heldout128-failed-attempt-v1',job_id=self.job,checkpoint=self.cp,
            assignment_sha256=digest(canonical(self.assignment)),read_plan_sha256=self.read_hash,phase='failed',
            full_readback_verified=True,scientific_success=False,archive={'log':dict(sha256=digest(b'log'),size=3)}))
        self.stage = self.destination.parent / ('.'+self.cp+'.hydrate-'+self.read_hash[:16]); self.stage.mkdir()
        self.work = self.stage / 'objects'; self.work.mkdir()
        (self.work/'a.safetensors.partial').write_bytes(b'al'); (self.work/'config.json').write_bytes(b'{}')
        save(self.stage/'binding.json',dict(plan_sha256=self.read_hash,checkpoint=self.cp,role='heldout128-evaluate',
            retained_UUID='fixture',descriptor_sha256=self.read['payload']['checkpoint_descriptor_sha256']))
        (self.stage/'last-failure.json').write_text('{}'); (self.directory/'worker.log').write_text('original log')
        self.body = dict(version='failed-heldout128-cache-retirement-v1',root=str(self.root),assignment=self.assignment,
            read_plan=self.read,failure_archive=self.archive,remove_complete_model=False)
        self.grant = self.root/'grant.json'; self.write_grant()

    def sign(self, body):
        return dict(payload=body,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(body)).signature).decode())

    def write_grant(self): self.grant.write_bytes(canonical(self.sign(self.body)))
    def run_cleanup(self): return cleanup.retire_failed(self.root,self.grant,self.authority)

    def test_partial_only_cleanup_keeps_original_logs_and_binding(self):
        result=self.run_cleanup(); self.assertEqual(result['bytes'],4); self.assertFalse(self.work.exists())
        self.assertTrue((self.stage/'binding.json').exists()); self.assertTrue((self.directory/'worker.log').exists())
        self.assertEqual(result,self.run_cleanup())

    def test_exhaustion_can_remove_only_full_hashed_completed_model(self):
        self.destination.mkdir(); (self.destination/'a.safetensors').write_bytes(b'alpha');(self.destination/'config.json').write_bytes(b'{}')
        self.body['remove_complete_model']=True;self.write_grant();self.run_cleanup();self.assertFalse(self.destination.exists())

    def test_retry_retains_completed_model(self):
        self.destination.mkdir();(self.destination/'keep').write_text('unchanged')
        self.run_cleanup();self.assertTrue((self.destination/'keep').exists())

    def test_interrupted_unlink_resumes_original_authorization(self):
        unlink=Path.unlink;count=0
        def interrupt(path,*a,**kw):
            nonlocal count
            if path.parent==self.work:
                count+=1
                if count==2:raise OSError('injected interruption')
            return unlink(path,*a,**kw)
        with patch.object(Path,'unlink',interrupt):
            with self.assertRaises(OSError):self.run_cleanup()
        self.assertEqual(self.run_cleanup()['phase'],'complete');self.assertFalse(self.work.exists())

    def test_unknown_staged_file_refuses_all_deletions(self):
        (self.work/'unknown').write_text('keep')
        with self.assertRaisesRegex(ValueError,'unexpected'):self.run_cleanup()
        self.assertEqual(len(list(self.work.iterdir())),3)

    def test_changed_binding_refuses(self):
        save(self.stage/'binding.json',{})
        with self.assertRaisesRegex(ValueError,'binding'):self.run_cleanup()
        self.assertTrue(self.work.exists())

    def test_forged_or_incomplete_archive_refuses(self):
        self.body['failure_archive']['payload']['full_readback_verified']=False; self.write_grant()
        with self.assertRaises(Exception):self.run_cleanup()
        self.body['failure_archive']=self.sign(self.body['failure_archive']['payload']);self.write_grant()
        with self.assertRaisesRegex(ValueError,'archived'):self.run_cleanup()

    def test_other_current_job_refuses(self):
        save(self.root/'active.json',dict(job_id='step-4-other',checkpoint=self.cp))
        with self.assertRaisesRegex(ValueError,'another current'):self.run_cleanup()

    def test_all_three_leases_refuse_concurrent_cleanup(self):
        with lease(self.root):
            with self.assertRaises(BlockingIOError):self.run_cleanup()
        for path in [self.directory/'evaluation.lease',self.destination.parent/('.'+self.cp+'.hydrate-lock')]:
            with path.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):self.run_cleanup()
        self.assertTrue(self.work.exists())

    def test_wrong_completed_hash_preserves_model_and_stage(self):
        self.destination.mkdir();(self.destination/'a.safetensors').write_bytes(b'wrong');(self.destination/'config.json').write_bytes(b'{}')
        self.body['remove_complete_model']=True;self.write_grant()
        with self.assertRaisesRegex(ValueError,'full hash'):self.run_cleanup()
        self.assertTrue(self.work.exists());self.assertTrue(self.destination.exists())

    def test_open_file_guard_refuses_before_unlink(self):
        with patch.object(cleanup, 'unused_model', side_effect=ValueError('open file')):
            with self.assertRaisesRegex(ValueError, 'open file'): self.run_cleanup()
        self.assertTrue(self.work.exists())

    def test_base_path_is_protected_even_in_signed_grant(self):
        self.body['read_plan']['payload']['destination']=str(self.root.parent/'base')
        self.body['read_plan']=self.sign(self.body['read_plan']['payload'])
        self.body['assignment']['payload']['program_files']['read-plan.json']=digest(canonical(self.body['read_plan']))
        self.body['assignment']=self.sign(self.body['assignment']['payload'])
        a=self.body['failure_archive']['payload'];a['read_plan_sha256']=digest(canonical(self.body['read_plan']));a['assignment_sha256']=digest(canonical(self.body['assignment']))
        self.body['failure_archive']=self.sign(a);self.write_grant()
        with self.assertRaisesRegex(ValueError,'base protected'):self.run_cleanup()

    def test_symlink_member_preserved(self):
        p=self.work/'a.safetensors.partial';p.unlink();outside=self.root/'outside';outside.write_text('original');p.symlink_to(outside)
        with self.assertRaises(ValueError):self.run_cleanup()
        self.assertEqual(outside.read_text(),'original')

if __name__=='__main__':unittest.main()
