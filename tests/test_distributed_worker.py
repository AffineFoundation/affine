import base64
import json
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.distributed_worker import Worker
from subnet.storage import canonical
from subnet.distributed_roles import digest


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.authority,self.folder.name)
        self.files={'config.json':hashlib.sha256(b'{}').hexdigest(),'model.safetensors':hashlib.sha256(b'weights').hexdigest()}
        self.job={'job_id':'verify-job','role':'verify','manifest':{'payload':{'checkpoint':{'id':'CP','files':self.files}}}}
        envelope={'signer':self.authority,'payload':self.job,'signature':base64.b64encode(self.key.sign(canonical(self.job)).signature).decode()}
        self.claim={'job':envelope,'job_sha256':digest(self.job),'token':'lease-secret','lease_until':time.time()+300,'attempt':1}

    def test_worker_rejects_plain_remote_http(self):
        with self.assertRaises(ValueError):Worker('http://remote:19080',bytes(SigningKey.generate()),self.authority,self.folder.name)

    def test_fresh_subprocess_and_exact_report_submission(self):
        report={'job_id':'verify-job','chain_transactions':False}
        self.worker.request=Mock(side_effect=[{'claim':self.claim},{'accepted':True}])
        def run(args,**kwargs):
            self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            workspace=Path(args[args.index('--workspace')+1]);out=workspace/'jobs'/'verify-job'
            out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(report))
            return SimpleNamespace(returncode=0)
        with patch('subnet.distributed_worker.subprocess.run',side_effect=run):self.assertTrue(self.worker.once())
        self.worker.request.assert_called_with('report',job_id='verify-job',token='lease-secret',report=report)
        self.assertTrue((Path(self.folder.name)/'verify-job'/'attempt-1'/'pending-report.json').is_file())

    def test_failed_job_reports_failure_not_fabricated_metrics(self):
        self.worker.request=Mock(side_effect=[{'claim':self.claim},{'status':'queued'}])
        with patch('subnet.distributed_worker.subprocess.run',return_value=SimpleNamespace(returncode=1)):
            self.assertTrue(self.worker.once())
        self.worker.request.assert_called_with('fail',job_id='verify-job',token='lease-secret')

    def test_retry_keeps_prior_workspace_and_uses_checked_shared_cache(self):
        cache=Path(self.folder.name)/'backend'/'checkpoints'/'CP';cache.mkdir(parents=True)
        (cache/'config.json').write_bytes(b'{}');(cache/'model.safetensors').write_bytes(b'weights')
        prior=Path(self.folder.name)/'verify-job'/'attempt-1';prior.mkdir(parents=True);(prior/'worker.log').write_text('old')
        self.claim['attempt']=2;self.worker.request=Mock(side_effect=[{'claim':self.claim},{'status':'failed'}])
        with patch('subnet.distributed_worker.subprocess.run',return_value=SimpleNamespace(returncode=1)) as run:
            self.worker.once()
        self.assertEqual((prior/'worker.log').read_text(),'old')
        self.assertIn('--checkpoint-cache',run.call_args.args[0]);self.assertIn(str(cache),run.call_args.args[0])

class CheckpointCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)/'cache';self.root.mkdir()
        self.files={'config.json':hashlib.sha256(b'{}').hexdigest(),'model.safetensors':hashlib.sha256(b'weights').hexdigest()}
        (self.root/'config.json').write_bytes(b'{}');(self.root/'model.safetensors').write_bytes(b'weights')
    def test_complete_inventory_is_reusable_without_model_import(self):
        from subnet.distributed_worker import complete_checkpoint_cache
        import sys
        before=set(sys.modules)
        self.assertTrue(complete_checkpoint_cache(self.root,self.files))
        self.assertEqual(set(sys.modules)-before,set())
    def test_partial_wrong_bytes_extra_entries_are_not_reusable(self):
        from subnet.distributed_worker import complete_checkpoint_cache
        self.assertFalse(complete_checkpoint_cache(self.root/'absent',self.files))
        (self.root/'model.safetensors').unlink();self.assertFalse(complete_checkpoint_cache(self.root,self.files))
        (self.root/'model.safetensors').write_bytes(b'wrong');self.assertFalse(complete_checkpoint_cache(self.root,self.files))
        (self.root/'model.safetensors').write_bytes(b'weights');extra=self.root/'model.safetensors.partial';extra.write_bytes(b'partial');self.assertFalse(complete_checkpoint_cache(self.root,self.files));extra.unlink()
        (self.root/'extra-directory').mkdir();self.assertFalse(complete_checkpoint_cache(self.root,self.files))
    def test_file_root_and_parent_symlinks_are_not_reusable(self):
        from subnet.distributed_worker import complete_checkpoint_cache
        target=Path(self.folder.name)/'outside';target.write_bytes(b'weights');weight=self.root/'model.safetensors';weight.unlink();weight.symlink_to(target)
        self.assertFalse(complete_checkpoint_cache(self.root,self.files));weight.unlink();weight.write_bytes(b'weights')
        alias=Path(self.folder.name)/'alias';alias.symlink_to(self.root,target_is_directory=True);self.assertFalse(complete_checkpoint_cache(alias,self.files));self.assertFalse(complete_checkpoint_cache(alias/'.',self.files))
    def test_storage_errors_propagate_and_invalid_inventory_is_not_a_miss(self):
        from subnet.distributed_worker import complete_checkpoint_cache
        for error in (OSError(101,'network unavailable'),PermissionError('unreadable')):
            with patch.object(Path,'open',side_effect=error),self.assertRaises(type(error)):complete_checkpoint_cache(self.root,self.files)
        with patch.object(Path,'iterdir',side_effect=OSError('inventory unavailable')),self.assertRaises(OSError):complete_checkpoint_cache(self.root,self.files)
        with self.assertRaises(ValueError):complete_checkpoint_cache(self.root,{'../private':'a'*64})

class RetryCacheSelection(WorkerTests):
    def test_invalid_retry_cache_uses_normal_signed_hydration(self):
        for kind in ('partial','extra','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as workspace:
                worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.authority,workspace)
                cache=Path(workspace)/'backend'/'checkpoints'/'CP';cache.mkdir(parents=True);(cache/'config.json').write_bytes(b'{}')
                if kind!='partial':(cache/'model.safetensors').write_bytes(b'weights')
                if kind=='extra':(cache/'extra').write_text('unexpected')
                if kind=='symlink':
                    (cache/'model.safetensors').unlink();target=Path(workspace)/'outside';target.write_bytes(b'weights');(cache/'model.safetensors').symlink_to(target)
                self.claim['attempt']=2;worker.request=Mock(side_effect=[{'claim':self.claim},{'status':'failed'}])
                with patch('subnet.distributed_worker.subprocess.run',return_value=SimpleNamespace(returncode=1)) as run:worker.once()
                self.assertNotIn('--checkpoint-cache',run.call_args.args[0]);self.assertTrue(cache.exists())

    def test_explicit_complete_cache_reuses_and_incomplete_mapping_hydrates(self):
        for complete in (True,False):
            with self.subTest(complete=complete),tempfile.TemporaryDirectory() as workspace:
                cache=Path(workspace)/'approved';cache.mkdir();(cache/'config.json').write_bytes(b'{}')
                if complete:(cache/'model.safetensors').write_bytes(b'weights')
                worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.authority,workspace,checkpoint_caches={'CP':cache});worker.request=Mock(side_effect=[{'claim':self.claim},{'status':'failed'}])
                with patch('subnet.distributed_worker.subprocess.run',return_value=SimpleNamespace(returncode=1)) as run:worker.once()
                self.assertEqual('--checkpoint-cache' in run.call_args.args[0],complete)

class CheckpointCandidateTests(CheckpointCacheTests):
    def test_candidate_selection_never_reads_weight_bytes(self):
        from subnet.distributed_worker import checkpoint_cache_candidate
        with patch.object(Path,'open',side_effect=AssertionError('weight read')):
            self.assertTrue(checkpoint_cache_candidate(self.root,self.files))

    def test_corruption_is_rejected_by_authoritative_backend_before_runtime(self):
        from subnet.distributed_worker import checkpoint_cache_candidate
        from subnet.backend_jobs import checkpoint
        (self.root/'model.safetensors').write_bytes(b'wrong')
        self.assertTrue(checkpoint_cache_candidate(self.root,self.files))
        manifest={'checkpoint':{'id':'candidate','files':self.files}}
        with self.assertRaisesRegex(ValueError,'approved cached checkpoint mismatch'):
            checkpoint(manifest,Path(self.folder.name)/'backend',cache=self.root)

    def test_candidate_inventory_rejects_symlink_extra_or_missing_files(self):
        from subnet.distributed_worker import checkpoint_cache_candidate
        weight=self.root/'model.safetensors';weight.unlink()
        self.assertFalse(checkpoint_cache_candidate(self.root,self.files))
        target=Path(self.folder.name)/'outside';target.write_bytes(b'weights');weight.symlink_to(target)
        self.assertFalse(checkpoint_cache_candidate(self.root,self.files));weight.unlink();weight.write_bytes(b'weights')
        (self.root/'extra').write_bytes(b'x')
        self.assertFalse(checkpoint_cache_candidate(self.root,self.files))

if __name__=='__main__':unittest.main()
