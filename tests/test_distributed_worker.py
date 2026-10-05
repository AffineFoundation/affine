import base64
import json
import hashlib
import tempfile
import time
import unittest
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.distributed_worker import Worker, ExpiredCompletedLease
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

class AutomaticLifetimeWorkerTests(WorkerTests):
    def test_lease_lost_during_backend_only_returns_after_original_backend_exits(self):
        targets=[];finished=[]
        stopped=Mock();stopped.wait.return_value=False;lost=threading.Event()
        def thread(target,daemon):
            targets.append(target)
            return SimpleNamespace(start=lambda:None,join=lambda **kwargs:None)
        def run(args,**kwargs):
            targets[0]()
            finished.append(True)
            return SimpleNamespace(returncode=0)
        self.claim['lease_until']=time.time()-1
        self.worker.request=Mock(side_effect=[{'claim':self.claim},ValueError('renewal refused')])
        with patch('subnet.distributed_worker.threading.Event',side_effect=[stopped,lost]),patch('subnet.distributed_worker.threading.Thread',side_effect=thread),patch('subnet.distributed_worker.subprocess.run',side_effect=run):
            with self.assertRaises(ExpiredCompletedLease):self.worker.once()
        self.assertEqual(finished,[True]);self.assertEqual(self.worker.request.call_count,2)
        diagnostic=json.loads((Path(self.folder.name)/'verify-job'/'attempt-1'/'expired-completed-lease.json').read_text())
        self.assertEqual(diagnostic['stage'],'backend_completed')
        self.assertTrue(diagnostic['backend_terminal']);self.assertEqual(diagnostic['backend_exit'],0)
        self.assertFalse(diagnostic['report_acknowledged']);self.assertNotIn('token',diagnostic)

    def test_successful_ack_retires_inputs_keeps_reports(self):
        def run(args,**kwargs):
            workspace=Path(args[args.index('--workspace')+1]);out=workspace/'jobs'/'verify-job';out.mkdir(parents=True)
            (out/'report.json').write_text('{}');(out/'submission-0.zip').write_bytes(b'x')
            cp=workspace/'checkpoints'/'CP';cp.mkdir(parents=True);(cp/'config.json').write_bytes(b'{}');(cp/'model.safetensors').write_bytes(b'weights')
            self.assertTrue(kwargs['pass_fds']);return SimpleNamespace(returncode=0)
        self.job['submissions']=[{'sha256':hashlib.sha256(b'x').hexdigest()}]
        self.claim['job']['signature']=base64.b64encode(self.key.sign(canonical(self.job)).signature).decode();self.claim['job_sha256']=digest(self.job)
        self.worker.request=Mock(side_effect=[{'claim':self.claim},{'accepted':True}])
        with patch('subnet.distributed_worker.subprocess.run',side_effect=run):self.worker.once()
        out=Path(self.folder.name)/'backend'/'jobs'/'verify-job'
        self.assertFalse((out/'submission-0.zip').exists());self.assertTrue((out/'report.json').exists())
    def test_missing_ack_preserves_submission_and_pending_report(self):
        def run(args,**kwargs):
            workspace=Path(args[args.index('--workspace')+1]);out=workspace/'jobs'/'verify-job';out.mkdir(parents=True)
            (out/'report.json').write_text('{}');(out/'submission-0.zip').write_bytes(b'x');return SimpleNamespace(returncode=0)
        self.worker.request=Mock(side_effect=[{'claim':self.claim},ValueError('no ACK')]);self.claim['lease_until']=time.time()-1
        with patch('subnet.distributed_worker.subprocess.run',side_effect=run),self.assertRaises(ValueError):self.worker.once()
        self.assertTrue((Path(self.folder.name)/'backend'/'jobs'/'verify-job'/'submission-0.zip').exists())
        self.assertTrue((Path(self.folder.name)/'verify-job'/'attempt-1'/'pending-report.json').exists())
        diagnostic=json.loads((Path(self.folder.name)/'verify-job'/'attempt-1'/'expired-completed-lease.json').read_text())
        self.assertEqual(diagnostic['stage'],'report_acknowledgment')
        self.assertTrue(diagnostic['backend_terminal'])
        self.assertFalse(diagnostic['report_acknowledged'])

class RootSourceRegistryTests(unittest.TestCase):
    def test_old_and_new_archives_route_exact_installed_code(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'old';b=Path(folder)/'new';a.mkdir();b.mkdir()
            (a/'backend.py').write_bytes(b'old');(b/'backend.py').write_bytes(b'new')
            files_old={'backend.py':hashlib.sha256(b'old').hexdigest()};files_new={'backend.py':hashlib.sha256(b'new').hexdigest()}
            registry={'6a':{'path':str(a),'source_files':files_old},'new':{'path':str(b),'source_files':files_new}}
            worker=Worker('http://localhost:1',bytes(SigningKey.generate()),'a'*64,folder,backend_source=str(a),source_registry=registry)
            def job(bundle,files):return {'manifest':{'payload':{'source_bundle':{'sha256':bundle,'path':'/miner/arbitrary'}}},'source_files':files}
            self.assertEqual(worker.source_for_job(job('6a',files_old)),str(a));self.assertEqual(worker.source_for_job(job('new',files_new)),str(b))
            with self.assertRaisesRegex(ValueError,'not in ROOT'):worker.source_for_job(job('unknown',files_new))
            with self.assertRaisesRegex(ValueError,'differs'):worker.source_for_job(job('6a',files_new))
            # Subsequent routing adds no new inventory hashing; backend still
            # performs its original per-job inventory validation independently.
            with patch.object(Path,'open',side_effect=AssertionError('extra source hash')):
                self.assertEqual(worker.source_for_job(job('6a',files_old)),str(a))
    def test_unpinned_fallback_or_modified_staged_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            worker=Worker('http://localhost:1',bytes(SigningKey.generate()),'a'*64,folder,backend_source=folder)
            with self.assertRaisesRegex(ValueError,'explicit pinned'):worker.source_for_job({})
            p=Path(folder)/'backend.py';p.write_bytes(b'changed');files={'backend.py':hashlib.sha256(b'old').hexdigest()}
            worker.source_registry={'6a':{'path':folder,'source_files':files}}
            with self.assertRaisesRegex(ValueError,'source changed'):worker.source_for_job({'manifest':{'payload':{'source_bundle':{'sha256':'6a'}}},'source_files':files})

class ReceiptInputPathTests(WorkerTests):
    def test_external_verified_cache_does_not_attest_same_id_unverified_owned_copy(self):
        root=Path(self.folder.name);external=root/'external';external.mkdir()
        (external/'config.json').write_bytes(b'{}');(external/'model.safetensors').write_bytes(b'weights')
        owned=root/'backend'/'checkpoints'/'CP';owned.mkdir(parents=True)
        (owned/'config.json').write_bytes(b'{}');(owned/'model.safetensors').write_bytes(b'unverified different bytes')
        self.worker.checkpoint_caches={'CP':external}
        self.worker.request=Mock(side_effect=[{'claim':self.claim},{'accepted':True}])
        def run(args,**kwargs):
            self.assertEqual(args[args.index('--checkpoint-cache')+1],str(external))
            out=root/'backend'/'jobs'/'verify-job';out.mkdir(parents=True);(out/'report.json').write_text('{}')
            return SimpleNamespace(returncode=0)
        with patch('subnet.distributed_worker.subprocess.run',side_effect=run):self.worker.once()
        self.assertFalse((root/'backend'/'.cache-lifecycle'/'CP.json').exists())
        self.assertEqual((owned/'model.safetensors').read_bytes(),b'unverified different bytes')
        self.assertTrue(external.exists())
