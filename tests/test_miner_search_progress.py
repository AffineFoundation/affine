"""CPU controls for persistent local search; no real keys, GPU, network or data."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from subnet import forced_sampling as forced
from subnet.cli import validate_search_budget
from subnet.miner import Miner, EpochClosed
from subnet.miner_search_state import SearchState, NoncesExhausted, digest
from subnet.storage import canonical
import test_fast_prefill_audit as fixture


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.manifest = fixture.Controls().support_runtime()
        self.manifest.update(K=4, L=4, max_batches=3, deadline=time.time()+3600,
                             source_bundle={'sha256': '9'*64},
                             environment={'id': 'math', 'version': 'v1'},
                             indices=list(range(8)), harness={'version': 'text-tools-v1', 'policy': 'autoregressive', 'max_output_tokens': 4, 'temperature': .7, 'top_p': 1.})
        self.manifest['sampling_contract'].update(version=forced.MINER_VERSION, max_attempts=1000)
        self.identity = SimpleNamespace(id='d'*64)
        self.context = forced.binding(self.manifest, self.identity.id)
        self.state = self.root/'complete.zip'
        self.calls = []
        self.runtime = SimpleNamespace(spec=SimpleNamespace(version='v1'))
        self.runtime.for_environment = lambda *args: self.runtime
        self.runtime.rollout = self.generate
        self.kind = lambda n: 'positive' if n < 32 else 'negative'
        self.make = self.enterContext(patch('subnet.miner.make_runtime', return_value=self.runtime))
        self.enterContext(patch('subnet.miner.check_runtime_profile'))

    def roll(self, index, nonce, kind=None, output=None):
        kind = self.kind(nonce) if kind is None else kind
        return dict(env_id='math', environment_version='v1', index=index, sample_index=index,
                    seed=nonce, sampling=forced.receipt(self.context, nonce), task_hash='c'*64,
                    classification=kind, reward=1 if kind=='positive' else 0 if kind=='negative' else .5,
                    turns=[dict(prompt=[1,2], output=output or [nonce+10,3])])

    def generate(self, index, nonce):
        self.calls.append(nonce)
        return self.roll(index, nonce), [np.array([[float(nonce)],[-.1]], dtype=np.float32)]

    def miner(self, **kwargs):
        result = Miner(self.identity, self.manifest, 'unused', capability={'put_url':'not-network'}, state_path=self.state, **kwargs)
        self.addCleanup(result.search_state.close)
        return result

    def unfinished(self, miner, budget=32, index=0, seed=0):
        with self.assertRaisesRegex(RuntimeError, 'search budget exhausted'):
            miner.search(index, seed=seed, max_attempts=budget, env_id='math')

    def test_repeated_budget32_preserves_successes_and_uses_fresh_nonces(self):
        miner=self.miner();self.unfinished(miner)
        self.assertEqual(miner.search_state.load('math',0)[0],32)
        self.assertEqual(len(miner.search_state.load('math',0)[2]),4)
        batch=miner.search(0,max_attempts=32)
        self.assertEqual(self.calls,list(range(36)))
        self.assertEqual([r['seed'] for r in batch['rollouts']],[0,1,2,3,32,33,34,35])
        self.assertEqual(len(miner.batches),1)

    def test_restart_restores_partial_proofs_exactly(self):
        miner=self.miner();self.unfinished(miner)
        restored=self.miner();batch=restored.search(0,max_attempts=32)
        self.assertEqual([r['seed'] for r in batch['rollouts']],[0,1,2,3,32,33,34,35])
        np.testing.assert_array_equal(restored.batches[0][1][2][0],np.array([[2.],[-.1]],dtype=np.float32))
        self.assertEqual(self.calls,list(range(36)))

    def test_one_attempt_budget_accumulates_across_eight_calls(self):
        self.kind=lambda n:'positive' if n%2==0 else 'negative'
        miner=self.miner()
        for _ in range(7):self.unfinished(miner,budget=1)
        batch=miner.search(0,max_attempts=1)
        self.assertEqual(set(r['seed']for r in batch['rollouts']),set(range(8)))

    def test_neutral_attempts_consume_nonces_but_not_quota(self):
        self.kind=lambda n:'neutral'
        miner=self.miner();self.unfinished(miner,budget=32)
        self.assertEqual(miner.search_state.load('math',0)[0],32)
        self.assertEqual(miner.search_state.load('math',0)[2],[])
        self.assertEqual(miner.batches,[])

    def test_interrupted_generation_consumes_reserved_nonce(self):
        miner=self.miner()
        with patch.object(self.runtime,'rollout',side_effect=RuntimeError('GPU interrupted')):
            with self.assertRaisesRegex(RuntimeError,'GPU interrupted'):miner.search(0,max_attempts=1)
        restored=self.miner();self.unfinished(restored,budget=1)
        self.assertEqual(self.calls,[1])

    def test_native_task_error_does_not_reuse_nonce_or_count_negative(self):
        from verifiers.v1.errors import TaskError
        miner=self.miner()
        with patch.object(self.runtime,'rollout',side_effect=TaskError('indeterminate')):self.unfinished(miner,budget=1)
        self.assertEqual(miner.search_state.load('math',0)[0],1)
        self.assertEqual(miner.search_state.load('math',0)[2],[])

    def test_explicit_seed_never_rewinds_saved_cursor(self):
        miner=self.miner();self.unfinished(miner,budget=1,seed=100)
        self.unfinished(miner,budget=1,seed=0)
        self.assertEqual(self.calls,[100,101])

    def test_nonce999_valid_then_exhausted_without_1000_generation(self):
        miner=self.miner();self.unfinished(miner,budget=1,seed=999)
        with self.assertRaises(NoncesExhausted):miner.search(0,max_attempts=32)
        restored=self.miner()
        with self.assertRaises(NoncesExhausted):restored.search(0,max_attempts=32)
        self.assertEqual(self.calls,[999])

    def test_all_thousand_neutral_attempts_are_unique(self):
        self.kind=lambda n:'neutral';miner=self.miner()
        for _ in range(31):self.unfinished(miner,budget=32)
        with self.assertRaises(NoncesExhausted):miner.search(0,max_attempts=32)
        self.assertEqual(self.calls,list(range(1000)))

    def test_duplicate_output_new_nonce_cannot_fill_extra_slots(self):
        miner=self.miner()
        def copied(i,n):
            self.calls.append(n)
            return self.roll(i,n,output=[10,3]),[np.zeros((2,1),dtype=np.float32)]
        with patch.object(self.runtime,'rollout',side_effect=copied):self.unfinished(miner,budget=8)
        self.assertEqual(len(miner.search_state.load('math',0)[2]),1)
        self.assertEqual(miner.search_state.load('math',0)[0],8)

    def test_previous_nonce_receipt_cannot_be_relabelled_as_new_attempt(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        with patch.object(self.runtime,'rollout',return_value=(self.roll(0,0),[np.zeros((2,1),dtype=np.float32)])):
            with self.assertRaisesRegex(ValueError,'reserved nonce'):miner.search(0,max_attempts=1)
        self.assertEqual(miner.search_state.load('math',0)[0],2)

    def test_complete_group_recovers_before_upload_without_generation(self):
        self.kind=lambda n:'positive' if n<4 else 'negative'
        miner=self.miner();miner.search(0,max_attempts=8)
        restored=self.miner()
        with patch.object(self.runtime,'rollout',side_effect=AssertionError('must not regenerate')):
            restored.search(0,max_attempts=1)
        self.assertEqual(len(restored.batches),1)

    def test_same_complete_task_search_does_not_append_duplicate_batch(self):
        self.kind=lambda n:'positive' if n<4 else 'negative'
        miner=self.miner();first=miner.search(0,max_attempts=8)
        self.assertIs(miner.search(0,max_attempts=8),first)
        self.assertEqual(len(miner.batches),1)
        self.assertEqual(len(self.calls),8)

    def test_upload_failure_still_preserves_durable_complete_batch_and_retires_partial(self):
        self.kind=lambda n:'positive' if n<4 else 'negative'
        miner=self.miner();miner.search(0,max_attempts=8)
        with patch('subnet.miner.requests.put',side_effect=RuntimeError('transport')):
            with self.assertRaisesRegex(RuntimeError,'transport'):miner.upload()
        self.assertTrue(self.state.exists())
        self.assertEqual(miner.search_state.load('math',0)[1:],(True,[],[]))
        restored=self.miner()
        self.assertEqual(len(restored.batches),1)

    def test_missing_complete_state_does_not_silently_restart_completed_task(self):
        self.kind=lambda n:'positive' if n<4 else 'negative'
        miner=self.miner();miner.search(0,max_attempts=8)
        with patch('subnet.miner.requests.put',return_value=SimpleNamespace(status_code=200,raise_for_status=lambda:None)):miner.upload()
        self.state.unlink()
        restored=self.miner()
        with self.assertRaisesRegex(ValueError,'missing durable batch'):restored.search(0,max_attempts=1)

    def test_binding_changes_refuse_restore_without_mutating_original(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        original=copy.deepcopy(self.manifest)
        for change in ({'epoch':'other'},{'checkpoint':{'id':'e'*64}},{'source_bundle':{'sha256':'e'*64}}, {'deadline':self.manifest['deadline']+1}):
            with self.subTest(change=change),patch.dict(self.manifest,change):
                with self.assertRaisesRegex(ValueError,'stale local search binding'):self.miner()
        self.assertEqual(miner.search_state.load('math',0)[0],1)
        self.assertEqual(self.manifest,original)

    def test_other_identity_cannot_restore_same_cache(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        with patch.object(self.identity,'id','e'*64):
            with self.assertRaisesRegex(ValueError,'stale local search binding'):self.miner()

    def test_authenticated_epoch_handover_retires_all_partials_and_old_instance(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        self.manifest['epoch']='next-epoch'
        self.manifest['deadline'] += 3600
        newer=self.miner(retire_previous_search=True)
        self.assertEqual(newer.search_state.load('math',0),(0,False,[],[]))
        with self.assertRaisesRegex(ValueError,'changed local search binding'):miner.search_state.load('math',0)
        self.context=forced.binding(self.manifest,self.identity.id)
        self.unfinished(newer,budget=1)
        self.assertEqual(self.calls,[0,0])

    def test_older_epoch_cannot_reset_nonce_ledger(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        self.manifest['epoch']='earlier-epoch';self.manifest['deadline']-=60
        with self.assertRaisesRegex(ValueError,'stale local search binding'):self.miner(retire_previous_search=True)

    def test_partial_bytes_capped_without_resetting_cursor(self):
        miner=self.miner()
        with patch('subnet.miner_search_state.MAX_PARTIAL_BYTES',10):
            with self.assertRaisesRegex(ValueError,'local cache bound'):miner.search(0,max_attempts=1)
        self.assertEqual(miner.search_state.load('math',0),(1,False,[],[]))

    def test_task_namespace_and_harness_cannot_change_on_restore(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        for changed in ({'indices':[1,2]}, {'harness':dict(self.manifest['harness'],temperature=.8)}):
            with patch.dict(self.manifest,changed):
                with self.assertRaisesRegex(ValueError,'stale local search binding'):self.miner()

    def test_same_epoch_checkpoint_change_cannot_use_retirement_override(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        self.manifest['checkpoint']['id']='e'*64
        with self.assertRaisesRegex(ValueError,'stale local search binding'):self.miner(retire_previous_search=True)

    def test_corrupt_cursor_or_proof_digest_fails_closed(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        db=miner.search_state.db
        with db:db.execute('UPDATE task SET next=0')
        with self.assertRaisesRegex(ValueError,'row digest'):self.miner()

    def test_corrupt_proof_bytes_fail_before_decode(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        db=miner.search_state.db
        with db:db.execute('UPDATE task SET proof=?',(b'garbled',))
        with patch('subnet.miner_search_state.unpack',side_effect=AssertionError('no decode')):
            with self.assertRaisesRegex(ValueError,'row digest'):self.miner()

    def test_bad_scope_digest_fails_closed(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        with miner.search_state.db:miner.search_state.db.execute('UPDATE scope SET sha=?',('0'*64,))
        with self.assertRaisesRegex(ValueError,'scope digest'):self.miner()

    def test_partial_proof_wrong_task_or_receipt_fails_even_with_new_local_hash(self):
        from subnet.batches import pack
        miner=self.miner();self.unfinished(miner,budget=1)
        journal=miner.search_state
        roll=self.roll(1,0)
        body=pack([(journal._batch('math',0,[roll]),[[np.zeros((2,1),dtype=np.float32)]])],stable=True)
        key=journal._key('math',0)
        with journal.db:journal._write(key,1,0,body)
        with self.assertRaisesRegex(ValueError,'same-task binding'):journal.load('math',0)

    def test_partial_LRU_eviction_preserves_nonce_cursors(self):
        miner=self.miner()
        with patch('subnet.miner_search_state.MAX_PARTIAL_TASKS',2):
            for index in range(3):self.unfinished(miner,budget=1,index=index)
            self.assertEqual(miner.search_state.load('math',0),(1,False,[],[]))
            self.assertEqual(len(miner.search_state.load('math',2)[2]),1)
            self.unfinished(miner,budget=1,index=0)
        self.assertEqual(self.calls,[0,0,0,1])

    def test_partial_write_failure_keeps_cursor_and_prior_group(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        with patch('subnet.miner_search_state.pack',side_effect=RuntimeError('disk simulation')):
            with self.assertRaisesRegex(RuntimeError,'disk simulation'):miner.search(0,max_attempts=1)
        nonce,_,rolls,_=miner.search_state.load('math',0)
        self.assertEqual(nonce,2);self.assertEqual([r['seed']for r in rolls],[0])

    def test_other_local_client_cannot_search_or_reset_while_locked(self):
        miner=self.miner();other=self.miner()
        with miner.search_state.locked():
            with self.assertRaisesRegex(ValueError,'already in use'):other.search(0,max_attempts=1)
            with self.assertRaisesRegex(ValueError,'already in use'):self.miner()
        self.assertEqual(self.calls,[])

    def test_local_search_lock_excludes_a_separate_process(self):
        import subprocess,sys
        miner=self.miner()
        script="import os,fcntl,sys;fd=os.open(sys.argv[1],os.O_RDWR);\ntry: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nsys.exit(1)"
        with miner.search_state.locked():
            result=subprocess.run([sys.executable,'-B','-c',script,str(miner.search_state.path)+'.lock'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_symlink_cache_cannot_be_followed(self):
        self.root.joinpath('complete.search.sqlite3').symlink_to(self.root/'outside')
        with self.assertRaises(OSError):self.miner()
        self.assertFalse((self.root/'outside').exists())

    def test_expired_epoch_cannot_generate(self):
        self.manifest['deadline']=time.time()-1
        miner=self.miner()
        with self.assertRaises(EpochClosed):miner.search(0,max_attempts=1)
        self.assertEqual(self.calls,[])

    def test_invalid_search_parameters_rejected(self):
        miner=self.miner()
        for seed in (True,-1,1000,1.5):
            with self.subTest(seed=seed),self.assertRaises(ValueError):miner.search(0,seed=seed)
        for budget in (True,0,-1,1.5):
            with self.subTest(budget=budget),self.assertRaises(ValueError):miner.search(0,max_attempts=budget)
        self.assertEqual(self.calls,[])

    def cli_args(self, budget=32):
        self.manifest['capabilities']={self.identity.id:{}}
        self.identity.decrypt=lambda value:{'put_url':'not-network'}
        return SimpleNamespace(search_budget=budget,env_id='math',indices=[0],cap_file=None,key='mock',
            state=str(self.root/'cli'),manifest_url='https://manifest.invalid',current_url=None,
            gateway='https://unused.invalid',authority='trusted',max_batches=None,once=True,source_bundle_sha256='9'*64)

    def test_actual_CLI_restart_completes_partial_group_and_only_uploads_complete(self):
        from subnet import cli
        args=self.cli_args()
        response=SimpleNamespace(status_code=200,raise_for_status=lambda:None)
        with patch.object(cli,'fetch_signed',return_value=self.manifest),patch.object(cli,'identity',return_value=self.identity),patch.object(cli,'check_runtime_profile'),patch.object(cli,'checkpoint_download',return_value='unused'),patch('subnet.miner.requests.put',return_value=response)as upload:
            cli.run(args)
            upload.assert_not_called()
            cli.run(args)
            self.assertEqual(upload.call_count,1)
        self.assertEqual(self.calls,list(range(36)))
        from subnet.batches import unpack
        rows=unpack(upload.call_args.kwargs['data'])
        self.assertEqual(len(rows),1);self.assertEqual(len(rows[0][0]['rollouts']),8)

    def test_actual_CLI_accepts_budget129_and_rejects1001_before_download(self):
        from subnet import cli
        args=self.cli_args(129);self.kind=lambda n:'neutral'
        with patch.object(cli,'fetch_signed',return_value=self.manifest),patch.object(cli,'identity',return_value=self.identity),patch.object(cli,'check_runtime_profile'),patch.object(cli,'checkpoint_download',return_value='unused')as download,patch('subnet.miner.requests.put')as upload:
            cli.run(args);self.assertEqual(self.calls,list(range(129)));upload.assert_not_called()
            args.search_budget=1001;download.reset_mock()
            with self.assertRaisesRegex(ValueError,'authorized limit'):cli.run(args)
            download.assert_not_called()

    def test_CLI_closes_journal_on_exit_and_error(self):
        from subnet import cli
        args=self.cli_args(1)
        made=[]
        def construct(*a,**k):
            instance=Miner(*a,**k);made.append((instance,instance.search_state));return instance
        with patch.object(cli,'fetch_signed',return_value=self.manifest),patch.object(cli,'identity',return_value=self.identity),patch.object(cli,'check_runtime_profile'),patch.object(cli,'checkpoint_download',return_value='unused'),patch.object(cli,'Miner',side_effect=construct):
            cli.run(args)
            self.assertIsNone(made[-1][1].db);self.assertIsNone(made[-1][1].lock_fd)
            with patch.object(self.runtime,'rollout',side_effect=ValueError('bad runtime')):
                with self.assertRaisesRegex(ValueError,'bad runtime'):cli.run(args)
            self.assertIsNone(made[-1][1].db);self.assertIsNone(made[-1][1].lock_fd)

    def test_CLI_handover_closes_old_journal_and_retires_only_local_partials(self):
        from subnet import cli
        args=self.cli_args(1);args.once=False
        next_manifest=copy.deepcopy(self.manifest);next_manifest['epoch']='next-epoch';next_manifest['deadline']+=3600
        made=[]
        def construct(*a,**k):
            instance=Miner(*a,**k);made.append((instance,instance.search_state));return instance
        def runtime(checkpoint,manifest,*a,**k):
            self.context=forced.binding(manifest,self.identity.id);return self.runtime
        with patch.object(cli,'fetch_signed',side_effect=[self.manifest,next_manifest]),patch.object(cli,'identity',return_value=self.identity),patch.object(cli,'check_runtime_profile'),patch.object(cli,'checkpoint_download',return_value='unused'),patch.object(cli,'Miner',side_effect=construct),patch('subnet.miner.make_runtime',side_effect=runtime),patch.object(cli.time,'sleep',side_effect=[None,RuntimeError('test stop')]),patch('subnet.miner.requests.put')as upload:
            with self.assertRaisesRegex(RuntimeError,'test stop'):cli.run(args)
            upload.assert_not_called()
        self.assertEqual(self.calls,[0,0]);self.assertEqual(len(made),2)
        for miner,journal in made:self.assertIsNone(journal.db);self.assertIsNone(journal.lock_fd)

    def test_epoch_retirement_reclaims_partial_disk_pages_automatically(self):
        miner=self.miner()
        tensor=np.random.default_rng(0).normal(size=(2,32768)).astype(np.float32)
        with patch.object(self.runtime,'rollout',return_value=(self.roll(0,0),[tensor])):self.unfinished(miner,budget=1)
        before=miner.search_state.path.stat().st_size
        self.manifest['epoch']='new-epoch';self.manifest['deadline']+=3600
        newer=self.miner(retire_previous_search=True)
        self.assertLess(newer.search_state.path.stat().st_size,before)
        self.assertEqual(newer.search_state.db.execute('SELECT COUNT(*) FROM task').fetchone()[0],0)

    def test_constructor_validation_failure_closes_SQLite_and_lock(self):
        miner=self.miner();self.unfinished(miner,budget=1)
        import os
        before=len(os.listdir('/proc/self/fd'))
        for _ in range(5):
            with patch.dict(self.manifest,{'deadline':self.manifest['deadline']+1}):
                with self.assertRaisesRegex(ValueError,'stale local search binding'):self.miner()
        self.assertEqual(len(os.listdir('/proc/self/fd')),before)

    def test_explicit_close_is_idempotent_and_cannot_restart_legacy_search(self):
        miner=self.miner();journal=miner.search_state
        miner.close();miner.close()
        self.assertIsNone(journal.db);self.assertIsNone(journal.lock_fd)
        with self.assertRaisesRegex(ValueError,'miner is closed'):miner.search(0,max_attempts=1)

    def test_internal_reservation_rejects_invalid_nonce_start(self):
        miner=self.miner()
        for start in (-1,True,1.5,1000):
            with self.subTest(start=start),self.assertRaises(ValueError):miner.search_state.reserve('math',0,start)
        self.assertEqual(miner.search_state.load('math',0)[0],0)

    def test_CLI_budget_follows_v5_manifest_but_keeps_legacy_behavior(self):
        for value in (1,32,50,128,129,1000):self.assertEqual(validate_search_budget(value,self.manifest),value)
        for value in (True,0,-1,1.5,'32',1001):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_search_budget(value,self.manifest)
        for old in ({},{'sampling_contract':{'version':forced.VERSION,'max_attempts':16}}):
            self.assertEqual(validate_search_budget(128,old),128)
            with self.assertRaises(ValueError):validate_search_budget(129,old)

if __name__=='__main__':unittest.main()
