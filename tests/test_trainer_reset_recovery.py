import copy
import hashlib
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ops/trainer_lifecycle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_trainer_reset_lifecycle import ResetTests
import trainer_reset_recovery as retry
import trainer_reset_lifecycle as reset
from subnet.optimizer_state_cache import StateCache, sha
from subnet import optimizer_state_cache as optimizer, trainer_cache_lifecycle as retention


class RetryTests(ResetTests):
    """State/signature/lineage controls; full scientific job validation is separate."""
    def setUp(self):
        super().setUp(); self.execute()
        old, self.new_manifest = self.fresh()
        old.update(created_at=150, expires_at=900, steps=1,
            persistent_training={'output_namespace':'private/new','binding_sha256':'a'*64},
            source_files={'x':'b'*64}, submissions=[{'unchanged':'input'}])
        self.failed = old
        self.retry_job = dict(copy.deepcopy(old), job_id='retry', created_at=250, expires_at=950,
            persistent_training={'output_namespace':'private/retry','binding_sha256':'a'*64})
        self.write(self.root/'new.json',self.sign(self.failed));self.write(self.root/'retry.json',self.sign(self.retry_job))
        self.status={'job_id':'new','phase':'failed','exit_code':1,'actual_wait':True}
        self.write(self.root/'runner-status/new.json',self.status)
        self.log=self.root/'new-worker.log'; self.log.write_bytes(b'cache_budget=local_cache.admit(\nValueError: '+retry.FAILURE.encode()+b'\n');self.log.chmod(0o600)
        self.grant=dict(version=retry.VERSION, original_job=self.sign(self.failed),new_job=self.sign(self.retry_job),
            original_status=self.status,original_worker_log_sha256=hashlib.sha256(self.log.read_bytes()).hexdigest(),
            reset_envelope_sha256=sha(self.envelope),created_at=200,expires_at=time.time()+100)
        self.grant['created_at']=time.time()-100
        self.signed_grant=self.sign(self.grant)
        self.optimizer.joinpath('current.json').unlink()
        # Fixture has deliberately tiny parameters. The separately run integration
        # invokes these unpatched validators on actual signed production jobs.
        a=patch('subnet.persistent_training_protocol.validate_job',return_value=(self.new_manifest['trainer_state_binding'],None))
        b=patch('subnet.unaudited_training_execution.validate',return_value={'same':'science','learning_rate_authorization':self.sign({'effective_learning_rate':5e-7})})
        c=patch('subnet.trainer_cache_lifecycle.live_original',return_value=False)
        for q in (a,b,c):q.start();self.addCleanup(q.stop)

    def check(self):return retry.validate(self.signed_grant,self.envelope,self.authority,self.root)

    def resign(self):self.signed_grant=self.sign(self.grant)

    def test_valid_retry_prepares_original_zero_state_without_reset(self):
        before=reset._read(self.lifecycle/'trainer-current-state.json')
        reset.install_for_train(self.envelope,self.authority,self.root,recovery_envelope=self.signed_grant)
        with StateCache(self.root,self.retry_job,self.new_manifest,self.authority) as cache:
            self.assertEqual(cache.prepare_parent(None,'4'*64),0)
        self.assertEqual(reset._read(self.lifecycle/'trainer-current-state.json'),before)
        self.assertFalse((self.optimizer/'current.json').exists())

    def test_changed_input_manifest_or_source_rejected(self):
        for name,value in [('submissions',[{'changed':'input'}]),('source_files',{'x':'c'*64}),('steps',2)]:
            new=dict(self.retry_job,**{name:value});self.grant['new_job']=self.sign(new);self.resign()
            self.write(self.root/'retry.json',self.grant['new_job'])
            with self.assertRaisesRegex(ValueError,'scientific'):self.check()

    def test_wrong_original_job_and_wrong_reset_rejected(self):
        self.grant['reset_envelope_sha256']='0'*64;self.resign()
        with self.assertRaisesRegex(ValueError,'exact original reset'):self.check()

    def test_modified_failure_evidence_rejected(self):
        self.log.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'failure evidence'):self.check()

    def test_nonterminal_failed_child_rejected(self):
        self.status['phase']='running';self.write(self.root/'runner-status/new.json',self.status);self.resign()
        with self.assertRaisesRegex(ValueError,'terminal'):self.check()

    def test_original_report_precludes_retry(self):
        self.write(self.root/'jobs/new/report.json',{'success':True})
        with self.assertRaisesRegex(ValueError,'failure evidence'):self.check()

    def test_pending_current_or_candidate_precludes_retry(self):
        for name in ('pending.json','current.json'):
            self.write(self.optimizer/name,{})
            with StateCache(self.root,self.retry_job,self.new_manifest,self.authority) as cache:
                with self.assertRaisesRegex(ValueError,'existing optimizer'):retry.require_zero_state(cache,self.envelope,self.failed,self.retry_job)
            self.optimizer.joinpath(name).unlink()
        with StateCache(self.root,self.retry_job,self.new_manifest,self.authority) as cache:
            cache.directory('retry').mkdir()
            with self.assertRaisesRegex(ValueError,'candidate'):retry.require_zero_state(cache,self.envelope,self.failed,self.retry_job)

    def test_expired_grant_historical_read_only_not_new_execution(self):
        self.grant.update(created_at=100,expires_at=1000);self.resign()
        with self.assertRaisesRegex(ValueError,'lifetime'):self.check()
        retry.validate(self.signed_grant,self.envelope,self.authority,self.root,historical=True)

    def test_original_first_job_no_longer_accepted(self):
        reset.install_for_train(self.envelope,self.authority,self.root,recovery_envelope=self.signed_grant)
        with StateCache(self.root,self.failed,self.new_manifest,self.authority) as cache:
            with self.assertRaisesRegex(ValueError,'exact authorized'):cache.prepare_parent(None,'4'*64)

    def test_real_retry_promotion_and_delayed_old_ack_fence(self):
        reset.install_for_train(self.envelope,self.authority,self.root,recovery_envelope=self.signed_grant)
        reset.install_for_retirement(self.envelope,self.authority,self.root,recovery_envelope=self.signed_grant)
        descriptor=dict(self.descriptor,optimizer_steps=1,genesis_sha256=self.newgenesis)
        state=dict(self.report['persistent_training_state'],descriptor=descriptor,descriptor_sha256=sha(descriptor))
        report=dict(self.report,job_id='retry',job_sha256=sha(self.retry_job),persistent_training_state=state)
        value=dict(self.ack['payload'],job_id='retry',job_sha256=sha(self.retry_job),report_sha256=sha(report),
            input_checkpoint=self.new_manifest['checkpoint'],trainer_state=dict(self.ack['payload']['trainer_state'],
                optimizer_steps=1,genesis_sha256=self.newgenesis,descriptor_sha256=sha(descriptor)))
        ack=self.sign(value)
        with StateCache(self.root,self.retry_job,self.new_manifest,self.authority) as owned:
            self.assertEqual(owned.prepare_parent(None,'4'*64),0)
            owned.begin_candidate()
            source=self.root/'owned-new-shard';source.write_bytes(b'actual-owned-old-state');source.chmod(0o600)
            row=self.descriptor['shards'][0];owned.retain(row['name'],source,row['sha256'],row['size'])
            self.write(self.root/'jobs/retry/report.json',report);owned.finish(descriptor)
        self.write(self.root/'runner-status/retry.json',dict(phase='complete',exit_code=0))
        with patch.object(retention,'_retire_owned',return_value=dict(status='complete')):
            result=retention.retire(ack,self.authority,self.root)
        self.assertTrue(result['optimizer_cache_promotion']['promoted'])
        marker=reset._read(self.lifecycle/'trainer-current-state.json')
        self.assertEqual(marker['ROOT_ack'],ack)
        self.assertEqual(retention.retire(self.ack,self.authority,self.root)['reason'],'retired-genesis')
        with self.assertRaisesRegex(ValueError,'retired-genesis'):optimizer.promote(self.ack,self.authority,self.root)
        self.assertEqual((self.optimizer/'candidate-retry'/row['name']).read_bytes(),b'actual-owned-old-state')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(RetryTests(name) for name in RetryTests.__dict__ if name.startswith('test_'))


if __name__=='__main__':unittest.main()
