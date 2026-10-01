import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.native_role_optimizer import AUTHORITY,OPTIMIZER,REVISION,validate_job,signed,source_map,runtime_environment

class NativeOptimizerTests(unittest.TestCase):
    def test_wrong_authority_before_any_source_artifact_reads(self):
        key=SigningKey.generate()
        with patch('subnet.native_role_optimizer.source_map',side_effect=AssertionError('unexpected source read')):
            with self.assertRaisesRegex(ValueError,'authority'):validate_job(signed({'role':'wrong'},key),AUTHORITY)
    def test_signed_wrong_role_never_enters_source_or_model_reads(self):
        key=SigningKey.generate();anchor=key.verify_key.encode().hex()
        with patch('subnet.native_role_optimizer.AUTHORITY',anchor),patch('subnet.native_role_optimizer.source_map',side_effect=AssertionError('unexpected source read')):
            with self.assertRaisesRegex(ValueError,'role/policy'):validate_job(signed({'role':'chain-weight-setter'},key),anchor)
    def test_malformed_optimizer_policy_rejected(self):
        key=SigningKey.generate();anchor=key.verify_key.encode().hex();policy=dict(OPTIMIZER,steps=2)
        with patch('subnet.native_role_optimizer.AUTHORITY',anchor):
            with self.assertRaisesRegex(ValueError,'role/policy'):validate_job(signed({'role':'native-agent-full-optimizer','revision':REVISION,'optimizer':policy,'payable':False,'chain_transactions':False},key),anchor)
    def test_checkpoint_destination_cannot_escape_native_scope(self):
        from subnet.native_role_optimizer import bounded_path
        for path in ('/tmp/weights','/home/const/subnet120-rewrite/state/native-tau2-training'):
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'path scope'):bounded_path(path,'state/native-tau2-training')
    def test_complete_signed_job_contract_with_ephemeral_authority(self):
        key=SigningKey.generate();anchor=key.verify_key.encode().hex()
        from subnet.native_role_optimizer import ROOT
        job={'role':'native-agent-full-optimizer','revision':REVISION,'optimizer':OPTIMIZER,'payable':False,'chain_transactions':False,'source_files':source_map(),'runtime_environment':runtime_environment(),'budget':{'max_parameters':200000000,'max_context':8192,'max_peak_rss_bytes':64*1024**3,'min_disk_free_bytes':2*1024**3},'positive':{'path':str(ROOT/'state/native-tau2-probe/positive')},'negative':{'path':str(ROOT/'state/native-tau2-probe/negative')},'checkpoint':{'path':str(ROOT/'state/service-conformance/checkpoints/approved')},'destination':str(ROOT/'state/native-tau2-training/control/checkpoint')}
        with patch('subnet.native_role_optimizer.AUTHORITY',anchor):self.assertEqual(validate_job(signed(job,key),anchor),job)

class NativeOptimizerLivenessTests(unittest.TestCase):
    def test_failed_namespace_cannot_be_blindly_retried(self):
        import pathlib,tempfile
        from subnet.native_role_optimizer import execute
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);destination=root/'state/native-tau2-training/control/checkpoint';job={'destination':str(destination)}
            with patch('subnet.native_role_optimizer.ROOT',root),patch('subnet.native_role_optimizer.validate_job',return_value=job),patch('subnet.native_role_optimizer._execute_claimed',side_effect=ValueError('controlled failure')) as work:
                with self.assertRaisesRegex(ValueError,'controlled failure'):execute({},'unused','unused')
                with self.assertRaisesRegex(ValueError,'already attempted'):execute({},'unused','unused')
                self.assertEqual(work.call_count,1)
    def test_active_lock_prevents_second_worker_before_artifact_reads(self):
        import fcntl,pathlib,tempfile
        from subnet.native_role_optimizer import execute
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);out=root/'state/native-tau2-training/control';out.mkdir(parents=True);job={'destination':str(out/'checkpoint')}
            with (out/'optimizer.lock').open('a+') as held:
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('subnet.native_role_optimizer.ROOT',root),patch('subnet.native_role_optimizer.validate_job',return_value=job),patch('subnet.native_role_optimizer._execute_claimed',side_effect=AssertionError('unexpected artifact reads')):
                    with self.assertRaisesRegex(ValueError,'already active'):execute({},'unused','unused')
