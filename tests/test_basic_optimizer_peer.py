"""Prospective basic-eligibility peer admission using portable signed fixtures."""
import copy
import unittest
from subnet import learner_selection_operator_bridge as peer, math_completion, training_task_representatives as rep
from test_training_task_representatives import AUTH, signed

class BasicOptimizerPeerTests(unittest.TestCase):
    def fixture(self, threaded=True):
        names=['sampling_uniqueness','trajectory_identity','batch_quotas','trainer_local_state',
               'math_completion','fp32_gradient_accumulation','unaudited_training_execution',
               'learning_rate_transition','training_task_representatives']
        if threaded: names.append('optimizer_threading')
        files={'subnet/'+name+'.py':'f'*64 for name in names}
        # Source inventory placeholders are test assertions, never execution authority.
        for i in range((187 if threaded else 186)-len(files)):files['subnet/fixture_'+str(i)+'.py']='e'*64
        operators={name:'d'*64 for name in peer.FILES}
        operators['subnet/training_task_representatives.py']=files['subnet/training_task_representatives.py']
        manifest=dict(epoch='fixture--1000-900',source_bundle={'sha256':'a'*64},K=4,L=4,max_batches=9,
            sampling_contract={'version':'forced-inverse-cdf-prefill-miner-bound-v5','max_attempts':1000},
            optimizer_state_export_policy='trainer-local-only-v1',
            environments=[{'env_id':'math','indices':[1,2],'spec':dict(id='affine_math',adapter='prime_v1',max_turns=1,
                version=math_completion.ENVIRONMENT_VERSION,config={math_completion.FIELD:math_completion.VERSION})}],
            training_representative_policy=dict(version=rep.PRIORITY_VERSION if threaded else rep.VERSION,
                max_candidate_documents=2304,max_total_input_bytes=4608000000,max_native_documents_per_wave=256,
                max_native_wall_seconds=600,exhausted_task_rule='advance-fixed-task-order'))
        if threaded:manifest[rep.FIELD].update(history_epochs=12,task_priority='untrained-in-prior12-first-stable-rank')
        policy=dict(version=peer.BASIC_THREAD_AUTH_VERSION if threaded else peer.BASIC_AUTH_VERSION,
            source_sha256='a'*64,scientific_source_files=files,operator_files=operators,minimum_round=900,
            epoch_prefix='fixture--',peer_entry_sha256='1'*64,peer_runner_sha256='2'*64,backend_execution_allowed=False)
        return manifest,policy
    def test_v11_no_assessment_and_v10_absent_thread_policy(self):
        for threaded in (True,False):
            manifest,policy=self.fixture(threaded)
            self.assertEqual(peer.approval(signed(policy),AUTH,manifest),policy)
            self.assertNotIn('learner_blacklist_selection_policy',manifest)
    def test_versions_cannot_silently_change_source_closure(self):
        manifest,policy=self.fixture();policy['version']=peer.BASIC_AUTH_VERSION
        with self.assertRaisesRegex(ValueError,'historical CPU peer'):peer.approval(signed(policy),AUTH,manifest)
        manifest,policy=self.fixture();policy['scientific_source_files'].pop('subnet/optimizer_threading.py')
        with self.assertRaisesRegex(ValueError,'threading dependency'):peer.approval(signed(policy),AUTH,manifest)
    def test_basic_path_refuses_reward_assessment_and_wrong_epoch(self):
        for key,value in [('learner_blacklist_selection_policy',{}),('learner_blacklist_selection_snapshot',{}),
                          ('epoch','fixture--malformed'),('learner_blacklist_selection_round',901)]:
            manifest,policy=self.fixture();manifest[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):peer.approval(signed(policy),AUTH,manifest)
    def test_representative_parser_must_be_same_pinned_module(self):
        manifest,policy=self.fixture();policy['operator_files']['subnet/training_task_representatives.py']='3'*64
        with self.assertRaisesRegex(ValueError,'same authenticated representative parser'):peer.approval(signed(policy),AUTH,manifest)

if __name__=='__main__':unittest.main()
