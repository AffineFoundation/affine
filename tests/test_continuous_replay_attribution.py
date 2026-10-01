import copy, unittest
from test_replay_training import ReplayTrainingTests
from ops.check_gpu_continuous_evidence import authenticated_training_pairs
from subnet import verified_replay_pool as r
from subnet.replay_training import REVISION

class AttributionTests(unittest.TestCase):
    def setUp(self):
        fixture=ReplayTrainingTests();fixture.setUp();self.fixture=fixture
        self.manifest=fixture.manifest;self.envelope=copy.deepcopy(fixture.inputs)
        pool=self.envelope['pool']['payload'];entry=pool['entries'][0]
        entry['target_sha256']=r.digest(dict(environment_id='e',environment_index=0,
            task_hash=entry['task_hash'],positive_rollout_sha256=r.digest(entry['positive']),
            negative_rollout_sha256=r.digest(entry['negative'])))
        pool['pool_sha256']=r.digest({k:v for k,v in pool.items() if k!='pool_sha256'})
        self.envelope['pool']=fixture.sign(pool)
        target=entry['target_sha256']
        self.report=dict(revision=REVISION,checks=[dict(env_id='e',index=0,target_sha256=target,
            current_checkpoint=self.manifest['checkpoint']['id'],historical_probabilities_used_as_reference=False,
            fresh_current_numerical_native_verification=True)],pool_sha256=pool['pool_sha256'],
            proposed_reuse_increments={target:1},optimizer_performed=False)
        self.metrics=dict(steps=1,replay_training=self.report,replay_inputs_sha256=r.digest(self.envelope))
    def check(self,job=None,report=None,metrics=None):
        return authenticated_training_pairs(self.manifest,job or {'replay':self.envelope},
            {'replay_training':report or self.report},metrics or self.metrics,[],self.fixture.authority)
    def test_authenticated_historical_pair(self):self.assertEqual(len(self.check()),1)
    def test_replay_uses_full_signed_approval_when_live_family_is_inactive(self):
        self.manifest=copy.deepcopy(self.manifest)
        self.manifest['environments'][0]['indices']=[]
        pair=self.check()[0]
        self.assertEqual(pair[0]['indices'],[0,1])
        self.assertEqual(pair[1]['index'],0)
    def test_no_unsigned_replay(self):
        with self.assertRaises(Exception):self.check(job={'steps':1})
    def test_false_fresh_verification_rejected(self):
        report=copy.deepcopy(self.report);report['checks'][0]['fresh_current_numerical_native_verification']=False
        with self.assertRaises(Exception):self.check(report=report)
    def test_changed_request_rejected(self):
        metrics={**self.metrics,'replay_inputs_sha256':'0'*64}
        with self.assertRaises(Exception):self.check(metrics=metrics)
    def test_no_unused_increments(self):
        report=copy.deepcopy(self.report);report['proposed_reuse_increments']['f'*64]=1
        with self.assertRaises(Exception):self.check(report=report)
    def test_fresh_only_remains_default(self):
        fresh=[('unchanged',1,2)]
        self.assertEqual(authenticated_training_pairs({}, {}, {}, {}, fresh,'unused'),fresh)
    def archive_case(self):
        self.manifest=copy.deepcopy(self.manifest)
        self.manifest['harness_source_hash']='1'*64
        current=copy.deepcopy(self.envelope['manifest']['payload'])
        current['harness_source_hash']='1'*64
        self.envelope['manifest']=self.fixture.sign(current)
        pool=copy.deepcopy(self.envelope['pool']['payload'])
        pool['current_manifest_sha256']=r.digest(self.envelope['manifest'])
        for entry in pool['entries']:
            entry['current_manifest_sha256']=pool['current_manifest_sha256']
        pool['pool_sha256']=r.digest({k:v for k,v in pool.items() if k!='pool_sha256'})
        self.envelope['pool']=self.fixture.sign(pool)
        self.report['pool_sha256']=pool['pool_sha256']
        self.metrics['replay_inputs_sha256']=r.digest(self.envelope)
    def archive_check(self,pin):
        return authenticated_training_pairs(self.manifest,{'replay':self.envelope},
            {'replay_training':self.report},self.metrics,[],self.fixture.authority,
            expected_archive_harness_source_hash=pin)
    def test_readonly_archive_attribution_uses_external_pin(self):
        self.archive_case()
        self.assertEqual(len(self.archive_check('1'*64)),1)
    def test_wrong_archive_pin_refused(self):
        self.archive_case()
        with self.assertRaises(ValueError):self.archive_check('2'*64)
    def test_archive_pin_cannot_be_selected_by_job(self):
        self.archive_case()
        with self.assertRaises(ValueError):self.check(job={'replay':self.envelope,
            'expected_archive_harness_source_hash':'1'*64,'skip_source_checks':True})
