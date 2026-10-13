"""Data-only public projection with signed CPU-only qualification fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest

from test_mimo_public_worker import PublicWorkerTests
from subnet.mimo_release_policy import ReleasePolicy, sign, digest
from subnet.mimo_audit_bridge import build_population
from subnet.storage import canonical
from dashboard.mimo_projection import project


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.f=PublicWorkerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        f=self.f;self.m=deepcopy(f.m);self.policy=ReleasePolicy(f.policy,f.operator.id,allow_public=True)
        self.opening=sign(f.operator,self.m);self.root=f.f.root/'projection'
        self.epoch=self.root/'epochs/0';(self.epoch/'coordinator').mkdir(parents=True)
        self.write(self.epoch/'coordinator/OPENING.json',self.opening)
        self.evals=self.root/'eval';self.evals.mkdir()
        self.config=dict(state_root=str(self.root),evaluation_directory=str(self.evals),first_round=182,
            started_at=self.m['start'],trainer_workers=['b'*64],evaluator_workers=['c'*64])
        self.old=dict(incentive={'block':99},incentive_history=[{'block':98}],incentive_prices={},
            epochs=[{'id':'old'}],evaluations=[{'private_url':'SECRET_OLD'}])
    def write(self,path,data):path.write_bytes(canonical(data))
    def test_open_epoch_does_not_publish_submissions_or_capabilities(self):
        result=project(self.old,self.config,self.policy,now=self.m['start']+1)
        self.assertEqual(result['incentive'],self.old['incentive'])
        self.assertEqual(result['epochs'][0]['source'],'live-reward-math')
        self.assertIsNone(result['epochs'][0]['batches']);self.assertEqual(result['evaluations'],[])
        raw=json.dumps(result)
        self.assertNotIn('X-Amz',raw);self.assertNotIn('capabilities',raw);self.assertNotIn('SECRET',raw)
    def finalize_empty(self):
        f=self.f;m=self.m
        dispatch=sign(f.operator,dict(version='mimo-complete-group-coordinator-v1',epoch=m['epoch'],run_id=m['run_id'],
            checkpoint=m['checkpoint']['id'],manifest_sha256=digest(m),selected=[],excluded=[],
            training_requires_sampling_audit=False,
            population_sha256=digest(dict(version='mimo-complete-group-coordinator-v1',epoch=m['epoch'],
                manifest_sha256=digest(m),captures=[],sealed_at=m['deadline'])),
            selected_task_count=0,selected_rollout_count=0))
        regs=sign(f.operator,dict(epoch=m['epoch'],snapshot_block=123,registrations={
            'fixture-hotkey':dict(uid=85,public_key=f.f.identity.id,snapshot_block=123)}))
        pop=build_population(self.opening,{}, {},dispatch,regs,round=182,committed_at=m['deadline'],
                            authority=f.operator.id,manifest_guard=self.policy.validate_manifest)
        self.write(self.epoch/'AUDIT-POPULATION.json',sign(f.operator,pop))
        self.write(self.epoch/'coordinator/DISPATCH.json',dispatch)
        self.final=sign(f.operator,dict(policy_sha256=self.policy.policy_sha256,epoch=m['epoch'],round=182,
            opening_sha256=digest(self.opening),dispatch_sha256=digest(dispatch),input_checkpoint=m['checkpoint']['id'],
            result=dict(status='no-update',checkpoint=m['checkpoint'],optimizer_step=0)))
        self.write(self.epoch/'FINALIZED.json',self.final)
    def test_empty_finalized_epoch_is_zero_batches_not_a_fake_optimizer_step(self):
        self.finalize_empty();result=project(self.old,self.config,self.policy,now=self.m['deadline']+1)
        row=result['epochs'][0];self.assertTrue(row['finalized']);self.assertEqual(row['batches'],0)
        self.assertEqual(row['training']['steps'],0);self.assertIsNone(row['training']['losses'])
        self.assertEqual(result['training_run']['model'],self.m['model_id'])
    def test_abandoned_preopening_directory_does_not_shift_global_round(self):
        self.finalize_empty()
        self.epoch.rename(self.root/'epochs/2')
        result=project(self.old,self.config,self.policy,now=self.m['deadline']+1)
        self.assertEqual(result['epochs'][0]['display_epoch'],1)
    def test_altered_finalization_fails_instead_of_exporting_numbers(self):
        self.finalize_empty();self.final['payload']['round']+=1
        self.write(self.epoch/'FINALIZED.json',self.final)
        with self.assertRaises(Exception):project(self.old,self.config,self.policy)
