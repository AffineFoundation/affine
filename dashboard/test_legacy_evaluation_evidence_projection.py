"""Synthetic pre-cached fixed32 provenance and missing-output controls."""
import base64
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey
from dashboard import legacy_evaluation_evidence_projection as p
from ops.publish_finalized_epoch_evidence import safe_projection


class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey(bytes(range(32)));self.authority=self.key.verify_key.encode().hex()
        self.override=patch.object(p,'AUTHORITY',self.authority);self.override.start()
        self.epoch='nonpayable-live-reward-math-v1--1700000000-14';self.jobid=self.epoch+'-eval-before-1234abcd'
        self.manifest=dict(epoch=self.epoch,checkpoint={'id':'a'*64},source_bundle={'sha256':sorted(p.SOURCES)[0]},
            environments=[{'env_id':'affine_math','spec':{'version':'synthetic','max_output_tokens':1024}}],
            model_runtime_revision='synthetic-runtime',backend_profile={'dtype':'float32'},harness_source_hash='b'*64)
        source={name:'c'*64 for name in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')}
        suite=dict(env_id='affine_math',harness=p.HARNESS,indices=p.INDICES,seeds=[20261002+i*1000 for i in p.INDICES])
        self.job=dict(role='evaluate',job_id=self.jobid,manifest=self.sign(self.manifest),heldout=[suite],
            created_at=1000,expires_at=2000,source_files=source,runtime_versions={'torch':'synthetic'})
        self.report=dict(role='evaluate',job_id=self.jobid,job_sha256=p.digest(self.job),epoch=self.epoch,
            checkpoint='a'*64,source_files=source,runtime_versions=self.job['runtime_versions'],chain_transactions=False,
            completed_at=1500,heldout=[dict(index=i,seed=20261002+i*1000,env_id='affine_math',classification='negative',
                reward=0,task_hash=f'{i:064x}',verified=True,SECRET='private') for i in p.INDICES],heldout_failures=[])
        frozen=dict(env_id='affine_math',environment=self.manifest['environments'][0]['spec'],harness=p.HARNESS,
            indices=p.INDICES,seeds=suite['seeds'],model_runtime_revision='synthetic-runtime',backend_profile={'dtype':'float32'},
            runtime_versions=self.job['runtime_versions'],harness_source_hash='b'*64,source_files=source)
        self.record=dict(remote_job_id=self.jobid,checkpoint='a'*64,epoch_id=self.epoch,env_id='affine_math',
            experiment_id='live-original-math94ae-fixed32-v1',harness_config=p.HARNESS,heldout_indices=p.INDICES,
            timestamp=1500,payable=False,weight_submission=False,dataset_id=p.digest(frozen),taskset_hash=p.digest(frozen),
            requested_count=32,attempted_count=32,completed_count=32,count=32,successes=0,
            task_hashes=[r['task_hash'] for r in self.report['heldout']],fixed_task_ids=[r['task_hash'] for r in self.report['heldout']],
            evaluation_failures=[],status='complete',mean_reward=0)
        self.summary=dict(version='independent-checkpoints-v1',status='complete',epoch=self.epoch,
            checkpoint='a'*64,phase='before',records=[self.record],completed_at=1500)

    def tearDown(self):self.override.stop()

    def sign(self,value):
        return dict(payload=value,signer=self.authority,signature=base64.b64encode(self.key.sign(p.canonical(value)).signature).decode())

    def project(self):return p.project_completed(self.sign(self.summary),self.sign(self.job),self.report)

    def test_original_verdicts_but_no_invented_output_or_signature(self):
        row=self.project();safe_projection(row)
        self.assertEqual(len(row['tasks']),32);self.assertTrue(row['signed_summary_authenticated'])
        self.assertFalse(row['individual_results_authenticated'])
        self.assertTrue(all(t['output_length'] is None and t['stop_reason'] is None and t['output_text'] is None for t in row['tasks']))
        self.assertNotIn('SECRET',str(row))

    def test_failed_task_keeps_denominator_without_fabricated_zero(self):
        row=self.report['heldout'].pop();failure={k:row[k] for k in ('index','seed','env_id')};failure['error']='PRIVATE failure'
        self.report['heldout_failures']=[failure];self.record.update(count=31,completed_count=31,
            task_hashes=[r['task_hash'] for r in self.report['heldout']],fixed_task_ids=[r['task_hash'] for r in self.report['heldout']],
            evaluation_failures=[failure],status='error',mean_reward=None)
        result=self.project();self.assertEqual(len(result['tasks']),32)
        error=next(t for t in result['tasks'] if t['verdict']=='evaluation_error')
        self.assertIsNone(error['reward']);self.assertNotIn('PRIVATE',str(result))

    def test_bad_signature_and_aggregate_mismatch_rejected(self):
        envelope=self.sign(copy.deepcopy(self.summary));envelope['payload']['records'][0]['successes']=1
        with self.assertRaises(Exception):p.project_completed(envelope,self.sign(self.job),self.report)
        self.record['successes']=1
        with self.assertRaisesRegex(ValueError,'aggregate'):self.project()

    def test_wrong_job_source_cohort_or_task_rejected(self):
        original=copy.deepcopy(self.report);self.report['heldout'][0]['seed']+=1
        with self.assertRaises(ValueError):self.project()
        self.report=original;self.job['heldout'][0]['harness']=dict(p.HARNESS,max_output_tokens=2048)
        with self.assertRaises(ValueError):self.project()

    def test_original_unsigned_failure_has_no_task_verdicts(self):
        terminal=dict(job_id=self.jobid,phase='failed',exit_code=1,started_at=1100,finished_at=1200,stderr='SECRET')
        row=p.project_failure(self.sign(self.job),terminal);safe_projection(row)
        self.assertTrue(row['infrastructure_error']);self.assertEqual(row['tasks'],[])
        self.assertFalse(row['terminal_authenticated']);self.assertNotIn('SECRET',str(row))
        terminal['job_id']='other'
        with self.assertRaises(ValueError):p.project_failure(self.sign(self.job),terminal)

    def test_collector_uses_signed_records_and_checkpoint_not_queue_pointers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();roles=root/p.BASE/'controller-state'/'roles';roles.mkdir(parents=True)
            cache=root/p.CACHE;cache.mkdir(parents=True)
            (cache/(self.epoch+'-evaluation-before.json')).write_bytes(p.canonical(self.sign(self.summary)))
            (roles/(self.jobid+'-job.json')).write_bytes(p.canonical(self.sign(self.job)))
            (roles/(self.jobid+'-report.json')).write_bytes(p.canonical(self.report))
            other='nonpayable-live-reward-math-v1--1700000100-15'
            finalized={self.epoch:{'input_checkpoint':'a'*64,'output_checkpoint':'d'*64},
                       other:{'input_checkpoint':'a'*64,'output_checkpoint':'e'*64},
                       'nonpayable-live-reward-math-v1--1700000200-19':{'input_checkpoint':'a'*64,'output_checkpoint':'f'*64}}
            by_epoch,issues=p.collect_legacy_evaluation_evidence(root,finalized)
            self.assertEqual(len(by_epoch[self.epoch]),1);self.assertEqual(len(by_epoch[other]),1)
            self.assertEqual(by_epoch[other][0]['job_id'],self.jobid);self.assertEqual(len(by_epoch),2)
            (cache/(self.epoch+'-evaluation-before.json')).unlink()
            pointers=roles.parent/'checkpoint-evaluation-references';pointers.mkdir()
            (pointers/(self.epoch+'-eval-before.json')).write_bytes(p.canonical({'status':'complete','checkpoint':'a'*64}))
            empty,_=p.collect_legacy_evaluation_evidence(root,finalized)
            self.assertFalse(any(empty.values()))


if __name__=='__main__':unittest.main()
