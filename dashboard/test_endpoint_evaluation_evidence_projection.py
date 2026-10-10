import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey
from dashboard import endpoint_evaluation_evidence_projection as p


class EndpointEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.directory=self.root/p.RELATIVE/'g/phases/evaluate-original128'
        (self.directory/'captured/output').mkdir(parents=True)
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.auth=patch.object(p,'AUTHORITY',self.authority);self.auth.start()
        self.files={'model.safetensors':'a'*64};self.checkpoint=p.digest(self.files)
        source={'subnet/cached_sampling.py':p.SAMPLER,'subnet/owned_cached_evaluation.py':p.OWNED}
        definition={'env_id':'affine_math','spec':{'max_turns':1,'config':{}},'indices':[999]}
        manifest={'environments':[definition],'model_runtime_revision':'runtime','backend_profile':{'dtype':'float32'}}
        suites=[dict(env_id='affine_math',harness=p.HARNESS,indices=list(range(i*32,(i+1)*32)),seeds=list(range(100+i*32,100+(i+1)*32))) for i in range(4)]
        cohorts=[p.digest(dict(version='owned-cached-native-evaluation-v1',env_id=s['env_id'],environment=definition['spec'],harness=p.HARNESS,indices=s['indices'],seeds=s['seeds'],model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],source_files=source)) for s in suites]
        self.plan=dict(version='fresh-run-disjoint-heldout128-v1',endpoint='g',exposed_cohort='original128',task_count=128,
            production_mutation=False,optimizer_allocation=False,program_sha256=p.PROGRAM,
            created_at=10,expires_at=500,checkpoint={'id':self.checkpoint,'files':self.files},
            source_files=source,scientific_files=source,manifest=self.sign(manifest),suites=suites,
            cohort_sha256=cohorts,excluded_diagnostic_indices=[],reserved_indices=list(range(128)))
        self.tasks={i:dict(index=i,seed=100+i,checkpoint=self.checkpoint,cohort_sha256=cohorts[i//32],env_id='affine_math',
            classification='positive',reward=1,native_graded=True,verified=False,proof_verification_performed=False,
            cap_hit=False,budget_exhausted=False,elapsed_seconds=1.,task_hash='b'*64,
            native_results=[dict(classification='positive',reward=1,done=True,observations=[{'secret':'PRIVATE'}])],
            raw_turns=[dict(prompt_tokens=[1],output_tokens=[2,151645],text='answer',seed=100+i,full_vocab_logprobs=[[-1]*100],private_url='PRIVATE')],
            error='PRIVATE?token=secret') for i in range(128)}
        self.save()

    def tearDown(self):
        self.auth.stop();self.temp.cleanup()

    def sign(self,x):
        return dict(payload=copy.deepcopy(x),signer=self.authority,signature=base64.b64encode(self.key.sign(p.canonical(x)).signature).decode())

    def save(self):
        envelope=self.sign(self.plan);plan_sha=p.digest(envelope)
        fingerprint=dict(matched=True,checkpoint=self.checkpoint,plan_sha256=plan_sha,actual_loaded={'sha256':'c'*64},expected_loaded={'sha256':'c'*64},at=12)
        summary=dict(version=self.plan['version'],plan_sha256=plan_sha,checkpoint=self.checkpoint,tasks=128,
            cohort_sha256=self.plan['cohort_sha256'],production_mutation=False,optimizer_allocation=False,completed_at=300,
            correct=sum(t['classification']=='positive' for t in self.tasks.values()),
            errors=sum(t['classification']=='error' for t in self.tasks.values()),
            unresolved=sum(t['classification']=='unresolved' for t in self.tasks.values()),
            cap_hits=sum(t['cap_hit'] for t in self.tasks.values()),budget_exhausted=sum(t['budget_exhausted'] for t in self.tasks.values()),
            raw_artifacts={f'task-{i}.json':p.digest(t) for i,t in self.tasks.items()})
        summary['accuracy']=summary['correct']/128
        objects={'plan.json':envelope,'loaded-model-fingerprint.json':fingerprint,'output/result.json':summary}
        objects.update({f'output/task-{i}.json':t for i,t in self.tasks.items()})
        pins={}
        for name,x in objects.items():
            raw=p.canonical(x);(self.directory/'captured'/name).write_bytes(raw)
            pins[name]=dict(sha256=p.sha(raw),bytes=len(raw),full_readback_verified=True,url='PRIVATE')
        archive=dict(version='private-learning-evidence-archive-v1',label='evaluate-original128',full_readback_verified=True,at=301,objects=pins)
        (self.directory/'archive.json').write_bytes(p.canonical(self.sign(archive)))

    def project(self):
        return p.project_phase(self.directory,'g','original128',{self.checkpoint})

    def test_exact_full_projection_omits_secrets_arrays_and_observations(self):
        result=self.project();raw=p.canonical(result)
        self.assertEqual(result['requested_count'],128)
        self.assertEqual(len(result['tasks']),128)
        for secret in (b'PRIVATE',b'full_vocab',b'observations',b'private_url'):
            self.assertNotIn(secret,raw)
        turn=result['tasks'][0]['raw_turns'][0]
        self.assertEqual(turn['output_token_ids'],[2,151645]);self.assertEqual(turn['output_text'],'answer')

    def test_native_neutral_survives_outer_error(self):
        t=self.tasks[0];t.update(classification='error',reward=0)
        t['native_results']=[dict(classification='neutral',reward=0,done=True)]
        self.save();row=self.project()['tasks'][0]
        self.assertEqual(row['outer_classification'],'error')
        self.assertEqual(row['native_classification'],'neutral')
        self.assertTrue(row['unresolved_or_incomplete']);self.assertFalse(row['infrastructure_error'])

    def test_infrastructure_error_keeps_fixed_denominator_without_output(self):
        self.tasks[0].update(classification='error',reward=0,native_graded=False,native_results=[],raw_turns=[])
        self.save();result=self.project()
        self.assertEqual(len(result['tasks']),128);self.assertEqual(result['infrastructure_error_count'],1)
        self.assertIsNone(result['tasks'][0]['output_length'])

    def test_recorded_budget_and_eos_at_cap_are_distinct(self):
        for exhausted in (True,False):
            t=self.tasks[0];t.update(cap_hit=True,budget_exhausted=exhausted)
            t['raw_turns'][0]['output_tokens']=[2]*2047+([2] if exhausted else [151645])
            self.save();turn=self.project()['tasks'][0]['raw_turns'][0]
            self.assertEqual(turn['stop_reason'],'budget' if exhausted else 'eos')

    def test_corrupt_task_bytes_fail_closed(self):
        path=self.directory/'captured/output/task-0.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.project()

    def test_wrong_seed_and_wrong_cohort_fail_even_when_archive_resigned(self):
        self.tasks[0]['seed']+=1;self.save()
        with self.assertRaises(ValueError):self.project()
        self.tasks[0]['seed']-=1;self.tasks[0]['cohort_sha256']='d'*64;self.save()
        with self.assertRaises(ValueError):self.project()

    def test_wrong_program_and_scientific_sampler_fail(self):
        self.plan['program_sha256']='e'*64;self.save()
        with self.assertRaises(ValueError):self.project()
        self.plan['program_sha256']=p.PROGRAM;self.plan['scientific_files']['subnet/cached_sampling.py']='e'*64;self.save()
        with self.assertRaises(ValueError):self.project()

    def test_unfinalized_checkpoint_never_reads_task_payloads(self):
        (self.directory/'captured/output/task-0.json').unlink()
        self.assertIsNone(p.project_phase(self.directory,'g','original128',{'f'*64}))

    def test_unknown_experiment_and_incomplete_archive_reject(self):
        with self.assertRaises(ValueError):p.project_phase(self.directory,'g','deepmath512',{self.checkpoint})
        del self.tasks[127];self.save()
        with self.assertRaises(ValueError):self.project()

    def test_collector_only_finalized_checkpoints_and_exact_known_paths(self):
        final={'epoch14':dict(input_checkpoint='e'*64,output_checkpoint=self.checkpoint)}
        rows,issues=p.collect_endpoint_evidence(self.root,final)
        self.assertEqual(len(rows['epoch14']),1)
        self.assertEqual(rows['epoch14'][0]['checkpoint_association'],['post_update'])
        self.assertEqual(len(issues),3)
        self.assertEqual(p.collect_endpoint_evidence(self.root,{})[0],{})


if __name__=='__main__':unittest.main()
