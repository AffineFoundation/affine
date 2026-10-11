import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from dashboard import serial128_evaluation_evidence_projection as p
from dashboard import endpoint_evaluation_evidence_projection as e
from dashboard import test_endpoint_evaluation_evidence_projection as fixtures

class SerialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.EndpointEvidenceTests();self.f.setUp();self.f.plan['runtime_versions']={'python':'fixture'};self.f.plan['program_sha256']=p.PROGRAM;self.base=copy.deepcopy(self.f.plan);self.archive_path=self.f.directory/'serial-archive.json';self.save()
    def tearDown(self):self.f.tearDown()
    def save(self):
        self.f.save();self.plan_path=self.f.directory/'captured/plan.json';self.captured=self.f.directory/'captured/output';env=json.loads(self.plan_path.read_bytes());plan=env['payload'];summary=json.loads((self.captured/'result.json').read_bytes());endpoint=json.loads((self.f.directory/'archive.json').read_bytes())['payload']
        pins={name.removeprefix('output/'):dict(sha256=r['sha256'],size=r['bytes'])for name,r in endpoint['objects'].items()if name.startswith('output/')};pins['plan.json']=dict(sha256=e.sha(self.plan_path.read_bytes()),size=self.plan_path.stat().st_size);pins['evaluate.py']=dict(sha256=p.PROGRAM,size=500)
        self.archive=dict(version='prospective-heldout128-durable-readback-v1',full_readback_verified=True,evaluation_lease_free_at_capture=True,production_mutation=False,optimizer_allocation=False,checkpoint=plan['checkpoint']['id'],task_count=128,plan_sha256=e.digest(env),at=301,result_sha256=e.sha((self.captured/'result.json').read_bytes()),archive=pins);self.write_archive()
    def write_archive(self):self.archive_path.write_bytes(e.canonical(self.f.sign(self.archive)))
    def project(self):return p.project_original(self.plan_path,self.archive_path,self.captured,self.base,{self.f.checkpoint},step=43)
    def test_all128_outputs_and_no_private_fields(self):
        row=self.project();self.assertEqual(row['suite'],'serial-heldout128');self.assertEqual(row['requested_count'],128);self.assertEqual(len(row['tasks']),128);self.assertEqual(row['output_text_status'],'retained')
        raw=e.canonical(row)
        for word in (b'PRIVATE',b'private_url',b'full_vocab',b'observations',b'authority_seed'):self.assertNotIn(word,raw)
        self.assertEqual(row['tasks'][0]['raw_turns'][0]['stop_reason'],'eos')
    def test_open_checkpoint_returns_before_archive_or_task_read(self):
        self.archive_path.unlink()
        for path in self.captured.glob('*'):path.unlink()
        self.assertIsNone(p.project_original(self.plan_path,self.archive_path,self.captured,self.base,{'0'*64},step=43))
    def test_archive_signature_tamper(self):
        env=json.loads(self.archive_path.read_bytes());env['payload']['at']+=1;self.archive_path.write_bytes(e.canonical(env))
        with self.assertRaises(Exception):self.project()
    def test_plan_signature_tamper(self):
        env=json.loads(self.plan_path.read_bytes());env['payload']['created_at']+=1;self.plan_path.write_bytes(e.canonical(env))
        with self.assertRaises(Exception):self.project()
    def test_changed_fixed_task_or_seed_rejected(self):
        self.f.plan['suites'][0]['seeds'][0]+=1;self.save()
        with self.assertRaisesRegex(ValueError,'original serial128 plan'):self.project()
    def test_missing_fixed_runtime_metadata_rejected(self):
        self.f.plan.pop('runtime_versions');self.base.pop('runtime_versions');self.save()
        with self.assertRaisesRegex(ValueError,'original serial128 plan'):self.project()
    def test_collector_associates_only_finalized_epochs_and_skips_open_payloads(self):
        monitor=self.f.root/'monitor';(monitor/'state/jobs').mkdir(parents=True)
        (monitor/'config.ROOT-SIGNED.private.json').write_text('{}')
        config=dict(base_plan=str(self.plan_path),base_archive=str(self.archive_path),base_raw_directory=str(self.captured))
        final={'closed':dict(input_checkpoint=self.f.checkpoint,output_checkpoint='0'*64)}
        with patch.object(p,'MONITOR',monitor),patch.object(p,'configuration',return_value=(config,self.base,'c'*64)):
            rows,issues=p.collect_serial128_evidence(self.f.root,final)
            self.assertEqual(list(rows),['closed']);self.assertEqual(issues,[])
            self.assertEqual(rows['closed'][0]['checkpoint_association'],['input_checkpoint'])
            self.assertEqual(len(rows['closed'][0]['tasks']),128)
            original_read=e.read
            def guarded_read(path,*args):
                if Path(path)==self.archive_path or self.captured in Path(path).parents:raise AssertionError('open payload read')
                return original_read(path,*args)
            # A checkpoint with no finalized association must not read raw task/summary/archive bodies.
            with patch.object(e,'read',side_effect=guarded_read):
                rows,issues=p.collect_serial128_evidence(self.f.root,{'closed':dict(input_checkpoint='1'*64,output_checkpoint='2'*64)})
            self.assertEqual(rows,{'closed':[]});self.assertEqual(issues,[])
    def test_consistently_mutated_sampler_rejected(self):
        for plan in (self.f.plan,self.base):
            plan['source_files']['subnet/cached_sampling.py']='0'*64;plan['scientific_files']['subnet/cached_sampling.py']='0'*64
        self.save()
        with self.assertRaisesRegex(ValueError,'sampler source'):self.project()
    def test_incomplete_archive_rejected(self):
        self.archive['evaluation_lease_free_at_capture']=False;self.write_archive()
        with self.assertRaisesRegex(ValueError,'complete signed'):self.project()
    def test_mutated_task_body_rejected(self):
        path=self.captured/'task-0.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'artifact SHA'):self.project()
    def test_missing_archive_member_rejected(self):
        self.archive['archive'].pop('task-0.json');self.write_archive()
        with self.assertRaisesRegex(ValueError,'denominator'):self.project()
    def test_authenticated_wrong_summary_counts_rejected(self):
        path=self.captured/'result.json';v=json.loads(path.read_bytes());v['correct']=127;path.write_bytes(e.canonical(v));self.archive['archive']['result.json']=dict(sha256=e.sha(path.read_bytes()),size=path.stat().st_size);self.archive['result_sha256']=e.sha(path.read_bytes());self.write_archive()
        with self.assertRaisesRegex(ValueError,'summary counts'):self.project()
    def test_late_result_rejected(self):
        path=self.captured/'result.json';v=json.loads(path.read_bytes());v['completed_at']=600;path.write_bytes(e.canonical(v));self.archive['archive']['result.json']=dict(sha256=e.sha(path.read_bytes()),size=path.stat().st_size);self.archive['result_sha256']=e.sha(path.read_bytes());self.write_archive()
        with self.assertRaisesRegex(ValueError,'terminal summary'):self.project()
    def test_assignment_step_checkpoint_required(self):
        path=self.f.root/'assignment.json';path.write_bytes(e.canonical(self.f.sign(dict(step=42,checkpoint=self.f.checkpoint))))
        with self.assertRaisesRegex(ValueError,'assignment checkpoint'):p.project_original(self.plan_path,self.archive_path,self.captured,self.base,{self.f.checkpoint},step=43,assignment_path=path)
    def test_budget_and_native_incomplete_are_retained(self):
        t=self.f.tasks[0];t.update(classification='unresolved',reward=0,cap_hit=True,budget_exhausted=True);t['native_results']=[dict(classification='unresolved',reward=0,done=False)];t['raw_turns'][0]['output_tokens']=[2]*2048;self.save();row=self.project()['tasks'][0]
        self.assertEqual(row['output_length'],2048);self.assertEqual(row['raw_turns'][0]['stop_reason'],'budget');self.assertTrue(row['unresolved_or_incomplete'])
    def test_native_neutral_not_reported_as_infrastructure_failure(self):
        t=self.f.tasks[0];t.update(classification='error',reward=0);t['native_results']=[dict(classification='neutral',reward=0,done=True)];self.save();row=self.project()['tasks'][0]
        self.assertTrue(row['unresolved_or_incomplete']);self.assertFalse(row['infrastructure_error'])
    def test_missing_outputs_remain_missing(self):
        t=self.f.tasks[0];t.update(classification='error',reward=0,native_graded=False,native_results=[],raw_turns=[]);self.save();row=self.project()['tasks'][0]
        self.assertEqual(row['raw_turns'],[]);self.assertIsNone(row['output_length']);self.assertTrue(row['infrastructure_error'])
    def test_symlink_task_rejected(self):
        path=self.captured/'task-0.json';raw=path.read_bytes();path.unlink();other=self.f.root/'other';other.write_bytes(raw);path.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'canonical'):self.project()

if __name__=='__main__':unittest.main(verbosity=2)
