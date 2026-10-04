"""Synthetic CPU controls for the isolated helper, never GPU qualification."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
from nacl.signing import SigningKey

from subnet import verified_distribution_sampling as cdf
from subnet.audit_policy import InvalidSample

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('isolated_cdf_diagnostic',ROOT/'ops/diagnostic_full_forward_cdf_gpu.py')
helper=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(helper)
SUPERVISOR_SPEC=importlib.util.spec_from_file_location('isolated_cdf_supervisor',ROOT/'ops/diagnostic_full_forward_cdf_supervisor.py')
supervisor=importlib.util.module_from_spec(SUPERVISOR_SPEC);SUPERVISOR_SPEC.loader.exec_module(supervisor)


class IsolatedCDFDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.source=self.directory/'source';self.source.mkdir();(self.source/'example.py').write_bytes(b'# synthetic source\n')
        self.archive=self.directory/'source.tar.gz';self.archive.write_bytes(b'synthetic archive')
        self.cp=self.directory/'checkpoint';self.cp.mkdir();(self.cp/'model.safetensors').write_bytes(b'synthetic weights')
        files={'model.safetensors':helper.file_sha(self.cp/'model.safetensors')}
        self.manifest=dict(epoch='nonpayable-lab-synthetic-cdf-v1',payable=False,checkpoint=dict(id=helper.sha(files),files=files),
            environment=dict(id='synthetic-math',max_turns=1),harness=dict(policy='autoregressive',top_p=1.,temperature=.8,max_output_tokens=2),
            indices=[73],sampling_contract=cdf.make_contract('a'*64,max_attempts=2),sampling_source_hash=cdf.source_hash())
        self.request=dict(version=helper.DOMAIN,run_id='synthetic-only',mode='generate-reference',created_at=10.,expires_at=3500.,
            helper_sha256=helper.file_sha(helper.__file__),candidate_path=cdf.__file__,candidate_sha256=cdf.source_hash(),
            source_path=str(self.source),source_files={'example.py':helper.file_sha(self.source/'example.py')},
            source_archive_path=str(self.archive),source_archive_sha256=helper.file_sha(self.archive),checkpoint_path=str(self.cp),
            checkpoint=self.manifest['checkpoint'],runtime_versions=dict(torch='synthetic',transformers='synthetic',toploc='synthetic',
                datasets='synthetic',verifiers='synthetic'),
            gpu=dict(index=0,name='NVIDIA H200',sm=[9,0],driver='synthetic',idle_fence_sha256='b'*64),
            workspace=str(self.directory/'diagnostic-original'),manifest=self.manifest,task_index=73,
            reference_artifacts={},reference_completion_path=None,reference_completion_sha256=None,
            task_asset_root=str(self.directory),task_asset_files={},diagnostic_only=True,normal_queue_allowed=False,
            chain_transactions_allowed=False,live_admission_allowed=False)

    def signed(self,payload):
        return dict(payload=payload,signer=self.authority,signature=base64.b64encode(self.key.sign(helper.canonical(payload)).signature).decode())

    def reference(self):
        request=copy.deepcopy(self.request);request['mode']='verify-reference'
        request['gpu'].update(name='NVIDIA B200',sm=[10,0])
        artifacts={name:dict(name=name+'.zip',sha256='c'*64,size=101,batch_sha256='d'*64,rollout_sha256='e'*64)
            for name in('honest','synthetic')}
        report=dict(version=helper.DOMAIN,mode='generate-reference',status='reference_ready',diagnostic_only=True,live_admission=False,
            normal_queue_used=False,chain_transactions=False,historical_execution_proven=False,
            source_archive_sha256=request['source_archive_sha256'],complete_source_files_sha256=helper.sha(request['source_files']),
            helper_sha256=request['helper_sha256'],candidate_sha256=request['candidate_sha256'],checkpoint=request['checkpoint']['id'],
            lab_manifest_sha256=helper.sha(request['manifest']),runtime_versions=request['runtime_versions'],
            hardware=dict(gpu_name='NVIDIA H200',sm=[9,0],compute_revision=helper.COMPUTE_REVISION),
            public_context_sha256=helper.sha(cdf.context_for_manifest(request['manifest'])),
            controls={name:dict(status=status,probability_TOPLOC_environment_passed=True,old_prefix_sampler_checked=False,
                live_admission=False,artifact=artifacts[name])for name,status in [('honest','accepted'),('synthetic','sampler_mismatch')]})
        request['reference_artifacts']={name:dict(artifact,path=str(self.directory/artifact['name']))for name,artifact in artifacts.items()}
        envelope=self.signed(report);path=self.directory/'completion.json';path.write_bytes(helper.canonical(envelope))
        request.update(reference_completion_path=str(path),reference_completion_sha256=helper.sha(envelope))
        return request,report,path

    def test_authentication_precedes_model_or_subprocess_access(self):
        envelope=self.signed(self.request)
        self.assertEqual(helper.authenticate(envelope,self.authority),self.request)
        envelope['payload']['live_admission_allowed']=True
        with self.assertRaises(Exception):helper.authenticate(envelope,self.authority)
        path=self.directory/'bad-request.json';path.write_bytes(helper.canonical(envelope))
        with patch('sys.argv',['diagnostic','--request',str(path),'--authority',self.authority]), \
                patch.object(helper,'preflight',side_effect=AssertionError('no GPU preflight before signature')), \
                patch.object(helper,'load_runtime',side_effect=AssertionError('no CUDA/model before signature')):
            with self.assertRaises(Exception):helper.main()
        self.assertFalse(Path(self.request['workspace']).exists())

    def test_scope_lifetime_attempt_budget_and_reference_gpu_are_fail_closed(self):
        helper.validate_request(self.request,11.)
        changes=[('live_admission_allowed',True),('normal_queue_allowed',True),('chain_transactions_allowed',True),
            ('expires_at',4000.),('created_at',12.),('workspace',str(self.source)),('helper_sha256','0'*64)]
        for field,value in changes:
            request=copy.deepcopy(self.request);request[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):helper.validate_request(request,11.)
        for attempts in(True,0,1,9):
            request=copy.deepcopy(self.request);request['manifest']['sampling_contract']['max_attempts']=attempts
            with self.subTest(attempts=attempts),self.assertRaises(ValueError):helper.validate_request(request,11.)
        request=copy.deepcopy(self.request);request['gpu'].update(name='NVIDIA H100',sm=[9,0])
        with self.assertRaises(ValueError):helper.validate_request(request,11.)

    def test_reference_requires_authority_readback_and_exact_scientific_and_control_lineage(self):
        request,report,path=self.reference()
        helper.validate_request(request,11.);self.assertEqual(helper.authenticate_reference(request,self.authority),report)
        mutations=[('checkpoint','0'*64),('candidate_sha256','0'*64),('lab_manifest_sha256','0'*64),('status','worker_claimed_admitted')]
        for field,value in mutations:
            changed=copy.deepcopy(report);changed[field]=value;envelope=self.signed(changed)
            path.write_bytes(helper.canonical(envelope));request['reference_completion_sha256']=helper.sha(envelope)
            with self.subTest(field=field),self.assertRaises(ValueError):helper.authenticate_reference(request,self.authority)
        for name,field,value in [('honest','status','numerical_ambiguity'),('synthetic','probability_TOPLOC_environment_passed',False),
                ('synthetic','old_prefix_sampler_checked',True)]:
            changed=copy.deepcopy(report);changed['controls'][name][field]=value;envelope=self.signed(changed)
            path.write_bytes(helper.canonical(envelope));request['reference_completion_sha256']=helper.sha(envelope)
            with self.subTest(field=field),self.assertRaises(ValueError):helper.authenticate_reference(request,self.authority)

    def test_unsigned_worker_completion_or_substituted_artifact_cannot_be_a_reference(self):
        request,report,path=self.reference()
        path.write_bytes(helper.canonical(report));request['reference_completion_sha256']=helper.sha(report)
        with self.assertRaises(ValueError):helper.authenticate_reference(request,self.authority)
        envelope=self.signed(report);path.write_bytes(helper.canonical(envelope));request['reference_completion_sha256']=helper.sha(envelope)
        request['reference_artifacts']['honest']['sha256']='f'*64
        with self.assertRaises(ValueError):helper.authenticate_reference(request,self.authority)

    def test_source_inventory_and_paths_cannot_silently_add_modules_or_follow_links(self):
        helper.verify_inventory(str(self.source),self.request['source_files'])
        (self.source/'extra.py').write_text('# extra')
        with self.assertRaises(ValueError):helper.verify_inventory(str(self.source),self.request['source_files'])
        (self.source/'extra.py').unlink();(self.source/'link.py').symlink_to(self.source/'example.py')
        with self.assertRaises(ValueError):helper.verify_inventory(str(self.source),self.request['source_files'])
        link=self.directory/'linked-archive';link.symlink_to(self.archive)
        with self.assertRaises(ValueError):helper.file_sha(link)

    def test_checkpoint_substitution_and_busy_gpu_are_rejected_before_cuda_import(self):
        request=copy.deepcopy(self.request);request['checkpoint']['files']['../outside']='0'*64
        with self.assertRaises(ValueError):helper.preflight(request)
        with patch.object(helper.subprocess,'check_output',return_value='12345\n'):
            with self.assertRaisesRegex(ValueError,'occupied'):helper.preflight(self.request)

    def test_artifact_bytes_context_and_decoded_hashes_are_all_checked(self):
        from subnet.batches import pack
        workspace=self.directory/'artifacts';workspace.mkdir()
        doc=dict(schema=2,env_id='synthetic-math',environment_version='synthetic',index=73,sample_index=73,turns=[])
        arrays=[np.zeros((1,3),np.float32)]
        descriptor=helper.artifact(workspace,'honest',self.request,doc,arrays);descriptor['path']=str(workspace/descriptor['name'])
        restored,probabilities=helper.read_artifact(descriptor,self.request)
        self.assertEqual(restored,doc);self.assertTrue(np.array_equal(probabilities[0],arrays[0]))
        for field,value in [('sha256','0'*64),('batch_sha256','0'*64),('rollout_sha256','0'*64),('size',descriptor['size']+1)]:
            changed=dict(descriptor);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):helper.read_artifact(changed,self.request)
        changed_request=copy.deepcopy(self.request);changed_request['task_index']=74
        with self.assertRaises(ValueError):helper.read_artifact(descriptor,changed_request)

    def fake_runtime(self):
        context=cdf.context_for_manifest(self.manifest);row=np.log(np.array([.1,.2,.3,.4,1e-40],np.float64))
        output=[cdf.select_token(row,cdf.uniform(context,'synthetic-math','d'*64,73,0,0,i),.8)for i in range(2)]
        receipt=dict(version=cdf.VERSION,binding_sha256=helper.sha(context),attempt=0)
        doc=dict(task_hash='d'*64,index=73,seed=0,sampling=receipt,turns=[dict(prompt=[0,1],output=output)])
        runtime=SimpleNamespace(spec=SimpleNamespace(id='synthetic-math'),harness=self.manifest['harness'],
            tokenizer=SimpleNamespace(eos_token_id=4),sampling_receipt=lambda seed:dict(sampling=dict(receipt,attempt=seed)),
            last_forward=dict(prompt=[0,1],output=output,inputs=[0,1]+output,full=np.tile(row,(4,1))),verify=lambda doc,arrays:True)
        return runtime,context,doc

    def test_independent_checks_then_all_public_choices_without_prefix_replay(self):
        runtime,context,doc=self.fake_runtime();result=helper.check(runtime,cdf,context,doc,[])
        self.assertEqual(result['status'],'accepted');self.assertEqual(result['checked_tokens'],2)
        self.assertTrue(result['probability_TOPLOC_environment_passed']);self.assertFalse(result['old_prefix_sampler_checked'])
        fake=copy.deepcopy(doc);fake['turns'][0]['output'][0]=(fake['turns'][0]['output'][0]+1)%4
        runtime.last_forward.update(output=fake['turns'][0]['output'],inputs=[0,1]+fake['turns'][0]['output'])
        result=helper.check(runtime,cdf,context,fake,[])
        self.assertEqual(result['status'],'sampler_mismatch');self.assertTrue(result['probability_TOPLOC_environment_passed'])
        self.assertFalse(result['live_admission']);self.assertFalse(result['historical_execution_proven'])
        fake['sampling']['binding_sha256']='0'*64
        with self.assertRaises(ValueError):helper.check(runtime,cdf,context,fake,[])

    def test_probability_or_toploc_drift_is_numeric_incompatibility_not_fraud_or_weakened_acceptance(self):
        runtime,context,doc=self.fake_runtime()
        for reason in('probabilities','TOPLOC','environment replay'):
            def failed(*args,reason=reason):raise InvalidSample(reason)
            runtime.verify=failed
            with patch.object(cdf,'verify_turn',side_effect=AssertionError('gate cannot bypass failed original checks')):
                result=helper.check(runtime,cdf,context,doc,[])
            self.assertEqual(result['status'],'source_or_environment_mismatch'if reason=='environment replay'else'cross_hardware_numerical_mismatch')
            self.assertFalse(result['probability_TOPLOC_environment_passed']);self.assertFalse(result['live_admission'])

    def test_numeric_statistics_are_bounded_without_tokens_or_distribution_values(self):
        runtime,context,doc=self.fake_runtime();claimed=runtime.last_forward['full'][1:3].astype(np.float32)
        claimed=claimed.copy();claimed[0,0]+=3e-5;claimed[1,0]=np.nan
        stats=helper.probability_statistics(runtime,[claimed])
        self.assertEqual(stats['claimed_nonfinite'],1);self.assertGreater(stats['max_absolute_difference'],1e-5)
        self.assertEqual(stats['outside_existing_tolerance'],1);self.assertEqual(stats['unchanged_logprob_atol'],1e-5)
        self.assertNotIn('tokens',stats);self.assertNotIn('probabilities',stats)
        self.assertTrue(all(not isinstance(v,np.ndarray)for v in stats.values()))

    def test_synthetic_control_changes_choice_without_introducing_invalid_stopping(self):
        self.assertEqual(helper.synthetic_output([4],5,4,2),[0,4])
        self.assertEqual(helper.synthetic_output([4],5,4,1),[0])
        self.assertEqual(helper.synthetic_output([0,4],5,4,2),[1,4])
        self.assertEqual(helper.synthetic_output([0,1],5,4,2),[1,1])

    def test_supervisor_actual_wait_observes_one_synthetic_cpu_child_and_disallows_uncertain_rerun(self):
        fake=self.directory/'diagnostic_full_forward_cdf_gpu.py'
        fake.write_text("print('synthetic CPU child; no scientific qualification')\nraise SystemExit(7)\n")
        request=self.directory/'request.json';request.write_text('{}')
        command=dict(argv=[sys.executable,'-I','-B',str(fake),'--request',str(request),'--authority',self.authority],
            helper_path=str(fake),helper_sha256=helper.file_sha(fake),request_path=str(request),request_sha256=helper.file_sha(request),
            root_scientific_execution_approved=True,diagnostic_only=True,normal_queue_allowed=False,live_admission_allowed=False,
            chain_transactions_allowed=False)
        self.assertEqual(supervisor.validate(command,sys.executable),command['argv'])
        wrong=copy.deepcopy(command);wrong['live_admission_allowed']=True
        with self.assertRaises(ValueError):supervisor.validate(wrong,sys.executable)
        path=self.directory/'command.json';path.write_bytes(helper.canonical(command));out=self.directory/'wait-v1'
        argv=['supervisor','--command-file',str(path),'--command-file-sha256',helper.file_sha(path),'--output',str(out)]
        with patch('sys.argv',argv),self.assertRaises(SystemExit)as stopped:supervisor.main()
        self.assertEqual(stopped.exception.code,7)
        receipt=json.loads((out/'completion.private.json').read_bytes())
        self.assertTrue(receipt['actual_child_wait_completed']);self.assertEqual(receipt['exit_code'],7)
        self.assertTrue(receipt['child_ticks']);self.assertFalse(receipt['live_admission'])
        with patch('sys.argv',argv),self.assertRaises(ValueError):supervisor.main()


if __name__=='__main__':unittest.main()
