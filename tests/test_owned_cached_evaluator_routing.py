import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.owned_cached_evaluation import POLICY
from subnet.trusted_native_evaluation import POLICY as TRUSTED
from subnet.checkpoint_evaluator import CONFIG_FIELDS,fingerprint
from subnet.gpu_service import evaluate,evaluation_policies,heldout
from test_gpu_service import GPUFixedHeldout

class OwnedRoutingControls(unittest.TestCase):
 def setUp(self):
  f=GPUFixedHeldout();f.setUp();self.f=f;self.config=copy.deepcopy(f.config);self.config['owned_evaluation_policy']=dict(POLICY);self.config['heldout'][0]['harness']['version']='text-tools-long-kv-v3'
 def report(self):
  report=self.f.report();report['owned_cached_evaluation']={'policy':dict(POLICY)}
  report['source_files'].update({'subnet/owned_cached_evaluation.py':'a'*64,'subnet/cached_sampling.py':'b'*64})
  for row in report['heldout']:row.update(verified=False,native_graded=True,proof_verification_performed=False,trust_scope=POLICY['trust_scope'])
  return report
 def test_real_queue_identity_policy_and_legacy_compatibility(self):
  plan=[dict(env_id='env',indices=[2,3],seeds=[2100,3100],harness=self.f.row['harness'])]
  manifest=self.f.manifest;config=self.f.config;bundle=manifest.get('source_bundle')
  source_identity={k:bundle.get(k)for k in ('sha256','format')}if isinstance(bundle,dict)else bundle
  from subnet.storage import canonical
  original=hashlib.sha256(canonical(dict(version='independent-checkpoints-v1',checkpoint=dict(id=manifest['checkpoint']['id'],files=manifest['checkpoint'].get('files')),heldout=plan,environments=[dict(env_id=r['env_id'],spec=r['spec'])for r in manifest['environments']],runtime={k:manifest.get(k)for k in ('model_runtime_revision','backend_profile','numerical_policy','harness_source_hash')},source_bundle=source_identity,model='HuggingFaceTB/SmolLM2-1.7B-Instruct',experiment_id='gpu-continuous-fixed128',evaluation_seed=20260930))).hexdigest()
  self.assertEqual(fingerprint(manifest,config,plan),original)
  owned=fingerprint(manifest,self.config,plan)
  self.assertNotEqual(owned,original)
  self.assertNotEqual(owned,fingerprint(manifest,dict(config,trusted_evaluation_policy=TRUSTED),plan))
  self.assertIn('owned_evaluation_policy',CONFIG_FIELDS)
 def test_policy_null_unknown_wrong_types_and_mixed_fail_closed(self):
  for value in [None,{},dict(POLICY,proof_reverification=0),dict(POLICY,version='future')]:
   with self.subTest(value=value),self.assertRaises(ValueError):evaluation_policies(dict(self.f.config,owned_evaluation_policy=value))
  with self.assertRaisesRegex(ValueError,'mutually'):evaluation_policies(dict(self.config,trusted_evaluation_policy=TRUSTED))
 def test_cached_config_cannot_silently_use_uncached_harness(self):
  with patch('subnet.gpu_service.definitions',return_value=[self.f.row]),self.assertRaisesRegex(ValueError,'cached evaluator harness'):heldout(dict(self.f.config,owned_evaluation_policy=POLICY),self.f.manifest)
 def test_passthrough_native_assurance_and_explicit_record(self):
  report=self.report();controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=report)))
  with tempfile.TemporaryDirectory()as d,patch('subnet.gpu_service.definitions',return_value=[self.f.row]):
   record=evaluate(controller,self.f.manifest,None,'before',10,dict(self.config,evaluation_state=d))[0]
   self.assertEqual(controller.jobs.run.call_args.kwargs['owned_evaluation_policy'],POLICY)
   self.assertNotIn('trusted_evaluation_policy',controller.jobs.run.call_args.kwargs)
   self.assertEqual(record['owned_evaluation_policy'],POLICY);self.assertFalse(record['verified']);self.assertFalse(record['proof_verification_performed']);self.assertEqual(record['completed_count'],2)
 def test_wrong_report_policy_or_assurance_rejected(self):
  for change in ['policy','verified','native','proof','scope','seed']:
   report=self.report()
   if change=='policy':report['owned_cached_evaluation']['policy']['version']='wrong'
   if change=='verified':report['heldout'][0]['verified']=True
   if change=='native':report['heldout'][0]['native_graded']=False
   if change=='proof':report['heldout'][0]['proof_verification_performed']=True
   if change=='scope':report['heldout'][0]['trust_scope']='untrusted'
   if change=='seed':report['heldout'][0]['seed']+=1
   with tempfile.TemporaryDirectory()as d,patch('subnet.gpu_service.definitions',return_value=[self.f.row]),self.subTest(change=change),self.assertRaises(ValueError):evaluate(SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=report))),self.f.manifest,None,'after',11,dict(self.config,evaluation_state=d))
 def test_infrastructure_failure_neutral_not_model_wrong(self):
  report=self.report();row=report['heldout'].pop();report['heldout_failures']=[dict(env_id='env',index=row['index'],seed=row['seed'],error_type='RuntimeError')]
  with tempfile.TemporaryDirectory()as d,patch('subnet.gpu_service.definitions',return_value=[self.f.row]):
   record=evaluate(SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=report))),self.f.manifest,None,'after',11,dict(self.config,evaluation_state=d))[0]
   self.assertEqual(record['status'],'error');self.assertIsNone(record['mean_reward']);self.assertIsNone(record['uncertainty']);self.assertEqual(record['requested_count'],2);self.assertEqual(record['completed_count'],1)
 def test_cached_dataset_cannot_merge_with_legacy_or_unchanged_trusted(self):
  controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=self.report())))
  with tempfile.TemporaryDirectory()as d,patch('subnet.gpu_service.definitions',return_value=[self.f.row]):
   cached=evaluate(controller,self.f.manifest,None,'before',10,dict(self.config,evaluation_state=d))[0]
   controller.jobs.run.return_value=self.f.report()
   legacy=evaluate(controller,self.f.manifest,None,'before',10,dict(self.f.config,evaluation_state=d))[0]
   self.assertNotEqual(cached['dataset_id'],legacy['dataset_id'])

class ContinuousClosureControls(unittest.TestCase):
 def test_uncommitted_parent_never_queued(self):
  from ops.continuous_owned_cached_evaluator import durable_checkpoint
  with tempfile.TemporaryDirectory()as d:
   (Path(d)/'controller.json').write_text(json.dumps(dict(persistent_state_committed=False)))
   self.assertIsNone(durable_checkpoint(d,'unused'))
 def test_genuine_completion_and_exact_pointer_required(self):
  from ops.continuous_owned_cached_evaluator import durable_checkpoint
  from subnet.storage import Identity,canonical
  import base64
  who=Identity(bytes(range(32)));closure=dict(epoch='nonpayable-test',checkpoint='parent10',next_checkpoint='parent11',completed_at=42,round=20)
  pointer=dict(inference_checkpoint='parent11',optimizer_steps=11,descriptor_sha256='d'*64)
  status=dict(persistent_state_committed=True,last_completed_epoch=closure,checkpoint=dict(id='parent11',files={}),trainer_state=pointer,public_optimizer_steps=11)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'controller.json').write_text(json.dumps(status));(p/'latest-trainer-state.json').write_text(json.dumps(pointer));(p/'nonpayable-test-signed-learner-completion.json').write_text(json.dumps(dict(payload=closure,signer=who.id,signature=base64.b64encode(who.key.sign(canonical(closure)).signature).decode())))
   self.assertEqual(durable_checkpoint(d,who.id)['optimizer_steps'],11)
   pointer['optimizer_steps']=12;(p/'latest-trainer-state.json').write_text(json.dumps(pointer))
   with self.assertRaisesRegex(ValueError,'lineage'):durable_checkpoint(d,who.id)
   pointer['optimizer_steps']=11;(p/'latest-trainer-state.json').write_text(json.dumps(pointer));closure['next_checkpoint']='fake';(p/'nonpayable-test-signed-learner-completion.json').write_text(json.dumps(dict(payload=closure,signer=who.id,signature=base64.b64encode(who.key.sign(canonical(closure)).signature).decode())))
   with self.assertRaisesRegex(ValueError,'completion'):durable_checkpoint(d,who.id)
 def test_wrong_config_policy_or_shared_production_queue_rejected(self):
  from ops.continuous_owned_cached_evaluator import config_admission
  c=dict(version='continuous-owned-cached-checkpoints-v1',dispatch_allowed=True,owned_evaluation_policy=POLICY,state='separate',production_state='original',source_sha256='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373',heldout=[dict(indices=list(range(32)),harness=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=128,temperature=.7,top_p=1.))],evaluation_mode='independent-checkpoints-v1');c['source_bundle']={'sha256':c['source_sha256']}
  c['legacy_evaluator_scheduler_must_remain_stopped']=True;config_admission(c)
  for change in [dict(dispatch_allowed=False),dict(state='original'),dict(trusted_evaluation_policy=TRUSTED),dict(owned_evaluation_policy=None)]:
   with self.subTest(change=change),self.assertRaises(ValueError):config_admission(dict(c,**change))
 def test_new_diagnostic_manifest_never_relabels_production(self):
  from ops.continuous_owned_cached_evaluator import queue_original
  base=dict(checkpoint=dict(id='1'*64),source_bundle={'sha256':'original'},payable=False,epoch='original-epoch');config=dict(source_bundle={'sha256':'qualified'})
  with patch('subnet.checkpoint_evaluator.enqueue',return_value='queued')as enqueue:
   self.assertEqual(queue_original(None,config,base,10,'before'),'queued')
  issued=enqueue.call_args.args[1];self.assertNotEqual(issued['epoch'],base['epoch']);self.assertEqual(base['source_bundle']['sha256'],'original');self.assertEqual(issued['source_bundle']['sha256'],'qualified')

if __name__=='__main__':unittest.main()
