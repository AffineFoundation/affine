"""Synthetic credentials only; no fixture constitutes GPU qualification."""
import copy,sys,unittest
from pathlib import Path
import test_unaudited_execution_release as release_fixtures
from subnet import unaudited_training_execution as a
from subnet import training_policy as policy
from subnet.training_receipts import sha
O=lambda n:dict(version=policy.OBJECTIVE_VERSION,positive_nll_weight=float(n))

class HorizonContract(unittest.TestCase):
 def setUp(self):
  self.fx=release_fixtures.ReleaseTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups);self.f=self.fx.f
  self.r=copy.deepcopy(self.fx.release['payload']);after=self.r['execution_source_files']
  after.update({'subnet/persistent_training_evidence.py':'7'*64,'subnet/training_policy.py':'8'*64})
  self.h=dict(version=policy.HORIZON_VERSION,first_optimizer_step=2,updates=16,initial_checkpoint=self.f.manifest['payload']['checkpoint']['id'],initial_parent_descriptor_sha256=sha({'step':2}),coefficient_during=1.,coefficient_after=0.)
  self.r.update(version=a.OBJECTIVE_RELEASE,method=a.OBJECTIVE_METHOD,training_objective=O(1),training_horizon=self.h,changed_source_files={n:h for n,h in after.items()if self.r['original_source_files'].get(n)!=h})
  q=copy.deepcopy(self.r['execution_qualification']['payload']);q.update(version=a.OBJECTIVE_QUALIFICATION,method=a.OBJECTIVE_METHOD,training_objective=O(1),tested_positive_nll_weights=[0.,1.],qualification_scope='private-nonzero-adam-mechanical-continuation-v1',live_optimizer_distribution_equivalence_claimed=False,initial_execution_source_bundle_sha256='9'*64,objective_changed=True,optimizer_reset=False,execution_source_files_sha256=sha(after),retained_parent_descriptor_sha256=self.h['initial_parent_descriptor_sha256'],retained_optimizer_step=2,retained_input_checkpoint=self.h['initial_checkpoint'])
  self.r['execution_qualification']=self.f.sign(q)
 def job(self,n=90,step=2):
  job=self.fx.epoch(n,step,n);job['source_files']=copy.deepcopy(self.r['execution_source_files']);return job
 def amended(self,job=None,release=None):
  job=job or self.job();release=release or self.f.sign(self.r)
  g=a.automatic_preparation(self.fx.controller,job,release)
  return a.attach(job,g,self.f.authority,self.f.sign)
 def test_exact_sixteen_successors_then_automatic_zero(self):
  for offset in range(18):
   job=self.job(90+offset,2+offset);before=copy.deepcopy(job['manifest']);out=self.amended(job)
   v=a.validate(self.f.sign(out),self.f.authority);expected=1 if offset<16 else 0
   self.assertEqual(v['training_objective'],O(expected));self.assertEqual(v['training_horizon'],self.h)
   self.assertEqual(out['manifest'],before);self.assertFalse(a.provenance(self.f.sign(out),self.f.authority)['optimizer_reset'])
   self.assertEqual(v['learning_rate_authorization']['payload']['effective_learning_rate'],5e-7)
 def test_failed_or_retried_epoch_does_not_consume_dose(self):
  job=self.job();first=self.amended(job);self.assertEqual(first,self.amended(job))
  later=self.amended(self.job(91,2));self.assertEqual(later[a.FIELD]['payload']['training_objective'],O(1))
 def test_default_parser_and_old_release_shape_unchanged(self):
  self.assertIsNone(a.configured_objective(self.fx.release['payload']))
  self.assertEqual(a.release(self.fx.release,self.f.authority),self.fx.release['payload'])
 def test_legacy_cannot_add_objective_or_horizon(self):
  for field,val in (('training_objective',O(1)),('training_horizon',self.h)):
   r=copy.deepcopy(self.fx.release['payload']);r[field]=val
   with self.assertRaises(ValueError):a.release(self.f.sign(r),self.f.authority)
 def test_only_zero_and_unit_numbers(self):
  for x in (True,False,-1,.5,2,'1',None,float('nan'),float('inf')):
   with self.subTest(x=x),self.assertRaises(ValueError):policy.objective_config(dict(version=policy.OBJECTIVE_VERSION,positive_nll_weight=x))
 def test_exact_horizon_fields_types_and_dose(self):
  for key,value in (('updates',15),('updates',17),('updates',True),('first_optimizer_step',0),('first_optimizer_step',True),('coefficient_during',2),('coefficient_during',True),('coefficient_after',1),('initial_checkpoint','bad')):
   h=dict(self.h);h[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):policy.objective_horizon(h)
  for h in ({},dict(self.h,unknown=1)):
   with self.assertRaises(ValueError):policy.objective_horizon(h)
 def test_single_update_and_lr_are_fixed(self):
  for k,v in (('steps',2),('steps',True),('effective_learning_rate',1e-6),('minimum_optimizer_step',3)):
   r=copy.deepcopy(self.r);r[k]=v
   with self.subTest(k=k),self.assertRaises(ValueError):a.release(self.f.sign(r),self.f.authority)
 def test_both_qualified_graphs_and_nonzero_parent_required(self):
  for k,v in (('actual_GPU_execution',False),('passed',False),('optimizer_reset',True),('objective_changed',False),('tested_positive_nll_weights',[1.]),('tested_positive_nll_weights',[True,False]),('retained_optimizer_step',0),('retained_optimizer_step',True),('retained_input_checkpoint','bad'),('retained_parent_descriptor_sha256','bad'),('qualification_scope','live-equivalent'),('live_optimizer_distribution_equivalence_claimed',True),('training_objective',dict(version=policy.OBJECTIVE_VERSION,positive_nll_weight=True))):
   r=copy.deepcopy(self.r);q=r['execution_qualification']['payload'];q[k]=v;r['execution_qualification']=self.f.sign(q)
   with self.subTest(k=k,v=v),self.assertRaises(ValueError):a.release(self.f.sign(r),self.f.authority)
 def test_private_mechanical_parent_does_not_claim_live_moment_equivalence(self):
  r=copy.deepcopy(self.r);q=r['execution_qualification']['payload']
  q.update(retained_optimizer_step=1,retained_input_checkpoint='7'*64,retained_parent_descriptor_sha256='8'*64)
  r['execution_qualification']=self.f.sign(q)
  value=a.validate(self.f.sign(self.amended(release=self.f.sign(r))),self.f.authority)
  self.assertEqual(value['optimizer_step_before'],2);self.assertEqual(value['training_horizon'],self.h)
  self.assertFalse(value['execution_qualification']['payload']['live_optimizer_distribution_equivalence_claimed'])
 def test_old_gpu_qualification_cannot_authorize_new_objective(self):
  self.r['execution_qualification']=self.fx.release['payload']['execution_qualification']
  with self.assertRaises(ValueError):self.amended()
 def test_exact_first_parent_and_checkpoint(self):
  for key,value in (('initial_parent_descriptor_sha256','0'*64),('initial_checkpoint','0'*64)):
   r=copy.deepcopy(self.r);r['training_horizon'][key]=value
   q=r['execution_qualification']['payload'];q['retained_parent_descriptor_sha256' if 'descriptor' in key else 'retained_input_checkpoint']=value;r['execution_qualification']=self.f.sign(q)
   with self.subTest(key=key),self.assertRaises(ValueError):self.amended(self.job(90 if 'descriptor' in key else 91),self.f.sign(r))
 def test_premature_zero_and_extended_nll_declined(self):
  for step,weight in ((2,0),(18,1)):
   out=self.amended(self.job(90+step,step));v=out[a.FIELD]['payload'];v['training_objective']=O(weight);out[a.FIELD]=self.f.sign(v)
   with self.assertRaises(ValueError):a.validate(self.f.sign(out),self.f.authority)
 def test_explicit_null_objective_after_horizon_declined(self):
  out=self.amended(self.job(108,18));v=out[a.FIELD]['payload'];v['training_objective']=None;out[a.FIELD]=self.f.sign(v)
  with self.assertRaises(ValueError):a.validate(self.f.sign(out),self.f.authority)
  self.assertEqual(policy.objective_config(None),O(0))
 def test_bad_signature_and_immutable_preparation(self):
  out=self.amended();out[a.FIELD]['payload']['training_horizon']['updates']=17
  with self.assertRaises(ValueError):a.validate(self.f.sign(out),self.f.authority)
  job=self.job(91);self.amended(job);path=self.fx.state/'roles/execution-preparations'/(job['manifest']['payload']['epoch']+'.ROOT-SIGNED.json');before=path.read_bytes()
  r=copy.deepcopy(self.r);r['expires_at']+=1
  with self.assertRaises(ValueError):self.amended(job,self.f.sign(r))
  self.assertEqual(path.read_bytes(),before)
 def test_provenance_binds_horizon_and_coefficient(self):
  out=self.amended();env=self.f.sign(out);report=dict(job_sha256=sha(out),source_files=out['source_files'],**{a.FIELD:a.provenance(env,self.f.authority)})
  a.validate_provenance(report,env,self.f.authority)
  for field,val in (('training_objective',O(0)),('training_horizon',dict(self.h,updates=17))):
   bad=copy.deepcopy(report);bad[a.FIELD][field]=val
   with self.assertRaises(ValueError):a.validate_provenance(bad,env,self.f.authority)
 def test_bootstrap_does_not_import_compute_before_loader(self):
  import subprocess
  root=Path(__file__).resolve().parents[1]
  code="from subnet import unaudited_training_execution as a; import sys; a.configured_objective({'version':a.VERSION}); a.configured_objective("+repr(self.r)+"); assert 'subnet.epoch_optimizer' not in sys.modules"
  p=subprocess.run([sys.executable,'-B','-c',code],env={**__import__('os').environ,'PYTHONPATH':str(root)},capture_output=True,text=True,timeout=20,cwd=root)
  self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
