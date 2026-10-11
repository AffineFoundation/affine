"""Synthetic ephemeral-key CPU contracts; no real release or GPU attestation.

Reuse historical native-admission fixtures so new routing crosses the original
signature, inventory, parent, capability, lifetime, and source checks.
"""
import copy
import builtins
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tests')]
from test_unaudited_execution_genesis import GenesisReleaseTests
from subnet import unaudited_training_execution as a
from subnet import learning_rate_transition as lr
from subnet.positive_only_objective import VERSION, COMPLETION_POLICY, TASK_WEIGHT_RULE
from subnet.training_receipts import sha


class PositiveSFTRouteTests(unittest.TestCase):
    def setUp(self):
        self.fx = GenesisReleaseTests(); self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.sign = self.fx.f.sign; self.authority = self.fx.f.authority
        self.objective = dict(version=VERSION, completion_policy=COMPLETION_POLICY,
            task_weight_rule=TASK_WEIGHT_RULE)
        g = self.fx.genesis
        self.fx.genesis = lr.genesis_document(g['parameters_sha256'],
            g['input_checkpoint'], g['run_id'], 1e-6)
        r = copy.deepcopy(self.fx.genesis_release['payload'])
        r.update(version=a.SFT_GENESIS_RELEASE, method=a.SFT_GENESIS_METHOD,
            genesis_document=self.fx.genesis, genesis_sha256=sha(self.fx.genesis),
            steps=8, effective_learning_rate=1e-6, training_objective=self.objective,
            tokenizer_eos_token_id=151645)
        after = r['execution_source_files']
        after.update({name:'8'*64 for name in a.SFT_SOURCE_FILES})
        after.update({'subnet/training_policy.py':'7'*64,
            'subnet/persistent_training_evidence.py':'6'*64})
        r['changed_source_files'] = {n:h for n,h in after.items()
            if r['original_source_files'].get(n) != h}
        self.fx.f.job['source_files'] = copy.deepcopy(after)
        q = dict(version=a.SFT_QUALIFICATION, method=a.SFT_METHOD,
            execution_source_bundle_sha256=r['training_source_bundle']['sha256'],
            execution_source_files_sha256=sha(after), runtime_versions=r['runtime_versions'],
            training_runtime_sha256=r['training_runtime_sha256'],
            actual_GPU_execution=True, passed=True, training_objective=self.objective,
            tokenizer_eos_token_id=151645, effective_learning_rate=1e-6,
            base_hyperparameters_sha256=a._base_hyperparameters_sha256(),
            state_version='persistent-fp32-trainer-state-v2-effective-lr',
            tested_methods=[a.SFT_GENESIS_METHOD,a.SFT_METHOD],
            research_kernel_evidence_sha256='1'*64,
            source_equivalence_evidence_sha256='2'*64,
            integration_evidence_sha256='3'*64,
            qualification_scope='positive-sft-fresh-genesis-and-continuation-integration-v1',
            research_kernel_GPU_execution=True, fresh_genesis_tested=True,
            continuation_tested=True, negative_gradients=0, EOS_included=True)
        r['execution_qualification'] = self.sign(q)
        self.release = self.sign(r)

    def job(self, n=90, step=0):
        job = self.fx.bound_epoch(n,step);job['steps']=8
        return job

    def amend(self,job,release=None):
        prep=a.automatic_preparation(self.fx.controller,job,release or self.release)
        return a.attach(job,prep,self.authority,self.sign)

    def test_fresh_eight_steps_then_successor_retains_exact_genesis_and_manifest(self):
        values=[]
        for n,step in ((90,0),(91,8),(92,16)):
            job=self.job(n,step);original=copy.deepcopy(job['manifest'])
            amended=self.amend(job);value=a.validate(self.sign(amended),self.authority)
            values.append(value)
            self.assertEqual(amended['manifest'],original)
            self.assertEqual(value['genesis_sha256'],sha(self.fx.genesis))
            self.assertEqual(value['execution_release_sha256'],sha(self.release))
            self.assertEqual(value['training_objective'],self.objective)
            self.assertEqual(value['tokenizer_eos_token_id'],151645)
            grant=value['learning_rate_authorization']['payload']
            self.assertEqual(grant['version'],lr.GENESIS_AUTH_VERSION if step==0 else lr.VERSION)
            self.assertEqual(grant['optimizer_step_before'],step)
            self.assertEqual(grant['steps'],8)
            evidence=a.provenance(self.sign(amended),self.authority)
            self.assertEqual(evidence['optimizer_reset'],step==0)
            self.assertEqual(evidence['optimizer_step_after'],step+8)
            self.assertEqual(evidence['tokenizer_eos_token_id'],151645)
            self.assertNotIn('training_horizon',evidence)
        self.assertEqual([v['version']for v in values],
            [a.SFT_GENESIS_VERSION,a.SFT_VERSION,a.SFT_VERSION])

    def test_continuation_only_release_does_not_reset(self):
        r=copy.deepcopy(self.release['payload']);r.pop('genesis_document')
        r.update(version=a.SFT_RELEASE,method=a.SFT_METHOD,minimum_optimizer_step=8)
        self.amend(self.job(91,8),self.sign(r))
        with self.assertRaises(ValueError):self.amend(self.job(),self.sign(r))

    def test_epoch_grant_is_idempotent_and_cannot_rebind_inventory(self):
        job=self.job();one=a.automatic_preparation(self.fx.controller,job,self.release)
        self.assertEqual(one,a.automatic_preparation(self.fx.controller,job,self.release))
        job['submissions']=[]
        with self.assertRaises(ValueError):a.automatic_preparation(self.fx.controller,job,self.release)

    def test_steps_eos_lr_and_objective_are_explicit_not_report_inferred(self):
        for field,value in [('steps',0),('steps',33),('steps',True),
                ('tokenizer_eos_token_id',True),('tokenizer_eos_token_id',-1),
                ('tokenizer_eos_token_id',None),('effective_learning_rate',5e-7),
                ('training_horizon',{}),('training_objective',{'version':'sequence-mean-preference-positive-nll-v1','positive_nll_weight':1.0})]:
            r=copy.deepcopy(self.release['payload']);r[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                a.release(self.sign(r),self.authority)
        r=copy.deepcopy(self.release['payload']);r['steps']=32
        self.assertEqual(a.release(self.sign(r),self.authority)['steps'],32)

    def test_no_cpu_or_research_only_claim_can_qualify_new_integration(self):
        for field,value in [('actual_GPU_execution',False),('research_kernel_GPU_execution',False),
                ('continuation_tested',False),('fresh_genesis_tested',False),
                ('tested_methods',[a.SFT_GENESIS_METHOD]),('negative_gradients',1),
                ('negative_gradients',False),('EOS_included',False),
                ('integration_evidence_sha256',None),('source_equivalence_evidence_sha256',None),
                ('research_kernel_evidence_sha256',None)]:
            r=copy.deepcopy(self.release['payload']);q=r['execution_qualification']['payload'];q[field]=value
            r['execution_qualification']=self.sign(q)
            with self.subTest(field=field),self.assertRaises(ValueError):a.release(self.sign(r),self.authority)

    def test_historical_qualification_cannot_be_reused_for_sft(self):
        r=copy.deepcopy(self.release['payload'])
        r['execution_qualification']=self.fx.genesis_release['payload']['execution_qualification']
        with self.assertRaises(ValueError):a.release(self.sign(r),self.authority)

    def test_exact_new_source_runtime_and_pinned_eos_bind_qualification(self):
        for field,value in [('execution_source_files_sha256','0'*64),
                ('execution_source_bundle_sha256','0'*64),('runtime_versions',{}),
                ('training_runtime_sha256','0'*64),('tokenizer_eos_token_id',151643)]:
            r=copy.deepcopy(self.release['payload']);q=r['execution_qualification']['payload'];q[field]=value
            r['execution_qualification']=self.sign(q)
            with self.subTest(field=field),self.assertRaises(ValueError):a.release(self.sign(r),self.authority)

    def test_source_delta_cannot_change_sampler_or_omit_sft(self):
        for name in ['subnet/cached_sampling.py','subnet/positive_only_training.py']:
            r=copy.deepcopy(self.release['payload'])
            if 'cached_sampling' in name:r['execution_source_files'][name]='0'*64
            else:r['execution_source_files'].pop(name)
            r['changed_source_files']={n:h for n,h in r['execution_source_files'].items()if r['original_source_files'].get(n)!=h}
            q=r['execution_qualification']['payload'];q['execution_source_files_sha256']=sha(r['execution_source_files']);r['execution_qualification']=self.sign(q)
            with self.subTest(name=name),self.assertRaises(ValueError):a.release(self.sign(r),self.authority)

    def test_new_route_preserves_foreign_signature_expiry_and_parent_guards(self):
        amended=self.amend(self.job())
        with self.assertRaises(ValueError):a.validate(self.sign(amended),self.authority,now=1000)
        for field,value in [('parent_descriptor_sha256','0'*64),('genesis_sha256','0'*64),
                ('optimizer_step_before',1),('input_inventory_sha256','0'*64),
                ('original_signed_manifest_sha256','0'*64)]:
            bad=copy.deepcopy(amended);v=bad[a.FIELD]['payload'];v[field]=value;bad[a.FIELD]=self.sign(v)
            with self.subTest(field=field),self.assertRaises(ValueError):a.validate(self.sign(bad),self.authority)
        bad=copy.deepcopy(amended);bad[a.FIELD]['payload']['steps']=7
        with self.assertRaises(ValueError):a.validate(self.sign(bad),self.authority)

    def test_existing_historical_provenance_shape_remains_unchanged(self):
        # Use a fresh original fixture unaffected by the prospective SFT source map.
        f=GenesisReleaseTests();f.setUp();self.addCleanup(f.doCleanups)
        for n,step in ((90,0),(91,1)):
            job=f.bound_epoch(n,step)
            p=a.automatic_preparation(f.controller,job,f.genesis_release)
            amended=a.attach(job,p,f.f.authority,f.f.sign)
            evidence=a.provenance(f.f.sign(amended),f.f.authority)
            self.assertEqual(evidence['version'],a.GENESIS_VERSION if step==0 else a.VERSION)
            self.assertNotIn('tokenizer_eos_token_id',evidence)
            self.assertNotIn('training_objective',evidence)

    def test_historical_objectives_do_not_import_new_positive_modules(self):
        from subnet.persistent_training_worker import objective_trainer
        original_import=builtins.__import__
        def checked(name,*args,**kwargs):
            if 'positive_only' in name:raise AssertionError('new dependency on historical path')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=checked):
            for coefficient in (None,0,1):
                objective=None if coefficient is None else dict(
                    version='sequence-mean-preference-positive-nll-v1',positive_nll_weight=coefficient)
                function,arguments=objective_trainer(objective)
                self.assertEqual(function.__module__,'subnet.task_normalized_training')
                self.assertEqual(arguments,dict(positive_nll_weight=coefficient or 0))

    def test_runtime_eos_comes_from_signed_pin_not_mutable_report(self):
        from subnet.persistent_training_worker import validate_objective_runtime,objective_trainer
        declaration=self.release['payload']
        validate_objective_runtime(self.objective,declaration,
            SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=151645)))
        for token in (151643,None,True):
            with self.subTest(token=token),self.assertRaises(ValueError):
                validate_objective_runtime(self.objective,declaration,
                    SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=token)))
        function,arguments=objective_trainer(self.objective)
        self.assertEqual(function.__module__,'subnet.positive_only_training')
        self.assertEqual(arguments,dict(objective=self.objective))

    def test_new_module_loader_discards_stale_bootstrap_objects(self):
        # Isolated process prevents this loader test from changing test imports.
        script='''import importlib,pathlib,sys,tempfile,types
sys.path.insert(0,sys.argv[1])
from subnet import backend_jobs as b
names=('positive_only_objective','positive_only_training','positive_only_training_evidence')
assert not any('positive_only' in n for n in b.SOURCE_FILES)
with tempfile.TemporaryDirectory() as directory:
 root=pathlib.Path(directory);(root/'subnet').mkdir()
 for name in names:
  (root/'subnet'/(name+'.py')).write_text('MARKER="authenticated-fresh-source"\\n')
  stale=types.ModuleType('subnet.'+name);stale.MARKER='untrusted-bootstrap';sys.modules['subnet.'+name]=stale
 b.install_source_loader(root,tuple('subnet/'+n+'.py'for n in names))
 for name in names:
  module=importlib.import_module('subnet.'+name)
  assert module.MARKER=='authenticated-fresh-source'
  assert pathlib.Path(module.__file__).parent==root/'subnet'
'''
        result=subprocess.run([sys.executable,'-B','-c',script,str(REPO)],
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_v6_state_output_accepts_actual_fresh_adam_descriptor_and_rejects_relabel(self):
        import torch
        from test_learning_rate_genesis import LRGenesisControls
        from subnet.persistent_training_protocol import validate_output
        f=LRGenesisControls();f.setUp();self.addCleanup(f.doCleanups)
        optimizer=f.initial();optimizer.step(gradients={'weight':torch.ones(3)})
        descriptor,_=f.export(optimizer,'fresh90','33'*32)
        binding=dict(parameters=f.inventory,parameters_sha256=sha(f.inventory),
            input_checkpoint='11'*32,parent=None,genesis_sha256=sha(f.document()),global_step_before=0)
        manifest=dict(epoch='fresh90',trainer_state_binding=binding)
        value=dict(version=a.SFT_GENESIS_VERSION,method=a.SFT_GENESIS_METHOD,
            execution_release_sha256='44'*32,learning_rate_authorization=optimizer.learning_rate_authorization,
            training_objective=self.objective,tokenizer_eos_token_id=2,steps=1,effective_learning_rate=1e-6)
        job=dict(job_id='freshjob90',steps=1,manifest=f.sign(manifest),
            persistent_training=dict(global_step_after=1,output_shards={x['name']:{}for x in descriptor['shards']}),
            unaudited_training_execution=f.sign(value))
        validate_output(descriptor,job,manifest)
        for field,replacement in [('method',a.GENESIS_METHOD),('version',a.SFT_VERSION),
                ('tokenizer_eos_token_id',None),('effective_learning_rate',5e-7)]:
            bad=copy.deepcopy(job);v=copy.deepcopy(value);v[field]=replacement
            bad[a.FIELD]=f.sign(v)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_output(descriptor,bad,manifest)


if __name__ == '__main__':unittest.main()
