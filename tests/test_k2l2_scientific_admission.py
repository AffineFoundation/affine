import copy
import json
from pathlib import Path
import unittest
from ops import durable_audit_services as g
from ops import durable_learner_service as runner
from ops import k2l2_scientific_admission as m
import test_durable_learner_service as legacy


class ScientificSuccessor(unittest.TestCase):
    def setUp(self):
        self.old = legacy.LearnerRecovery('test_restart_preserves_original_and_allows_epoch_advance')
        self.old.setUp(); self.addCleanup(self.old.doCleanups)
        f = self.old.fixture; self.f = f; self.p = copy.deepcopy(self.old.p)
        self.previous = self.p['source_approval']; predecessor = g.read(self.previous['path'])['payload']
        self.sha = 'b' * 64
        if self.sha == f.source: self.sha = 'c' * 64
        for name in m.REQUIRED_RUNTIME_ADDITIONS:
            (f.runtime / name).write_text('# new scientific module\n')
        runtime = {str(p.relative_to(f.runtime)):g.file_hash(p) for p in f.runtime.rglob('*.py')}
        cfg = g.read(f.config)
        cfg['persistent_training_admission'].update(parameters={'weight':[1]},parameters_sha256='7'*64,genesis_round=0,genesis_checkpoint='8'*64,genesis_sha256='9'*64)
        f.config.write_text(json.dumps(cfg));self.p['config']['file_sha256']=g.file_hash(f.config)
        predecessor_config=f.root/'preserved-old-config.json';predecessor_config.write_bytes(f.config.read_bytes())
        original_policy=copy.deepcopy(self.p);original_policy['config']['path']=str(predecessor_config)
        self.previous_policy=f.document('preserved-old-policy.json',original_policy)
        cfg.update(K=2, L=2, commitment_max_batches=3,
            sampling_policy={'version':m.SAMPLER, 'max_attempts':1000,
                'support_adjudication':'exact-cached-replay-v1','calibration':{'original':'support-calibration'}},
            probability_artifact_policy={'version':'selected-token-logprobs-v1'},
            submission_transport_policy='small-commitment-pairs-v2')
        cfg['source_bundle']['sha256']=self.sha
        cfg['persistent_training_admission']['source_sha256']=self.sha
        self.cfg=cfg
        self.source=dict(version=m.SOURCE,approved=True,source_sha256=self.sha,
            optimizer_reset=False,historical_relabel=False,full_source_files=runtime,
            runtime_source_files=runtime,runtime_execution_files=sorted(runtime),
            evidence=predecessor['evidence'],predecessor_source_approval=self.previous,
            predecessor_learner_policy=self.previous_policy,
            runtime_changes={name:{'before':predecessor['runtime_source_files'].get(name),'after':value}
                for name,value in runtime.items() if predecessor['runtime_source_files'].get(name)!=value},
            contract_sha256=g.digest(m.contract(cfg)))
        self.report=dict(version=m.GPU_RESULT,candidate_source_sha256=self.sha,
            runtime_inventory_sha256=g.digest(runtime),contract_sha256=self.source['contract_sha256'],
            parent_checkpoint='1'*64,parent_optimizer_sha256='2'*64,original_terminal_sha256='3'*64,
            actual_original_wait0=True,controls={name:True for name in m.CONTROLS},
            optimizer_reset=False,historical_relabel=False,
            qualification_class='disposable-sm90-one-step-local-model-no-state-promotion-v1',
            hardware={'name':'NVIDIA H100 PCIe','uuid':'GPU-test-fixture','sm':[9,0]},
            runtime_profile='cuda-fp32-eager-sm90-v1',model_exported=True,optimizer_exported=False,
            model_export_destination='local-isolated-smoke',model_uploaded=False,model_promoted=False)
        self.scope=dict(version='K2L2-miner-bound-v5-CP33-realGPU-smoke-v2',steps=1,
            objective='unchanged-task-normalized-pairwise',optimizer_disposition='isolated-smoke-no-continuation-no-promotion',source_sha256=self.sha,candidate_source_bundle_sha256=self.sha,
            full_source_files=runtime,full_source_inventory_sha256=g.digest(runtime),checkpoint_id='1'*64,
            parent_descriptor_sha256='2'*64,production_mutations=False,network_operations=False,
            parent_step=33,output_step=34,GPU_inventory=[{'name':'NVIDIA H100 PCIe','uuid':'GPU-test-fixture','memory_total_MiB':80000}])
        self.report['original_scope']=f.document('gpu-original-scope.json',self.scope)
        self.raw=dict(version='K2L2-miner-bound-v5-CP33-realGPU-smoke-v2',scope_sha256=g.digest(self.scope),new_source_sha256=self.sha,parent_checkpoint='1'*64,
            parent_descriptor_sha256='2'*64,parent_step=33,output_step=34,
            actual_native_labels=['positive','positive','negative','negative'],full_four_rollout_verification=True,
            probability_artifact_policy={'version':'selected-token-logprobs-v1'},selected_token_probability_transport_bound=True,
            optimizer_state_exported=False,optimizer_state_durable=False,production_pointer_writes=False,
            model_state_exported=True,model_export_destination='local-isolated-smoke',model_state_uploaded=False,
            network_operations=False,heldout_gain_claimed=False,complete=False)
        self.terminal=dict(exit_code=0,actual_child_wait_completed=True,timed_out=False,
            scope_sha256=g.digest(self.scope),production_mutations=False)
        self.close_originals()
        self.q=dict(version=m.QUALIFICATION,approved=True,candidate_source_sha256=self.sha,
            translation_path=cfg['persistent_training_qualification_translation']['path'],
            translation_file_sha256=cfg['persistent_training_qualification_translation']['sha256'])
        self.ack=dict(version=m.ACK,candidate_source_sha256=self.sha,
            report_payload_sha256=g.digest(self.report),original_terminal_sha256=self.report['original_terminal_sha256'],
            original_scope_payload_sha256=g.digest(self.scope),
            original_result_file_sha256=self.report['original_result']['file_sha256'],
            full_independent_metadata_readback=True,actual_original_wait0=True)
        self.close_evidence()

    def close_originals(self):
        for key,value in [('original_result',self.raw),('original_terminal',self.terminal)]:
            p=self.f.root/(key+'.json');p.write_text(json.dumps(value))
            self.report[key]={'path':str(p),'file_sha256':g.file_hash(p)}
        self.report['original_terminal_sha256']=self.report['original_terminal']['file_sha256']

    def close_evidence(self):
        self.q['original_gpu_report']=self.f.document('new-gpu-report.json',self.report)
        self.ack['report_payload_sha256']=g.digest(self.report)
        self.q['durable_readback_ack']=self.f.document('new-archive-ACK.json',self.ack)

    def validate(self):
        return m.validate(self.source,self.q,self.cfg,self.sha,self.f.auth,runner.verify_admission)

    def test_actual_signed_successor_derives_closure(self):
        before=self.old.status.read_bytes()
        result=self.validate()
        self.assertEqual(result['runtime_count'],len(self.source['runtime_source_files']))
        self.assertEqual(result['runtime_count'],179)
        self.assertEqual(before,self.old.status.read_bytes())

    def test_entire_new_runner_policy_and_old_signature_not_reused(self):
        f=self.f;p=self.p;p.update(version=m.POLICY,source_sha256=self.sha,scientific_admission_file_sha256=g.file_hash(m.__file__))
        p['source_approval']=f.document('new-source-approval.json',self.source)
        p['qualification_approval']=f.document('new-qualification.json',self.q)
        self.cfg['persistent_training_qualification_translation'].update(
            approval_path=p['qualification_approval']['path'],approval_sha256=p['qualification_approval']['file_sha256'])
        p['reward_activation']=f.document('new-reward.json',{'approved_sources':[self.sha]})
        self.cfg['continuous_reward_activation_document']=g.read(p['reward_activation']['path'])
        f.config.write_text(json.dumps(self.cfg));p['config']['file_sha256']=g.file_hash(f.config)
        before=self.old.status.read_bytes()
        runner.validate_policy(f.sign(p),f.auth)
        self.assertEqual(before,self.old.status.read_bytes())
        helper_hash=p['scientific_admission_file_sha256'];p['scientific_admission_file_sha256']='0'*64
        with self.assertRaises(ValueError):runner.validate_policy(f.sign(p),f.auth)
        p['scientific_admission_file_sha256']=helper_hash
        p.pop('scientific_admission_file_sha256');p['version']=runner.VERSION
        with self.assertRaises(ValueError):runner.validate_policy(f.sign(p),f.auth)

    def test_old_ordinary_policy_rejects_extended_closure(self):
        with self.assertRaises(ValueError):self.old.validate()

    def test_contract_mutations_rejected(self):
        original=copy.deepcopy(self.cfg)
        for field,value in [('K',1),('L',1),('K',True),('commitment_max_batches',4),
            ('token_artifact_policy',{'version':'token-only-v1'}),
            ('training_input_policy','audited-only'),('probability_artifact_policy',None)]:
            self.cfg=copy.deepcopy(original);self.cfg[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.validate()
        for field,value in [('max_attempts',999),('max_attempts',True),('version','forced-inverse-cdf-prefill-support-v3'),
            ('support_adjudication','threeway')]:
            self.cfg=copy.deepcopy(original);self.cfg['sampling_policy'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.validate()

    def test_runtime_delta_removal_unlisted_and_wrong_hash_rejected(self):
        original=copy.deepcopy(self.source)
        for alteration in ('remove','add','delta','unsorted','wrongfull'):
            self.source=copy.deepcopy(original);runtime=self.source['runtime_source_files']
            if alteration=='remove':runtime.pop(next(iter(runtime)))
            elif alteration=='add':runtime['subnet/unapproved.py']='e'*64
            elif alteration=='delta':self.source['runtime_changes']={}
            elif alteration=='unsorted':self.source['runtime_execution_files'].reverse()
            else:self.source['full_source_files']['subnet/sampling_uniqueness.py']='e'*64
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):self.validate()

    def test_fresh_qualification_every_scientific_control_required(self):
        original=copy.deepcopy(self.report)
        for control in sorted(m.CONTROLS):
            self.report=copy.deepcopy(original);self.report['controls'][control]=False;self.close_evidence()
            with self.subTest(control=control),self.assertRaises(ValueError):self.validate()

    def test_old_flags_cannot_authorize_new_science(self):
        self.source['version']='ordinary-orchestration-only-source-approval-v1'
        with self.assertRaises(ValueError):self.validate()

    def test_no_optimizer_reset_or_history_relabel(self):
        for key in ('optimizer_reset','historical_relabel'):
            self.source[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
            self.source[key]=False

    def test_unsigned_edited_original_gpu_evidence_rejected(self):
        row=self.q['original_gpu_report'];doc=g.read(row['path']);doc['payload']['parent_checkpoint']='9'*64
        Path(row['path']).write_text(json.dumps(doc));row['file_sha256']=g.file_hash(row['path'])
        row['payload_sha256']=g.digest(doc['payload'])
        with self.assertRaises(Exception):self.validate()

    def test_missing_durable_original_exit_or_readback_rejected(self):
        self.ack['full_independent_metadata_readback']=False;self.close_evidence()
        with self.assertRaises(ValueError):self.validate()
        self.ack['full_independent_metadata_readback']=True;self.report['actual_original_wait0']=False;self.close_evidence()
        with self.assertRaises(ValueError):self.validate()

    def test_original_optimizer_genesis_and_parameter_inventory_immutable(self):
        for key in ('parameters','parameters_sha256','genesis_round','genesis_checkpoint','genesis_sha256'):
            original=copy.deepcopy(self.cfg['persistent_training_admission'])
            self.cfg['persistent_training_admission'][key]='changed'
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
            self.cfg['persistent_training_admission']=original

    def test_actual_original_scope_result_terminal_binding(self):
        original=copy.deepcopy(self.raw)
        for key,value in [('scope_sha256','0'*64),('new_source_sha256','0'*64),
            ('actual_native_labels',['positive','negative']),('probability_artifact_policy',None),
            ('selected_token_probability_transport_bound',False),('version','K2L2-miner-bound-v5-CP33-realGPU-smoke-v1'),('optimizer_state_durable',True),
            ('production_pointer_writes',True),('output_step',35)]:
            self.raw=copy.deepcopy(original);self.raw[key]=value;self.close_originals();self.close_evidence()
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
        self.raw=original;self.terminal['timed_out']=True;self.close_originals();self.close_evidence()
        with self.assertRaises(ValueError):self.validate()

    def test_actual_hardware_not_relabelled_h200(self):
        self.report['hardware']['name']='NVIDIA H200';self.close_evidence()
        with self.assertRaises(ValueError):self.validate()
        self.report['hardware']['name']='NVIDIA H100 PCIe';self.report['hardware']['sm']=[8,0];self.close_evidence()
        with self.assertRaises(ValueError):self.validate()

    def test_metadata_ack_does_not_claim_checkpoint_state_durability(self):
        self.report['model_uploaded']=True;self.close_evidence()
        with self.assertRaises(ValueError):self.validate()
        self.report['model_uploaded']=False;self.report['optimizer_exported']=True;self.close_evidence()
        with self.assertRaises(ValueError):self.validate()

    def test_foreign_qualification_candidate_rejected(self):
        self.q['candidate_source_sha256']='f'*64
        with self.assertRaises(ValueError):self.validate()

    def test_ordinary_predecessor_runtime_count_remains_177(self):
        row=copy.deepcopy(self.previous);doc=g.read(row['path'])['payload']
        doc['runtime_source_files'].pop(next(iter(doc['runtime_source_files'])))
        self.source['predecessor_source_approval']=self.f.document('short-predecessor.json',doc)
        with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':unittest.main()
