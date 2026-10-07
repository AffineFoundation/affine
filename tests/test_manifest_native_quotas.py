"""Manifest quotas are CPU support, not live or GPU qualification evidence."""
import base64
import copy
import unittest
import torch
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.task_normalized_training import task_groups,accumulate_tasks
from ops.native_training_eligibility import bind_subset
from ops.native_training_outcome_filter import (MULTI_VERSION,K2L2_VERSION,validate_limits,digest,
    _complete_document_pairs,_filter_admitted_pairs,document_pair_quota)
from test_k2l2_training_cutover import pair
import test_k2l2_scientific_admission as scientific_fixture
from ops import k2l2_scientific_admission as science
from ops import durable_audit_services as guards

class ManifestQuotas(unittest.TestCase):
    def population(self):return [pair(0,2*i+1,2*i+2)for i in range(4)]
    def test_four_disjoint_pairs_one_task_equal_weight(self):
        pairs,tasks,groups,_=task_groups(self.population(),1,'ab'*32,required_pairs_per_task=4)
        value=torch.nn.Parameter(torch.zeros(1))
        rows=accumulate_tasks(torch,lambda i:value[0],[0.]*4,tasks,groups[0])
        self.assertEqual(len(rows),4);self.assertTrue(all(r['gradient_weight']==.25 for r in rows))
        torch.testing.assert_close(value.grad,torch.tensor([-.05]))
    def test_missing_duplicate_and_shared_rollouts_fail(self):
        cases=[self.population()[:3],self.population()+[pair(0,9,10)],
               self.population()[:3]+[copy.deepcopy(self.population()[0])],
               self.population()[:3]+[pair(0,1,8)]]
        for rows in cases:
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):
                task_groups(rows,1,'ab'*32,required_pairs_per_task=4)
    def test_native_grades_eight_and_document_is_atomic(self):
        policy=dict(version=MULTI_VERSION,workers=2,max_pairs=1024,per_grade_seconds=2,wall_seconds=10,max_reply_bytes=1024)
        rows=self.population()
        accepted,receipt=_filter_admitted_pairs(rows,policy,lambda *a:('gold',format(1,'064x'),8,{0},20),lambda t:str(t[0]),lambda gold,reply,timeout:(int(int(reply)%2==1),None))
        self.assertEqual(accepted,rows);self.assertEqual(sum(len(r['grades'])for r in receipt['rows']),8)
        decisions=[dict(pair_sha256=[digest(list(p))for p in rows],accepted=False)]
        self.assertEqual(_complete_document_pairs(rows,decisions,4),[])
        decisions[0]['pair_sha256'].pop()
        with self.assertRaises(ValueError):_complete_document_pairs(rows,decisions,4)
        self.assertEqual(validate_limits(policy)['max_pairs'],1024)
        with self.assertRaises(ValueError):validate_limits(dict(policy,version=K2L2_VERSION))
    def test_signed_subset_requires_all_four_pairs(self):
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        def sign(p):return dict(payload=p,signer=authority,signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
        obj=dict(sha256='a'*64,size=123,learner_admission=sign({'test':'admission'}))
        context=sign(dict(submissions=[obj],original_signed_manifest=sign(dict(K=4,L=4))))
        ids=[format(i+1,'064x')for i in range(4)]
        rows=[dict(pair_sha256=i,status='accepted_native_labels',grades=[dict(claim='positive',native_score=1,label_matches=True),dict(claim='negative',native_score=0,label_matches=True)])for i in ids]
        receipt=dict(version=MULTI_VERSION,context_sha256=digest(context),sampling_assurance='unaudited',proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False,rows=rows,document_decisions=[dict(document_sha256=obj['sha256'],learner_admission_sha256=digest(obj['learner_admission']),pair_sha256=ids,accepted=True)])
        self.assertEqual(bind_subset(context,receipt,[obj],authority)[0],[obj])
        for mutation in (ids[:2],ids[:3]+[ids[0]]):
            bad=copy.deepcopy(receipt);bad['document_decisions'][0]['pair_sha256']=mutation
            with self.assertRaises(ValueError):bind_subset(context,bad,[obj],authority)
        with self.assertRaises(ValueError):bind_subset(context,dict(receipt,version=K2L2_VERSION),[obj],authority)
    def test_training_report_cannot_hide_extra_unpaired_rollout(self):
        from subnet.persistent_training_evidence import validate_updates
        from subnet.persistent_cpu_adamw import POLICY
        manifest=dict(K=4,L=4,epoch='cpu-test',checkpoint={'id':'a'*64},trainer_state_binding={'global_step_before':0},environments=[{'env_id':'math'}],training_coverage={'seed':'ab'*32})
        diagnostics=dict(training_policy=POLICY,optimizer_steps=1,global_optimizer_step_before=0,global_optimizer_step_after=1,epoch='cpu-test',input_checkpoint='a'*64,heldout_gain_claimed=False,state_publication_required=True)
        rolls=[r for _,p,n in self.population()for r in (p,n)]
        report=dict(training={'updates':[{}],'persistent_diagnostics':diagnostics},training_admissions=[{'claimed_batch':{'env_id':'math','rollouts':rolls+[pair(0,9,10)[1]]}}])
        job=dict(steps=1,training_input_policy='committed-unaudited-training-v1')
        with self.assertRaisesRegex(ValueError,'complete manifest task rollout quotas'):validate_updates(report,job,manifest)

    def test_quotas_reject_boolean_unbalanced_and_unbounded(self):
        for K,L in ((True,True),(4,2),(0,0),(65,65)):
            with self.subTest(K=K,L=L),self.assertRaises(ValueError):document_pair_quota(dict(K=K,L=L))
    def fixture(self):
        f=scientific_fixture.ScientificSuccessor('test_actual_signed_successor_derives_closure');f.setUp();self.addCleanup(f.doCleanups);return f
    def test_historical_four_gpu_evidence_cannot_qualify_eight(self):
        f=self.fixture();f.cfg.update(K=4,L=4);f.source['contract_sha256']=guards.digest(science.contract(f.cfg));f.report['contract_sha256']=f.source['contract_sha256'];f.close_evidence()
        with self.assertRaisesRegex(ValueError,'qualification closure'):f.validate()
    def test_distinct_eight_scientific_evidence_binds_quota(self):
        f=self.fixture();f.cfg.update(K=4,L=4);f.source['contract_sha256']=guards.digest(science.contract(f.cfg))
        f.report.update(version=science.MULTI_GPU_RESULT,contract_sha256=f.source['contract_sha256'],controls={k:True for k in science.MULTI_CONTROLS})
        f.scope.update(version=science.MULTI_SCOPE,K=4,L=4)
        f.report['original_scope']=f.f.document('gpu-original-scope.json',f.scope)
        f.raw.update(version=science.MULTI_SCOPE,scope_sha256=guards.digest(f.scope),actual_native_labels=['positive']*4+['negative']*4,full_rollout_verification=True)
        f.terminal['scope_sha256']=guards.digest(f.scope);f.close_originals()
        f.ack.update(original_scope_payload_sha256=guards.digest(f.scope),original_result_file_sha256=f.report['original_result']['file_sha256'],original_terminal_sha256=f.report['original_terminal_sha256']);f.close_evidence()
        self.assertEqual(f.validate()['source'],f.sha)
        f.scope['K']=2;f.report['original_scope']=f.f.document('gpu-original-scope.json',f.scope);f.close_evidence()
        with self.assertRaisesRegex(ValueError,'scope binds'):f.validate()

if __name__=='__main__':unittest.main()
