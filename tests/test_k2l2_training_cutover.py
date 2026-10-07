"""CPU cutover guards; no claim of GPU/runtime or learning qualification."""
import copy
import unittest
import torch
import base64
from nacl.signing import SigningKey
from subnet.storage import canonical
from ops.native_training_eligibility import bind_subset
from subnet.task_normalized_training import task_groups, accumulate_tasks
from ops.native_training_outcome_filter import VERSION,K2L2_VERSION,validate_limits,_filter_admitted_pairs,_complete_document_pairs,digest


def pair(task,p,n):
    def rollout(token,classification):
        return dict(env_id='math',index=task,task_hash=format(task+1,'064x'),classification=classification,
            turns=[dict(prompt=[10],output=[token])])
    return dict(env_id='math'),rollout(p,'positive'),rollout(n,'negative')


class Cutover(unittest.TestCase):
    def groups(self,population):
        return task_groups(population,1,'ab'*32,required_pairs_per_task=2)

    def test_four_rollouts_two_disjoint_pairs_no_cartesian_product(self):
        pairs,tasks,groups,ids=self.groups([pair(0,1,2),pair(0,3,4)])
        self.assertEqual(len(pairs),2)
        self.assertEqual(len(tasks),1)
        self.assertEqual(len(tasks[0]['pair_indices']),2)
        self.assertEqual(len(ids),2)

    def test_incomplete_task_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete two-pair'):self.groups([pair(0,1,2)])

    def test_extra_pair_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete two-pair'):self.groups([pair(0,1,2),pair(0,3,4),pair(0,5,6)])

    def test_same_positive_cannot_fill_two_pairs(self):
        with self.assertRaisesRegex(ValueError,'shared rollout'):self.groups([pair(0,1,2),pair(0,1,4)])

    def test_same_negative_cannot_fill_two_pairs(self):
        with self.assertRaisesRegex(ValueError,'shared rollout'):self.groups([pair(0,1,2),pair(0,3,2)])

    def test_metadata_repack_does_not_create_new_content(self):
        second=pair(0,1,4);second[1]['attempt']=999;second[1]['turns'][0]['proof']='replacement'
        with self.assertRaisesRegex(ValueError,'shared rollout'):self.groups([pair(0,1,2),second])

    def test_cross_class_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'shared rollout'):self.groups([pair(0,1,2),pair(0,3,1)])

    def test_exact_pair_clone_rejected(self):
        p=pair(0,1,2)
        with self.assertRaisesRegex(ValueError,'duplicate pair'):self.groups([p,copy.deepcopy(p)])

    def test_equal_task_weight_mean_two_pairs(self):
        pairs,tasks,groups,_=self.groups([pair(0,1,2),pair(0,3,4),pair(1,5,6),pair(1,7,8)])
        values=torch.nn.Parameter(torch.zeros(2))
        observations=accumulate_tasks(torch,lambda i:values[pairs[i][1]['index']],[0.]*4,tasks,groups[0])
        torch.testing.assert_close(values.grad,torch.tensor([-.025,-.025]))
        self.assertEqual(len(observations),4)
        self.assertTrue(all(r['gradient_weight']==.25 for r in observations))

    def test_native_rejects_entire_document_when_one_pair_fails(self):
        pairs=[pair(0,1,2),pair(0,3,4),pair(1,5,6),pair(1,7,8)]
        decisions=[dict(pair_sha256=[digest(list(p))for p in pairs[:2]],accepted=False),
                   dict(pair_sha256=[digest(list(p))for p in pairs[2:]],accepted=True)]
        self.assertEqual(_complete_document_pairs(pairs,decisions),pairs[2:])
        with self.assertRaises(ValueError):_complete_document_pairs(pairs,decisions[:1])
        decisions[1]['pair_sha256'][1]=decisions[0]['pair_sha256'][0]
        with self.assertRaises(ValueError):_complete_document_pairs(pairs,decisions)

    def test_historical_k1_policy_still_accepts_original(self):
        p=pair(0,1,2)
        self.assertEqual(len(task_groups([p,copy.deepcopy(p)],1,'ab'*32)[0]),1)

    def test_new_filter_admits_512_pairs_without_changing_old_cap(self):
        policy=dict(version=K2L2_VERSION,workers=4,max_pairs=512,per_grade_seconds=2,wall_seconds=10,max_reply_bytes=1024)
        self.assertEqual(validate_limits(policy)['max_pairs'],512)
        with self.assertRaises(ValueError):validate_limits(dict(policy,version=VERSION))
        with self.assertRaises(ValueError):validate_limits(dict(policy,max_pairs=513))
        with self.assertRaises(ValueError):validate_limits(dict(policy,max_pairs=True))

    def test_subset_requires_two_dispositions_and_rejects_legacy_version(self):
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        def sign(payload):
            return dict(payload=payload,signer=authority,signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
        obj=dict(sha256='a'*64,size=123,learner_admission=sign({'test':'admission'}))
        context=sign(dict(submissions=[obj],original_signed_manifest=sign(dict(K=2,L=2))))
        rows=[]
        for identity in ('b'*64,'c'*64):
            rows.append(dict(pair_sha256=identity,status='accepted_native_labels',grades=[
                dict(claim='positive',native_score=1,label_matches=True),
                dict(claim='negative',native_score=0,label_matches=True)]))
        receipt=dict(version=K2L2_VERSION,context_sha256=digest(context),sampling_assurance='unaudited',
            proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False,rows=rows,
            document_decisions=[dict(document_sha256=obj['sha256'],learner_admission_sha256=digest(obj['learner_admission']),
                                    pair_sha256=['b'*64,'c'*64],accepted=True)])
        accepted,_=bind_subset(context,receipt,[obj],authority)
        self.assertEqual(accepted,[obj])
        with self.assertRaisesRegex(ValueError,'new native filter version'):
            bind_subset(context,dict(receipt,version=VERSION),[obj],authority)
        broken=copy.deepcopy(receipt);broken['document_decisions'][0]['pair_sha256']=['b'*64]
        with self.assertRaisesRegex(ValueError,'requires two pairs'):bind_subset(context,broken,[obj],authority)
        bad=copy.deepcopy(receipt);bad['rows'][1]['status']='excluded_label_mismatch'
        bad['rows'][1]['grades'][0].update(native_score=0,label_matches=False)
        bad['document_decisions'][0]['accepted']=False
        self.assertEqual(bind_subset(context,bad,[obj],authority)[0],[])

    def test_filter_receipt_names_new_policy_and_grades_all_four(self):
        policy=dict(version=K2L2_VERSION,workers=2,max_pairs=512,per_grade_seconds=2,wall_seconds=10,max_reply_bytes=1024)
        pairs=[pair(0,1,2),pair(0,3,4)]
        accepted,receipt=_filter_admitted_pairs(pairs,policy,lambda *a:('gold',format(1,'064x'),8,{0},20),
            lambda tokens:str(tokens[0]),lambda gold,reply,timeout:(int(int(reply)%2==1),None))
        self.assertEqual(accepted,pairs)
        self.assertEqual(receipt['version'],K2L2_VERSION)
        self.assertEqual(sum(len(r['grades'])for r in receipt['rows']),4)
        self.assertFalse(receipt['proof_verification_performed'])

if __name__=='__main__':unittest.main()
