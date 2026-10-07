import copy
import unittest
from nacl.signing import SigningKey
import test_paired_quota_batch_adapter as fixtures
from ops.paired_quota_nested_selector import GRADE_VERSION, select_nested, selection_scope
from ops.paired_quota_qualification import digest, identities
from ops.paired_quota_research_ledger import validate_revision
from subnet.storage import canonical


class NestedQuotaControls(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CurrentWireAdapterControls();self.f.setUp()
        self.key=SigningKey.generate();self.miner=self.key.verify_key.encode().hex()
        self.eos=(999,)
        self.rows=[]
        for rollout in self.f.rolls:
            rollout['turns'][0]['output'].append(999)
            rollout['turns'][0]['text']=self.f.decode(rollout['turns'][0]['output'])
            self.rows.extend(self.f.adapter.normalize(self.f.batch([rollout])))

    def record(self,row,*,status='native-graded',native_done=True,reward=None,classification=None,attempt=None):
        attempt=row['attempt'] if row is not None else attempt
        if status=='native-graded':
            classification=classification or row['classification']
            reward=int(classification=='positive') if reward is None else reward
        else: classification=reward=native_done=None
        payload=dict(status=status,classification=classification,reward=reward,native_done=native_done,
                     row_sha256=digest(row) if row is not None else None,attempt=attempt)
        signature=self.key.sign(canonical(payload)).signature.hex()
        return dict(attempt=attempt,row=copy.deepcopy(row),evidence=dict(payload=payload,signature=signature))

    def authenticate(self,record,binding):
        ev=record['evidence'];p=ev['payload'];self.key.verify_key.verify(canonical(p),bytes.fromhex(ev['signature']))
        # The test boundary independently binds original admitted normalized row
        # and exact prescribed task context, rather than trusting callback True.
        self.assertEqual(p['row_sha256'],digest(record['row']) if record['row'] is not None else None)
        self.assertEqual(p['attempt'],record['attempt'])
        self.assertEqual(binding['selection_scope_sha256'],digest(selection_scope(self.f.adapter.task,self.miner,self.eos)))
        return dict(binding,version=GRADE_VERSION,**{k:p[k]for k in ('status','classification','reward','native_done')})

    def select(self,records,**extra):
        return select_nested(self.f.adapter.task,self.miner,records,eos_token_ids=self.eos,
                             authenticate_admitted_native=self.authenticate,enabled=True,**extra)

    def test_shuffled_uploads_select_nested_prescribed_prefix_no_cartesian(self):
        records=[self.record(r)for r in self.rows]
        first=self.select(records);second=self.select(list(reversed(records)))
        self.assertEqual(first,second)
        self.assertEqual((first['first_K1_prefix_length'],first['first_K2_prefix_length']),(3,4))
        self.assertEqual(first['K1L1']['pairs'],first['K2L2']['pairs'][:1])
        self.assertEqual(len(first['K2L2']['pairs']),2)
        expected=[identities(self.f.adapter.task,r)for r in self.rows]
        for pair,p,n in zip(first['K2L2']['pairs'],expected[:2],expected[2:]):
            self.assertEqual(pair['positive']['execution_id'],p[0]);self.assertEqual(pair['negative']['execution_id'],n[0])
        for arm in ('K1L1','K2L2'):
            validate_revision(first[arm]);self.assertEqual(first[arm]['contribution_units'],1)
            self.assertEqual(len(first[arm]['pairs'])*first[arm]['pair_weight_within_task'],1.)

    def test_content_hash_sort_does_not_override_first_attempt(self):
        later=copy.deepcopy(self.rows[1]);first_hash=identities(self.f.adapter.task,self.rows[0])[1]
        for token in range(100,2000):
            later['turns'][0]['output']=[token,999]
            if identities(self.f.adapter.task,later)[1]<first_hash:break
        self.assertLess(identities(self.f.adapter.task,later)[1],first_hash)
        result=self.select([self.record(r)for r in [later,self.rows[3],self.rows[2],self.rows[0]]])
        self.assertEqual([m['attempt']for m in result['selected_members']],[0,1,2,3])
        self.assertEqual(result['K1L1']['pairs'][0]['positive']['content_id'],first_hash)

    def test_duplicate_execution_and_content_do_not_fill(self):
        row=copy.deepcopy(self.rows[0]);row['attempt']=1
        records=[self.record(self.rows[0]),self.record(self.rows[0]),self.record(row),self.record(self.rows[2])]
        result=self.select(records)
        self.assertTrue(result['complete_K1']);self.assertFalse(result['complete_K2'])
        self.assertEqual(result['duplicate_executions'],1);self.assertEqual(result['duplicate_contents'],1)
        self.assertEqual(result['supply'][1]['reason'],'duplicate-content')
        self.assertEqual(result['per_arm_task_contribution_units'],dict(K1L1=1,K2L2=0))

    def test_conflicting_same_execution_and_same_content_labels_refuse(self):
        changed=copy.deepcopy(self.rows[0]);changed['classification']='negative'
        for attempt in (0,1):
            changed['attempt']=attempt
            with self.assertRaisesRegex(ValueError,'conflicting'):
                self.select([self.record(self.rows[0]),self.record(changed)])

    def test_missing_indeterminate_incomplete_and_capped_failures_all_reported(self):
        capped=copy.deepcopy(self.rows[1]);capped['turns'][0]['output'].pop()
        records=[self.record(self.rows[0],status='indeterminate'),self.record(capped),
                 self.record(self.rows[2],native_done=False),self.record(None,attempt=3,status='infrastructure')]
        result=self.select(records)
        self.assertFalse(result['complete_K1']);self.assertFalse(result['complete_K2'])
        self.assertEqual([r['reason']for r in result['supply']],
            ['indeterminate','no-terminal-EOS','native-incomplete','infrastructure','not-observed','not-observed','not-observed','not-observed'])
        self.assertEqual(result['observed_unique_attempts'],4)
        self.assertIsNone(result['K1L1']);self.assertIsNone(result['K2L2'])

    def test_grade_reward_claim_consistency_and_nonbinary_refused(self):
        for kwargs in [dict(reward=.5),dict(reward=0),dict(reward=True),dict(classification='negative')]:
            with self.assertRaisesRegex(ValueError,'grade/classification'):
                self.select([self.record(self.rows[0],**kwargs)])

    def test_indeterminate_cannot_masquerade_as_binary_and_attempt_order_frozen(self):
        record=self.record(self.rows[0],status='indeterminate')
        payload=record['evidence']['payload'];payload.update(classification='positive',reward=1,native_done=True)
        record['evidence']['signature']=self.key.sign(canonical(payload)).signature.hex()
        with self.assertRaisesRegex(ValueError,'indeterminate'):
            self.select([record])
        from dataclasses import replace
        task=replace(self.f.adapter.task,approved_attempts=(1,0,2,3,4,5,6,7))
        with self.assertRaisesRegex(ValueError,'ascending'):
            select_nested(task,self.miner,[],eos_token_ids=self.eos,authenticate_admitted_native=self.authenticate,enabled=True)

    def test_tampered_native_grade_and_boolean_auth_refused(self):
        record=self.record(self.rows[0]);record['evidence']['payload']['reward']=0
        with self.assertRaises(Exception):self.select([record])
        with self.assertRaisesRegex(ValueError,'authenticated'):
            select_nested(self.f.adapter.task,self.miner,[self.record(self.rows[0])],eos_token_ids=self.eos,authenticate_admitted_native=lambda *args:True,enabled=True)

    def test_token_trace_conflicting_observations_refused(self):
        row=copy.deepcopy(self.rows[0]);row['attempt']=1;row['turns'][0]['observations'][0]['content']='rewritten'
        with self.assertRaisesRegex(ValueError,'token trace'):
            self.select([self.record(self.rows[0]),self.record(row)])

    def test_false_eos_metadata_and_forged_member_ids_have_no_effect(self):
        row=copy.deepcopy(self.rows[0]);row.update(execution_id='a'*64,content_id='b'*64,uid=85)
        row['turns'][0]['output'].pop();row['turns'][0].update(done=True,eos=True,stop_reason='EOS')
        result=self.select([self.record(row)])
        self.assertEqual(result['supply'][0]['reason'],'no-terminal-EOS');self.assertFalse(result['complete_K1'])
        row=copy.deepcopy(self.rows[0]);row.update(execution_id='a'*64,content_id='b'*64)
        result=self.select([self.record(row),self.record(None,attempt=1,status='infrastructure'),self.record(self.rows[2])])
        self.assertEqual(result['K1L1']['pairs'][0]['positive']['execution_id'],identities(self.f.adapter.task,row)[0])
        self.assertNotEqual(result['K1L1']['pairs'][0]['positive']['execution_id'],'a'*64)

    def test_unapproved_attempt_wrong_task_and_missing_eos_pin_refused(self):
        for change in ('attempt','task'):
            row=copy.deepcopy(self.rows[0])
            if change=='attempt':row['attempt']=8
            else:row['task_sha256']='f'*64
            with self.assertRaises(ValueError):self.select([self.record(row)])
        with self.assertRaises(ValueError):select_nested(self.f.adapter.task,self.miner,[],eos_token_ids=(),authenticate_admitted_native=self.authenticate,enabled=True)

    def test_one_task_four_rollouts_retains_equal_weight_and_records_complete_empty_task(self):
        full=self.select([self.record(r)for r in self.rows]);empty=self.select([])
        self.assertEqual(full['K1L1']['pair_weight_within_task'],1.)
        self.assertEqual(full['K2L2']['pair_weight_within_task'],.5)
        for revision in [full['K1L1'],full['K2L2']]:
            self.assertEqual(sum(revision['pair_weight_within_task']/128 for _ in revision['pairs']),1/128)
        self.assertEqual(len(empty['supply']),len(self.f.adapter.task.approved_attempts));self.assertFalse(empty['complete_K2'])
        self.assertEqual(empty['observed_unique_attempts'],0)

    def test_missing_lower_attempt_blocks_K1_and_K2_without_dropping_supply(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:row['attempt']+=1
        result=self.select([self.record(r)for r in rows])
        self.assertFalse(result['complete_K1']);self.assertFalse(result['complete_K2'])
        self.assertEqual(result['completion_prefix_gaps'],dict(K1L1=[0],K2L2=[0]))
        self.assertIsNone(result['first_K1_prefix_length']);self.assertEqual(result['selected_members'],[])
        self.assertEqual(result['observed_unique_attempts'],4)
        self.assertEqual(result['supply'][0]['reason'],'not-observed')

    def test_missing_between_completed_K1_and_K2_blocks_only_K2(self):
        rows=copy.deepcopy([self.rows[0],self.rows[2],self.rows[1],self.rows[3]])
        for row,attempt in zip(rows,[0,1,3,4]):row['attempt']=attempt
        result=self.select([self.record(r)for r in rows])
        self.assertTrue(result['complete_K1']);self.assertFalse(result['complete_K2'])
        self.assertEqual(result['completion_prefix_gaps'],dict(K1L1=[],K2L2=[2]))
        self.assertEqual(result['first_K1_prefix_length'],2);self.assertIsNone(result['first_K2_prefix_length'])
        self.assertEqual([m['attempt']for m in result['selected_members']],[0,1])

    def test_missing_after_completed_K2_allows_stopped_collection(self):
        result=self.select([self.record(r)for r in self.rows])
        self.assertTrue(result['complete_K2']);self.assertEqual(result['first_K2_prefix_length'],4)
        self.assertEqual(result['completion_prefix_gaps'],dict(K1L1=[],K2L2=[]))
        self.assertTrue(all(r['reason']=='not-observed'for r in result['supply'][4:]))

    def test_authenticated_failed_attempt_in_prefix_is_complete_observation(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:row['attempt']+=1
        records=[self.record(None,attempt=0,status='admission-rejected')]+[self.record(r)for r in rows]
        result=self.select(records)
        self.assertTrue(result['complete_K1']);self.assertTrue(result['complete_K2'])
        self.assertEqual(result['first_K2_prefix_length'],5)
        self.assertEqual(result['supply'][0]['reason'],'admission-rejected')

    def test_default_off_requires_explicit_research_opt_in(self):
        with self.assertRaisesRegex(ValueError,'opt-in'):
            select_nested(self.f.adapter.task,self.miner,[],eos_token_ids=self.eos,authenticate_admitted_native=self.authenticate)

    def test_input_rows_and_evidence_unchanged(self):
        records=[self.record(r)for r in self.rows];before=copy.deepcopy(records);self.select(records);self.assertEqual(records,before)

if __name__=='__main__':unittest.main()
