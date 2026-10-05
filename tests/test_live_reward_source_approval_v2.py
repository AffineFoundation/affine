"""Actual signed v1→v2 source chains, immutable history and explicit retirements."""
import copy,hashlib,json,unittest
from pathlib import Path
import test_live_reward_source_approval as source_fixture
from ops.live_reward_exporter import sign,epoch_anchor
from ops.live_reward_source_approval import apply_source_approvals,source_verifiers
from subnet.live_reward_bridge import sha

class ReplacementApprovalTests(unittest.TestCase):
 source_chain=source_fixture.ApprovalTests.source_chain
 def setUp(self):
  source_fixture.ApprovalTests.setUp(self)
  self.c['verifier_identities']=self.ids[:]
  self.cutover=sign(self.c,self.key)
  self.payload['original_cutover_sha256']=sha(self.cutover)
 def replacement_chain(self,rosters=None):
  five=self.ids+[format(5,'064x')]
  rosters=rosters or [five,self.ids+[format(6,'064x'),format(7,'064x')]]
  docs=self.source_chain(rosters)
  previous_ids=set(rosters[0])
  for i in range(1,len(docs)):
   p=copy.deepcopy(docs[i]['payload']);previous=docs[i-1]
   p.update(version='live-compute-source-approval-v2',previous_authorization_sha256=sha(previous),retirements=[])
   for identity in sorted(previous_ids-set(rosters[i])):
    evidence=Path(self.t.name)/('retirement-'+str(i)+'-'+identity+'.json')
    evidence.write_text(json.dumps(dict(operator_observed='deleted node and unavailable key',identity=identity)))
    retirement=dict(version='live-compute-verifier-retirement-v1',original_cutover_sha256=sha(self.cutover),previous_anchor_sha256=sha(previous['payload']['anchor_document']),previous_authorization_sha256=sha(previous),previous_source_sha256=previous['payload']['source']['sha256'],verifier_identity=identity,effective_at=p['effective_at'],reason='signing_key_unavailable',evidence_path=str(evidence),evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    p['retirements'].append(sign(retirement,self.key))
   docs[i]=sign(p,self.key);previous_ids=set(rosters[i])
  return docs
 def apply_chain(self,docs):return apply_source_approvals(self.c,self.anchor,self.auth,self.cutover,docs)
 def test_deleted_fifth_key_replaced_by_two_live_keys_and_historical_scopes_preserved(self):
  docs=self.replacement_chain();before=copy.deepcopy((docs,self.c,self.anchor,self.cutover))
  c,anchor=self.apply_chain(docs)
  self.assertEqual(before,(docs,self.c,self.anchor,self.cutover))
  for d in docs:
   p=d['payload'];m=dict(source_bundle={'sha256':p['source']['sha256']},start=999,epoch=p['epoch_prefix']+'NEW');j=dict(source_files=p['runtime_source_files'],runtime_versions=p['runtime_versions'])
   self.assertEqual(source_verifiers(c,m,j),p['verifier_identities'])
  self.assertEqual(source_verifiers(c,{'source_bundle':{'sha256':self.old}},{}),self.ids)
  self.assertEqual(c['approved_source_anchors'][self.old],self.anchor)
  self.assertEqual(epoch_anchor({'source_bundle':{'sha256':self.old}},anchor,self.auth,c['approved_source_anchors']),self.anchor)
  self.assertEqual(c['verifier_identities'],self.ids)
 def test_v2_can_retire_original_identity_only_with_explicit_evidence_for_future_source(self):
  five=self.ids+[format(5,'064x')];new=self.ids[1:]+[format(i,'064x')for i in (5,6)]
  docs=self.replacement_chain([five,new]);c,_=self.apply_chain(docs)
  self.assertEqual(source_verifiers(c,{'source_bundle':{'sha256':self.old}},{}),self.ids)
  self.assertEqual(c['_source_authorizations'][docs[-1]['payload']['source']['sha256']]['verifier_identities'],new)
 def test_removed_identity_missing_retirement_is_rejected(self):
  docs=self.replacement_chain();p=copy.deepcopy(docs[-1]['payload']);p['retirements']=[];docs[-1]=sign(p,self.key)
  with self.assertRaisesRegex(ValueError,'every removed identity'):self.apply_chain(docs)
 def test_retirement_must_be_root_signed_not_unsigned_or_other_key(self):
  from nacl.signing import SigningKey
  for kind in ('unsigned','different-authority'):
   docs=self.replacement_chain();p=copy.deepcopy(docs[-1]['payload']);r=p['retirements'][0]['payload'];p['retirements'][0]={'payload':r}if kind=='unsigned'else sign(r,SigningKey.generate());docs[-1]=sign(p,self.key)
   with self.assertRaises(Exception):self.apply_chain(docs)
 def test_retirement_wrong_binding_time_identity_reason_and_evidence_refused(self):
  changes={'original_cutover_sha256':'b'*64,'previous_anchor_sha256':'b'*64,'previous_authorization_sha256':'b'*64,'previous_source_sha256':'b'*64,'verifier_identity':self.ids[0],'effective_at':True,'reason':'timeout','evidence_sha256':'b'*64}
  for field,value in changes.items():
   with self.subTest(field=field):
    docs=self.replacement_chain();p=copy.deepcopy(docs[-1]['payload']);r=copy.deepcopy(p['retirements'][0]['payload']);r[field]=value;p['retirements'][0]=sign(r,self.key);docs[-1]=sign(p,self.key)
    with self.assertRaises(ValueError):self.apply_chain(docs)
 def test_corrupt_or_symlink_retirement_record_refused(self):
  for mode in ('corrupt','symlink'):
   docs=self.replacement_chain();p=copy.deepcopy(docs[-1]['payload']);r=copy.deepcopy(p['retirements'][0]['payload']);path=Path(r['evidence_path'])
   if mode=='corrupt':path.write_bytes(b'changed')
   else:
    alias=path.with_suffix('.alias');alias.symlink_to(path);r['evidence_path']=str(alias);p['retirements'][0]=sign(r,self.key);docs[-1]=sign(p,self.key)
   with self.assertRaises(ValueError):self.apply_chain(docs)
 def test_surviving_duplicate_or_unnecessary_retirement_refused(self):
  for mode in ('duplicate','survivor','unknown'):
   docs=self.replacement_chain();p=copy.deepcopy(docs[-1]['payload'])
   if mode=='duplicate':p['retirements']*=2
   else:
    r=copy.deepcopy(p['retirements'][0]['payload']);r['verifier_identity']=self.ids[0]if mode=='survivor'else 'f'*64;p['retirements'].append(sign(r,self.key))
   docs[-1]=sign(p,self.key)
   with self.assertRaisesRegex(ValueError,'distinct removed'):self.apply_chain(docs)
 def test_new_roster_bounds_duplicates_and_unordered_approval_refused(self):
  for field,value in [('verifier_identities',self.ids[:3]),('verifier_identities',self.ids+[format(i,'064x')for i in (6,7,8)]),('verifier_identities',self.ids+[self.ids[0]]),('previous_authorization_sha256','b'*64),('effective_at',199)]:
   docs=self.replacement_chain();p=copy.deepcopy(docs[-1]['payload']);p[field]=value;docs[-1]=sign(p,self.key)
   with self.assertRaises(ValueError):self.apply_chain(docs)
 def test_retired_identity_cannot_silently_reappear_on_later_source(self):
  rosters=[self.ids+[format(5,'064x')],self.ids+[format(6,'064x'),format(7,'064x')],self.ids+[format(5,'064x'),format(7,'064x')]]
  with self.assertRaisesRegex(ValueError,'cannot silently return'):self.apply_chain(self.replacement_chain(rosters))
 def test_v1_after_v2_is_refused_because_v1_cannot_express_retirement(self):
  docs=self.replacement_chain([self.ids+[format(5,'064x')],self.ids+[format(6,'064x')],self.ids+[format(6,'064x')]])
  p=copy.deepcopy(docs[-1]['payload']);p.update(version='live-compute-source-approval-v1');p.pop('retirements');p.pop('previous_authorization_sha256');docs[-1]=sign(p,self.key)
  with self.assertRaisesRegex(ValueError,'v1 cannot follow'):self.apply_chain(docs)
 def test_new_source_effective_time_and_exact_job_pins_remain_enforced(self):
  docs=self.replacement_chain();c,_=self.apply_chain(docs);p=docs[-1]['payload'];m=dict(source_bundle={'sha256':p['source']['sha256']},start=p['effective_at']-1,epoch=p['epoch_prefix']+'NEW');j=dict(source_files=p['runtime_source_files'],runtime_versions=p['runtime_versions'])
  with self.assertRaisesRegex(ValueError,'precedes opening'):source_verifiers(c,m,j)
  m['start']=p['effective_at'];j['source_files']={}
  with self.assertRaisesRegex(ValueError,'exact new-source'):source_verifiers(c,m,j)

 def test_actual_writer_worker_authorization_uses_correct_historical_source_roster(self):
  from ops.verifier_workforce import authorize_worker
  docs=self.replacement_chain();c,_=self.apply_chain(docs)
  removed=format(5,'064x');replacement=format(6,'064x')
  for ordinal,document in enumerate(docs):
   p=document['payload'];m=dict(source_bundle={'sha256':p['source']['sha256']},start=p['effective_at'],epoch=p['epoch_prefix']+'NEW');j=dict(source_files=p['runtime_source_files'],runtime_versions=p['runtime_versions'])
   existing=source_verifiers(c,m,j)
   allowed=removed if ordinal==0 else replacement
   self.assertIsNone(authorize_worker(allowed,m,j,{},None,existing,{}))
   denied=replacement if ordinal==0 else removed
   with self.assertRaisesRegex(ValueError,'approved verifier identity'):authorize_worker(denied,m,j,{},None,existing,{})

 def test_first_v2_binds_original_cutover_and_cannot_predate_it(self):
  p=copy.deepcopy(self.payload);p.update(version='live-compute-source-approval-v2',previous_authorization_sha256=sha(self.cutover),retirements=[])
  c,_=self.apply_chain([sign(p,self.key)])
  self.assertEqual(c['_source_authorizations'][p['source']['sha256']]['verifier_identities'],self.ids)
  p['effective_at']=99
  with self.assertRaisesRegex(ValueError,'prospective replacement'):self.apply_chain([sign(p,self.key)])
