import copy,gzip,hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from ops.live_reward_exporter import sign,epoch_anchor
from ops.live_reward_source_approval import apply_source_approvals,source_verifiers
from subnet.live_reward_bridge import sha
from subnet.source_bootstrap import TASK_ASSET
class ApprovalTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex();self.old='a'*64;self.ids=[format(i,'064x')for i in range(1,5)]
  anchor=dict(version='live-reward-cutover-v1',netuid=120,owner_hotkey='owner',effective_at=100,compute_epoch_prefix='nonpayable-live-reward-math-v1-',live_epoch_prefix='live-math-reward-v1-',cutover_id='ORIGINAL',approved_compute_sources=[self.old]);self.anchor=sign(anchor,self.key)
  self.c=dict(verifier_identities=self.ids[:2],runtime_versions={'torch':'2.14.0','transformers':'5.14.1','toploc':'0.1.6'},approved_sources={},approved_source_anchors={self.old:self.anchor});self.cutover=sign(self.c,self.key)
  data={'subnet/__init__.py':b'', 'subnet/cli.py':b'',TASK_ASSET:b'{}','subnet/backend_jobs.py':b"SOURCE_FILES=('subnet/backend_jobs.py',)\n"};raw=io.BytesIO()
  with tarfile.open(fileobj=raw,mode='w')as tar:
   for n,b in data.items():i=tarfile.TarInfo(n);i.size=len(b);tar.addfile(i,io.BytesIO(b))
  body=gzip.compress(raw.getvalue(),mtime=0);self.digest=hashlib.sha256(body).hexdigest();root=Path(self.t.name);p=root/'source.tar.gz';p.write_bytes(body);d=root/'descriptor.json';d.write_text(json.dumps(sign({'sha256':self.digest,'size':len(body)},self.key)))
  new=copy.deepcopy(anchor);new['approved_compute_sources'].append(self.digest)
  self.payload=dict(version='live-compute-source-approval-v1',original_cutover_sha256=sha(self.cutover),previous_anchor_sha256=sha(self.anchor),effective_at=200,source=dict(sha256=self.digest,archive_path=str(p),descriptor_path=str(d)),runtime_source_files={n:hashlib.sha256(b).hexdigest()for n,b in data.items()if n.endswith('.py')},runtime_versions=self.c['runtime_versions'],verifier_identities=self.ids,anchor_document=sign(new,self.key),epoch_prefix=anchor['compute_epoch_prefix'],registration_policy='all_activated_subnet')
 def apply(self,p=None):return apply_source_approvals(self.c,self.anchor,self.auth,self.cutover,[sign(p or self.payload,self.key)])
 def test_additive_approval_preserves_original_documents_and_worker_scope(self):
  before=copy.deepcopy((self.c,self.anchor,self.cutover));c,a=self.apply();self.assertEqual(before,(self.c,self.anchor,self.cutover));self.assertEqual(c['verifier_identities'],self.ids[:2]);self.assertEqual(c['approved_source_anchors'][self.old],self.anchor);self.assertEqual(epoch_anchor({'source_bundle':{'sha256':self.old}},a,self.auth,c['approved_source_anchors']),self.anchor)
  m={'source_bundle':{'sha256':self.digest},'start':201,'epoch':'nonpayable-live-reward-math-v1-NEW'};j={'source_files':self.payload['runtime_source_files'],'runtime_versions':self.c['runtime_versions']};self.assertEqual(source_verifiers(c,m,j),self.ids);m['source_bundle']['sha256']=self.old;self.assertEqual(source_verifiers(c,m,{}),self.ids[:2])
 def test_tampered_signature_and_wrong_original_authority_refused(self):
  doc=sign(self.payload,self.key);doc['payload']['effective_at']=1
  with self.assertRaises(Exception):apply_source_approvals(self.c,self.anchor,self.auth,self.cutover,[doc])
  p=copy.deepcopy(self.payload);p['original_cutover_sha256']='b'*64
  with self.assertRaisesRegex(ValueError,'writer authority'):self.apply(p)
 def test_anchor_change_removal_and_reorder_refused(self):
  for mutate in [lambda a:a.update(effective_at=0),lambda a:a.update(approved_compute_sources=[self.digest]),lambda a:a.update(approved_compute_sources=[self.digest,self.old])]:
   p=copy.deepcopy(self.payload);a=copy.deepcopy(p['anchor_document']['payload']);mutate(a);p['anchor_document']=sign(a,self.key)
   with self.assertRaises(ValueError):self.apply(p)
 def test_full_inventory_runtime_and_worker_changes_refused(self):
  for field,value in [('runtime_source_files',{}),('runtime_versions',{'torch':'OTHER'}),('verifier_identities',self.ids[1:]+['f'*64]),('registration_policy','allowlist'),('effective_at',True),('previous_anchor_sha256','b'*64)]:
   p=copy.deepcopy(self.payload);p[field]=value
   with self.assertRaises(ValueError):self.apply(p)
 def test_corrupt_archive_refused(self):
  Path(self.payload['source']['archive_path']).write_bytes(b'bad')
  with self.assertRaises(ValueError):self.apply()
 def test_early_opening_wrong_epoch_or_job_pins_refused(self):
  c,_=self.apply();m={'source_bundle':{'sha256':self.digest},'start':199,'epoch':'nonpayable-live-reward-math-v1-new'};j={'source_files':self.payload['runtime_source_files'],'runtime_versions':self.c['runtime_versions']}
  with self.assertRaisesRegex(ValueError,'precedes opening'):source_verifiers(c,m,j)
  m['start']=200;m['epoch']='nonpayable-OLD'
  with self.assertRaises(ValueError):source_verifiers(c,m,j)
  m['epoch']='nonpayable-live-reward-math-v1-new';j['source_files']={}
  with self.assertRaises(ValueError):source_verifiers(c,m,j)
 def test_duplicates_and_unknown_fields_refused(self):
  doc=sign(self.payload,self.key)
  with self.assertRaises(ValueError):apply_source_approvals(self.c,self.anchor,self.auth,self.cutover,[doc,doc])
  p=copy.deepcopy(self.payload);p['source_waiver']=True
  with self.assertRaises(ValueError):self.apply(p)
if __name__=='__main__':unittest.main()
