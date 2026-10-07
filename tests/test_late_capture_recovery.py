import base64,copy,tempfile,time,unittest
from unittest.mock import patch
from subnet.storage import Identity
from subnet.commitment_transport import canonical,sha,freeze,FreezeMetadataIncomplete
from subnet import late_capture_recovery as r
from test_commitment_freeze_prefetch import BoundedCommitmentFreeze
from test_capture_journal import commitment_gateway,CaptureJournal

class LateRecovery(unittest.TestCase):
 def fixture(self):
  g,ids,_=BoundedCommitmentFreeze().fixture(3);s=g.epochs['e'];s['commitment_binding']['freeze_until']=25
  key=Identity();m=dict(epoch='e',start=0,deadline=20,source_bundle={'sha256':'b'*64},checkpoint={'id':'a'*64})
  first=self.sign(key,m);p=self.payload(s,'e',first,30,50);return g,key,first,self.sign(key,p)
 def sign(self,key,p):return dict(payload=p,signature=base64.b64encode(key.key.sign(canonical(p)).signature).decode(),signer=key.id)
 def payload(self,s,e,first,start,until):
  b=s['commitment_binding'];return dict(version=r.VERSION,epoch=e,start=s['start'],deadline=s['deadline'],original_freeze_until=b['freeze_until'],source=b['source'],checkpoint=b['checkpoint'],binding_sha256=sha(canonical(b)),miners_sha256=sha(canonical(sorted(s['miners']))),first_signed_manifest_sha256=sha(canonical(first)),operational_start=start,operational_until=until,reason='original-controller-infrastructure-recovery')
 def test_actual_late_capture_does_not_change_original_upload_binding(self):
  g,k,f,d=self.fixture();s=g.epochs['e'];original=copy.deepcopy(s['commitment_binding'])
  with patch.object(r,'AUTHORITY',k.id),patch('time.time',return_value=31):
   r.attach(g,'e',d,f);out=freeze(g,'e');self.assertEqual(len(out),3)
   self.assertEqual(s['commitment_binding'],original);self.assertEqual(s['deadline'],20)
   self.assertEqual(s[r.FIELD],[d]);self.assertEqual(r.cutoff(s,'e'),50)
   self.assertTrue(r.valid_capture_time(s,'e',31));self.assertFalse(r.valid_capture_time(s,'e',26))
 def test_no_authority_no_deadline_extension(self):
  g,k,f,d=self.fixture()
  with patch('time.time',return_value=31):
   with self.assertRaises(FreezeMetadataIncomplete):freeze(g,'e')
  self.assertFalse(g.epochs['e'].get('rejections'))
 def test_wrong_signature_scope_and_original_manifest_reject(self):
  for field,value in [('epoch','other'),('deadline',21),('source','c'*64),('original_freeze_until',26),('operational_until',1000),('binding_sha256','c'*64),('miners_sha256','c'*64)]:
   g,k,f,d=self.fixture();p=dict(d['payload'],**{field:value});d=self.sign(k,p)
   with patch.object(r,'AUTHORITY',k.id),self.assertRaises(ValueError):r.attach(g,'e',d,f,at=31)
  g,k,f,d=self.fixture();d['signature']=base64.b64encode(b'x'*64).decode()
  with patch.object(r,'AUTHORITY',k.id),self.assertRaises(Exception):r.attach(g,'e',d,f,at=31)
  g,k,f,d=self.fixture();f=copy.deepcopy(f);f['payload']['deadline']=21
  with patch.object(r,'AUTHORITY',k.id),self.assertRaises(ValueError):r.attach(g,'e',d,f,at=31)
 def test_future_or_expired_window_cannot_attach_or_launch(self):
  for at in (29,50):
   g,k,f,d=self.fixture()
   with patch.object(r,'AUTHORITY',k.id),self.assertRaises(ValueError):r.attach(g,'e',d,f,at=at)
  g,k,f,d=self.fixture()
  with patch.object(r,'AUTHORITY',k.id):
   r.attach(g,'e',d,f,at=31)
   self.assertEqual(r.cutoff(g.epochs['e'],'e',at=51),25)
 def test_old_pending_discovery_reused_and_changed_mutable_etag_not_read(self):
  g,k,f,d=self.fixture();bucket=g.bucket;miner=sorted(g.epochs['e']['miners'])[0];bucket.fail_copy=lambda key:miner in key
  with patch('time.time',return_value=21):
   with self.assertRaises(RuntimeError):freeze(g,'e')
  pending=copy.deepcopy(g.epochs['e']['commitment_pending']);discovery=copy.deepcopy(g.epochs['e']['commitment_discovery'])
  bucket.fail_copy=None
  with patch.object(r,'AUTHORITY',k.id),patch('time.time',return_value=31):
   r.attach(g,'e',d,f)
   with patch.object(bucket,'get_object',side_effect=AssertionError('no mutable reread')):out=freeze(g,'e')
  self.assertEqual(len(out),3);self.assertEqual({m:{k:v for k,v in row.items()if k in ('etag','document','sha256','size','received_at','key','root')}for m,row in g.epochs['e']['commitment_pending'].items()},{m:{k:v for k,v in row.items()if k in ('etag','document','sha256','size','received_at','key','root')}for m,row in pending.items()});self.assertEqual(g.epochs['e']['commitment_discovery'],discovery)
 def test_late_upload_metadata_rejected_even_under_recovery(self):
  import datetime
  g,k,f,d=self.fixture();old=g.bucket.get_object
  def get(**kw):v=old(**kw);v['LastModified']=datetime.datetime.fromtimestamp(30,datetime.timezone.utc);return v
  g.bucket.get_object=get
  with patch.object(r,'AUTHORITY',k.id),patch('time.time',return_value=31):
   r.attach(g,'e',d,f);self.assertEqual(freeze(g,'e'),{})
  self.assertEqual(set(g.epochs['e']['rejections'].values()),{'commitment time'})
 def test_transport_uncertainty_stays_infrastructure_not_penalty(self):
  g,k,f,d=self.fixture()
  with patch.object(r,'AUTHORITY',k.id),patch('time.time',return_value=31):
   r.attach(g,'e',d,f)
   with patch.object(g.bucket,'get_object',side_effect=TimeoutError('network')):
    with self.assertRaises(FreezeMetadataIncomplete):freeze(g,'e')
  self.assertEqual(g.epochs['e']['rejections'],{})
 def test_installed_authorization_restart_after_expiry_and_next_epoch_is_idempotent(self):
  g,k,f,d=self.fixture();s=g.epochs['e'];writes=[];g.persist=lambda:writes.append(1)
  with patch.object(r,'AUTHORITY',k.id):
   r.attach(g,'e',d,f,at=31);self.assertEqual(len(writes),1)
   r.attach(g,'e',d,f,at=1000);self.assertEqual(len(writes),1)
   g.epochs['next']=copy.deepcopy(s);g.epochs['next'].pop(r.FIELD);g.epochs['next']['commitment_binding']['freeze_until']=1200
   before=copy.deepcopy(g.epochs['next']);r.attach(g,'e',d,f,at=1001)
   self.assertEqual(g.epochs['next'],before);self.assertEqual(r.cutoff(s,'e',at=1001),25)
   self.assertEqual(r.cutoff(g.epochs['next'],'next',at=1001),1200)
 def test_tampered_installed_authorization_cannot_bypass_restart_validation(self):
  g,k,f,d=self.fixture()
  with patch.object(r,'AUTHORITY',k.id):
   r.attach(g,'e',d,f,at=31);g.epochs['e'][r.FIELD][0]['payload']['operational_until']=90000
   with self.assertRaises(Exception):r.attach(g,'e',d,f,at=1000)
 def test_runner_policy_allows_installed_later_phase_but_initial_wrong_epoch_rejects(self):
  import json
  from pathlib import Path
  from ops import durable_learner_service as runner
  from ops import durable_audit_services as guards
  g,k,f,d=self.fixture();state=g.epochs['e']
  with tempfile.TemporaryDirectory()as td:
   root=Path(td);first=root/'first.json';first.write_bytes(canonical(f));auth=root/'authorization.json';auth.write_bytes(canonical(d))
   row=dict(authorization={'path':str(auth),'file_sha256':sha(auth.read_bytes()),'payload_sha256':sha(canonical(d['payload']))},first_signed_manifest={'path':str(first),'file_sha256':sha(first.read_bytes())},epoch='e')
   policy=dict(capture_recovery=row,source_sha256='b'*64,operator_overlay={'overrides':{'subnet/late_capture_recovery.py':'c'*64}})
   cfg={'state':str(root)}
   def save(active):
    (root/'controller.json').write_bytes(canonical({'active':active}));serial=copy.deepcopy(state);serial['miners']=sorted(serial['miners']);(root/'gateway.json').write_bytes(canonical({'epochs':{'e':serial}}))
   save({'epoch':'e','phase':'collect'});runner.validate_capture_recovery(policy,cfg,k.id)
   save({'epoch':'next','phase':'opening'})
   with self.assertRaises(ValueError):runner.validate_capture_recovery(policy,cfg,k.id)
   with patch.object(r,'AUTHORITY',k.id):r.attach(g,'e',d,f,at=31)
   save({'epoch':'next','phase':'opening'});runner.validate_capture_recovery(policy,cfg,k.id)
   save(None);runner.validate_capture_recovery(policy,cfg,k.id)
 def test_durable_token_journal_actual_late_time_replays_after_window(self):
  with tempfile.TemporaryDirectory()as td:
   g,*_=commitment_gateway(td,True);g.bucket.json=lambda key,value:g.bucket.put(key,canonical(value));s=g.epochs['bounded-test'];key=Identity();now=time.time();s['commitment_binding']['freeze_until']=now-1
   first=self.sign(key,dict(epoch='bounded-test',start=s['start'],deadline=s['deadline'],source_bundle={'sha256':s['commitment_binding']['source']},checkpoint={'id':s['commitment_binding']['checkpoint']}))
   # Synthetic original upload window is in the past; retain original stored bytes and metadata.
   s['deadline']=now-2;first=self.sign(key,dict(first['payload'],deadline=s['deadline']))
   doc=self.sign(key,self.payload(s,'bounded-test',first,now,now+30))
   with patch.object(r,'AUTHORITY',key.id):
    r.attach(g,'bounded-test',doc,first,at=now+1)
    with patch('time.time',return_value=now+1):freeze(g,'bounded-test')
    rows=s['training_document_snapshots'];self.assertEqual(sum(map(len,rows.values())),1)
    receipt=next(iter(next(iter(rows.values())).values()));self.assertEqual(receipt['captured_at'],now+1)
    with patch('time.time',return_value=now+40):
     wal=CaptureJournal(g,'bounded-test');wal.replay();wal.close()
    self.assertEqual(s['commitment_binding']['freeze_until'],now-1)
