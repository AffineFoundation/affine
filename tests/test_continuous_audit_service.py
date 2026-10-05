import datetime,io,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from botocore.exceptions import ClientError
from nacl.signing import SigningKey
from test_continuous_audit_policy import signed
from subnet.continuous_audit_policy import VERSION,digest
from subnet.continuous_audit_service import ContinuousAuditor,InvalidCommittedArtifact,atomic
class Bucket:
 def __init__(self):self.name='bucket';self.client=self;self.error=None;self.body=b'proof';self.copies=[]
 def head_object(self,**kw):
  if self.error:raise self.error
  return dict(ContentLength=5,LastModified=datetime.datetime.fromtimestamp(15,datetime.timezone.utc),ETag='original-etag')
 def copy(self,key,frozen,expected_etag):self.copies.append((key,frozen,expected_etag))
 def get_object(self,**kw):return {'Body':io.BytesIO(self.body)}
 def presign(self,key):return 'https://private/'+key
class Queue:
 workers={}
 def __init__(self):self.envelopes=[]
 def status(self,j):return {'status':'queued'}
 def enqueue(self,e):self.envelopes.append(e)
 def archive(self,*a):pass
class ServiceControls(unittest.TestCase):
 def setUp(self):
  self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.key=SigningKey.generate();self.root=self.key.verify_key.encode().hex();self.bucket=Bucket();self.queue=Queue()
  self.controller=SimpleNamespace(authority=SimpleNamespace(id=self.root),bucket=self.bucket,signed=lambda p:signed(self.key,p))
  policy=dict(version=VERSION,recent_epochs=4,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.25,zero_epoch_after=3,blacklist_after=4,blacklist_epochs=2)
  self.service=ContinuousAuditor(self.controller,self.queue,directory=self.directory.name,approved_sources={},job_metadata={},audit_policy=policy,max_inflight=1)
  import hashlib
  self.row=dict(epoch='e1',round=1,checkpoint='a'*64,miner='b'*64,env_id='math',index=0,batch_sha256='c'*64,proof_sha256=hashlib.sha256(b'proof').hexdigest(),commitment_sha256='e'*64,verifier_contract_sha256='f'*64,committed_at=20)
  artifact=dict(slot=0,batch_sha256='c'*64,sha256=self.row['proof_sha256'],size=5,key='original',frozen_key='immutable')
  self.p=dict(manifest_document=signed(self.key,dict(epoch='e1',start=10,deadline=20)),receipts={'b'*64:{'artifacts':[artifact]}},records=[self.row])
  self.service.state['populations']['e1']=signed(self.key,self.p)
 def test_hour_boundary_includes_all_new_completed_epochs_once(self):
  self.service.state['populations']['e2']=self.service.state['populations']['e1'];calls=[]
  def snapshot(epoch,round,checkpoint,cutoff):
   calls.append(epoch);return signed(self.key,dict(version=VERSION,epoch=epoch,round=round,checkpoint=checkpoint,cutoff=cutoff,policy=self.service.policy,population_sha256='1'*64,points={'b'*64:1}))
  completed=[dict(epoch='e1',round=1,checkpoint='a'*64,completed_at=1),dict(epoch='e2',round=2,checkpoint='a'*64,completed_at=3600),dict(epoch='old',round=0,checkpoint='a'*64,completed_at=0),dict(epoch='future',round=3,checkpoint='a'*64,completed_at=3601)]
  with patch.object(self.service,'hourly_snapshot',side_effect=snapshot),patch.object(self.service,'publish_immutable'):
   document=self.service.hourly_completed(completed,3600);self.service.hourly_completed(completed,3600)
  self.assertEqual(calls,['e1','e2']);self.assertEqual(document['payload']['points']['b'*64],2.)
 def test_actual_conditional_copy_full_hash_before_capability(self):
  _,_,captured=self.service._capture(self.row,self.p);self.assertEqual(self.bucket.copies,[('original','immutable','original-etag')]);self.assertEqual(captured['read_url'],'https://private/immutable')
 def test_corrupt_full_bytes_cannot_claim_verified(self):
  self.bucket.body=b'wrong'
  with self.assertRaises(InvalidCommittedArtifact):self.service._capture(self.row,self.p)
 def test_deadline_overwrite_rejected(self):
  original=self.bucket.head_object;self.bucket.head_object=lambda **kw:dict(original(**kw),LastModified=datetime.datetime.fromtimestamp(20,datetime.timezone.utc))
  with self.assertRaises(InvalidCommittedArtifact):self.service._capture(self.row,self.p)
  self.assertFalse(self.bucket.copies)
 def test_infra_retries_same_draw_but_missing_is_invalid(self):
  self.bucket.error=ClientError({'Error':{'Code':'503'}},'HeadObject');first=self.service.tick(now=25);draw=dict(self.service.state['draws']);self.service.tick(now=26);self.assertEqual(draw,self.service.state['draws']);self.assertEqual(first['selected'],1);self.assertEqual(next(iter(self.service.state['capture_failures'].values()))['kind'],'infrastructure_error')
  self.bucket.error=ClientError({'Error':{'Code':'NoSuchKey'}},'HeadObject');self.service.tick(now=27);self.assertEqual(next(iter(self.service.state['capture_failures'].values()))['kind'],'confirmed_invalid_artifact');self.assertEqual(self.service.tick(now=28)['retried'],0)
 def test_restart_reuses_original_request_without_mutable_head(self):
  identity=digest(self.row);jobid='continuous-audit-'+identity[:32];self.service.state['draws'][identity]=dict(row=self.row,seed='1'*64,selected_at=25);job=dict(job_id=jobid,created_at=25,expires_at=100,manifest='immutable');envelope=signed(self.key,job);atomic(Path(self.directory.name)/(jobid+'-job.json'),envelope)
  with patch.object(self.service,'_capture',side_effect=AssertionError('must not redraw/rehead')):self.service.tick(now=26)
  self.assertEqual(self.queue.envelopes,[envelope])
 def test_actual_queued_status_enforces_backpressure(self):
  self.service.state['jobs']['job']=dict(row_sha256=digest(self.row));self.assertTrue(self.service.tick(now=25)['backpressure'])
if __name__=='__main__':unittest.main()
