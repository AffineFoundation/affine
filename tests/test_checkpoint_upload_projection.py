import copy, hashlib, json, time, unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.checkpoint_upload_projection import VERSION, authorize, project_manifest, ProjectedUploadJobs
from subnet.storage import canonical
from subnet.distributed_roles import digest
import base64

def sign(key, data):return {'payload':data,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(data)).signature).decode()}

class ProjectionTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.now=time.time()
  self.m={'epoch':'nonpayable-e21','source_bundle':{'sha256':'a'*64},'checkpoint':{'id':'b'*64,'files':{str(i):'c'*64 for i in range(10)}},'training_startup_recovery':sign(self.key,{'version':'terminal-parent-restore-pre-update-bootstrap-recovery-v3','epoch':'nonpayable-e21','padding':'x'*4_050_000})}
  self.job={'role':'upload','job_id':'failed-original','expires_at':self.now+600,'manifest':sign(self.key,self.m),'source_files':{'subnet/f%d.py'%i:'d'*64 for i in range(177)},'runtime_versions':{'torch':'x','transformers':'y','toploc':'z'},'put_urls':{str(i):'original-cap%d'%i for i in range(10)}}
  self.raw=canonical(sign(self.key,self.job));self.projected=project_manifest(self.m)
  self.v={'version':VERSION,'original_upload_file_sha256':hashlib.sha256(self.raw).hexdigest(),'original_job_sha256':digest(self.job),'original_manifest_sha256':digest(self.m),'projected_manifest_sha256':digest(self.projected),'original_label':'nonpayable-e21-publish-bbbbbbbb','replacement_label':'nonpayable-e21-publish-recovery-fresh','source_sha256':'a'*64,'source_files':self.job['source_files'],'runtime_versions':self.job['runtime_versions'],'created_at':self.now-1,'expires_at':self.now+300,'helper_sha256':hashlib.sha256(Path(__import__('subnet.checkpoint_upload_projection',fromlist=['x']).__file__).read_bytes()).hexdigest(),**{k:'e'*64 for k in ('prelaunch_witness_sha256','completed_training_report_sha256','independent_reader_receipt_sha256','independent_reader_terminal_sha256')}}
 def test_only_train_declaration_removed_original_immutable(self):
  before=copy.deepcopy(self.m);v,j,m,p=authorize(sign(self.key,self.v),self.key.verify_key.encode().hex(),self.raw,self.now)
  self.assertEqual(before,self.m);self.assertEqual(set(m)-set(p),{'training_startup_recovery'});self.assertLess(len(canonical(sign(self.key,dict(j,manifest=sign(self.key,p))))),4_000_000)
 def test_mutations_rejected(self):
  for field,value in [('source_sha256','f'*64),('source_files',{}),('runtime_versions',{}),('projected_manifest_sha256','f'*64),('helper_sha256','f'*64),('original_upload_file_sha256','f'*64),('expires_at',self.now+900),('replacement_label',self.v['original_label']),('prelaunch_witness_sha256',None)]:
   with self.subTest(field=field):
    v=dict(self.v);v[field]=value
    with self.assertRaises((ValueError,TypeError)):authorize(sign(self.key,v),self.key.verify_key.encode().hex(),self.raw,self.now)
 def test_delegation_fresh_label_preserves_original_caps_other_roles(self):
  class Jobs:
   def __init__(self):self.calls=[]
   def run(self,*args,**kw):self.calls.append((args,kw));return 'report'
  jobs=Jobs();adapter=ProjectedUploadJobs(jobs,self.v,self.job,self.m,self.projected)
  self.assertEqual(adapter.run(self.v['original_label'],'upload',self.m,'original-output',put_urls={str(i):'regenerated%d'%i for i in range(10)}),'report')
  args,kw=jobs.calls[0];self.assertEqual(args[0],self.v['replacement_label']);self.assertEqual(args[2],self.projected);self.assertEqual(kw['put_urls'],self.job['put_urls'])
  adapter.run('original-training','train',self.m,steps=1);self.assertEqual(jobs.calls[1][0][0],'original-training')
  altered=copy.deepcopy(self.m);altered['checkpoint']['id']='f'*64
  with self.assertRaises(ValueError):adapter.run(self.v['original_label'],'upload',altered,put_urls=self.job['put_urls'])
 def test_null_unknown_recovery_refused(self):
  for value in (None,{}, {'payload':{'version':'foreign'}}):
   m=dict(self.m,training_startup_recovery=value)
   with self.assertRaises(ValueError):project_manifest(m)
