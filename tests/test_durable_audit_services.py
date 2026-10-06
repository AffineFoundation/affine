import base64,copy,hashlib,json,os,pathlib,sqlite3,tempfile,unittest
from nacl.signing import SigningKey
from ops import durable_audit_services as m

class DurableServices(unittest.TestCase):
 def setUp(self):
  previous=os.umask(0o077);self.addCleanup(os.umask,previous);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name);self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex()
  self.state=self.root/'state';self.state.mkdir();self.seed=self.state/'authority.seed';self.seed.write_text(self.key.encode().hex());self.seed.chmod(0o600)
  self.db=self.state/'queue.sqlite3';db=sqlite3.connect(self.db);db.execute('pragma journal_mode=wal');db.execute('create table jobs(id text primary key,payload text)');db.execute('insert into jobs values (?,?)',('original','signed-body'));db.commit();schema=m.digest(m.queue_schema(db));db.close()
  self.runtime=self.root/'runtime';(self.runtime/'subnet').mkdir(parents=True)
  files={}
  for name in ('__init__','backend_jobs','remote_backend'):
   p=self.runtime/'subnet'/f'{name}.py';p.write_text('# unchanged\n');files[f'subnet/{name}.py']=m.file_hash(p)
  self.op=self.root/'operator';self.op.mkdir();(self.op/'unchanged.py').write_text('# pinned\n')
  self.config=self.root/'config.json';self.config.write_text(json.dumps({'state':str(self.state)}))
  self.old={'sources':{str(i)*64:{'runtime_files':files,'runtime_versions':{},'sampling_versions':[]}for i in range(8)}};self.new=copy.deepcopy(self.old);self.source='f'*64;self.new['sources'][self.source]=copy.deepcopy(next(iter(self.old['sources'].values())))
  self.oldrow=self.document('old.json',self.old);self.newrow=self.document('new.json',self.new)
  s=self.db.stat();self.p={'version':m.VERSION,'kind':'API','execute_allowed':True,'authority':self.auth,'identity':{'uid':os.getuid(),'machine_id_sha256':hashlib.sha256(pathlib.Path('/etc/machine-id').read_bytes()).hexdigest()},'config':{'path':str(self.config),'file_sha256':m.file_hash(self.config)},'queue':{'path':str(self.db),'inode':[s.st_dev,s.st_ino],'schema_sha256':schema},'singleton_lock':str(self.root/'lock'),'excluded_units':['original-api.service'],'runner_file_sha256':m.file_hash(pathlib.Path(m.__file__).resolve()),'operator':{'root':str(self.op),'files':{'unchanged.py':m.file_hash(self.op/'unchanged.py')},'retry_helper':None,'overlay':None},'admission':self.newrow,'historical_admission':self.oldrow,'source':self.source,'source_trees':{k:str(self.runtime)for k in self.new['sources']},'API_dependency':None,'authority_seed':{'path':str(self.seed),'file_sha256':m.file_hash(self.seed)}}
 def sign(self,p):return {'payload':p,'signer':self.auth,'signature':base64.b64encode(self.key.sign(m.canonical(p)).signature).decode()}
 def document(self,name,p):
  f=self.root/name;f.write_text(json.dumps(self.sign(p)));return {'path':str(f),'file_sha256':m.file_hash(f),'payload_sha256':m.digest(p)}
 def validate(self,p=None):return m.validate_policy(self.sign(p or self.p),self.auth)
 def test_real_sqlite_restart_and_no_deadline(self):
  self.validate();self.validate();db=sqlite3.connect(self.db);self.assertEqual(db.execute('select * from jobs').fetchall(),[('original','signed-body')]);db.close()
 def test_config_runtime_and_runner_drift(self):
  for key in ('config','runner_file_sha256'):
   p=copy.deepcopy(self.p)
   if key=='config':p[key]['file_sha256']='0'*64
   else:p[key]='0'*64
   with self.assertRaises(ValueError):self.validate(p)
  (self.op/'unchanged.py').write_text('tampered')
  with self.assertRaises(ValueError):self.validate()
 def test_historical_rewrite_and_source_omission(self):
  for new in ({'sources':{self.source: self.new['sources'][self.source]}},dict(self.new,sources=dict(self.new['sources'],**{'0'*64:{}}))):
   p=copy.deepcopy(self.p);p['admission']=self.document('bad.json',new)
   with self.assertRaises(ValueError):self.validate(p)
 def test_queue_inode_and_schema(self):
  p=copy.deepcopy(self.p);p['queue']['inode'][1]+=1
  with self.assertRaises(ValueError):self.validate(p)
  db=sqlite3.connect(self.db);db.execute('create table foreign_table(x)');db.close()
  with self.assertRaises(ValueError):self.validate()
 def test_authority_and_seed_identity(self):
  doc=self.sign(self.p);doc['signer']='0'*64
  with self.assertRaises(ValueError):m.validate_policy(doc,self.auth)
  self.seed.write_text(SigningKey.generate().encode().hex());p=copy.deepcopy(self.p);p['authority_seed']['file_sha256']=m.file_hash(self.seed)
  with self.assertRaises(ValueError):self.validate(p)
 def test_full_source_inventory_and_forged_registry(self):
  (self.runtime/'subnet'/'new.py').write_text('foreign')
  with self.assertRaises(ValueError):self.validate()
  doc=self.sign(self.p);doc['signature']=self.key.sign(m.canonical(self.p)).signature.hex()
  with self.assertRaises(Exception):m.validate_policy(doc,self.auth)
 def test_singleton_and_symlink(self):
  with m.singleton(self.p['singleton_lock']):
   with self.assertRaises(BlockingIOError):
    with m.singleton(self.p['singleton_lock']):pass
  pathlib.Path(self.p['singleton_lock']).unlink();pathlib.Path(self.p['singleton_lock']).symlink_to(self.seed)
  with self.assertRaises(OSError):
   with m.singleton(self.p['singleton_lock']):pass
 def test_expiry_identity_schema_and_execution_refused(self):
  for change in ({'expires_at':1},{'execute_allowed':False},{'identity':dict(self.p['identity'],uid=-1)}):
   p=copy.deepcopy(self.p);p.update(change)
   with self.assertRaises(ValueError):self.validate(p)
 def test_auditor_overlay_and_historical_cutoff(self):
  old={'approved_sources':{str(i)*64:{'old':i}for i in range(8)},'job_metadata':{str(i)*64:{'old':i}for i in range(8)},'execution_evidence_policy':{'effective_cutoff':10,'sources':{str(i)*64:{'cutoff':i}for i in range(8)}}};new=copy.deepcopy(old)
  for field in ('approved_sources','job_metadata'):new[field][self.source]={'new':True}
  new['execution_evidence_policy']['sources'][self.source]={'cutoff':20}
  apirow=self.document('api-policy.json',self.p);p=copy.deepcopy(self.p);p['API_dependency']={'unit':'durable-api.service','policy_path':apirow['path'],'policy_file_sha256':apirow['file_sha256']};p['kind']='auditor';p['historical_admission']=self.document('audit-old.json',old);p['admission']=self.document('audit-new.json',new);p['source_trees']={};cfg={'state':str(self.state),'continuous_audit_service':{'source_admission':m.read(p['admission']['path'])}};self.config.write_text(json.dumps(cfg));p['config']['file_sha256']=m.file_hash(self.config)
  helper=self.root/'retry.py';helper.write_text('# immutable BEGIN/status retry\n');p['operator']['retry_helper']={'path':str(helper),'file_sha256':m.file_hash(helper)};p['operator']['overlay']={'root':str(self.op),'files':dict(p['operator']['files'])};self.validate(p)
  helper.write_text('drift')
  with self.assertRaises(ValueError):self.validate(p)
  helper.write_text('# immutable BEGIN/status retry\n');new['execution_evidence_policy']['sources']['0'*64]['cutoff']=999;p['admission']=self.document('audit-bad.json',new);cfg['continuous_audit_service']['source_admission']=m.read(p['admission']['path']);self.config.write_text(json.dumps(cfg));p['config']['file_sha256']=m.file_hash(self.config)
  with self.assertRaises(ValueError):self.validate(p)

if __name__=='__main__':unittest.main()
