"""Real hash/atomic transfer with bounded simulated HTTP; no model/GPU."""
import hashlib,json,os,pathlib,tempfile,threading,time,unittest
from unittest.mock import patch
from subnet import backend_jobs as b
from subnet.cache_lifecycle import CacheLifecycle

class Response:
 def __init__(self,data,owner):self.data=data;self.owner=owner;self.status_code=200
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def iter_content(self,size):
  with self.owner.lock:self.owner.active+=1;self.owner.maximum=max(self.owner.maximum,self.owner.active)
  try:
   time.sleep(.035)
   yield self.data
  finally:
   with self.owner.lock:self.owner.active-=1
class Client:
 def __init__(self,owner):self.owner=owner
 def __enter__(self):return self
 def __exit__(self,*a):self.owner.closed+=1
 def get(self,url,**kw):self.owner.calls.append(url);return Response(self.owner.data[url.rsplit('/',1)[1]],self.owner)
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name);self.lock=threading.Lock();self.active=self.maximum=self.closed=0;self.calls=[]
  self.data={f'file{i}.json':bytes([i+1])*8192 for i in range(6)};self.files={n:hashlib.sha256(v).hexdigest()for n,v in self.data.items()};self.cp={'id':'1'*64,'files':self.files,'read_urls':{n:'https://approved.example/'+n for n in self.files}}
  self.manifest={'checkpoint':self.cp,'checkpoint_download_policy':{'version':'bounded-parallel-checkpoint-GET-v1','workers':3,'file_sizes':{n:len(v)for n,v in self.data.items()},'disk_floor_bytes':2*1024**3}}
  self.target=self.root/'checkpoints'/self.cp['id'];self.target.mkdir(parents=True);self.life=CacheLifecycle(self.root)
  self.addCleanup(patch.stopall);patch('requests.Session',side_effect=lambda:Client(self)).start();patch.object(b,'r2_url',side_effect=lambda u,m:u).start()
 def run_fetch(self):b._parallel_checkpoint(self.cp,self.target,self.root,b.parallel_checkpoint_policy(self.manifest),self.life)
 def test_bounded_parallel_real_SHA_atomic_journal_and_reuse(self):
  with self.life.lease_checkpoint(self.cp['id']):self.run_fetch()
  self.assertEqual(self.maximum,3);self.assertEqual(self.closed,6);self.assertEqual({n:b.digest(self.target/n)for n in self.files},self.files);self.assertFalse(list(self.target.glob('*.partial')))
  receipt=json.loads(self.life._receipt(self.cp['id']).read_bytes());self.assertEqual(set(receipt['files']),set(self.files));calls=len(self.calls)
  with self.life.lease_checkpoint(self.cp['id']):self.run_fetch()
  self.assertEqual(len(self.calls),calls)
 def test_same_checkpoint_concurrent_hydration_GETs_once(self):
  errors=[]
  def go():
   try:self.run_fetch()
   except Exception as e:errors.append(e)
  threads=[threading.Thread(target=go)for _ in range(2)]
  for t in threads:t.start()
  for t in threads:t.join()
  self.assertEqual(errors,[]);self.assertEqual(len(self.calls),6)
 def test_digest_failure_never_admitted_partial_removed_completed_members_owned(self):
  self.data['file0.json']=b'forged'
  with self.assertRaises(b.ArtifactRejected):self.run_fetch()
  self.assertFalse((self.target/'file0.json').exists());self.assertFalse(list(self.target.glob('*.partial')))
  if self.life._receipt(self.cp['id']).exists():
   receipt=json.loads(self.life._receipt(self.cp['id']).read_bytes());self.assertNotIn('file0.json',receipt['members'])
 def test_signed_sizes_disk_floor_and_bool_workers_fail_closed(self):
  self.manifest['checkpoint_download_policy']['workers']=True
  with self.assertRaises(ValueError):b.parallel_checkpoint_policy(self.manifest)
  self.manifest['checkpoint_download_policy']['workers']=3
  with patch('shutil.disk_usage',return_value=type('Usage',(),{'free':0})()):
   with self.assertRaises(ValueError):self.run_fetch()
  self.assertEqual(self.calls,[])
 def test_foreign_symlink_hardlink_partial_and_changed_existing_refused(self):
  name=next(iter(self.files));foreign=self.root/'foreign';foreign.write_bytes(self.data[name]);p=self.target/name;p.symlink_to(foreign)
  with self.assertRaises(ValueError):self.run_fetch()
  p.unlink();os.link(foreign,p)
  with self.assertRaises(ValueError):self.run_fetch()
  p.unlink();p.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.run_fetch()
  p.unlink();partial=p.with_suffix(p.suffix+'.partial');partial.symlink_to(foreign)
  with self.assertRaises(ValueError):self.run_fetch()
  self.assertTrue(partial.is_symlink());self.assertEqual(foreign.read_bytes(),self.data[name])
 def test_absent_option_preserves_serial_checkpoint_behavior(self):
  self.assertIsNone(b.parallel_checkpoint_policy({'checkpoint':self.cp}))
  with patch.object(b,'get_object',side_effect=lambda u,s,p,l:p.write_bytes(self.data[p.name]))as get,patch('subnet.model.model_files',return_value=self.files):
   self.assertEqual(b.checkpoint({'checkpoint':self.cp},self.root),self.target)
  self.assertEqual(get.call_count,6)
 def test_active_outer_lease_prevents_retirement(self):
  with self.life.lease_checkpoint(self.cp['id']):
   self.run_fetch();self.life.evict_checkpoints(keep=0)
   self.assertTrue((self.target/next(iter(self.files))).exists())
if __name__=='__main__':unittest.main()
