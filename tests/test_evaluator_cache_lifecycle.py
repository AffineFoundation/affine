import hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from subnet.storage import Identity
from subnet.remote_optimizer_readback import sign
from subnet.evaluator_cache_lifecycle import adopt,retain,VERSION
from subnet.cache_lifecycle import CacheLifecycle
class EvaluatorRetention(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name);self.a=Identity();self.a.sign=lambda x:sign(x,self.a.key)
  self.old='a'*64;self.new='b'*64;self.rows=[]
  for cp in [self.old,self.new]:
   d=self.root/'checkpoints'/cp;d.mkdir(parents=True);(d/'model').write_bytes(cp.encode());files={'model':hashlib.sha256(cp.encode()).hexdigest()};self.rows.append({'relative_path':'checkpoints/'+cp,'checkpoint_document':self.a.sign({'id':cp,'files':files})})
  self.value={'version':VERSION,'created_at':time.time()-1,'expires_at':time.time()+60,'quiescent_readers_confirmed':True,'roots':[{'root':str(self.root),'checkpoints':self.rows,'current_checkpoint':self.new}]}
 def apply(self):return adopt(self.a.sign(self.value),self.a.id)
 def test_verified_catalog_removes_old_keeps_current_and_evidence(self):
  (self.root/'report.json').write_text('keep');r=self.apply();self.assertEqual(r[0]['removed'],[self.old]);self.assertTrue((self.root/'checkpoints'/self.new).exists());self.assertTrue((self.root/'report.json').exists())
 def test_tampered_file_or_signature_never_adopts_or_deletes(self):
  (self.root/'checkpoints'/self.new/'model').write_bytes(b'changed')
  with self.assertRaises(ValueError):self.apply()
  self.assertTrue((self.root/'checkpoints'/self.old).exists());self.assertFalse((self.root/'.cache-lifecycle'/(self.old+'.json')).exists())
 def test_live_original_evaluator_blocks_adoption(self):
  p=self.root/'runner-status';p.mkdir();ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19];(p/'old.json').write_text(json.dumps({'child_pid':os.getpid(),'child_pid_ticks':ticks}))
  with self.assertRaises(ValueError):self.apply()
  self.assertTrue((self.root/'checkpoints'/self.old).exists())
 def test_expired_catalog_rejected(self):
  self.value['expires_at']=0
  with self.assertRaises(ValueError):self.apply()
 def test_inherited_lease_blocks_normal_retention(self):
  c=CacheLifecycle(self.root)
  for cp,row in zip([self.old,self.new],self.rows):
   c.adopt_checkpoint(cp,self.root/row['relative_path'],row['checkpoint_document']['payload']['files'],row['checkpoint_document'])
  with c.lease_checkpoint(self.old):self.assertEqual(retain(self.root,self.new)['removed'],[])
  self.assertEqual(retain(self.root,self.new)['removed'],[self.old])
 def test_unknown_cache_is_preserved(self):
  r=retain(self.root,self.new);self.assertEqual(r['removed'],[]);self.assertTrue((self.root/'checkpoints'/self.old).exists())
if __name__=='__main__':unittest.main()
