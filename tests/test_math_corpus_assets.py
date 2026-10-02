import copy,gzip,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from subnet.math_corpus import SYSTEM
from subnet.math_corpus_assets import admit_bytes,hydrate
from subnet.math_corpus_provider import VERSION,validate

def fixture():
 raw=json.dumps([{'task_class':'MathTask','task_config':{},'data':{'problem':'1+1?','prompt':'1+1?','answer':'2','system_prompt':SYSTEM}}]).encode();body=gzip.compress(raw,mtime=0);sha=hashlib.sha256(raw).hexdigest()
 binding=dict(version=VERSION,corpus='DeepMath-103K',upstream_revision='5cf055d1fe3d7a2eb19719ac020211469736ae44',catalog_sha256='a'*64,fold='train',shard=0,rows=1,sha256=sha,size=len(raw),compressed_sha256=hashlib.sha256(body).hexdigest(),compressed_size=len(body),path='assets/math-corpora/'+sha+'.tasks.json',provider_sha256=hashlib.sha256(Path(__import__('subnet.math_corpus_provider',fromlist=['']).__file__).read_bytes()).hexdigest())
 spec=SimpleNamespace(id='math_corpus_deepmath103k_train_000',adapter='prime_v1',version=VERSION,num_samples=1,max_turns=1,success_reward=1.0,config={'task_snapshot':binding['path'],'math_corpus_asset':binding})
 return body,binding,spec
class Tests(unittest.TestCase):
 def test_original_alias_and_origin_refusals(self):
  body,b,s=fixture();validate(s)
  for field,value in [('version','bad'),('upstream_revision','0'*40),('fold','heldout'),('rows',True),('provider_sha256','0'*64),('path','../secret')]:
   changed=copy.deepcopy(s);changed.config['math_corpus_asset'][field]=value
   with self.assertRaises(ValueError):validate(changed)
  s.version='bad'
  with self.assertRaises(ValueError):validate(s)
 def test_corrupt_compressed_and_raw_membership(self):
  body,b,s=fixture()
  with self.assertRaises(ValueError):admit_bytes(body+b'x',b)
  raw=json.dumps([{'task_class':'Other','task_config':{},'data':{}}]).encode();bad=gzip.compress(raw,mtime=0);changed=dict(b,size=len(raw),sha256=hashlib.sha256(raw).hexdigest(),compressed_size=len(bad),compressed_sha256=hashlib.sha256(bad).hexdigest());changed['path']='assets/math-corpora/'+changed['sha256']+'.tasks.json'
  with self.assertRaises(ValueError):admit_bytes(bad,changed)
 def test_hydration_exact_atomic_and_no_redownload(self):
  body,b,s=fixture();calls=[]
  def fetch(url,limit):calls.append(limit);return body
  with tempfile.TemporaryDirectory() as root:
   path=hydrate(root,b,'https://a.r2.cloudflarestorage.com/x?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=a',fetch);validate(s,root)
   hydrate(root,b,'unused',fetch);self.assertEqual(calls,[len(body)])
   path.chmod(0o600);path.write_bytes(b'changed')
   with self.assertRaises(ValueError):hydrate(root,b,'unused',fetch)
 def test_symlink_and_oversize_before_network(self):
  body,b,s=fixture()
  with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as target:
   (Path(root)/'assets').symlink_to(target,target_is_directory=True)
   with self.assertRaises(ValueError):hydrate(root,b,'unused',lambda *x:self.fail('network'))
  b['size']=33*1024**2
  with self.assertRaises(ValueError):admit_bytes(body,b)
if __name__=='__main__':unittest.main()

class ExternalCacheTests(unittest.TestCase):
 def test_external_root_exact_bytes_and_canonical_signed_path(self):
  from unittest.mock import patch
  from subnet.math_corpus_provider import asset_path
  from subnet.environments import _snapshot_path
  body,b,s=fixture()
  with tempfile.TemporaryDirectory() as source,tempfile.TemporaryDirectory() as cache:
   with patch.dict('os.environ',{'AFFINE_MATH_CORPUS_ASSET_ROOT':cache}):
    path=Path(cache)/b['path'];path.parent.mkdir(parents=True);path.write_bytes(admit_bytes(body,b));validate(s,source)
    self.assertEqual(asset_path(b,source),path);self.assertEqual(_snapshot_path(s.config),path);self.assertEqual(s.config['task_snapshot'],b['path'])
    path.write_bytes(b'forged')
    with self.assertRaises(ValueError):validate(s,source)
 def test_external_alias_root_refused(self):
  from unittest.mock import patch
  from subnet.math_corpus_provider import asset_path
  body,b,s=fixture()
  with tempfile.TemporaryDirectory() as parent,tempfile.TemporaryDirectory() as target:
   alias=Path(parent)/'alias';alias.symlink_to(target,target_is_directory=True)
   with patch.dict('os.environ',{'AFFINE_MATH_CORPUS_ASSET_ROOT':str(alias)}):
    with self.assertRaises(ValueError):asset_path(b,target)
   with patch.dict('os.environ',{'AFFINE_MATH_CORPUS_ASSET_ROOT':'relative'}):
    with self.assertRaises(ValueError):asset_path(b,target)
