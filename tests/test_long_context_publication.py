import base64,hashlib,json,pathlib,subprocess,sys,tempfile,unittest
from nacl.signing import SigningKey
from ops.publish_long_context_checkpoint import REMOTE
from subnet.long_context_runtime import canonical,digest

class PublicationTests(unittest.TestCase):
 def test_untrusted_signature_precedes_checkpoint_reads(self):
  value={'signer':'d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f','payload':{'checkpoint':{'path':'/untrusted-missing'}},'signature':base64.b64encode(bytes(64)).decode()}
  p=subprocess.run([sys.executable,'-c',REMOTE],input=canonical(value),capture_output=True)
  self.assertNotEqual(p.returncode,0);self.assertIn(b'BadSignatureError',p.stderr);self.assertNotIn(b'FileNotFoundError',p.stderr)
 def test_valid_delegation_checks_remote_bytes_even_when_object_exists(self):
  key=SigningKey(bytes(32));authority=key.verify_key.encode().hex()
  code=REMOTE.replace('d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f',authority)
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'model.safetensors';path.write_bytes(b'approved')
   files={path.name:hashlib.sha256(b'approved').hexdigest()}
   payload={'role':'long-context-exact-checkpoint-upload-v1','checkpoint':{'path':d,'id':digest(files),'files':files},'capabilities':{path.name:None},'payable':False,'chain_transactions':False}
   def run():
    envelope={'payload':payload,'signer':authority,'signature':base64.b64encode(key.sign(canonical(payload)).signature).decode()}
    return subprocess.run([sys.executable,'-c',code],input=canonical(envelope),capture_output=True)
   p=run();self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['files'][path.name]['size'],8)
   path.write_bytes(b'mutated');p=run();self.assertNotEqual(p.returncode,0);self.assertIn(b'remote checkpoint hash',p.stderr)
