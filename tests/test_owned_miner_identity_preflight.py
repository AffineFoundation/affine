import os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from nacl.signing import SigningKey
from subnet.backend_jobs import owned_miner_identity,execute
class MinerIdentityPreflight(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'miner.seed';self.key=SigningKey.generate();self.job={'role':'mine','source_files':{},'runtime_versions':{},'miner_identity_file':str(self.path),'miner_id':self.key.verify_key.encode().hex()};self.manifest={'submission_transport_policy':'small-commitment-pairs-v2'}
 def write(self,mode=0o600):self.path.write_text(self.key.encode().hex());self.path.chmod(mode)
 def test_real_valid_private_seed_binds_public_identity(self):
  self.write();self.assertEqual(owned_miner_identity(self.job).id,self.job['miner_id'])
 def test_missing_identity_fails_before_runtime_factory_and_assets(self):
  factory=Mock();hydrate=Mock()
  with patch('subnet.backend_jobs._validate',return_value=(self.job,self.manifest)),patch('subnet.backend_jobs.install_source_loader'),patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch('subnet.task_assets.hydrate_manifest',hydrate):
   with self.assertRaises(FileNotFoundError):execute({},'operator',self.tmp.name,runtime_factory=factory)
  factory.assert_not_called();hydrate.assert_not_called()
 def test_public_seed_mode_rejected(self):
  self.write(0o644)
  with self.assertRaises(ValueError):owned_miner_identity(self.job)
 def test_wrong_public_signer_rejected(self):
  self.write();self.job['miner_id']='0'*64
  with self.assertRaisesRegex(ValueError,'signer binding'):owned_miner_identity(self.job)
 def test_symlink_rejected(self):
  other=Path(self.tmp.name)/'real.seed';other.write_text(self.key.encode().hex());other.chmod(0o600);self.path.symlink_to(other)
  with self.assertRaises(ValueError):owned_miner_identity(self.job)
if __name__=='__main__':unittest.main()
