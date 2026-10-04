import copy,hashlib,unittest
from pathlib import Path
import test_backend_jobs as fixtures
from subnet.backend_jobs import validate,_validate
from subnet.native_math_grader import dependency_binding

class NativeMathSourcePins(unittest.TestCase):
 def fixture(self,marked=True,pinned=True,legacy=False):
  fixture=fixtures.BackendJobAuthorization('test_honest_signed_verify_policy');fixture.setUp()
  spec={'id':'affine_math','adapter':'prime_v1','config':{'dependency_versions':{'verifiers':'0.3.1',**(dependency_binding() if marked else {})}}}
  if legacy:fixture.manifest['environment']=spec
  else:fixture.manifest['environments']=[{'env_id':'affine_math','spec':spec}]
  fixture.job['manifest']=fixture.sign(fixture.manifest)
  if pinned:fixture.job['source_files']['subnet/native_math_grader.py']=hashlib.sha256(Path('subnet/native_math_grader.py').read_bytes()).hexdigest()
  return fixture
 def test_prospective_missing_helper_pin_rejected_before_imports(self):
  fixture=self.fixture(pinned=False)
  for resolve in (False,True):
   with self.assertRaisesRegex(ValueError,'native MATH grader source pin'):_validate(fixture.sign(fixture.job),fixture.authority,now=50,resolve_source=resolve)
 def test_genuine_helper_pin_authorizes_prospective_registry(self):
  fixture=self.fixture();job,_=validate(fixture.sign(fixture.job),fixture.authority,now=50)
  self.assertIn('subnet/native_math_grader.py',job['source_files'])
 def test_old_contract_requires_no_new_helper(self):
  fixture=self.fixture(marked=False,pinned=False);job,_=validate(fixture.sign(fixture.job),fixture.authority,now=50)
  self.assertNotIn('subnet/native_math_grader.py',job['source_files'])
 def test_single_environment_manifest_also_requires_pin(self):
  fixture=self.fixture(pinned=False,legacy=True)
  with self.assertRaisesRegex(ValueError,'native MATH grader source pin'):validate(fixture.sign(fixture.job),fixture.authority,now=50)

if __name__=='__main__':unittest.main()
