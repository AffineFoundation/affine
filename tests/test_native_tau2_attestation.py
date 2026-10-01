import copy,hashlib,json,pathlib,sys,tempfile,unittest
from importlib.metadata import version
from nacl.signing import SigningKey
from subnet.native_tau2_model import envelope
from subnet.native_tau2_attestation import PACKAGES,REQUIRED,canonical,require_source_closure,sha

class NativeAttestationTests(unittest.TestCase):
    def fixture(self,path):
        key=SigningKey.generate();plan={'source_hash':'0'*64,'harness_source_hash':'1'*64}
        sources={n:'2'*64 for n in REQUIRED};sources['subnet/native_tau2_model.py']=plan['source_hash'];sources['subnet/harness.py']=plan['harness_source_hash']
        closure={'plan_hash':sha(canonical(plan)),'sources':sources,'package_versions':{n:version(n) for n in PACKAGES},'python_version':sys.version,'interpreter_hash':sha(pathlib.Path(sys.executable).resolve().read_bytes())}
        return key,plan,closure
    def test_wrong_interpreter_fails_before_model_or_untrusted_sources(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);key,plan,closure=self.fixture(p);closure['interpreter_hash']='3'*64
            (p/'operator-source-closure.json').write_text(json.dumps(envelope(closure,key)))
            with self.assertRaisesRegex(ValueError,'interpreter'):require_source_closure(p,plan,key.verify_key.encode().hex())
    def test_wrong_package_version_fails_before_model_or_untrusted_sources(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);key,plan,closure=self.fixture(p);closure['package_versions']['toploc']='unapproved'
            (p/'operator-source-closure.json').write_text(json.dumps(envelope(closure,key)))
            with self.assertRaisesRegex(ValueError,'package'):require_source_closure(p,plan,key.verify_key.encode().hex())

if __name__=='__main__':unittest.main()
