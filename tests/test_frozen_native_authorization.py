import json
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey
from ops.frozen_native_authorization import install,digest


def signed(payload,key):
    data=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(payload=payload,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(data).signature).decode())


class Frozen(unittest.TestCase):
    def fixture(self,old_changes=None):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        current=dict(version='native',execution_root='/new/operator',grader='same',source_files={'same':'bytes'},no_relabel=True)
        old=dict(current,execution_root='/original/operator',**(old_changes or{}))
        self.current=signed(current,self.key);self.old=signed(old,self.key)
        class Selector:
            def select(instance,manifest,submissions):
                if getattr(instance,'raise_after_check',False):raise RuntimeError('original guard')
                return instance.authorization,submissions
        module=SimpleNamespace(NativeEligibilitySelector=Selector,_load=lambda p:Path(p).read_bytes())
        instance=Selector();instance.controller=SimpleNamespace(state=Path(self.temp.name));instance.authorization=self.current;instance.policy=current
        root=instance.controller.state/'native-outcome-eligibility/epoch-62';root.mkdir(parents=True)
        for name,payload in [('context',dict(authorization_sha256=digest(self.old))),('grades',{'original':True}),('subset',{'original':True})]:
            (root/(name+'.ROOT-SIGNED.json')).write_text(json.dumps(signed(payload,self.key)))
        install(module,[self.old],self.authority)
        return instance,root

    def test_completed_path_only_move_preserves_receipts_and_restores_current_policy(self):
        instance,root=self.fixture();before={p.name:p.read_bytes()for p in root.iterdir()}
        authorization,inputs=instance.select({'epoch':'epoch-62'},['original-input'])
        self.assertEqual(authorization,self.old);self.assertEqual(inputs,['original-input'])
        self.assertEqual(instance.authorization,self.current)
        self.assertEqual(before,{p.name:p.read_bytes()for p in root.iterdir()})
        authorization,_=instance.select({'epoch':'epoch-63'},[]);self.assertEqual(authorization,self.current)

    def test_changed_grader_cannot_reuse_old_authorization(self):
        instance,unused=self.fixture({'grader':'different'})
        with self.assertRaisesRegex(ValueError,'grading semantics'):instance.select({'epoch':'epoch-62'},[])
        self.assertEqual(instance.authorization,self.current)

    def test_partial_journal_and_tampered_signature_are_rejected(self):
        instance,root=self.fixture();(root/'subset.ROOT-SIGNED.json').unlink()
        with self.assertRaisesRegex(ValueError,'completed original'):instance.select({'epoch':'epoch-62'},[])
        (root/'subset.ROOT-SIGNED.json').write_text(json.dumps(self.old));doc=json.loads((root/'grades.ROOT-SIGNED.json').read_text());doc['payload']['original']=False;(root/'grades.ROOT-SIGNED.json').write_text(json.dumps(doc))
        with self.assertRaises(Exception):instance.select({'epoch':'epoch-62'},[])
        self.assertEqual(instance.authorization,self.current)

    def test_original_selector_failure_does_not_leave_historical_policy_installed(self):
        instance,unused=self.fixture();instance.raise_after_check=True
        with self.assertRaisesRegex(RuntimeError,'original guard'):instance.select({'epoch':'epoch-62'},[])
        self.assertEqual(instance.authorization,self.current)


if __name__=='__main__':unittest.main()
