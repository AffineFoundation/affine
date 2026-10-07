import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from ops import durable_learner_service as runner
from ops.native_training_eligibility import FutureNativeEligibilitySelector
from ops.native_training_outcome_filter import AUTHORIZATION_VERSION
from ops.native_training_lifecycle import LIFECYCLE_POLICY

class NativeRunner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'scientific';self.overlay=self.root/'cpu'
        self.operator=self.root/'operator';self.state=self.root/'state'
        for path in (self.source,self.overlay,self.operator,self.state):path.mkdir()
        (self.overlay/'subnet').mkdir();(self.overlay/'subnet/__init__.py').write_text('')
        (self.overlay/'subnet/distributed_roles.py').write_text('from nacl.signing import VerifyKey\nimport json,base64\ndef authenticate(e,a):\n VerifyKey(bytes.fromhex(a)).verify(json.dumps(e["payload"],sort_keys=True,separators=(",",":")).encode(),base64.b64decode(e["signature"]))\n return e["payload"]\n')
        (self.overlay/'subnet/committed_training_inputs.py').write_text('def coverage_manifest(*a,**k):return None\n')
        (self.overlay/'subnet/training_receipts.py').write_text('def computation_binding(m):return m\n')
        (self.overlay/'subnet/gpu_service.py').write_text('class RemoteController:\n def __init__(self,state,authority):\n  self.state=state\n  self.authority=authority\n')
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        auth=dict(version=AUTHORIZATION_VERSION,source_sha256='a'*64,source_root=str(self.source),execution_root=str(self.overlay),source_files={'subnet/math.py':'b'*64})
        data=json.dumps(auth,sort_keys=True,separators=(',',':')).encode()
        self.auth=self.root/'authorization';self.auth.write_text(json.dumps(dict(payload=auth,signer=self.authority,signature=base64.b64encode(self.key.sign(data).signature).decode())))
        pins={}
        for leaf in ('native_training_outcome_filter.py','native_training_eligibility.py','native_training_lifecycle.py'):
            data=(Path(__file__).resolve().parents[1]/'ops'/leaf).read_bytes();(self.operator/leaf).write_bytes(data);pins[leaf]=hashlib.sha256(data).hexdigest()
        self.boundary=dict(version='future-native-eligibility-boundary-v1',epoch_prefix='production',earliest_round=34,minimum_parent_step=24,contract_fields=dict(K=1,L=1,training_policy='unchanged',training_input_policy='committed-unaudited-training-v1'))
        self.policy=dict(source_root=str(self.source),source_sha256='a'*64,operator_overlay=dict(root=str(self.overlay),overrides={'subnet/persistent_training_controller.py':'c'*64}),native_training_eligibility=dict(version='pinned-native-eligibility-operator-v1',root=str(self.operator),files=pins,authorization=dict(path=str(self.auth),file_sha256=hashlib.sha256(self.auth.read_bytes()).hexdigest(),payload_sha256=runner.guards.digest(auth)),tokenizer_root='/authenticated/tokenizer',interpreter='/authenticated/python',boundary=self.boundary,lifecycle_policy=LIFECYCLE_POLICY))
        self.saved={k:v for k,v in sys.modules.items() if k=='subnet' or k.startswith('subnet.')}
        self.oldpath=list(sys.path)
        self.addCleanup(self.restore)
    def restore(self):
        sys.path[:]=self.oldpath
        for k in list(sys.modules):
            if k=='subnet' or k.startswith('subnet.') or k.startswith('_root_pinned_native_eligibility'):sys.modules.pop(k)
        sys.modules.update(self.saved)
    def controller(self):
        service=runner.prepare_runtime(self.policy)
        self.assertEqual(Path(service.__file__),self.overlay/'subnet/gpu_service.py')
        controller=service.RemoteController(self.state,SimpleNamespace(id=self.authority))
        self.assertEqual(controller.native_training_eligibility_selector.__class__.__module__,'_root_pinned_native_eligibility.native_training_eligibility')
        return controller
    def manifest(self,round_number=34):
        return dict(epoch=f'production--1234-{round_number}',source_bundle={'sha256':'a'*64},trainer_state_binding={'global_step_before':24},**self.boundary['contract_fields'])
    def test_actual_prepare_runtime_constructor_installs_private_pinned_selector(self):
        self.assertTrue(self.controller().native_training_eligibility_selector.applies_to(self.manifest()))
    def test_prior_round_and_existing_original_jobs_skip(self):
        selector=self.controller().native_training_eligibility_selector
        self.assertFalse(selector.applies_to(self.manifest(33)))
        (self.state/'roles').mkdir();(self.state/'roles/production--1234-34-train.json').write_text('immutable-original')
        self.assertFalse(selector.applies_to(self.manifest()))
        self.assertEqual(selector.select(self.manifest(),['untouched']), (self.manifest(),['untouched']))
    def test_new_round_source_contract_parent_fail_closed(self):
        selector=self.controller().native_training_eligibility_selector
        for changed in (dict(source_bundle={'sha256':'d'*64}),dict(K=2),dict(trainer_state_binding={'global_step_before':23}),dict(epoch='foreign--1234-34')):
            with self.subTest(changed=changed),self.assertRaises(ValueError):selector.applies_to(dict(self.manifest(),**changed))
    def test_native_issued_subset_requires_authenticated_restart_not_original_skip(self):
        selector=self.controller().native_training_eligibility_selector
        (self.state/'roles').mkdir();(self.state/'roles/production--1234-34-train.json').write_text('original')
        directory=self.state/'native-outcome-eligibility/production--1234-34';directory.mkdir(parents=True);(directory/'subset.ROOT-SIGNED.json').write_text('untrusted')
        self.assertTrue(selector.applies_to(self.manifest()))
        # select must authenticate context/population; existence never supplies acceptance.
        with self.assertRaises(FileNotFoundError):selector.select(self.manifest(),[])
    def test_tampered_operator_refused_before_constructor_import(self):
        (self.operator/'native_training_eligibility.py').write_text('raise RuntimeError("untrusted")')
        with self.assertRaises(ValueError):runner.prepare_runtime(self.policy)
    def test_authorized_execution_root_distinct_from_scientific_root(self):
        runner.validate_native_operator(self.policy,self.authority,{'runtime_source_files':{'subnet/math.py':'b'*64}})
        self.policy['operator_overlay']['root']=str(self.source)
        with self.assertRaises(ValueError):runner.validate_native_operator(self.policy,self.authority,{'runtime_source_files':{'subnet/math.py':'b'*64}})
    def test_default_off_constructor_unchanged(self):
        del self.policy['native_training_eligibility']
        service=runner.prepare_runtime(self.policy)
        self.assertFalse(hasattr(service.RemoteController(self.state,SimpleNamespace(id=self.authority)),'native_training_eligibility_selector'))
