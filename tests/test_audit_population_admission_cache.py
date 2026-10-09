import copy
import types
import unittest

from ops import audit_population_admission_cache as module

class Tests(unittest.TestCase):
    def setUp(self):
        self.auth = []
        def authenticate(document, authority):
            self.auth.append(authority)
            if document['signature'] != 'valid': raise ValueError('signature')
            return document['payload']
        class Auditor:
            def __init__(self):
                self.controller=types.SimpleNamespace(authority=types.SimpleNamespace(id='authority'))
                self.state={'populations':{}}
                self.validations=0; self.writes=0
            def persist(self): self.writes += 1
            def admit(self, document):
                self.validations += 1
                payload=authenticate(document,self.controller.authority.id)
                if not payload['canonical']: raise ValueError('canonical population')
                epoch=payload['manifest_document']['payload']['epoch']
                prior=self.state['populations'].get(epoch)
                if prior is not None and prior != document:raise ValueError('immutable conflict')
                self.state['populations'][epoch]=document
                self.persist()
        self.service=types.SimpleNamespace(ContinuousAuditor=Auditor,authenticate=authenticate)
        module.install(self.service)
        self.a=Auditor()
        self.doc={'signature':'valid','payload':{'canonical':True,'manifest_document':{'payload':{'epoch':'epoch1'}},'data':[1,2]}}
    def test_existing_validates_once_authenticates_every_time_no_rewrite(self):
        self.a.state['populations']['epoch1']=copy.deepcopy(self.doc)
        self.a.admit(copy.deepcopy(self.doc)); self.a.admit(copy.deepcopy(self.doc))
        self.assertEqual((self.a.validations,self.a.writes,len(self.auth)),(1,0,3))
        self.assertNotIn('persist',self.a.__dict__)
    def test_new_population_written_once(self):
        self.a.admit(copy.deepcopy(self.doc)); self.a.admit(copy.deepcopy(self.doc))
        self.assertEqual((self.a.validations,self.a.writes),(1,1))
    def test_conflicting_data_rejected(self):
        self.a.admit(copy.deepcopy(self.doc)); changed=copy.deepcopy(self.doc);changed['payload']['data']=[3]
        with self.assertRaisesRegex(ValueError,'immutable conflict'):self.a.admit(changed)
        self.assertEqual(self.a.state['populations']['epoch1'],self.doc)
        self.assertEqual(self.a.writes,1)
    def test_signature_rechecked_on_cached_payload(self):
        self.a.admit(copy.deepcopy(self.doc));changed=copy.deepcopy(self.doc);changed['signature']='forged'
        with self.assertRaisesRegex(ValueError,'signature'):self.a.admit(changed)
    def test_failed_semantics_not_cached_and_persist_restored(self):
        bad=copy.deepcopy(self.doc);bad['payload']['canonical']=False
        self.a.state['populations']['epoch1']=bad
        for _ in range(2):
            with self.assertRaisesRegex(ValueError,'canonical population'):self.a.admit(copy.deepcopy(bad))
        self.assertEqual(self.a.validations,2)
        self.assertNotIn('persist',self.a.__dict__)
        self.assertEqual(self.a.writes,0)
    def test_restart_revalidates_without_rewrite(self):
        self.a.admit(copy.deepcopy(self.doc))
        other=self.service.ContinuousAuditor();other.state=copy.deepcopy(self.a.state)
        other.admit(copy.deepcopy(self.doc))
        self.assertEqual((other.validations,other.writes),(1,0))
    def test_existing_persist_override_restored(self):
        self.a.state['populations']['epoch1']=copy.deepcopy(self.doc)
        method=lambda:None;self.a.persist=method
        self.a.admit(copy.deepcopy(self.doc));self.assertIs(self.a.persist,method)
    def test_install_idempotent(self):
        first=self.service.ContinuousAuditor.admit;module.install(self.service)
        self.assertIs(first,self.service.ContinuousAuditor.admit)

if __name__=='__main__':unittest.main()
