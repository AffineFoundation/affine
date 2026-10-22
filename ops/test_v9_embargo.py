"""v9 embargo tests: miner training inputs are not read or published until EMBARGO_EPOCHS later epochs finalized."""
import unittest
from unittest.mock import patch
from ops import mimo_finalized_evidence as m
from ops import test_mimo_finalized_evidence as base
class EmbargoTests(base.EvidenceTests):
    def test_latest_epoch_embargoed_and_private_docs_never_read(self):
        self.trained(); rows=m.collect(self.f.p,self.bucket,25)
        ti=rows[0]['categories']['training_inputs']
        self.assertEqual(ti['status'],'embargoed'); self.assertEqual(ti['groups'],[]); self.assertEqual(ti['embargo_epochs'],3)
        self.assertFalse([k for k in self.bucket.reads if 'private/' in str(k)])
    def test_released_when_embargo_lifted(self):
        self.trained()
        with patch.object(m,'EMBARGO_EPOCHS',0):
            rows=m.collect(self.f.p,self.bucket,25)
        self.assertEqual(rows[0]['categories']['training_inputs']['status'],'available')
        self.assertEqual(rows[0]['categories']['training_inputs']['groups'][0]['document']['batch']['rollouts'][0]['turns'][0]['output'],[3,4])
    def test_embargo_rule(self):
        inv=dict(epochs={str(e):'0'*64 for e in range(0,71)})
        self.assertTrue(all(m.embargoed(str(e),inv) for e in (68,69,70)))
        self.assertFalse(any(m.embargoed(str(e),inv) for e in (0,40,67)))
for _n in dir(base.EvidenceTests):   # only the v9 tests here; the inherited suite runs (with EMBARGO_EPOCHS=0) in its own module
    if _n.startswith('test_') and _n not in EmbargoTests.__dict__: setattr(EmbargoTests, _n, None)
if __name__=='__main__':unittest.main()
