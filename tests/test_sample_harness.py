import copy
import unittest
from subnet.sample_harness import VERSION,validate,resolve

class IndexedHarnessTests(unittest.TestCase):
    def config(self):
        def row(text):return dict(version='text-tools-window-v1',policy='candidates',candidates=[text,'wrong'],max_output_tokens=256,temperature=4.,top_p=1.)
        return dict(version=VERSION,by_index={'0':row('route A'),'3':row('route B')})
    def test_task_choice_preserves_signed_window_and_input(self):
        p=self.config();original=copy.deepcopy(p);a=resolve(p,0,[0,3]);b=resolve(p,3,[0,3]);self.assertEqual(a['candidates'][0],'route A');self.assertEqual(b['candidates'][0],'route B');self.assertEqual(a['history_window_messages'],2);self.assertEqual(p,original)
        a['candidates'].append('altered')
        self.assertEqual(resolve(p,0,[0,3])['candidates'],['route A','wrong'])
    def test_missing_extra_or_alias_indices_refused(self):
        for entries in [{'0':self.config()['by_index']['0']},{**self.config()['by_index'],'4':self.config()['by_index']['0']},{'00':self.config()['by_index']['0'],'3':self.config()['by_index']['3']}]:
            with self.assertRaisesRegex(ValueError,'coverage'):validate(dict(version=VERSION,by_index=entries),[0,3])
    def test_heldout_and_boolean_index_refused(self):
        for index in [4,False,-1,'0']:
            with self.subTest(index=index),self.assertRaises(ValueError):resolve(self.config(),index,[0,3])
        for indices in [[],[0,0],[False,3],list(range(10001))]:
            with self.assertRaises(ValueError):validate(self.config(),indices)
    def test_nested_and_legacy_silent_policy_changes_refused(self):
        p=self.config();p['by_index']['0']=self.config()
        with self.assertRaisesRegex(ValueError,'nested'):validate(p,[0,3])
        p=self.config();p['version']='text-tools-v1'
        with self.assertRaisesRegex(ValueError,'versioned'):validate(p,[0,3])
        p=self.config();p['fallback']={}
        with self.assertRaisesRegex(ValueError,'fields'):validate(p,[0,3])
    def test_ordinary_harness_preserves_existing_normalization(self):
        p=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=128,temperature=.7,top_p=1.)
        self.assertEqual(resolve(p,3,[0,3]),validate(p,[0,3]))
        with self.assertRaises(ValueError):resolve(p,4,[0,3])

if __name__=='__main__':unittest.main()
