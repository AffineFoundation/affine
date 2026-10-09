import tempfile,time,unittest
import test_current_assessment_zero_sink as fixture

POLICY='current-hotkey-snapshot-v1'

class RegistrationChurnControls(unittest.TestCase):
    adapter = fixture.ZeroSinkControls.adapter
    def submit(self, adapter, points, before, fresh, **kwargs):
        adapter.registrations=lambda:fresh
        return adapter.submit_hour(points,before,int(time.time())//3600*3600,
                                   registration_change_policy=POLICY,**kwargs)
    def test_departed_miner_does_not_block_remaining_weights(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.adapter(d)
            r=self.submit(a,{'gone':100,'live':3},{'gone':{'uid':85,'public_key':'gone'},
                'live':{'uid':90,'public_key':'live'}},{'live':{'uid':90,'public_key':'live','snapshot_block':99},
                'replacement':{'uid':85,'public_key':'replacement','snapshot_block':99}})
            self.assertEqual(r['status'],'planned');self.assertEqual(r['uids'],[90]);self.assertEqual(r['weights'],[1.])
            self.assertEqual(r['excluded_unregistered'],['gone']);self.assertEqual(r['excluded_stale'],[])
    def test_retained_hotkey_maps_to_current_uid(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.submit(self.adapter(d),{'live':3},{'live':{'uid':85,'public_key':'live'}},
                {'live':{'uid':90,'public_key':'live','snapshot_block':99}})
            self.assertEqual(r['uids'],[90]);self.assertEqual(r['remapped_uids'],[{'hotkey':'live','from_uid':85,'to_uid':90}])
    def test_final_queries_use_same_verified_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.adapter(d);old=a.query;blocks=[]
            def query(n,p,b):blocks.append(b);return old(n,p,b)
            a.query=query
            r=self.submit(a,{'live':3},{'live':{'uid':85,'public_key':'live'}},
                {'live':{'uid':85,'public_key':'live','snapshot_block':99}})
            self.assertEqual(r['registration_snapshot_block'],99)
            self.assertEqual(blocks[0],100);self.assertTrue(all(b==99 for b in blocks[1:]))
    def test_inconsistent_or_missing_snapshot_refused(self):
        for fresh in ({'a':{'uid':1,'public_key':'a'}},
                      {'a':{'uid':1,'public_key':'a','snapshot_block':99},'b':{'uid':2,'public_key':'b','snapshot_block':100}}):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):self.submit(self.adapter(d),{'a':1},{'a':{'uid':1,'public_key':'a'}},fresh)
    def test_public_identity_mismatch_still_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'public identity'):
                self.submit(self.adapter(d),{'a':1},{'a':{'uid':1,'public_key':'original'}},
                    {'a':{'uid':1,'public_key':'wrong','snapshot_block':99}})
    def test_all_departed_uses_explicit_zero_policy(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.submit(self.adapter(d),{'gone':1},{'gone':{'uid':85,'public_key':'gone'}},{},zero_total_policy='no-owner-retain-v1')
            self.assertEqual(r['status'],'zero_points_no_submission');self.assertNotIn('uids',r);self.assertEqual(r['excluded_unregistered'],['gone'])

if __name__=='__main__':unittest.main()
