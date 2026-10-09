import unittest
from ops import resource_accounting as peer
class ResourceTests(unittest.TestCase):
 def test_actual(self):
  s=dict(file=144875200512,shmem=91387527168,inactive_file=37638184960,file_dirty=4096,file_writeback=0,file_mapped=24576,unevictable=0)
  r=peer.cgroup_headroom(267361714176,146420314112,s);self.assertGreater(r['usable_bytes'],145057269760);self.assertEqual(r['conservative_clean_inactive_file_bytes'],37638156288)
 def test_missing(self):self.assertEqual(peer.cgroup_headroom(100,80,{})['usable_bytes'],20)
 def test_shmem_only(self):self.assertEqual(peer.cgroup_headroom(100,80,dict(file=80,shmem=80,inactive_file=80))['usable_bytes'],20)
 def test_active_only(self):self.assertEqual(peer.cgroup_headroom(100,80,dict(file=80,active_file=80))['usable_bytes'],20)
 def test_exclusions(self):self.assertEqual(peer.cgroup_headroom(100,80,dict(file=60,shmem=30,inactive_file=50,file_dirty=4,file_writeback=3,file_mapped=2,unevictable=1))['usable_bytes'],40)
 def test_clamp(self):self.assertEqual(peer.cgroup_headroom(100,80,dict(file=500,inactive_file=500))['usable_bytes'],100)
 def test_invalid(self):
  for x in[-1,True,'3']:
   with self.assertRaises(ValueError):peer.cgroup_headroom(100,80,dict(file=x))
 def test_over_limit_never_counts_more_than_maximum(self):
  result=peer.cgroup_headroom(100,120,dict(file=120,inactive_file=120));self.assertEqual(result['hard_headroom_bytes'],0);self.assertEqual(result['usable_bytes'],100)
 def test_inactive_file_does_not_subtract_shmem_twice(self):
  result=peer.cgroup_headroom(200,180,dict(file=150,shmem=90,inactive_file=50));self.assertEqual(result['usable_bytes'],70)
if __name__=='__main__':unittest.main()
