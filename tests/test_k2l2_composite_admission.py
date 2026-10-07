"""Composite admission retains old evidence labels and an exact calibration-only delta."""
import tempfile,unittest
from pathlib import Path
from ops import k2l2_composite_admission as m
class Controls(unittest.TestCase):
 def maps(self):
  old={k:'a'*64 for k in m.DELTA|{'subnet/forced_sampling.py','subnet/task_normalized_training.py'}}
  new=dict(old);new.update({k:'b'*64 for k in m.DELTA|m.TEST_ADDITIONS})
  declared={k:dict(before=old.get(k),after=v)for k,v in new.items()if old.get(k)!=v};return old,new,declared
 def test_exact_new_calibration_and_three_named_tests(self):self.assertEqual(set(m.exact_delta(*self.maps())),m.DELTA|m.TEST_ADDITIONS)
 def test_no_extra_scientific_core_changes(self):
  old,new,d=self.maps();new['subnet/forced_sampling.py']='c'*64;d['subnet/forced_sampling.py']=dict(before=old['subnet/forced_sampling.py'],after='c'*64)
  with self.assertRaises(ValueError):m.exact_delta(old,new,d)
 def test_no_extra_qualification_test(self):
  old,new,d=self.maps();new['tests/unreviewed.py']='c'*64
  with self.assertRaises(ValueError):m.exact_delta(old,new,d)
 def test_no_undeclared_calibration_change(self):
  old,new,d=self.maps();d.pop('subnet/backend_jobs.py')
  with self.assertRaises(ValueError):m.exact_delta(old,new,d)
 def test_only_evaluate_calibration_branch_can_change(self):
  with tempfile.TemporaryDirectory()as temp:
   a=Path(temp)/'old';b=Path(temp)/'new';a.write_text("def validate(job):\n if job['role']=='evaluate' and job.get('successor_calibration'):\n  request(job)\n return True\n")
   b.write_text(a.read_text().replace('request(job)','request(job); context(job)'));m.validate_backend_delta(a,b)
   b.write_text(b.read_text().replace('return True','return False'))
   with self.assertRaises(ValueError):m.validate_backend_delta(a,b)
if __name__=='__main__':unittest.main()
