import sys,json,tempfile,unittest,threading,time,types
from pathlib import Path
from unittest.mock import Mock,patch
from ops.trainer_lifecycle import training_ack_ordering as ordering
from subnet import persistent_training_controller as training
from subnet.remote_backend import save
class Ordering(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'roles').mkdir();self.job={'job_id':'exact-original'};self.report={'same':'original'};self.pointer={'same':'pointer'};self.path=self.root/'roles/exact-original-trainer-cache-cleanup.json';self.good={'status':'complete','optimizer_cache_promotion':{'promoted':True}}
  self.jobs=types.SimpleNamespace(prepare_training_cache_ack=Mock(),retire_training_cache=Mock(return_value=self.good));self.c=types.SimpleNamespace(state=self.root,jobs=self.jobs)
  self.old=training._cleanup_threads;training._cleanup_threads={};self.addCleanup(setattr,training,'_cleanup_threads',self.old)
 def test_completed_ack_does_not_run_again(self):
  save(self.path,self.good);self.assertEqual(ordering.await_cleanup(self.c,self.job,self.report,self.pointer),self.good);self.jobs.retire_training_cache.assert_not_called()
 def test_absent_thread_recovers_same_genuine_ack(self):
  ordering.await_cleanup(self.c,self.job,self.report,self.pointer);self.jobs.prepare_training_cache_ack.assert_called_once_with(self.job,self.report,self.pointer);self.jobs.retire_training_cache.assert_called_once_with(self.job,self.report,self.pointer)
 def test_deferred_record_retries_original_identity(self):
  save(self.path,{'status':'deferred','reason':'workspace-role-in-flight'});ordering.await_cleanup(self.c,self.job,self.report,self.pointer);self.assertEqual(json.loads(self.path.read_bytes()),self.good)
 def test_waits_existing_thread_instead_of_concurrent_action(self):
  event=threading.Event()
  def finish():time.sleep(.025);save(self.path,self.good);event.set()
  thread=threading.Thread(target=finish);training._cleanup_threads[str(self.path)]=thread;thread.start();ordering.await_cleanup(self.c,self.job,self.report,self.pointer);self.assertTrue(event.is_set());self.jobs.retire_training_cache.assert_not_called()
 def test_live_thread_timeout_never_calls_retirement_or_calibration(self):
  done=threading.Event();thread=threading.Thread(target=done.wait);training._cleanup_threads[str(self.path)]=thread;thread.start()
  try:
   with self.assertRaises(TimeoutError):ordering.await_cleanup(self.c,self.job,self.report,self.pointer,budget=.001)
   self.jobs.retire_training_cache.assert_not_called()
  finally:done.set();thread.join()
 def test_unpromoted_complete_is_not_accepted(self):
  save(self.path,{'status':'complete','optimizer_cache_promotion':{'promoted':False}})
  with self.assertRaises(ValueError):ordering.await_cleanup(self.c,self.job,self.report,self.pointer)
 def test_second_deferred_keeps_calibration_closed(self):
  self.jobs.retire_training_cache.return_value={'status':'deferred','reason':'workspace-role-in-flight'}
  with self.assertRaises(ValueError):ordering.await_cleanup(self.c,self.job,self.report,self.pointer)
 def test_original_failure_not_hidden(self):
  self.jobs.retire_training_cache.side_effect=ValueError('original ACK binding changed')
  with self.assertRaisesRegex(ValueError,'binding'):ordering.await_cleanup(self.c,self.job,self.report,self.pointer)
 def test_expired_wait_cannot_launch_new_action(self):
  with self.assertRaises(TimeoutError):ordering.await_cleanup(self.c,self.job,self.report,self.pointer,budget=-1)
  self.jobs.retire_training_cache.assert_not_called()
 def test_initial_genesis_skips_ack_and_keeps_original_calibration(self):
  from subnet import successor_calibration
  old=successor_calibration.before_open;original=Mock(return_value='original calibration');successor_calibration.before_open=original
  self.addCleanup(setattr,successor_calibration,'before_open',old)
  ordering.install();result=successor_calibration.before_open(self.c,{}, {'trainer_state':None}, {'contract':'unchanged'});self.assertEqual(result,'original calibration');original.assert_called_once()
if __name__=='__main__':unittest.main()
