import unittest
import test_owned_cached750_group_operator as fixtures
from ops.owned_cached750_group_ack_relay import relay_step
class RelayControls(unittest.TestCase):
 setUp=fixtures.OperatorTests.setUp
 digest=fixtures.OperatorTests.digest
 sign=fixtures.OperatorTests.sign
 publisher=fixtures.OperatorTests.publisher
 complete=fixtures.OperatorTests.complete
 def test_all24_fullACKs_installed_with_actual_publisher_and_tail14(self):
  for n in range(24):self.complete(n,ack=False)
  test=self
  class Observer:
   scope=test.scope
   installed=set()
   def read(self,jid):return dict(terminal=test.transport.statuses[jid],report=test.transport.reports[jid],physical_original_absent=True)
   def install_ack(self,jid,ack):test.transport.acks[jid]=ack
  result=relay_step(self.publisher(),Observer())
  self.assertEqual(result['status'],'all-24-durable-ACKs-installed');self.assertEqual(result['durable_ACK_count'],24)
  self.assertEqual(len(self.bucket.objects),96)
 def test_failed_original_stops_relay_without_partial_score(self):
  self.complete(0,ack=False);jid=self.jobs[0]['payload']['job_id'];self.transport.statuses[jid].update(phase='failed',exit_code=1);test=self
  class Observer:
   scope=test.scope
   def read(self,jid):return dict(terminal=test.transport.statuses[jid],report=test.transport.reports[jid],physical_original_absent=True)
   def install_ack(self,jid,ack):raise AssertionError('failed original cannot be acknowledged as score')
  result=relay_step(self.publisher(),Observer());self.assertEqual(result['status'],'original-infrastructure-failure');self.assertIsNone(result['model_reward']);self.assertEqual(self.bucket.objects,{})
