import ast,copy,json,pathlib,unittest
from test_owned_cached_group_operator import OperatorTests as _Fixture
from ops.owned_cached_group_ack_relay import emit_read,relay_step
class Observer:
 def __init__(self,test,publisher):self.test=test;self.scope=publisher.scope;self.calls=[];self.installed=set();self.live=False
 def read(self,jid):
  if jid not in self.test.transport.reports:return None
  return dict(original_job=self.test.jobs[int(jid[-1])],report=self.test.transport.reports[jid],terminal=self.test.transport.statuses[jid],physical_original_absent=not self.live)
 def install_ack(self,jid,envelope):self.test.transport.acks[jid]=envelope;self.calls.append(jid)
class RelayTests(unittest.TestCase):
 setUp=_Fixture.setUp;sign=_Fixture.sign;digest=_Fixture.digest;publisher=_Fixture.publisher;complete=_Fixture.complete
 def test_passive_loop_installs_four_exact_ACKs_once_no_dispatch(self):
  for n in range(4):self.complete(n,ack=False)
  publisher=self.publisher();observer=Observer(self,publisher);r=relay_step(publisher,observer);self.assertEqual(r['durable_ACK_count'],4);self.assertFalse(r['GPU_dispatch']);self.assertEqual(len(observer.calls),4);relay_step(publisher,observer);self.assertEqual(len(observer.calls),4);self.assertFalse(self.transport.launched)
 def test_no_report_or_live_original_never_receives_ACK(self):
  p=self.publisher();o=Observer(self,p);self.assertEqual(relay_step(p,o)['durable_ACK_count'],0);self.complete(0,ack=False);o.live=True;self.assertEqual(relay_step(p,o)['status'],'observing-original');self.assertFalse(o.calls)
 def test_published_report_running_marker_waits_then_ACKs_same_original(self):
  self.complete(0,ack=False);p=self.publisher();o=Observer(self,p);original=copy.deepcopy(self.transport.statuses['original-group-0'])
  self.transport.statuses['original-group-0'].update(phase='running',exit_code=None,finished_at=None);o.live=True
  r=relay_step(p,o);self.assertEqual(r['status'],'observing-original');self.assertEqual(r['original_phase'],'running');self.assertFalse(o.calls);self.assertFalse(self.bucket.objects);self.assertFalse(self.transport.launched)
  self.transport.statuses['original-group-0']=original;o.live=False
  relay_step(p,o);self.assertEqual(o.calls,['original-group-0']);self.assertFalse(self.transport.launched)
 def test_pending_marker_with_existing_report_is_not_completed(self):
  self.complete(0,ack=False);p=self.publisher();o=Observer(self,p);self.transport.statuses['original-group-0'].update(phase='pending',exit_code=None,finished_at=None)
  self.assertEqual(relay_step(p,o)['status'],'observing-original');self.assertFalse(o.calls);self.assertFalse(self.bucket.objects)
 def test_nonterminal_changed_identity_or_terminal_fields_rejected(self):
  for change in ({'job_id':'foreign-original'},{'exit_code':0},{'finished_at':50}):
   with self.subTest(change=change):
    self.complete(0,ack=False);p=self.publisher();o=Observer(self,p);self.transport.statuses['original-group-0'].update(phase='running',exit_code=None,finished_at=None);self.transport.statuses['original-group-0'].update(change)
    with self.assertRaisesRegex(ValueError,'nonterminal original'):relay_step(p,o)
    self.assertFalse(o.calls)
 def test_unknown_phase_report_is_not_silently_pending(self):
  self.complete(0,ack=False);p=self.publisher();o=Observer(self,p);self.transport.statuses['original-group-0']['phase']='unknown'
  with self.assertRaises(ValueError):relay_step(p,o)
  self.assertFalse(o.calls)
 def test_scope_or_bucket_corruption_prevents_ACK(self):
  self.complete(0,ack=False);p=self.publisher();o=Observer(self,p);o.scope=copy.deepcopy(o.scope);o.scope['workspace']='/different'
  with self.assertRaises(ValueError):relay_step(p,o)
  o.scope=p.scope;self.bucket.bad=True
  with self.assertRaises(ValueError):relay_step(p,o)
  self.assertFalse(o.calls)
 def test_exact_emitted_remote_read_compiles_with_full177map_and_metachar_data(self):
  p=dict(workspace=str(self.root),source_path='/qualified/SOURCE',source_files=self.files,machine_id_sha256='a'*64,gpu_uuid='GPU-safe',job_id='original-group-0',envelope_sha256='b'*64,inert="');raise RuntimeError('injected');#");code=emit_read(p);compile(code,'exact-real-request-script','exec');self.assertIn('Strict',pathlib.Path('ops/owned_cached_group_ack_relay.py').read_text());self.assertIn('st_nlink==1',code)
 def test_relay_has_no_gpu_launch_or_cache_disposal_paths(self):
  tree=ast.parse(pathlib.Path('ops/owned_cached_group_ack_relay.py').read_text());text=ast.unparse(tree);self.assertNotIn('Popen',text);self.assertNotIn('evict_checkpoints',text);self.assertNotIn('launch_runner',text)
del _Fixture
if __name__=='__main__':unittest.main()
