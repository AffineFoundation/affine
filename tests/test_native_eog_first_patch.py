import unittest
from ops.probe_native_eog_first_patch import FirstPatchActor
class Actor:
 def __init__(self):self.calls=[]
 def call(self,name,args):self.calls.append((name,args));return 'observation'
class FirstPatchTests(unittest.TestCase):
 def test_only_first_mutating_action_changes_without_changing_original_arguments(self):
  actor=Actor();wrapped=FirstPatchActor(actor);args={'location':'TechCorp Main Campus - Building 1, Conference Room B'}
  wrapped.call('list_events',{});wrapped.call('patch_event',args);wrapped.call('patch_event',args);wrapped.call('patch_event',args)
  self.assertIn('Building 2',actor.calls[1][1]['location']);self.assertIn('Building 1',actor.calls[2][1]['location']);self.assertIn('Building 1',actor.calls[3][1]['location']);self.assertIn('Building 1',args['location'])
 def test_unexpected_public_location_fails_instead_of_guessing(self):
  with self.assertRaisesRegex(ValueError,'public requested location'):FirstPatchActor(Actor()).call('patch_event',{'location':'other'})
