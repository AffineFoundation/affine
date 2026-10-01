import copy
import unittest
from subnet.native_agent_isolation import REVISION, validate_descriptor, docker_command, checked_grade

def descriptor():
    public = {n:'1'*64 for n in ('tools.py','base.py','db.json','instruction.md','worker.py')}
    private = dict(public, **{'gold.json':'2'*64,'original-taskset.py':'3'*64})
    return dict(revision=REVISION, task_name='controlled', actor_image='sha256:'+'4'*64,
                grader_image='sha256:'+'5'*64, public_files=public, private_files=private,
                original_grader_sha256='3'*64,
                dependency_scope='immutable-controlled-images-not-full-upstream-closure')

class IsolationControls(unittest.TestCase):
    def test_private_gold_or_grader_cannot_enter_actor_contract(self):
        for name in ('gold.json','original-taskset.py','task.toml'):
            value=descriptor(); value['public_files'][name]='6'*64
            with self.assertRaisesRegex(ValueError,'private grader leaked'):validate_descriptor(value)

    def test_distinct_immutable_images_and_no_host_mount_or_network(self):
        value=descriptor(); actor=docker_command(value,'actor'); grader=docker_command(value,'grader')
        self.assertIn(value['actor_image'],actor); self.assertIn(value['grader_image'],grader)
        self.assertNotIn('--mount',actor); self.assertNotIn('-v',actor)
        self.assertIn('none',actor); self.assertIn('--read-only',actor)
        self.assertIn('no-new-privileges',actor)
        value['actor_image']='latest'
        with self.assertRaisesRegex(ValueError,'immutable'):validate_descriptor(value)

    def test_original_grader_identity_and_solved_semantics_are_not_miner_claims(self):
        value=descriptor(); response=dict(original_source_sha256='3'*64,reward=1,
                                           metrics=dict(db_hash=0,verify=1))
        self.assertEqual(checked_grade(response,value)['reward'],1)
        mutations=[dict(response,reward=0),dict(response,reward=True),
                   dict(response,original_source_sha256='9'*64),
                   dict(response,metrics=dict(db_hash=0,verify=.5))]
        for mutation in mutations:
            with self.assertRaises(ValueError):checked_grade(mutation,value)

    def test_filename_traversal_and_false_full_closure_fail(self):
        value=descriptor(); value['public_files']['../gold.json']='a'*64
        with self.assertRaises(ValueError):validate_descriptor(value)
        value=descriptor(); value['dependency_scope']='full-closure'
        with self.assertRaises(ValueError):validate_descriptor(value)

if __name__=='__main__':unittest.main()
