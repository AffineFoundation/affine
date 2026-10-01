import copy,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from subnet.native_eog_deployment import private_fixture_hash

class PortableIdentityTests(unittest.TestCase):
    def test_relocated_identical_seed_preserves_identity(self):
        with TemporaryDirectory() as directory:
            a=Path(directory)/'a.sql';b=Path(directory)/'b.sql';a.write_bytes(b'original seed');b.write_bytes(a.read_bytes())
            first=dict(data=dict(name='original-task',services=[dict(seed_file=str(a),context={'fixture':'context'})],verifiers=[dict(query='native SQL')]))
            second=copy.deepcopy(first);second['data']['services'][0]['seed_file']=str(b)
            self.assertEqual(private_fixture_hash(first),private_fixture_hash(second))
            b.write_bytes(b'forged seed');self.assertNotEqual(private_fixture_hash(first),private_fixture_hash(second))
    def test_private_grader_is_committed_and_original_not_mutated(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'seed.sql';p.write_text('original')
            task=dict(data=dict(services=[dict(seed_file=str(p))],verifiers=[dict(query='SELECT 1')]))
            before=json.dumps(task,sort_keys=True);digest=private_fixture_hash(task)
            self.assertEqual(before,json.dumps(task,sort_keys=True))
            task['data']['verifiers'][0]['query']='SELECT 2';self.assertNotEqual(digest,private_fixture_hash(task))

if __name__=='__main__':unittest.main()
