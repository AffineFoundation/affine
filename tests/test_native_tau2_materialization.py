import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops.materialize_native_tau2_common import materialize,REVISION

class Materialization(unittest.TestCase):
    def fixture(self,root):
        root.mkdir();(root/'.tau2_revision').write_text(REVISION)
        d=root/'tau2/domains/telecom';d.mkdir(parents=True)
        for n in ('tasks_full.json','tasks.json','db.toml','user_db.toml','main_policy.md'):(d/n).write_text('{}')
    def rows(self):return json.dumps({'rows':[{'id':str(i),'user_instructions':'PRIVATE_USER','evaluation_criteria':{'expected':'PRIVATE_GOLD'}} for i in range(4)],'pool_count':2171,'excluded_base_count':114})
    def test_public_private_boundary_and_disjoint_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td)/'data';self.fixture(data);out=Path(td)/'out'
            with patch('ops.materialize_native_tau2_common.subprocess.check_output',return_value=self.rows()):r=materialize(data,out,4)
            public=json.loads((out/'public-tasks.json').read_text());self.assertEqual(public['mining_indices'],[0,1]);self.assertEqual(public['heldout_indices'],[2,3]);self.assertNotIn('PRIVATE_',json.dumps(public));self.assertIn('PRIVATE_GOLD',(out/'private-tasks.json').read_text());self.assertEqual((out/'private-tasks.json').stat().st_mode&0o777,0o600);self.assertFalse(r['simulation_ran'])
    def test_missing_original_resource_rejected_before_loader(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td)/'data';self.fixture(data);(data/'tau2/domains/telecom/db.toml').unlink()
            with patch('ops.materialize_native_tau2_common.subprocess.check_output') as load:
                with self.assertRaises(ValueError):materialize(data,Path(td)/'out',4)
                load.assert_not_called()
    def test_existing_task_commitments_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td)/'data';self.fixture(data);out=Path(td)/'out'
            with patch('ops.materialize_native_tau2_common.subprocess.check_output',return_value=self.rows()):materialize(data,out,4)
            old=(out/'private-tasks.json').read_bytes();different=json.loads(self.rows());different['rows'][0]['user_instructions']='MUTATED'
            with patch('ops.materialize_native_tau2_common.subprocess.check_output',return_value=json.dumps(different)):
                with self.assertRaises(ValueError):materialize(data,out,4)
            self.assertEqual((out/'private-tasks.json').read_bytes(),old)
if __name__=='__main__':unittest.main()
