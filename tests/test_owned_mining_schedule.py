import copy
import unittest
from unittest.mock import patch
from subnet.gpu_service import owned_mining_job_fields


class OwnedScheduleTests(unittest.TestCase):
    def setUp(self):
        self.definition=dict(env_id='affine_math',spec={},indices=[0,1,2,3],harness={})
        self.schedule=[{'affine_math':[0,1]},{'affine_math':[2,3]}]

    def test_operator_rotation_preserves_whole_public_pool(self):
        original=copy.deepcopy(self.definition)
        with patch('subnet.protocol.entries',return_value=[self.definition]):
            for round_number,expected in [(0,[0,1]),(1,[2,3]),(2,[0,1])]:
                fields=owned_mining_job_fields({'owned_mining_schedule':self.schedule},{},round_number)
                self.assertEqual(fields,{'mining_subset':{'affine_math':expected}})
        self.assertEqual(self.definition,original)

    def test_ambiguous_schedule_and_unauthorized_indices_refused(self):
        for config in [{'owned_mining_schedule':[]},
                {'owned_mining_schedule':self.schedule,'owned_mining_subset':self.schedule[0]},
                {'owned_mining_schedule':[{'affine_math':[99]}]}]:
            with self.subTest(config=config),patch('subnet.protocol.entries',return_value=[self.definition]):
                with self.assertRaises(ValueError):owned_mining_job_fields(config,{},0)
        with self.assertRaises(ValueError):
            owned_mining_job_fields({'owned_mining_schedule':self.schedule},{},True)

    def test_unconfigured_controller_does_not_change_existing_jobs(self):
        self.assertEqual(owned_mining_job_fields({}, {}, 0),{})


if __name__=='__main__':unittest.main()
