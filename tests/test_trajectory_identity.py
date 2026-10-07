import copy
import hashlib
import unittest
from ops.paired_quota_qualification import digest
from subnet.storage import canonical
from subnet.training_receipts import sha
from subnet.trajectory_identity import token_trace_sha256


class TokenTraceIdentity(unittest.TestCase):
    def setUp(self):
        self.turns = [dict(prompt=[1, 2], output=[3, 4], text='first',
                           observations=[dict(role='user', content='tool')]),
                      dict(prompt=[1, 2, 3, 4, 5], output=[6], done=True)]

    def test_frozen_existing_learner_and_research_digest_agree(self):
        trace = [dict(prompt=t['prompt'], output=t['output']) for t in self.turns]
        golden = hashlib.sha256(b'[{"output":[3,4],"prompt":[1,2]},{"output":[6],"prompt":[1,2,3,4,5]}]').hexdigest()
        self.assertEqual(token_trace_sha256(self.turns), golden)
        self.assertEqual(token_trace_sha256(self.turns), sha(trace))
        self.assertEqual(token_trace_sha256(self.turns), digest(trace))

    def test_repack_labels_and_observations_cannot_create_new_token_trace(self):
        modified = copy.deepcopy(self.turns)
        for turn in modified:
            turn.update(classification='negative', reward=0, seed=999,
                        text='replacement', done=False, observations=[], actions=[])
        self.assertEqual(token_trace_sha256(modified), token_trace_sha256(self.turns))

    def test_prompt_output_and_turn_order_distinguish_traces(self):
        variants = [copy.deepcopy(self.turns), copy.deepcopy(self.turns), list(reversed(self.turns))]
        variants[0][0]['prompt'][0] = 7
        variants[1][0]['output'][0] = 7
        for value in variants:
            self.assertNotEqual(token_trace_sha256(value), token_trace_sha256(self.turns))

    def test_pure_projection_preserves_input(self):
        before = copy.deepcopy(self.turns)
        token_trace_sha256(self.turns)
        self.assertEqual(self.turns, before)


if __name__ == '__main__':
    unittest.main()
