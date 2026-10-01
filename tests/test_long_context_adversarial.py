import unittest
from unittest.mock import Mock
from ops.probe_long_context_adversarial import verify_bound
from subnet.long_context_runtime import digest
class InputBindingTests(unittest.TestCase):
    def test_self_consistent_model_proof_cannot_replace_authorized_input(self):
        runtime=Mock();runtime.verify.return_value=True
        with self.assertRaisesRegex(ValueError,'expected-input'):verify_bound(runtime,{'prompt':[2,3]},None,digest([1,3]))
        runtime.verify.assert_not_called()
    def test_authorized_input_still_requires_numerical_verification(self):
        runtime=Mock();runtime.verify.side_effect=ValueError('wrong model')
        with self.assertRaisesRegex(ValueError,'wrong model'):verify_bound(runtime,{'prompt':[1,3]},None,digest([1,3]))
        runtime.verify.assert_called_once()
