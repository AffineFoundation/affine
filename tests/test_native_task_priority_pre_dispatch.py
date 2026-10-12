"""Portable policy-projection controls; no operator files or worker calls.

The temporary transition stands in for an authenticated historical dispatcher.
Cryptographic authentication belongs to the caller's guards; these tests cover
the effective-policy boundary after that authentication.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ops.native_task_priority_pre_dispatch import historical_effective_policy


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HistoricalEffectivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        helper = root / "original_checker.py"
        helper.write_text("# immutable historical checker fixture\n")
        self.preflight = {
            "version": "actual-signed-independent-pre-dispatch-execute-check-v1",
            "helper": {"path": str(helper), "sha256": file_hash(helper)},
            "timeout_seconds": 120,
            "source_sha256": "a" * 64,
        }
        transition = root / "original_transition.py"
        transition.write_text(
            "FIELD = 'independent_training_dispatch'\n"
            "VERSION = 'basic-eligibility-authenticated-peer-dispatch136-v1'\n"
            "def preflight():\n"
            f"    return {self.preflight!r}\n"
            "def independent(manifest):\n"
            "    return manifest['independent']\n"
        )
        self.transition = transition
        self.policy = {
            "authority": "test-authority",
            "nll_pre_dispatch": {"version": "inherited-H5-checker"},
            "independent_training_dispatch": {
                "version": "basic-eligibility-authenticated-peer-dispatch136-v1",
                "helper": {"path": str(transition), "sha256": file_hash(transition)},
                "preflight": copy.deepcopy(self.preflight),
            },
            "checkpoint": "unchanged-checkpoint",
            "optimizer_step": 66,
        }
        self.row = {"historical_helper": copy.deepcopy(self.preflight["helper"])}
        self.job = {"manifest": {"test_authenticated": True, "independent": True}}

        def signed(document):
            if document.get("test_authenticated") is not True:
                raise ValueError("fixture manifest authentication refused")
            return document

        self.guards = SimpleNamespace(file_hash=file_hash, signed=signed)

    def project(self, policy=None, row=None, job=None):
        return historical_effective_policy(
            self.policy if policy is None else policy,
            self.job if job is None else job,
            self.row if row is None else row,
            "test-authority",
            self.guards,
        )

    def test_raw_policy_fails_original_gate_but_effective_policy_passes(self):
        original = copy.deepcopy(self.policy)

        def original_checker_gate(policy):
            if policy["nll_pre_dispatch"] != self.preflight:
                raise ValueError("exact pinned NLL pre-dispatch closure")

        with self.assertRaisesRegex(ValueError, "exact pinned NLL"):
            original_checker_gate(self.policy)
        effective = self.project()
        original_checker_gate(effective)
        self.assertEqual(self.policy, original)
        self.assertEqual(
            effective, dict(original, nll_pre_dispatch=copy.deepcopy(self.preflight))
        )
        # No alias permits downstream edits to mutate the authenticated input.
        effective["nll_pre_dispatch"]["helper"]["sha256"] = "b" * 64
        self.assertEqual(self.policy, original)

    def test_earlier_manifest_keeps_original_historical_policy(self):
        job = copy.deepcopy(self.job)
        job["manifest"]["independent"] = False
        self.assertIs(self.project(job=job), self.policy)

    def test_other_authority_is_rejected(self):
        policy = copy.deepcopy(self.policy)
        policy["authority"] = "foreign-authority"
        with self.assertRaisesRegex(ValueError, "historical preflight authority"):
            self.project(policy=policy)

    def test_unknown_transition_is_rejected(self):
        policy = copy.deepcopy(self.policy)
        policy["independent_training_dispatch"]["version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "exact authenticated H10"):
            self.project(policy=policy)

    def test_changed_pinned_transition_bytes_are_rejected(self):
        self.transition.write_text(self.transition.read_text() + "# changed\n")
        with self.assertRaisesRegex(ValueError, "pinned pre-dispatch helper"):
            self.project()

    def test_changed_nested_preflight_is_rejected(self):
        for field, value in [("timeout_seconds", 121), ("source_sha256", "b" * 64)]:
            with self.subTest(field=field):
                policy = copy.deepcopy(self.policy)
                policy["independent_training_dispatch"]["preflight"][field] = value
                with self.assertRaisesRegex(ValueError, "preflight declaration"):
                    self.project(policy=policy)

    def test_different_historical_checker_is_rejected(self):
        row = copy.deepcopy(self.row)
        row["historical_helper"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "same authenticated historical checker"):
            self.project(row=row)

    def test_manifest_authentication_is_still_required(self):
        job = copy.deepcopy(self.job)
        job["manifest"]["test_authenticated"] = False
        with self.assertRaisesRegex(ValueError, "manifest authentication refused"):
            self.project(job=job)


if __name__ == "__main__":
    unittest.main()
