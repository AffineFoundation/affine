import base64
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey

from dashboard import evaluation_evidence_projection as p


class EvaluationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey(bytes(range(32)))
        self.authority = self.key.verify_key.encode().hex()

    def sign(self, payload):
        return {"payload": payload, "signer": self.authority,
                "signature": base64.b64encode(self.key.sign(p.canonical(payload)).signature).decode()}

    def fixture(self):
        cp = {"id": "a" * 64, "files": {}}
        manifest = {"checkpoint": cp, "epoch": "test-evaluation", "source_bundle": {"sha256": "b" * 64}}
        job = {"role": "evaluate", "job_id": "fixed32-test", "manifest": self.sign(manifest),
               "created_at": 10, "expires_at": 30, "runtime_versions": {"torch": "test"},
               "source_files": {"subnet/cached_sampling.py": "c" * 64},
               "heldout": [{"indices": list(range(32)), "seeds": list(range(100, 132)),
                            "harness": {"max_output_tokens": 1024}}]}
        report = {"role": "evaluate", "job_id": job["job_id"], "job_sha256": p.digest(job),
                  "checkpoint": cp["id"], "epoch": manifest["epoch"], "source_files": job["source_files"],
                  "runtime_versions": job["runtime_versions"], "chain_transactions": False,
                  "completed_at": 20, "heldout_failures": [], "heldout": [
                      {"index": i, "seed": i + 100, "checkpoint": cp["id"], "task_hash": "d" * 64,
                       "classification": "positive", "reward": 1, "native_graded": True,
                       "proof_verification_performed": False,
                       "turns": [{"output_tokens": 100, "output_sha256": "e" * 64,
                                  "prompt_tokens": 60, "secret_url": "https://secret?token=xyz"}]} for i in range(32)]}
        terminal = {"job_id": job["job_id"], "phase": "complete", "exit_code": 0,
                    "started_at": 12, "finished_at": 21}
        originals = {"original-job": self.sign(job), "original-report": report, "original-terminal": terminal}
        ack = {"version": "owned-cached-evaluation-durable-ack-v1", "checkpoint": cp,
               "durable_report_full_readback": True, "original_job": originals["original-job"],
               "original_report": report, "original_terminal": terminal,
               "job_sha256": p.digest(job), "report_sha256": p.digest(report),
               "original_terminal_sha256": p.digest(terminal),
               "full_readback_objects": {k: {"sha256": p.digest(v), "bytes": len(p.canonical(v))}
                                         for k, v in originals.items()}}
        return self.sign(ack)

    def test_cap_length_does_not_fabricate_truncation(self):
        row = p.project_turn({"output_tokens": 1024}, 1024, eos_only_early_stop=True)
        self.assertEqual(row["stop_reason"], "eos_or_budget")
        self.assertTrue(row["reached_token_budget"])
        self.assertIsNone(row["output_token_ids"])

    def test_short_output_inference_is_explicit_and_source_gated(self):
        self.assertIsNone(p.project_turn({"output_tokens": 10}, 1024)["stop_reason"])
        row = p.project_turn({"output_tokens": 10}, 1024, eos_only_early_stop=True)
        self.assertEqual(row["stop_reason"], "eos")
        self.assertTrue(row["stop_reason_basis"].startswith("inferred"))

    def test_recorded_eos_at_cap_and_token_ids_preserved(self):
        row = p.project_turn({"output": [1, 2], "stop_reason": "eos"}, 2)
        self.assertEqual(row["stop_reason"], "eos")
        self.assertEqual(row["output_token_ids"], [1, 2])

    def test_bad_length_rejected(self):
        for value in (-1, True, 1025, [False], [1.2]):
            with self.assertRaises(ValueError):
                p.project_turn({"output_tokens": value}, 1024)

    def test_signed_original_projection_omits_unlisted_fields(self):
        with patch.object(p, "AUTHORITY", self.authority):
            result = p.project_ack(self.fixture(), "b" * 64)
        self.assertEqual(len(result["tasks"]), 32)
        self.assertNotIn("secret", p.canonical(result).decode())
        self.assertEqual(result["tasks"][0]["output_length"], 100)

    def test_tampered_report_or_wrong_checkpoint_is_rejected(self):
        document = self.fixture()
        document["payload"]["original_report"]["checkpoint"] = "f" * 64
        document = self.sign(document["payload"])
        with patch.object(p, "AUTHORITY", self.authority), self.assertRaises(ValueError):
            p.project_ack(document, "b" * 64)

    def test_wrong_scope_is_rejected(self):
        with patch.object(p, "AUTHORITY", self.authority), self.assertRaises(ValueError):
            p.project_ack(self.fixture(), "f" * 64)

    def test_legacy_root_ack_preserves_report_without_inventing_terminal(self):
        payload = copy.deepcopy(self.fixture()["payload"])
        for key in ("original_terminal", "original_terminal_sha256", "full_readback_objects"):
            del payload[key]
        with patch.object(p, "AUTHORITY", self.authority):
            result = p.project_ack(self.sign(payload), "b" * 64)
        self.assertFalse(result["original_terminal_retained"])
        self.assertEqual(len(result["tasks"]), 32)

    def test_signed_but_inconsistent_verdict_is_rejected(self):
        payload = self.fixture()["payload"]
        payload["original_report"]["heldout"][0]["reward"] = 0
        payload["report_sha256"] = p.digest(payload["original_report"])
        payload["full_readback_objects"]["original-report"] = {
            "sha256": payload["report_sha256"], "bytes": len(p.canonical(payload["original_report"]))}
        with patch.object(p, "AUTHORITY", self.authority), self.assertRaises(ValueError):
            p.project_ack(self.sign(payload), "b" * 64)

    def test_no_evidence_remains_explicitly_missing(self):
        with tempfile.TemporaryDirectory() as root:
            rows = p.collect_evaluation_evidence(root, {"epoch-14": {
                "input_checkpoint": "a" * 64, "output_checkpoint": "b" * 64}})
        self.assertEqual(rows["epoch-14"]["availability"], "not_retained_or_not_evaluated")
        self.assertEqual(rows["epoch-14"]["evaluations"], [])
        self.assertEqual(rows["epoch-14"]["status"], "unavailable")

    def test_private_cap_study_in_same_queue_is_not_published(self):
        document = self.fixture()
        ack = document["payload"]
        job = ack["original_job"]["payload"]
        manifest = job["manifest"]["payload"]
        manifest["epoch"] = "nonpayable-fixed32-cap2048-20261009-private"
        job["manifest"] = self.sign(manifest)
        job["heldout"][0]["harness"]["max_output_tokens"] = 2048
        ack["original_job"] = self.sign(job)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "queue"
            (state / "durable-evaluation-acks").mkdir(parents=True)
            (state / "durable-evaluation-acks/private.json").write_bytes(p.canonical(self.sign(ack)))
            (root / "state/dashboard").mkdir(parents=True)
            scope = self.sign({"states": [str(state)], "indices": list(range(32)),
                               "source_sha256": "b" * 64})
            (root / "state/dashboard/cached-evaluator-sources.ROOT-SIGNED.json").write_bytes(p.canonical(scope))
            with patch.object(p, "AUTHORITY", self.authority):
                rows = p.collect_evaluation_evidence(root, {"epoch-14": {
                    "input_checkpoint": "a" * 64, "output_checkpoint": "f" * 64}})
        self.assertEqual(rows["epoch-14"]["evaluations"], [])
        self.assertEqual(rows["epoch-14"]["collection_issues"], [])


if __name__ == "__main__":
    unittest.main()
