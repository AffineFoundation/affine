"""Failure-oriented controls for the inactive uncertain-submission helper."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from subnet.weight_submission_reconciliation import (
    BittensorReadonlyReader, collect_evidence, journal_intent, make_intent,
    reconcile, validate_intent,
)


def intent(**changes):
    values = dict(owner="public-owner", owner_uid=0, netuid=120, mecid=0,
                  window_end=1791486000, expected_weights=[[2, 65535], [5, 1234]],
                  assessment_sha256="a" * 64, policy_sha256="b" * 64,
                  registrations_sha256="c" * 64, attempt_start_block=100,
                  nonce_hint=4111)
    values.update(changes)
    return make_intent(**values)


class Reader:
    """No execution API, keys, network access, or mutable chain state."""

    def __init__(self):
        self.calls = []
        self.head = {"number": 105, "hash": "head-105"}
        self.state = {"block": 105, "block_hash": "head-105", "last_update": 102,
                      "weights": [[5, 1234], [2, 65535]], "owner_pending": False}
        self.commit = dict(signer="public-owner", netuid=120, mecid=0, nonce=4111,
                           extrinsic_hash="actual-signed-hash", extrinsic_index=5,
                           call_function="commit_timelocked_mechanism_weights",
                           success=True)

    def finalized_head(self):
        self.calls.append("finalized_head")
        return copy.deepcopy(self.head)

    def owner_state(self, frozen_intent, head):
        self.calls.append("owner_state")
        return copy.deepcopy(self.state)

    def block(self, number):
        self.calls.append(number)
        return {"number": number, "hash": f"block-{number}",
                "commits": [copy.deepcopy(self.commit)] if number == 102 else []}


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.intent = intent()
        self.reader = Reader()

    def evidence(self):
        return collect_evidence(self.reader, self.intent)

    def assert_fenced(self, evidence, reason=None, **kwargs):
        result = reconcile(self.intent, evidence, **kwargs)
        self.assertTrue(result["preserve_fence"])
        self.assertFalse(result["allow_current_window_submission"])
        self.assertFalse(result["old_window_replay_allowed"])
        if reason:
            self.assertEqual(result["reason"], reason)
        return result

    def test_timeout_after_success_can_resolve_from_finalized_reveal(self):
        result = reconcile(self.intent, self.evidence())
        self.assertEqual(result["status"], "submitted_revealed")
        self.assertEqual(result["resolved_window_end"], 1791486000)
        self.assertEqual(result["commit"]["block"], 102)
        self.assertFalse(result["preserve_fence"])
        self.assertTrue(result["allow_current_window_submission"])
        self.assertFalse(result["old_window_replay_allowed"])
        self.assertEqual(self.reader.calls, ["finalized_head", "owner_state", 100, 101, 102])

    def test_pending_reveal_never_allows_resubmission(self):
        self.reader.state["owner_pending"] = True
        result = self.assert_fenced(self.evidence(), "owner_timelock_pending")
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["finalized_commit_bound"])

    def test_actual_signed_identity_resolves_finalized_commit_before_reveal(self):
        self.reader.state["owner_pending"] = True
        observation = dict(intent_sha256=self.intent["sha256"],
                           extrinsic_hash="actual-signed-hash", nonce=4111)
        result = reconcile(self.intent, self.evidence(), actual_signed_observation=observation)
        self.assertEqual(result["status"], "submitted_finalized")
        self.assertFalse(result["preserve_fence"])
        self.assertFalse(result["revealed_weights_asserted"])

    def test_separate_plan_hash_does_not_identify_executed_commit(self):
        observation = dict(intent_sha256=self.intent["sha256"],
                           extrinsic_hash="different-plan-hash", nonce=4111)
        self.assert_fenced(self.evidence(), "missing_or_ambiguous_finalized_commit",
                           actual_signed_observation=observation)

    def test_pre_read_nonce_is_only_hint(self):
        self.intent = intent(nonce_hint=4110)
        self.assertEqual(reconcile(self.intent, self.evidence())["status"], "submitted_revealed")

    def test_wrong_actual_signed_nonce_is_not_hint(self):
        observation = dict(intent_sha256=self.intent["sha256"],
                           extrinsic_hash="actual-signed-hash", nonce=4110)
        self.assert_fenced(self.evidence(), "missing_or_ambiguous_finalized_commit",
                           actual_signed_observation=observation)

    def test_one_u16_difference_prevents_false_success(self):
        self.reader.state["weights"][0][1] += 1
        self.assert_fenced(self.evidence(), "revealed_vector_mismatch")

    def test_de_registration_zero_cannot_silently_drop_expected_recipient(self):
        self.reader.state["weights"][0][1] = 0
        self.assert_fenced(self.evidence(), "revealed_vector_mismatch")

    def test_unrelated_zero_storage_row_does_not_change_vector(self):
        self.reader.state["weights"].append([99, 0])
        self.assertEqual(reconcile(self.intent, self.evidence())["status"], "submitted_revealed")

    def test_old_or_later_last_update_does_not_bind_original_commit(self):
        for block in (99, 103):
            with self.subTest(last_update=block):
                self.reader.state["last_update"] = block
                self.assert_fenced(self.evidence(), "missing_or_ambiguous_finalized_commit")

    def test_failed_extrinsic_cannot_be_a_success_receipt(self):
        self.reader.commit["success"] = False
        self.assert_fenced(self.evidence(), "missing_or_ambiguous_finalized_commit")

    def test_owner_netuid_mechanism_and_method_are_all_required(self):
        for key, value in (("signer", "other-owner"), ("netuid", 121),
                           ("mecid", 1), ("call_function", "set_weights")):
            with self.subTest(field=key):
                evidence = self.evidence()
                evidence["blocks"][-1]["commits"][0][key] = value
                self.assert_fenced(evidence, "missing_or_ambiguous_finalized_commit")

    def test_two_successful_owner_commits_in_same_block_are_ambiguous(self):
        evidence = self.evidence()
        other = copy.deepcopy(evidence["blocks"][-1]["commits"][0])
        other.update(extrinsic_hash="second-hash", extrinsic_index=6, nonce=4112)
        evidence["blocks"][-1]["commits"].append(other)
        self.assert_fenced(evidence, "missing_or_ambiguous_finalized_commit")

    def test_exact_signed_hash_can_disambiguate_same_block(self):
        evidence = self.evidence()
        other = copy.deepcopy(evidence["blocks"][-1]["commits"][0])
        other.update(extrinsic_hash="second-hash", extrinsic_index=6, nonce=4112)
        evidence["blocks"][-1]["commits"].append(other)
        observation = dict(intent_sha256=self.intent["sha256"],
                           extrinsic_hash="actual-signed-hash", nonce=4111)
        self.assertEqual(reconcile(self.intent, evidence,
                                   actual_signed_observation=observation)["status"],
                         "submitted_finalized")

    def test_unknown_pending_state_retains_fence(self):
        for value in (None, "false", 0):
            with self.subTest(pending=value):
                self.reader.state["owner_pending"] = value
                self.assert_fenced(self.evidence(), "unknown_pending_state")

    def test_mixed_unfinalized_or_gapped_chain_evidence_is_rejected(self):
        for case in ("different_hash", "unfinalized", "missing_block"):
            with self.subTest(case=case):
                evidence = self.evidence()
                if case == "different_hash":
                    evidence["owner_state"]["block_hash"] = "other-fork"
                elif case == "unfinalized":
                    evidence["finalized_head"]["number"] = 101
                else:
                    evidence["blocks"].pop(1)
                self.assert_fenced(evidence)

    def test_scan_cap_never_becomes_proof_of_absence_or_expiry(self):
        evidence = collect_evidence(self.reader, self.intent, max_blocks=2)
        self.assertEqual([block["number"] for block in evidence["blocks"]], [100, 101])
        self.assert_fenced(evidence, "missing_or_ambiguous_finalized_commit")

    def test_timeout_and_rpc_error_retain_fence_without_leaking_exception(self):
        times = iter([0, 0, 0, 1, 61])
        evidence = collect_evidence(self.reader, self.intent, monotonic=lambda: next(times))
        self.assertEqual(evidence["read_error"], "TimeoutError")
        self.assert_fenced(evidence, "incomplete_read")

        def unavailable(number):
            raise ConnectionError("private-rpc-token-in-provider-message")

        self.reader.block = unavailable
        evidence = self.evidence()
        self.assertNotIn("private-rpc-token", json.dumps(evidence))
        self.assert_fenced(evidence, "incomplete_read")

    def test_evidence_for_other_frozen_assessment_is_not_reusable(self):
        evidence = self.evidence()
        evidence["intent_sha256"] = intent(assessment_sha256="d" * 64)["sha256"]
        self.assert_fenced(evidence, "wrong_intent")

    def test_journal_is_durable_create_once_and_never_replaces_old_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent.json"
            journal_intent(path, self.intent)
            old = path.read_bytes()
            self.assertEqual(validate_intent(json.loads(old)), self.intent["payload"])
            with self.assertRaises(FileExistsError):
                journal_intent(path, intent(window_end=1791489600))
            self.assertEqual(path.read_bytes(), old)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_partial_and_symlink_journals_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "partial"
            target.write_bytes(b'{"partial":')
            link = Path(directory) / "link"
            link.symlink_to(target)
            for path in (target, link):
                with self.assertRaises(FileExistsError):
                    journal_intent(path, self.intent)
            self.assertEqual(target.read_bytes(), b'{"partial":')

    def test_invalid_frozen_inputs_never_reach_chain_reader(self):
        for values in ({"expected_weights": []}, {"expected_weights": [[0, 65535]]},
                       {"expected_weights": [[2, 1.0]]},
                       {"expected_weights": [[2, 65535], [2, 1]]},
                       {"expected_weights": [[2, 65535], [5, 0]]},
                       {"expected_weights": [[2, 65536]]}, {"netuid": True},
                       {"window_end": 1791486001}, {"assessment_sha256": "invalid"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                intent(**values)
        corrupt = copy.deepcopy(self.intent)
        corrupt["payload"]["expected_weights"][0][1] -= 1
        with self.assertRaises(ValueError):
            collect_evidence(self.reader, corrupt)
        self.assertEqual(self.reader.calls, [])


class SDKReadAdapterTests(unittest.TestCase):
    def setUp(self):
        self.queries = []
        self.queues = {}
        self.storage = dict(CommitRevealWeightsVersion=4, Uids=0, Keys="public-owner",
                            LastUpdate=[102], Weights=[[2, 65535], [5, 1234]])
        args = dict(netuid=120, mecid=0, reveal_round=32894943)
        self.block_value = dict(header={"number": 102, "hash": "block-102"}, extrinsics=[
            dict(address="public-owner", nonce=4111, extrinsic_hash="actual-signed-hash",
                 call=dict(call_module="SubtensorModule",
                           call_function="commit_timelocked_mechanism_weights",
                           call_args=[dict(name=name, value=value) for name, value in args.items()]))])
        self.events = [dict(extrinsic_idx=0, module_id="System", event_id="ExtrinsicSuccess"),
                       dict(extrinsic_idx=0, module_id="SubtensorModule",
                            event_id="TimelockedWeightsCommitted")]
        self.wrapped_rpc = False

        def rpc(method, params):
            value = "head-105" if method == "chain_getFinalizedHead" else {"number": "0x69"}
            return {"result": value} if self.wrapped_rpc else value

        def query(module, name, params, *, block_hash):
            self.queries.append((module, name, tuple(params), block_hash))
            return copy.deepcopy(self.storage[name])

        def query_map(module, name, params, *, block_hash):
            self.queries.append((module, name, tuple(params), block_hash))
            return copy.deepcopy(self.queues.get(name, []))

        substrate = SimpleNamespace(raw=SimpleNamespace(rpc_request=rpc), query=query,
                                     query_map=query_map,
                                     get_block=lambda block_number: copy.deepcopy(self.block_value),
                                     events=lambda block_hash: copy.deepcopy(self.events))
        chain = SimpleNamespace(block=105, _call=lambda value: value,
                                _client=SimpleNamespace(_substrate=substrate))
        self.reader = BittensorReadonlyReader(chain)
        self.head = self.reader.finalized_head()

    def test_actual_sdk_rpc_return_shapes_and_exact_hash_storage_reads(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                self.wrapped_rpc = wrapped
                head = self.reader.finalized_head()
                state = self.reader.owner_state(intent()["payload"], head)
                self.assertEqual(state["block"], 105)
                self.assertFalse(state["owner_pending"])
                self.assertTrue(all(query[3] == "head-105" for query in self.queries))

    def test_pending_queues_discriminate_owner_and_reject_unknown_shapes(self):
        self.queues["TimelockedWeightCommits"] = [(10, [("other-owner", 100, "ciphertext", 123)])]
        self.assertFalse(self.reader.owner_state(intent()["payload"], self.head)["owner_pending"])
        self.queues["CRV3WeightCommits"] = [(10, [("public-owner", "ciphertext", 123)])]
        self.assertTrue(self.reader.owner_state(intent()["payload"], self.head)["owner_pending"])
        self.queues["CRV3WeightCommitsV2"] = [(10, [("unknown-short-format",)])]
        with self.assertRaisesRegex(ValueError, "unknown pending"):
            self.reader.owner_state(intent()["payload"], self.head)

    def test_dispatch_and_commit_events_must_match_exact_extrinsic(self):
        self.assertTrue(self.reader.block(102)["commits"][0]["success"])
        self.events[0]["extrinsic_idx"] = 1
        self.assertFalse(self.reader.block(102)["commits"][0]["success"])
        self.events[0]["extrinsic_idx"] = 0
        self.events[1]["event_id"] = "UnrelatedEvent"
        self.assertFalse(self.reader.block(102)["commits"][0]["success"])

    def test_undecoded_extrinsic_or_unfinalized_block_is_not_skipped(self):
        with self.assertRaisesRegex(ValueError, "not finalized"):
            self.reader.block(106)
        self.block_value["extrinsics"].append(None)
        with self.assertRaisesRegex(ValueError, "undecoded extrinsic"):
            self.reader.block(102)

    def test_changed_owner_mapping_or_unknown_runtime_variant_fails_closed(self):
        for key, value in (("Uids", 1), ("Keys", "other-owner"),
                           ("CommitRevealWeightsVersion", 5)):
            with self.subTest(field=key):
                original = self.storage[key]
                self.storage[key] = value
                with self.assertRaises(ValueError):
                    self.reader.owner_state(intent()["payload"], self.head)
                self.storage[key] = original


if __name__ == "__main__":
    unittest.main()
