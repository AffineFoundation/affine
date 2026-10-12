import copy
import unittest

from ops.original_job_completion import (
    digest, exact_job_completion_allowed, extend_checked_predicate,
)


class CompletionAdmission(unittest.TestCase):
    def setUp(self):
        self.manifest = {"epoch": "original-157", "checkpoint": "checkpoint66"}
        self.job = {"job_id": "original-157-train", "role": "train",
                    "created_at": 100, "expires_at": 200,
                    "manifest": self.manifest, "optimizer_step": 66}
        self.prior = {"job_id": self.job["job_id"], "checkpoint": "checkpoint66"}
        self.scope = {
            "authority": "operator", "job_id": self.job["job_id"],
            "actual_started_at": 190, "completion_before": 3790,
            "original_created_at": 100, "original_expires_at": 200,
            "original_job_file": {"sha256": "original-file"},
            "original_job_payload_sha256": digest(self.job),
            "original_prior": self.prior, "manifest_sha256": digest(self.manifest),
        }

    def allowed(self, completed, **changes):
        fields = dict(authority="operator", job=self.job,
                      report={"job_id": self.job["job_id"], "completed_at": completed},
                      prior=self.prior, manifest=self.manifest,
                      original_job_file_sha256="original-file")
        fields.update(changes)
        return exact_job_completion_allowed(self.scope, **fields)

    def test_late_original_allowed_without_changing_its_deadline(self):
        before = copy.deepcopy(self.job)
        self.assertTrue(self.allowed(201))
        self.assertTrue(self.allowed(3789.999))
        self.assertEqual(self.job, before)

    def test_time_boundaries_and_nonfinite_values(self):
        for value in (189.9, 3790, float("nan"), float("inf"), "201", None, True):
            with self.subTest(value=value):
                self.assertFalse(self.allowed(value))
        self.scope["completion_before"] += 1
        self.assertFalse(self.allowed(201))

    def test_cannot_extend_a_job_that_started_after_original_deadline(self):
        self.scope["actual_started_at"] = 200
        self.assertFalse(self.allowed(201))

    def test_another_job_or_modified_optimizer_cannot_use_scope(self):
        for key, value in (("job_id", "replacement-job"), ("expires_at", 201),
                           ("role", "evaluate"), ("optimizer_step", 65)):
            job = dict(self.job, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(self.allowed(202, job=job))

    def test_authority_manifest_prior_and_file_are_bound(self):
        for fields in ({"authority": "foreign"}, {"manifest": {"epoch": "158"}},
                       {"prior": {"job_id": "other"}},
                       {"original_job_file_sha256": "modified"},
                       {"report": {"job_id": "other", "completed_at": 201}}):
            self.assertFalse(self.allowed(201, **fields))

    def test_report_body_and_original_exception_remain_intact(self):
        namespace = {"_completion_time_allowed": lambda report: report["ordinary"]}
        exec("def checked(report):\n"
             "    if not report['valid_source']: raise ValueError('source')\n"
             "    if not _completion_time_allowed(report): raise ValueError('time')\n"
             "    if not report['valid_optimizer']: raise ValueError('optimizer')\n"
             "    return 'accepted'\n", namespace)
        original = namespace["checked"]
        wrapped = extend_checked_predicate(
            original, expected_code=original.__code__,
            additional_predicate=lambda report: report.get("exact_original", False))
        self.assertIs(wrapped.__code__, original.__code__)
        ordinary = dict(ordinary=True, valid_source=True, valid_optimizer=True)
        special = dict(ordinary=False, exact_original=True,
                       valid_source=True, valid_optimizer=True)
        self.assertEqual(wrapped(ordinary), "accepted")
        self.assertEqual(wrapped(special), "accepted")
        with self.assertRaisesRegex(ValueError, "time"):
            original(special)
        for field in ("valid_source", "valid_optimizer"):
            with self.assertRaises(ValueError):
                wrapped(dict(special, **{field: False}))

    def test_foreign_checked_body_refused(self):
        with self.assertRaisesRegex(ValueError, "exact original"):
            extend_checked_predicate(lambda: None, expected_code=(lambda x: x).__code__,
                                     additional_predicate=lambda: True)


if __name__ == "__main__":
    unittest.main()
