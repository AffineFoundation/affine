The continuous auditor can load a separately ROOT-signed historical numerical resolution policy from `continuous_audit_service.numerical_resolution`. This option is off by default. Its exact configuration fields are `policy_document` and `reference_archives`; each archive entry contains `ack_path` and `archive_path`.

Startup authenticates the policy and full original reference archives using their ROOT-signed acknowledgments. Symlinks, truncated archives, unknown fields and invalid signatures are refused. After the signed effective cutoff, snapshots use the same UNKNOWN coverage policy as the hourly writer. Earlier snapshots and existing immutable hourly files remain unchanged. Original reports are retained; numerical uncertainty grants no VALID credit and establishes neither sampler nor grader completion.

This is a CPU accounting overlay, independent of model inference thresholds. Historical allowlists do not resolve newer cases or qualify a prospective numerical runtime. Production requires an explicitly reviewed signed durable service/config admission; updating repository source alone does not deploy it.

Portable startup controls use generated test authorities and authentic test reference archives, without private state: `python -m unittest discover -s tests -p 'test_continuous_auditor_numerical_startup.py'`.
