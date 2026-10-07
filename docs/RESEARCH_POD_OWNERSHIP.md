Research pod ownership (review-only; not deployed)

The pod reaper legitimately releases owned-name pods that have no local registry owner after 90 minutes. The isolated eighth H200 was rented without that registration, so the reaper destroyed it during a reference run. Registering a remote GPU lock does not establish local reaper ownership.

`ops/research_pod_ownership.py` supplies the lifecycle guard for reviewed provisioning and research supervisors. `rent_once` registers the intended name before the one original provider rental, then binds the actual provider pod ID. A rental observation timeout leaves the reserved name intact and refuses a second rental. `before_launch` verifies the exact retained owner, pod ID, and original rental-intent digest before calling the launcher. Observers call `heartbeat`. Terminal evidence calls `completed`, which retains the pod for subsequent qualification/reference/study jobs. It never marks a job's completion as pod retirement.

Use this guard from every provisioning and launch caller; adding the module alone does not protect a deployment. The reviewer supplies the authenticated original rental binding and explicitly authorizes retention. The local registry module is `/home/const/subnet120/ops/pods/registry.py`; use its existing locking and atomic-write implementation. This does not change the global reaper or protect unrelated unregistered pods. Separate reviewed retirement remains responsible for deleting intentionally retained rentals.

The retained H100 already has a static, indefinite retained ownership record. No change to it was required. The deleted H200 and its interrupted original reference execution must not be revived or represented as a completed scientific job.

Validation: eleven CPU tests cover registration before billing/GPU launch, failed registry writes preventing action, timeout without reissue, foreign ownership/provider substitution refusal, heartbeat, and retention after job completion. No rentals, production registration, reaper restart, or live deployment occurred in these tests.
