# Native task snapshot deployment

Executable archive verification alone does not establish native environment readiness. The current approved MATH snapshot is 8,146,242 bytes with SHA-256 `77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3`. The original signed source archive `7459c28c…` contains those exact bytes. The FE archive omitted the snapshot; its original calibration failed before model creation and remains a failed historical execution.

Run `ops/native_task_asset_deployment.py --descriptor <original ROOT-signed source descriptor>` during deployment, before a calibration, miner, grader, or trainer can resolve its native task specification. The software verifies the signed archive descriptor, downloads and checks the complete archive SHA, extracts only the exact regular snapshot member, and atomically installs the verified dataset outside immutable executable source. Warm deployment verifies the local file instead of downloading again. Corrupt or aliased existing files fail closed and are preserved.

The default content-addressed cache is `/var/tmp/affine-approved-native-task-assets/<snapshot SHA>/original-math7496.tasks.json`. The controller and every executing role need their own verified copy. Changing `task_snapshot` changes the native environment source hash because that hash includes configuration. Recompute that hash under the exact executable source, check native sessions on all executing roles, then approve the new path-bound specification. Do not relabel historical specifications or retry their failed jobs as though they had succeeded.

Owned-fleet recovery is separate from public miner compatibility. External miners need the descriptor/bootstrap data contract and the approved declared path before the FE archive can be described as self-contained or automatically runnable. This CPU deployment command neither dispatches calibration nor creates a model or signs authority documents.

For the next immutable source, `ops.native_source_snapshot_guard.seal_source_archive`
automatically includes each relative native `task_snapshot` from an authenticated
approved source inventory, even when the asset is ignored by Git. Its complete
inventory contains the data file and the archive is checked again before it can
be published. Call `publish_source_bundle(..., environments=approved_envs,
approved_files=approved_inventory)` to reject missing, changed, duplicate, or
nonregular dependencies before any bucket operation. Existing historical calls
remain unchanged. Absolute external recovery paths are deliberately rejected by
this self-contained packaging gate; use the original relative native spec for a
new complete archive. The authenticated client bootstrap then extracts the
snapshot with the other approved source files.
