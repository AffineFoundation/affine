"""Parallel per-batch controller for SN120 (shadow mode first).

One process on the validator host owns this controller. Every other part can
die at any time: rented GPU boxes, the R2 network path, worker threads, and
the controller process itself. The controller recovers from two things only:
its local journal and R2.

Module map
----------
``journal``    append-only, hash-chained, fsync per append, atomic rename.
``core``       pure state machine. No I/O, no clock. ``state = fold(journal)``.
``layout``     R2 key layout and strict document parsers.
``interfaces`` the seams: object store, provider, box agent, batch source, judge.
``guard``      shadow-mode hard denies (R2 prefix, state root, forbidden names).
``effects``    idempotent side effects (rent, launch, poll, settle, apply, ...).
``timers``     deadlines and cost, folded from the journal.
``controller`` the runtime loop: recover, reconcile, plan, run, commit.
``modelcheck`` exhaustive state-space checker over the real ``core``.
``sim``        fake provider / R2 / boxes for the fault-injection harness.

Rules the code keeps
--------------------
* A HEAD exists only after the full state (fp32 master, bf16 weights, manifest
  with sha256s) and the batch result are verified in R2 by size and sha256.
* The HEAD pointer is the only mutable R2 object. It moves by compare-and-swap
  on the expected parent. Only the controller writes it.
* A box writes only under ``staging/<batch>/<fence>/``. The fence is a number
  that is unique per lease. A box whose lease is gone cannot commit.
* A batch keeps its queue position for life. A lost attempt is retried on a
  fresh box. After N lost attempts the batch is credited as ``lost_infra``.
"""

VERSION = 'affine-batchctl-v1'
MODE_SHADOW = 'shadow'

# Box names must not start with ``affine-pe-``: the live rental code counts
# every ``affine-pe-*`` pod against its node cap, and the PIT host reaps
# unknown ``affine-pe-pit-*`` pods. This prefix keeps the two systems apart.
BOX_PREFIX_SHADOW = 'affine-bctl-shadow-'

# Every R2 key of a shadow run starts with this root.
R2_SHADOW_ROOT = 'shadow/batchctl/'

__all__ = ['VERSION', 'MODE_SHADOW', 'BOX_PREFIX_SHADOW', 'R2_SHADOW_ROOT']
