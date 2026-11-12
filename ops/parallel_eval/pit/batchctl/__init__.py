"""Parallel per-batch controller for SN120 (two modes: shadow and live).

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
``guard``      hard denies of both modes (R2 prefix, state root, box names, forbidden names).
``effects``    idempotent side effects (rent, launch, poll, settle, apply, ...).
``timers``     deadlines and cost, folded from the journal.
``controller`` the runtime loop: recover, reconcile, plan, run, commit.
``modelcheck`` exhaustive state-space checker over the real ``core``.
``sim``        fake provider / R2 / boxes for the fault-injection harness.

Modes
-----
``shadow`` is the default. ``live`` is refused unless the caller gives its
explicit flag (``guard.check_mode``). Each mode has its own R2 root, its own
box name prefix and its own state root marker. A run of one mode cannot reach
the objects, the boxes or the files of a run of the other mode.

Rules the code keeps
--------------------
* A HEAD exists only after the full state (fp32 master, bf16 weights, manifest
  with sha256s) and the batch result are verified in R2 by size and sha256.
* The HEAD pointer is the only mutable R2 object. It moves by compare-and-swap
  on the expected parent. Only the controller writes it.
* A box writes only under ``staging/<batch>/<fence>/``. The fence is a number
  that is unique per lease. A box whose lease is gone cannot commit.
* A box has one base: the newest HEAD at the moment that the box is bound to
  its batch. Every box runs to the end against its base. A HEAD that moves
  stops no box.
* A decision is final. GREEN against its base is paid; it is applied only if
  it is the first GREEN on a base that is still the HEAD. RED against its
  base is forgotten: not paid, and never trained again. No decided batch is
  ever trained on a newer HEAD.
* A box that is lost before a decision is not a decision. The batch keeps its
  queue position and gets a fresh box, which trains on the newest HEAD of its
  own moment. After N lost boxes the batch is credited as ``lost_infra``.
"""

# v2: the base belongs to the box (the newest HEAD when it is bound), not to
# the batch. A v1 journal names one base per batch and is refused.
VERSION = 'affine-batchctl-v2'
MODE_SHADOW = 'shadow'
MODE_LIVE = 'live'
MODES = (MODE_SHADOW, MODE_LIVE)

# Box names must not start with ``affine-pe-``: the live rental code counts
# every ``affine-pe-*`` pod against its node cap, and the PIT host reaps
# unknown ``affine-pe-pit-*`` pods. These prefixes keep the two systems apart.
BOX_PREFIX_SHADOW = 'affine-bctl-shadow-'
BOX_PREFIX_LIVE = 'affine-bctl-live-'

# Every R2 key of a shadow run starts with this root. It is inside the one
# shadow prefix of the PIT package (``pit-shadow/``, see ``pit.shadow_guard``).
R2_SHADOW_ROOT = 'pit-shadow/batchctl/'
# Every R2 key of a live run starts with this root. It is outside the shadow prefix ``pit-shadow/``.
R2_LIVE_ROOT = 'pit/batchctl/'
R2_ROOTS = {MODE_SHADOW: R2_SHADOW_ROOT, MODE_LIVE: R2_LIVE_ROOT}
BOX_PREFIXES = {MODE_SHADOW: BOX_PREFIX_SHADOW, MODE_LIVE: BOX_PREFIX_LIVE}

__all__ = [
    'VERSION', 'MODE_SHADOW', 'MODE_LIVE', 'MODES', 'BOX_PREFIX_SHADOW', 'BOX_PREFIX_LIVE', 'BOX_PREFIXES',
    'R2_SHADOW_ROOT', 'R2_LIVE_ROOT', 'R2_ROOTS',
]
