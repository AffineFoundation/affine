"""Shared PE life-state path resolution (fail closed; no host-path defaults).

Two distinct roots (do not conflate):

* ``PE_LIFE_STATE`` — PE life-state dir: ``pending/`` + ``apply/`` (pe_live /
  apply_live). Not the training lifecycle tree.
* ``LIFE_ROOT`` / ``PE_LIFE`` — training lifecycle root: ``CURRENT.json`` +
  ``epochs/`` (queue_ingest floor). Never read via this module.
* ``PE_STATE`` — GREENS / HEAD-CHAIN / pit uploads.

``PE_SYNC_EXPECTED_HASH*`` is checked by cutover preflight separately.
"""
from __future__ import annotations

import os
from pathlib import Path


def pe_life_state_root() -> Path:
    env = os.environ.get('PE_LIFE_STATE')
    if not env:
        raise RuntimeError(
            'PE_LIFE_STATE is unset (PE pending/apply dir; required with PE_FLAG / '
            'PE_STATE / LIFE_ROOT / PE_SYNC_EXPECTED_HASH* before cutover restart; '
            'no host-path defaults)'
        )
    return Path(env)


def pe_state_root() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if not env:
        raise RuntimeError(
            'PE_STATE is unset (required with PE_FLAG / PE_LIFE_STATE / '
            'LIFE_ROOT / PE_SYNC_EXPECTED_HASH* before cutover restart; '
            'no host-path defaults)'
        )
    return Path(env)


def pending_dir() -> Path:
    """``PE_LIFE_STATE/pending`` — same root pe_live.apply_tick / reconcile read."""
    return pe_life_state_root() / 'pending'


def apply_records_dir() -> Path:
    return pe_life_state_root() / 'apply'
