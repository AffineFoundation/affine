"""Shared lifecycle path resolution (fail closed; no host-path defaults).

``PE_LIFE_STATE`` owns pending/ + apply/. ``PE_STATE`` owns GREENS / HEAD-CHAIN.
Both are required for live cutover; ``PE_SYNC_EXPECTED_HASH*`` is checked by
cutover preflight separately.
"""
from __future__ import annotations

import os
from pathlib import Path


def pe_life_state_root() -> Path:
    env = os.environ.get('PE_LIFE_STATE')
    if not env:
        raise RuntimeError(
            'PE_LIFE_STATE is unset (required with PE_FLAG / PE_STATE / '
            'PE_SYNC_EXPECTED_HASH* before cutover restart; no host-path defaults)'
        )
    return Path(env)


def pe_state_root() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if not env:
        raise RuntimeError(
            'PE_STATE is unset (required with PE_FLAG / PE_LIFE_STATE / '
            'PE_SYNC_EXPECTED_HASH* before cutover restart; no host-path defaults)'
        )
    return Path(env)


def pending_dir() -> Path:
    """``PE_LIFE_STATE/pending`` — same root pe_live.apply_tick / reconcile read."""
    return pe_life_state_root() / 'pending'


def apply_records_dir() -> Path:
    return pe_life_state_root() / 'apply'
