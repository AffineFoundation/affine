"""Durable boundary for one hourly weight transaction.

The caller holds the existing global writer lock; ChainAdapter additionally
holds weights.lock. All planning, preparation and signing are pre-broadcast.
Only submit_prepared() can send, exactly once, after fsynced intent + signed
identity + cursor. No old transaction is ever rebroadcast during recovery.

SDK execute() has no signed-call hook and replans. This adapter deliberately
uses its documented prepare/attach/submit-signed transport flow instead; the
locally inspected SDK seams are pinned by the operator's writer module policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from subnet.weight_submission_reconciliation import (
    validate_era,
    BittensorReadonlyReader, collect_evidence, journal_intent, make_intent,
    reconcile, validate_intent,
)


def sdk_seam_paths():
    # Candidate uses documented transport methods through private Client members;
    # these precise Python seams must be independently pinned before activation.
    import importlib.util
    modules = ('bittensor.intents.weights', 'bittensor.executor', 'bittensor._substrate',
               'bittensor._transport.interface', 'bittensor._transport.codec',
               'bittensor._transport.contract', 'bittensor.signing')
    return {str(Path(importlib.util.find_spec(name).origin).resolve()) for name in modules}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def durable_atomic(path, value):
    """Unique temporary, fsync file, rename, fsync directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp.exists():
            temp.unlink()


def make_recorded_intent(bt, *, netuid, uids, weights, version_key):
    """Same SDK SetWeights recipe, exposing the vector used by THIS plan.

    No global monkeypatch. The supported Intent subclass seam delegates to the
    installed weight helpers; those private helper files must be policy-pinned.
    Only direct timelocked CR4/mecid0 is qualified here.
    """
    from bittensor.intents.weights import _preflight, _conform, _build_timelocked

    class RecordedSetWeights(bt.SetWeights):
        async def build(self, substrate, wallet):
            preflight = await _preflight(substrate, self.hotkey_address(wallet), self.netuid, self.mechid)
            norm_uids, values = _conform(self.uids, self.weights, preflight, self.netuid)
            if self.mechid != 0 or not preflight.commit_reveal:
                raise ValueError('only current direct timelocked mechanism is qualified')
            if preflight.uid in norm_uids:
                raise ValueError('owner allocation is forbidden')
            built = await _build_timelocked(substrate, self.hotkey_public_key(wallet),
                self.netuid, self.mechid, norm_uids, values, self.version_key, 4)
            self.journal_vector = sorted([int(uid), int(value)] for uid, value in zip(norm_uids, values))
            self.journal_owner_uid = int(preflight.uid)
            self.journal_call_sha256 = hashlib.sha256(built.call.data).hexdigest()
            return built

    return RecordedSetWeights(netuid=netuid, uids=uids, weights=weights, version_key=version_key)


def prepare_exact_plan(chain, plan, intent, wallet):
    """Read, sign and assemble but do NOT transmit. Returns exact SDK object."""
    if not plan.ok or plan.signer != 'hotkey' or plan.signer_address != wallet.hotkey.ss58_address:
        raise ValueError('actual direct owner plan required')
    if hashlib.sha256(plan.call.data).hexdigest() != intent.journal_call_sha256:
        raise ValueError('normalized vector is not bound to this plan')
    unsigned = chain.prepare_call(plan.call, address=plan.signer_address,
                                  crypto_type=wallet.hotkey.crypto_type, period=64)
    if unsigned.address != plan.signer_address or unsigned.call_data != plan.call.data:
        raise ValueError('prepared transaction changed owner or call')
    # The actual prepared mortal era is signed into this transaction. Period64
    # has phase granularity1, so the SDK's normalized current is its birth.
    if not isinstance(unsigned.era, dict) or set(unsigned.era) != {'period', 'current'}:
        raise ValueError('actual normalized mortal era required')
    intent.journal_era = validate_era(dict(period=unsigned.era['period'],
        birth=unsigned.era['current'], death=unsigned.era['current'] + 64,
        block_hash=unsigned.era_block_hash))
    signature = wallet.hotkey.sign(unsigned.payload)
    # Documented SDK transport assembly method, no network broadcast. Pin SDK.
    signed = chain._call(chain._client._substrate.raw.attach_signature(unsigned, signature))
    expected_hash = '0x' + hashlib.blake2b(signed.data, digest_size=32).hexdigest()
    if signed.extrinsic_hash != expected_hash:
        raise ValueError('SDK signed extrinsic hash mismatch')
    return signed, int(unsigned.nonce)


def submit_prepared(chain, signed, wallet):
    """Only broadcast path. No planning, re-signing, retry or second ciphertext."""
    return chain._call(chain._client._substrate.submit_signed(
        signed, wallet.hotkey, wait_for_inclusion=True, wait_for_finalization=True))


class SubmissionJournal:
    def __init__(self, cursor_path, chain_state, *, assessment_sha256=None,
                 policy_sha256=None, authority=None):
        self.cursor_path, self.chain_state = Path(cursor_path), Path(chain_state)
        self.assessment_sha256, self.policy_sha256 = assessment_sha256, policy_sha256
        self.authority = authority

    def begin(self, *, owner, owner_uid, netuid, window_end, vector, registrations,
              attempt_start_block, signed, nonce, era=None):
        """Fails before broadcast. Orphan journals are never treated as sent."""
        if self.cursor_path.exists():
            cursor = json.loads(self.cursor_path.read_text())
            if cursor.get('status') == 'submitting':
                raise RuntimeError('uncertain chain outcome must be reconciled')
        document = make_intent(owner=owner, owner_uid=owner_uid, netuid=netuid,
            window_end=window_end, expected_weights=vector,
            assessment_sha256=self.assessment_sha256, policy_sha256=self.policy_sha256,
            registrations_sha256=digest(registrations), attempt_start_block=attempt_start_block,
            nonce_hint=nonce)
        directory = self.cursor_path.parent / 'attempts' / (str(window_end) + '-' + uuid.uuid4().hex)
        directory.mkdir(parents=True, mode=0o700)
        for parent in (directory.parent, directory.parent.parent):
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        journal_intent(directory / 'intent.json', document)
        observation = dict(intent_sha256=document['sha256'], extrinsic_hash=signed.extrinsic_hash,
                           nonce=nonce, signed_bytes_sha256=hashlib.sha256(signed.data).hexdigest())
        if era is not None: observation['era'] = validate_era(era)
        durable_atomic(directory / 'signed-observation.json', observation)
        cursor = dict(status='submitting', window_end=window_end,
            assessment_sha256=self.assessment_sha256, policy_sha256=self.policy_sha256,
            intent_sha256=document['sha256'], attempt_directory=str(directory),
            signed_observation_sha256=digest(observation))
        # This fsynced fence is the last local operation before the sole send.
        durable_atomic(self.cursor_path, cursor)
        return cursor

    def recover(self, chain, *, reader=None):
        """Caller holds both existing writer locks. Proven terminal states only."""
        cursor = json.loads(self.cursor_path.read_text())
        if cursor.get('status') != 'submitting':
            return dict(status='no_recovery_required')
        # An old cursor has no actual signed identity. Never guess/clear it.
        if not cursor.get('attempt_directory'):
            raise RuntimeError('legacy uncertain chain outcome requires actual-chain reconciliation')
        directory = Path(cursor['attempt_directory'])
        if directory.parent.resolve() != (self.cursor_path.parent / 'attempts').resolve() or directory.is_symlink():
            raise ValueError('attempt journal outside local cursor namespace')
        document = json.loads((directory / 'intent.json').read_text())
        intent = validate_intent(document)
        observation = json.loads((directory / 'signed-observation.json').read_text())
        if (document['sha256'] != cursor['intent_sha256'] or
            digest(observation) != cursor['signed_observation_sha256'] or
            intent['window_end'] != cursor['window_end'] or
            intent['assessment_sha256'] != cursor['assessment_sha256'] or
            intent['policy_sha256'] != cursor['policy_sha256'] or
            observation['intent_sha256'] != document['sha256']):
            raise ValueError('attempt journal integrity mismatch')
        evidence = collect_evidence(reader or BittensorReadonlyReader(chain), document,
                                    max_blocks=128, budget_seconds=90, actual_signed_observation=observation)
        decision = reconcile(document, evidence, actual_signed_observation=observation)
        receipt = dict(original_cursor=cursor, evidence=evidence, decision=decision)
        durable_atomic(directory / ('reconciliation-' + digest(receipt) + '.json'), receipt)
        if decision['preserve_fence']:
            return decision
        if decision['status'] in ('failed_finalized', 'expired_unincluded'):
            # A proven terminal non-submission does NOT count as paid. Preserve
            # the old chain cursor; the writer may now attempt the current hour.
            terminal = dict(cursor, status=decision['status'], recovered=True,
                            reconciliation_sha256=digest(receipt))
            durable_atomic(directory / 'terminal-cursor.json', terminal)
            durable_atomic(self.cursor_path, terminal)
            return decision
        status_path = self.chain_state / 'weights.json'
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        resolved = intent['window_end']
        if state.get('last_submitted_window', -1) <= resolved:
            state['last_submitted_window'] = max(state.get('last_submitted_window', -1), resolved)
            state['latest'] = dict(status='submitted', window_end=resolved,
                recovered=True, assessment_sha256=intent['assessment_sha256'],
                reconciliation_sha256=digest(receipt), block_hash=decision['commit']['block_hash'],
                extrinsic_hash=decision['commit']['extrinsic_hash'])
            durable_atomic(status_path, state)
        # State first: a crash between writes only causes another read-only recovery.
        durable_atomic(self.cursor_path, dict(cursor, status='submitted', recovered=True,
                        reconciliation_sha256=digest(receipt)))
        return decision
